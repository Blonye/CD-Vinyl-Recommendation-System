import json
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, MinMaxScaler
import torch

# ---------- 1. 加载评论数据 ----------
def load_reviews(path='data/CDs_and_Vinyl.csv.gz'):
    """
    读取 2023 5-core CSV（gzip 压缩）
    """
    df = pd.read_csv(path, compression='gzip')
    df = df.rename(columns={
        'user_id':     'user',
        'parent_asin': 'item',
        'rating':      'rating',
        'timestamp':   'timestamp'
    })
    df['timestamp'] = (df['timestamp'] / 1000).astype(int)
    # 添加占位列（兼容旧代码）
    df['verified'] = True
    df['format']   = 'unknown'
    print(f"Reviews loaded  : {len(df):,} rows | "
          f"{df['user'].nunique():,} users | "
          f"{df['item'].nunique():,} items")
    return df

# ---------- 2. 加载元数据 ----------
def load_metadata(path='data/meta_CDs_and_Vinyl.jsonl'):
    """
    读取 JSONL 元数据，categories 保留为列表
    """
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            cats = d.get('categories', [])
            if not isinstance(cats, list):
                cats = []
            records.append({
                'item':       d.get('parent_asin', ''),
                'brand':      d.get('store', 'unknown'),
                'price':      d.get('price', None),
                'categories': cats   # 保持列表形式
            })
    df = pd.DataFrame(records).drop_duplicates('item')
    print(f"Metadata loaded : {len(df):,} items")
    return df

# ---------- 3. 编码与留一法划分 ----------
def encode_and_split(df):
    """
    留一法划分：每个用户的最后一次交互作为测试集
    """
    df = df.sort_values(['user', 'timestamp']).reset_index(drop=True)
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df['user_id'] = user_enc.fit_transform(df['user'])
    df['item_id'] = item_enc.fit_transform(df['item'])
    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()

    # 留一法
    last_idx = df.groupby('user_id')['timestamp'].idxmax()
    test_df  = df.loc[last_idx].reset_index(drop=True)
    train_df = df.drop(last_idx).reset_index(drop=True)

    print(f"Train rows      : {len(train_df):,}")
    print(f"Test users      : {len(test_df):,}")
    print(f"Users / Items   : {n_users:,} / {n_items:,}")

    # 评分矩阵
    train_matrix = csr_matrix(
        (train_df['rating'].values,
         (train_df['user_id'].values, train_df['item_id'].values)),
        shape=(n_users, n_items)
    )
    train_binary = (train_matrix > 0).astype(np.float32)

    # 用户已购物品集合
    user_seen = (train_df.groupby('user_id')['item_id']
                       .apply(set).to_dict())

    return {
        'train_df':     train_df,
        'test_df':      test_df,
        'train_matrix': train_matrix,
        'train_binary': train_binary,
        'user_seen':    user_seen,
        'user_enc':     user_enc,
        'item_enc':     item_enc,
        'n_users':      n_users,
        'n_items':      n_items
    }

# ---------- 4. 增强版物品特征构建（优化版）----------
def build_item_features(train_df, meta_df, item_enc,
                        top_genres=50, top_artists=200):
    """
    构建物品特征：
      - 价格（归一化）
      - 平均评分（归一化）
      - log(评论数+1)（归一化）
      - 流派（前 8 个 token，保留 top_genres 个类别）
      - 艺术家（保留 top_artists 个品牌）
    """
    n_items = len(item_enc.classes_)
    known = set(item_enc.classes_)
    meta = meta_df[meta_df['item'].isin(known)].copy()
    meta['item_id'] = item_enc.transform(meta['item'])

    # ---------- 价格 ----------
    meta['price_num'] = pd.to_numeric(meta['price'], errors='coerce').fillna(0.0)
    price_scaler = MinMaxScaler()
    meta['price_norm'] = price_scaler.fit_transform(meta[['price_num']])

    # ---------- 平均评分 & 评论数 ----------
    item_stats = train_df.groupby('item_id')['rating'].agg(['mean', 'count']).reset_index()
    item_stats.columns = ['item_id', 'avg_rating', 'review_count']
    meta = meta.merge(item_stats, on='item_id', how='left')
    meta['avg_rating'] = meta['avg_rating'].fillna(3.0)
    meta['review_count'] = meta['review_count'].fillna(0.0)

    rating_scaler = MinMaxScaler()
    meta['avg_rating_norm'] = rating_scaler.fit_transform(meta[['avg_rating']])
    # 对评论数取 log1p 后再归一化
    meta['log_review_count'] = np.log1p(meta['review_count'])
    log_scaler = MinMaxScaler()
    meta['log_review_count_norm'] = log_scaler.fit_transform(meta[['log_review_count']])

    # ---------- 流派：每个物品取 categories 列表的前 8 个单词（去重）----------
    def extract_tokens(cat_list):
        tokens = []
        for cat in cat_list:
            for token in cat.lower().split():
                token = token.strip('.,!?;:()[]{}"\'')
                if token and len(token) > 2:
                    tokens.append(token)
        seen = set()
        uniq = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq[:8]

    meta['genre_tokens'] = meta['categories'].apply(extract_tokens)
    mlb = MultiLabelBinarizer()
    genre_mat = mlb.fit_transform(meta['genre_tokens'])
    top_idx = np.argsort(-genre_mat.sum(axis=0))[:top_genres]
    genre_mat = genre_mat[:, top_idx].astype(np.float32)

    # ---------- 艺术家 ----------
    top_artist_names = meta['brand'].value_counts().index[:top_artists].tolist()
    meta['brand_clean'] = meta['brand'].apply(
        lambda b: b.lower().strip() if b in top_artist_names else '__other__'
    )
    artist_dummies = pd.get_dummies(meta['brand_clean']).values.astype(np.float32)

    # ---------- 合并 ----------
    feat = np.hstack([
        meta['price_norm'].values.reshape(-1, 1),
        meta['avg_rating_norm'].values.reshape(-1, 1),
        meta['log_review_count_norm'].values.reshape(-1, 1),
        genre_mat,
        artist_dummies
    ]).astype(np.float32)

    feature_dim = feat.shape[1]
    matrix = np.zeros((n_items, feature_dim), dtype=np.float32)
    for _, row in meta.iterrows():
        matrix[row['item_id']] = feat[_]   # 使用索引

    return torch.tensor(matrix), feature_dim