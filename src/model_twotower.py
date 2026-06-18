import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_reviews, load_metadata, encode_and_split, build_item_features
from evaluate import hit_rate_and_ndcg


# ---------- 1. 用户历史索引 ----------
def build_user_history(train_df):
    """返回 dict: user_id -> list of item_ids"""
    return train_df.groupby('user_id')['item_id'].apply(list).to_dict()


# ---------- 2. 硬负样本索引（改进版：按流行度排序，避免遗漏重要物品）----------
def build_hard_negative_index(train_df, meta_df, item_enc,
                              n_hard=50,
                              max_items_per_group=500,
                              max_neighbors_per_item=30,
                              use_popularity=True):
    """
    为每个物品预计算流派/艺术家邻居，按流行度排序后取 top K。
    
    参数:
        train_df: 训练集（含 user_id, item_id, rating）—— 用于计算流行度
        meta_df: 元数据（含 item, brand, categories）
        item_enc: LabelEncoder 已拟合 item
        n_hard: 每个物品最终保留的硬负样本数量（去重后）
        max_items_per_group: 每个流派/品牌组最多保留的物品数量（按流行度取前 N）
        max_neighbors_per_item: 每个物品从流派/品牌中各取多少个候选（最后合并去重）
        use_popularity: 是否按流行度排序，False 则随机采样
    """
    from collections import defaultdict
    n_items = len(item_enc.classes_)
    known   = set(item_enc.classes_)
    meta    = meta_df[meta_df['item'].isin(known)].copy()
    meta['item_id'] = item_enc.transform(meta['item'])
    
    # 计算每个物品的流行度（训练集中的交互次数）
    item_pop = train_df.groupby('item_id').size().to_dict()
    meta['pop'] = meta['item_id'].map(item_pop).fillna(0).astype(int)
    
    hard_neg_dict = defaultdict(list)
    
    # ---------- 流派邻居（按主类别分组） ----------
    def get_primary_genre(cats):
        if isinstance(cats, list) and len(cats) > 0:
            return cats[0].lower().strip()
        return 'unknown'
    
    meta['primary_genre'] = meta['categories'].apply(get_primary_genre)
    
    for genre, group in meta.groupby('primary_genre'):
        if genre == 'unknown':
            continue
        if use_popularity:
            # 按流行度降序排序，取前 max_items_per_group 个
            group = group.sort_values('pop', ascending=False).head(max_items_per_group)
        else:
            if len(group) > max_items_per_group:
                group = group.sample(max_items_per_group, random_state=42)
        items = group['item_id'].values
        if len(items) < 2:
            continue
        for item_id in items:
            others = [x for x in items if x != item_id]
            if len(others) > max_neighbors_per_item:
                others = others[:max_neighbors_per_item]
            hard_neg_dict[item_id].extend(others)
    
    # ---------- 艺术家/厂牌邻居 ----------
    for brand, group in meta.groupby('brand'):
        if brand == 'unknown':
            continue
        if use_popularity:
            group = group.sort_values('pop', ascending=False).head(max_items_per_group)
        else:
            if len(group) > max_items_per_group:
                group = group.sample(max_items_per_group, random_state=42)
        items = group['item_id'].values
        if len(items) < 2:
            continue
        for item_id in items:
            others = [x for x in items if x != item_id]
            if len(others) > max_neighbors_per_item:
                others = others[:max_neighbors_per_item]
            hard_neg_dict[item_id].extend(others)
    
    # ---------- 去重并截断到 n_hard ----------
    result = {}
    filled = 0
    for i in range(n_items):
        unique = list(dict.fromkeys(hard_neg_dict[i]))[:n_hard]
        result[i] = np.array(unique, dtype=np.int64)
        if len(unique) > 0:
            filled += 1
    
    avg_cands = np.mean([len(v) for v in result.values()])
    print(f"Hard negative index built: "
          f"{filled}/{n_items} items have candidates "
          f"(avg {avg_cands:.1f} per item)")
    return result


# ---------- 3. Dataset ----------
class PairDataset(Dataset):
    def __init__(self, train_df, user_history, max_hist=50):
        self.users        = train_df['user_id'].values
        self.pos_items    = train_df['item_id'].values
        self.user_history = user_history
        self.max_hist     = max_hist

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u   = int(self.users[idx])
        pos = int(self.pos_items[idx])
        hist = self.user_history.get(u, [])
        hist = [h for h in hist if h != pos]
        if len(hist) == 0:
            hist = [pos]
        if len(hist) > self.max_hist:
            hist = np.random.choice(hist, self.max_hist, replace=False).tolist()
        hist_len    = len(hist)
        hist_padded = hist + [0] * (self.max_hist - hist_len)
        return (torch.tensor(u,           dtype=torch.long),
                torch.tensor(pos,         dtype=torch.long),
                torch.tensor(hist_padded, dtype=torch.long),
                torch.tensor(hist_len,    dtype=torch.long))


# ---------- 4. 模型定义 ----------
class UserTower(nn.Module):
    def __init__(self, n_users, embed_dim=64):
        super().__init__()
        self.id_emb    = nn.Embedding(n_users, embed_dim)
        self.hist_proj = nn.Linear(embed_dim, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, embed_dim)
        )

    def forward(self, user_ids, hist_vecs):
        u = self.id_emb(user_ids)
        h = self.hist_proj(hist_vecs)
        return F.normalize(self.mlp(torch.cat([u, h], dim=-1)), dim=-1)


class ItemTower(nn.Module):
    def __init__(self, n_items, feature_dim, embed_dim=64):
        super().__init__()
        self.id_emb   = nn.Embedding(n_items, embed_dim)
        self.feat_mlp = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, embed_dim)
        )
        self.fusion = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, item_ids, item_feats):
        id_vec   = self.id_emb(item_ids)
        feat_vec = self.feat_mlp(item_feats)
        return F.normalize(self.fusion(torch.cat([id_vec, feat_vec], dim=-1)), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_items, feature_dim, embed_dim=64):
        super().__init__()
        self.user_tower = UserTower(n_users, embed_dim)
        self.item_tower = ItemTower(n_items, feature_dim, embed_dim)

    def encode_users(self, user_ids, hist_vecs):
        return self.user_tower(user_ids, hist_vecs)

    def encode_items(self, item_ids, item_feats):
        return self.item_tower(item_ids, item_feats)


# ---------- 5. Loss 函数 ----------
def info_nce_loss(user_vecs, item_vecs, temperature=0.07):
    logits = user_vecs @ item_vecs.T / temperature
    labels = torch.arange(len(user_vecs), device=user_vecs.device)
    return F.cross_entropy(logits, labels)


def mixed_info_nce_loss(user_vecs, pos_item_vecs, hard_neg_vecs, temperature=0.07):
    B = user_vecs.shape[0]
    inbatch_logits = user_vecs @ pos_item_vecs.T / temperature
    hard_logits = torch.bmm(
        user_vecs.unsqueeze(1),
        hard_neg_vecs.transpose(1, 2)
    ).squeeze(1) / temperature
    combined = torch.cat([inbatch_logits, hard_logits], dim=1)
    labels = torch.arange(B, device=user_vecs.device)
    return F.cross_entropy(combined, labels)


# ---------- 6. 训练函数 ----------
def train_two_tower(data, item_feat_tensor, feature_dim,
                    embed_dim=64, epochs=20, batch_size=1024, lr=1e-3,
                    temperature=0.07,
                    warmup_epochs=5,
                    n_hard_neg=8,
                    hard_neg_index=None,
                    device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    use_hard = (hard_neg_index is not None)
    print(f"Training Two-Tower on {device}")
    if use_hard:
        print(f"  Warm-up:    epochs 1-{warmup_epochs}  (random in-batch negatives)")
        print(f"  Hard-neg:   epochs {warmup_epochs+1}-{epochs}  (K={n_hard_neg} genre/artist negatives added)")
    else:
        print(f"  Strategy: InfoNCE with in-batch negatives only")

    n_users = data['n_users']
    n_items = data['n_items']
    item_feat_tensor = item_feat_tensor.to(device)

    model = TwoTowerModel(n_users, n_items, feature_dim, embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    user_history = build_user_history(data['train_df'])
    dataset = PairDataset(data['train_df'], user_history)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0, drop_last=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        use_hard_this_epoch = use_hard and (epoch >= warmup_epochs)

        for user_ids, pos_ids, hist_ids, hist_lens in loader:
            user_ids = user_ids.to(device)
            pos_ids = pos_ids.to(device)
            hist_ids = hist_ids.to(device)
            hist_lens = hist_lens.to(device)
            B = user_ids.shape[0]

            # Masked mean pooling of history item embeddings
            with torch.no_grad():
                hist_embs = model.item_tower.id_emb(hist_ids)
                mask = (torch.arange(hist_ids.size(1), device=device)
                        .unsqueeze(0) < hist_lens.unsqueeze(1)).float()
                hist_vecs = ((hist_embs * mask.unsqueeze(-1)).sum(1) /
                              mask.sum(1, keepdim=True).clamp(min=1))

            pos_feats = item_feat_tensor[pos_ids]
            user_vecs = model.encode_users(user_ids, hist_vecs)
            item_vecs = model.encode_items(pos_ids, pos_feats)

            if use_hard_this_epoch:
                # 为 batch 内每个正样本查找 K 个硬负样本
                hard_ids_np = np.zeros((B, n_hard_neg), dtype=np.int64)
                for b, pos_id in enumerate(pos_ids.cpu().numpy()):
                    candidates = hard_neg_index.get(int(pos_id), np.array([], dtype=np.int64))
                    if len(candidates) >= n_hard_neg:
                        chosen = np.random.choice(candidates, n_hard_neg, replace=False)
                    elif len(candidates) > 0:
                        chosen = np.random.choice(candidates, n_hard_neg, replace=True)
                    else:
                        chosen = np.random.randint(0, n_items, n_hard_neg)
                    hard_ids_np[b] = chosen

                hard_ids = torch.tensor(hard_ids_np, dtype=torch.long, device=device)
                hard_feats = item_feat_tensor[hard_ids.view(-1)].view(B, n_hard_neg, -1)
                hard_vecs = model.encode_items(
                    hard_ids.view(-1),
                    hard_feats.view(-1, feature_dim)
                ).view(B, n_hard_neg, -1)

                loss = mixed_info_nce_loss(user_vecs, item_vecs, hard_vecs, temperature)
            else:
                loss = info_nce_loss(user_vecs, item_vecs, temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        phase = "HardNeg" if use_hard_this_epoch else "WarmUp"
        print(f"Epoch {epoch+1:02d}/{epochs} [{phase}] | Loss: {avg_loss:.4f}")

    return model


# ---------- 7. 评分函数（带用户历史缓存）----------
def create_score_fn(model, item_feat_tensor, data, device='cpu'):
    model.eval()
    item_feat_tensor = item_feat_tensor.to(device)
    user_history = build_user_history(data['train_df'])

    user_hist_vec_cache = {}
    with torch.no_grad():
        for uid, hist_list in user_history.items():
            if len(hist_list) == 0:
                hist_vec = torch.zeros(model.user_tower.hist_proj.in_features, device=device)
            else:
                hist_ids = torch.tensor(hist_list[:50], dtype=torch.long, device=device)
                hist_embs = model.item_tower.id_emb(hist_ids)
                hist_vec = hist_embs.mean(dim=0)
            user_hist_vec_cache[uid] = hist_vec

    def score_fn(user_id, item_ids):
        with torch.no_grad():
            hist_vec = user_hist_vec_cache.get(
                user_id,
                torch.zeros(model.user_tower.hist_proj.in_features, device=device)
            ).unsqueeze(0)
            u_ids = torch.tensor([user_id], dtype=torch.long, device=device)
            u_vec = model.encode_users(u_ids, hist_vec)
            i_ids = torch.tensor(item_ids, dtype=torch.long, device=device)
            i_vecs = model.encode_items(i_ids, item_feat_tensor[i_ids])
            return (u_vec * i_vecs).sum(dim=-1).cpu().numpy()

    return score_fn


# ---------- 8. 推荐函数 ----------
def recommend_two_tower(model, user_id, item_feat_tensor, train_binary,
                        item_enc, data, top_k=10, device='cpu'):
    model.eval()
    item_feat_tensor = item_feat_tensor.to(device)
    user_history = build_user_history(data['train_df'])

    with torch.no_grad():
        hist_list = user_history.get(user_id, [])
        if hist_list:
            hist_ids = torch.tensor(hist_list[:50], dtype=torch.long, device=device)
            hist_vec = model.item_tower.id_emb(hist_ids).mean(0, keepdim=True)
        else:
            hist_vec = torch.zeros(model.user_tower.hist_proj.in_features,
                                   device=device).unsqueeze(0)

        u_ids = torch.tensor([user_id], dtype=torch.long, device=device)
        u_vec = model.encode_users(u_ids, hist_vec)
        all_ids = torch.arange(item_feat_tensor.shape[0], device=device)
        i_vecs = model.encode_items(all_ids, item_feat_tensor)
        scores = (u_vec * i_vecs).sum(dim=-1).cpu().numpy()

    scores[train_binary[user_id].indices] = -np.inf
    return item_enc.inverse_transform(np.argsort(-scores)[:top_k])


# ---------- 9. 单独运行（训练并评估）----------
if __name__ == '__main__':
    # 注意：这里的路径要根据你的实际位置调整
    REVIEWS_PATH = 'data/CDs_and_Vinyl.csv.gz'
    METADATA_PATH = 'data/meta_CDs_and_Vinyl.jsonl'

    print("Loading data...")
    reviews_df = load_reviews(REVIEWS_PATH)
    meta_df = load_metadata(METADATA_PATH)
    data = encode_and_split(reviews_df)

    print("Building item features...")
    item_feat_tensor, feature_dim = build_item_features(
        data['train_df'], meta_df, data['item_enc']
    )

    print("Building hard negative index (with popularity)...")
    hard_neg_index = build_hard_negative_index(
        train_df=data['train_df'],
        meta_df=meta_df,
        item_enc=data['item_enc'],
        n_hard=50,
        max_items_per_group=500,
        max_neighbors_per_item=30,
        use_popularity=True
    )

    print("Starting training...")
    model = train_two_tower(
        data, item_feat_tensor, feature_dim,
        epochs=20,
        batch_size=1024,
        embed_dim=64,
        lr=1e-3,
        warmup_epochs=5,
        n_hard_neg=8,
        hard_neg_index=hard_neg_index
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    score_fn = create_score_fn(model, item_feat_tensor, data, device)
    results = hit_rate_and_ndcg(
        score_fn, data['test_df'], data['n_items'],
        K=10, user_seen=data['user_seen']
    )
    print("Two-Tower (hard neg) evaluation (100-neg):", results)

    # 可选：保存模型
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/tt_model_hard.pth')
    print("Model saved to models/tt_model_hard.pth")