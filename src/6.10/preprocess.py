import json
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder


# ── Load Reviews ──────────────────────────────────────────────────────────────
def load_reviews(path='data/CDs_and_Vinyl.csv.gz'):
    """
    Reads the 2023 5-core CSV (which is actually a .gz compressed file).
    pandas handles the decompression automatically — no extraction needed.

    Columns in the raw file:
        user_id | parent_asin | rating | timestamp

    Timestamp is in milliseconds in 2023 → divide by 1000 to get seconds.
    """
    df = pd.read_csv(path, compression='gzip')

    df = df.rename(columns={
        'user_id':      'user',
        'parent_asin':  'item',
        'rating':       'rating',
        'timestamp':    'timestamp'
    })

    # Convert milliseconds → seconds
    df['timestamp'] = (df['timestamp'] / 1000).astype(int)

    # 2023 5-core CSV has no verified or format fields — add placeholders
    df['verified'] = True
    df['format']   = 'unknown'

    print(f"Reviews loaded  : {len(df):,} rows | "
          f"{df['user'].nunique():,} users | "
          f"{df['item'].nunique():,} items")
    return df


# ── Load Metadata ─────────────────────────────────────────────────────────────
def load_metadata(path='data/meta_CDs_and_Vinyl.jsonl'):
    """
    Reads the 2023 metadata file.
    This file is a plain (uncompressed) JSONL — one JSON object per line.

    Key 2023 field names (different from 2018):
        parent_asin  →  item ID   (was 'asin' in 2018)
        store        →  artist    (was 'brand' in 2018)
        categories   →  flat list (was list-of-lists in 2018)
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

            records.append({
                'item':       d.get('parent_asin', ''),
                'brand':      d.get('store', 'unknown'),    # artist / label name
                'price':      d.get('price', None),
                # categories in 2023 is already a flat list: ["CDs & Vinyl", "Jazz"]
                'categories': ' '.join(d.get('categories', []))
                              if isinstance(d.get('categories'), list)
                              else ''
            })

    df = pd.DataFrame(records).drop_duplicates('item')
    print(f"Metadata loaded : {len(df):,} items")
    return df


# ── Encode IDs and Build Sparse Matrix ───────────────────────────────────────
def encode_and_split(df):
    """
    Leave-one-out split:
      - Sort each user's reviews by timestamp
      - Hold out the LAST review per user as the test item
      - Everything else is training data

    Returns a dict with everything needed by the three models.
    """
    df = df.sort_values(['user', 'timestamp']).reset_index(drop=True)

    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df['user_id'] = user_enc.fit_transform(df['user'])
    df['item_id'] = item_enc.fit_transform(df['item'])

    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()

    # Leave-one-out: last interaction per user → test set
    last_idx = df.groupby('user_id')['timestamp'].idxmax()
    test_df  = df.loc[last_idx].reset_index(drop=True)
    train_df = df.drop(last_idx).reset_index(drop=True)

    print(f"Train rows      : {len(train_df):,}")
    print(f"Test users      : {len(test_df):,}")
    print(f"Users / Items   : {n_users:,} / {n_items:,}")

    # Sparse matrix — values = rating
    train_matrix = csr_matrix(
        (train_df['rating'].values,
         (train_df['user_id'].values, train_df['item_id'].values)),
        shape=(n_users, n_items)
    )

    # Binary version: 1 wherever the user interacted (used by BPR)
    train_binary = (train_matrix > 0).astype(np.float32)

    # Per-user seen-item sets for fast negative sampling
    user_seen = (
        train_df.groupby('user_id')['item_id']
        .apply(set)
        .to_dict()
    )

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