import numpy as np


def hit_rate_and_ndcg(score_fn, test_df, n_items, K=10, n_negatives=100,
                      user_seen=None, seed=42):
    """
    Leave-one-out evaluation (standard for this dataset).

    For each test user:
      1. Take their held-out last-purchased item.
      2. Sample 100 random negatives (items they never bought).
      3. Score all 101 candidates via score_fn.
      4. HR@K  = 1 if test item appears in top-K ranked results; else 0.
      5. NDCG@K = 1/log2(rank+2) if hit; else 0.

    Args:
        score_fn    : callable(user_id: int, item_ids: np.array) -> scores: np.array
        test_df     : DataFrame with columns user_id, item_id
        n_items     : total number of items
        K           : cutoff rank (default 10)
        n_negatives : number of random negatives to sample (default 100)
        user_seen   : dict {user_id: set(item_ids)} — excludes seen items from negatives
        seed        : random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    all_items = np.arange(n_items)
    hits, ndcgs = [], []

    for _, row in test_df.iterrows():
        user_id   = int(row['user_id'])
        test_item = int(row['item_id'])

        # Build negative pool — exclude seen items and test item
        # np.setdiff1d runs in C speed: ~1000x faster than a Python list comprehension
        seen     = user_seen.get(user_id, set()) if user_seen else set()
        seen_arr = np.array(list(seen) + [test_item], dtype=np.int64)
        pool     = np.setdiff1d(all_items, seen_arr)

        negs       = rng.choice(pool, size=min(n_negatives, len(pool)), replace=False)
        candidates = np.append(negs, test_item)   # test item always last

        scores = score_fn(user_id, candidates)
        ranked = candidates[np.argsort(-scores)]  # descending by score

        hit = int(test_item in ranked[:K])
        hits.append(hit)

        if hit:
            rank = int(np.where(ranked[:K] == test_item)[0][0])
            ndcgs.append(1.0 / np.log2(rank + 2))
        else:
            ndcgs.append(0.0)

    return {
        f'HR@{K}':   round(float(np.mean(hits)),  4),
        f'NDCG@{K}': round(float(np.mean(ndcgs)), 4),
        'n_users':   len(hits)
    }