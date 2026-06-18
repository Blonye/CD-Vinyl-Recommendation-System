import numpy as np
import implicit
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_reviews, encode_and_split
from evaluate import hit_rate_and_ndcg


def train_bpr(data, factors=64, lr=0.01, reg=0.01, iterations=100):
    """
    Train BPR using the `implicit` library.
    Expects a (items × users) binary matrix.
    """
    model = implicit.bpr.BayesianPersonalizedRanking(
        factors=factors,
        learning_rate=lr,
        regularization=reg,
        iterations=iterations,
        verify_negative_samples=True,
        random_state=42
    )

    model.fit(data['train_binary'])
    return model


def bpr_score_fn(model):
    """Wraps model into the score_fn interface expected by evaluate.py"""
    user_vecs = model.user_factors   # (n_users, factors)
    item_vecs = model.item_factors   # (n_items, factors)

    def fn(user_id, item_ids):
        u = user_vecs[user_id]         # (factors,)
        i = item_vecs[item_ids]        # (len(item_ids), factors)
        return i @ u                   # dot product → (len(item_ids),)

    return fn


def recommend_top_k(model, user_id, train_binary, item_enc, k=10):
    """
    Returns top-K recommended ASINs for a given internal user_id integer.
    Filters out items the user has already purchased.
    """
    scores = model.user_factors[user_id] @ model.item_factors.T  # (n_items,)
    seen   = train_binary[user_id].indices
    scores[seen] = -np.inf
    top_ids = np.argsort(-scores)[:k]
    return item_enc.inverse_transform(top_ids)   # returns ASINs


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df   = load_reviews('data/CDs_and_Vinyl.csv.gz')
    data = encode_and_split(df)

    model   = train_bpr(data)
    score_fn = bpr_score_fn(model)

    results = hit_rate_and_ndcg(
        score_fn, data['test_df'], data['n_items'],
        K=10, user_seen=data['user_seen']
    )
    print("BPR:", results)

    # Recommend for the first user in the dataset
    recs = recommend_top_k(model, 0, data['train_binary'], data['item_enc'])
    print("Top-10 ASINs for user 0:", recs)
