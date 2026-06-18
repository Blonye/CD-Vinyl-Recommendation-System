import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import load_reviews, load_metadata, encode_and_split, build_item_features
from evaluate import hit_rate_and_ndcg
from model_bpr import train_bpr, bpr_score_fn
from model_twotower import train_two_tower, create_score_fn   # 注意这里变了

REVIEWS_PATH = 'data/CDs_and_Vinyl.csv.gz'
METADATA_PATH = 'data/meta_CDs_and_Vinyl.jsonl'
TEST_MODE = True

print("="*50)
print("STEP 1: Loading data")
print("="*50)
reviews_df = load_reviews(REVIEWS_PATH)
meta_df = load_metadata(METADATA_PATH)
data = encode_and_split(reviews_df)

# 构建物品特征（增强版）
item_feat_tensor, feature_dim = build_item_features(
    data['train_df'], meta_df, data['item_enc']
)

results = {}

# BPR 训练
print("\n" + "="*50)
print("STEP 2: Training BPR")
print("="*50)
bpr_model = train_bpr(data, iterations=10 if TEST_MODE else 100)
results['BPR'] = hit_rate_and_ndcg(
    bpr_score_fn(bpr_model),
    data['test_df'], data['n_items'],
    K=10, user_seen=data['user_seen']
)
print("BPR done:", results['BPR'])

# Two-Tower 训练（使用 InfoNCE + User History）
print("\n" + "="*50)
print("STEP 3: Training Two-Tower")
print("="*50)
epochs = 2 if TEST_MODE else 20
batch_size = 512 if TEST_MODE else 1024
tt_model = train_two_tower(
    data, item_feat_tensor, feature_dim,
    epochs=epochs, batch_size=batch_size, embed_dim=64, lr=1e-3
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 使用新的评分函数创建器
score_fn = create_score_fn(tt_model, item_feat_tensor, data, device)
results['Two-Tower'] = hit_rate_and_ndcg(
    score_fn,
    data['test_df'], data['n_items'],
    K=10, user_seen=data['user_seen']
)
print("Two-Tower done:", results['Two-Tower'])

# 输出结果
print("\n" + "="*50)
print(f"FINAL RESULTS ({'TEST MODE' if TEST_MODE else 'FULL RUN'})")
print("="*50)
print(f"{'Model':<14}  {'HR@10':>8}  {'NDCG@10':>10}")
print("-"*36)
for name, r in results.items():
    print(f"{name:<14}  {r['HR@10']:>8.4f}  {r['NDCG@10']:>10.4f}")

if TEST_MODE:
    print("\n*** TEST MODE: scores will be low — set TEST_MODE = False for real results ***")
