import sys
import os
import torch
import gzip
import joblib

# 文件路径
REVIEWS_PATH  = 'data/CDs_and_Vinyl.csv.gz'
METADATA_PATH = 'data/meta_CDs_and_Vinyl.jsonl'

if not os.path.exists(METADATA_PATH):
    gz_path = METADATA_PATH + '.gz'
    if os.path.exists(gz_path):
        print("解压 metadata...")
        with gzip.open(gz_path, 'rb') as fin:
            with open(METADATA_PATH, 'wb') as fout:
                fout.write(fin.read())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import load_reviews, load_metadata, encode_and_split, build_item_features
from evaluate import hit_rate_and_ndcg
from model_bpr import train_bpr, bpr_score_fn
# from model_lightfm import build_lightfm_data, train_lightfm, lfm_score_fn   # 已注释
from model_twotower import train_two_tower, create_score_fn   # 注意：使用 create_score_fn



# 全量训练参数
print("=" * 50)
print("STEP 1: Loading data")
print("=" * 50)
reviews_df = load_reviews(REVIEWS_PATH)
meta_df    = load_metadata(METADATA_PATH)
data       = encode_and_split(reviews_df)

# 构建物品特征（增强版，已在 preprocess.py 中实现）
item_feat_tensor, feature_dim = build_item_features(
    data['train_df'], meta_df, data['item_enc']
)

results = {}

# ---- BPR 训练 ----
print("\n" + "=" * 50)
print("STEP 2: Training BPR")
print("=" * 50)
bpr_model = train_bpr(data, iterations=100)   # 全量 100 轮
results['BPR'] = hit_rate_and_ndcg(
    bpr_score_fn(bpr_model),
    data['test_df'], data['n_items'],
    K=10, user_seen=data['user_seen']
)
print("BPR done:", results['BPR'])

# ---- LightFM 训练（已注释） ----
# print("\n" + "=" * 50)
# print("STEP 3: Training LightFM")
# print("=" * 50)
# dataset, interactions, weights, item_feat_matrix = build_lightfm_data(
#     reviews_df, meta_df
# )
# lfm_model = train_lightfm(interactions, item_feat_matrix, weights, loss='warp')
# results['LightFM'] = hit_rate_and_ndcg(
#     lfm_score_fn(lfm_model, dataset, item_feat_matrix,
#                  data['user_enc'], data['item_enc']),
#     data['test_df'], data['n_items'],
#     K=10, user_seen=data['user_seen']
# )
# print("LightFM done:", results['LightFM'])

# ---- Two‑Tower 训练（优化版） ----
print("\n" + "=" * 50)
print("STEP 4: Training Two-Tower")
print("=" * 50)
tt_model = train_two_tower(
    data, item_feat_tensor, feature_dim,
    epochs=20,          # 全量 20 轮
    batch_size=1024,
    embed_dim=64,
    lr=1e-3
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
score_fn = create_score_fn(tt_model, item_feat_tensor, data, device)
results['Two-Tower'] = hit_rate_and_ndcg(
    score_fn,
    data['test_df'], data['n_items'],
    K=10, user_seen=data['user_seen']
)
print("Two-Tower done:", results['Two-Tower'])

# 保存模型与组件
os.makedirs('models', exist_ok=True)
torch.save(tt_model.state_dict(), 'models/tt_model.pth')
torch.save(item_feat_tensor, 'models/item_feat_tensor.pt')
joblib.dump(data['user_enc'], 'models/user_enc.pkl')
joblib.dump(data['item_enc'], 'models/item_enc.pkl')
joblib.dump(data['train_binary'], 'models/train_binary.pkl')
with open('models/feature_dim.txt', 'w') as f:
    f.write(str(feature_dim))
print("模型已保存到 models/ 目录")

# ---- 最终结果表 ----
print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)
print(f"{'Model':<14}  {'HR@10':>8}  {'NDCG@10':>10}")
print("-" * 36)
for name, r in results.items():
    print(f"{name:<14}  {r['HR@10']:>8.4f}  {r['NDCG@10']:>10.4f}")