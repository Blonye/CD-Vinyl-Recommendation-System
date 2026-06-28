

import sys
import os
import torch

# 确保能找到 src 目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 尝试导入所需的模块
try:
    from preprocess import load_reviews, load_metadata, encode_and_split, build_item_features
    from evaluate import hit_rate_and_ndcg
    from model_bpr import train_bpr, bpr_score_fn
    from model_twotower import train_two_tower, create_score_fn
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 src 目录下有 preprocess.py, evaluate.py, model_bpr.py, model_twotower.py")
    sys.exit(1)

# 数据文件路径 (相对于项目根目录)
REVIEWS_PATH = os.path.join(current_dir, 'data', 'CDs_and_Vinyl.csv.gz')
METADATA_PATH = os.path.join(current_dir, 'data', 'meta_CDs_and_Vinyl.jsonl')

# 如果 data 目录下没有 .jsonl 但有 .jsonl.gz，自动解压
if not os.path.exists(METADATA_PATH):
    gz_path = METADATA_PATH + '.gz'
    if os.path.exists(gz_path):
        import gzip
        print(f"发现压缩的元数据文件 {gz_path}，正在解压...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(METADATA_PATH, 'wb') as f_out:
                f_out.write(f_in.read())
        print("解压完成。")
    else:
        print(f"错误: 找不到元数据文件 {METADATA_PATH} 或 {gz_path}")
        sys.exit(1)

TEST_MODE = True   # 改为 False 进行全量训练

print("="*50)
print("STEP 1: Loading data")
print("="*50)

# 加载数据
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

# ── 验证是否有数据泄露 ──────────────────────────────────────────
print("\n" + "="*50)
print("数据泄露检查")
print("="*50)
leakage_count = 0
for _, row in data['test_df'].head(100).iterrows():
    uid = int(row['user_id'])
    test_item = int(row['item_id'])
    seen = data['user_seen'].get(uid, set())
    if test_item in seen:
        leakage_count += 1
print(f"结果: {leakage_count}/100 个测试物品出现在训练集中")
print("（正常应为 0，若不为 0 则存在数据泄露）")