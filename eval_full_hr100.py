import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import load_reviews, load_metadata, encode_and_split, build_item_features
from evaluate import hit_rate_and_ndcg
from model_twotower import TwoTowerModel, create_score_fn

# 加载数据
reviews_df = load_reviews('data/CDs_and_Vinyl.csv.gz')
meta_df    = load_metadata('data/meta_CDs_and_Vinyl.jsonl')
data       = encode_and_split(reviews_df)
item_feat_tensor, feature_dim = build_item_features(data['train_df'], meta_df, data['item_enc'])

# 加载旧模型（最佳模型）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TwoTowerModel(data['n_users'], data['n_items'], feature_dim, embed_dim=64).to(device)
model.load_state_dict(torch.load('models/tt_model.pth', map_location=device))
model.eval()

score_fn = create_score_fn(model, item_feat_tensor, data, device)

# 抽样 1000 用户，全量排名，K=100
sample_test = data['test_df'].sample(1000, random_state=42)
print("评估全量排名 HR@100 ...")
results = hit_rate_and_ndcg(
    score_fn, sample_test, data['n_items'],
    K=100,                           # ← 改这里
    user_seen=data['user_seen'],
    n_negatives=data['n_items']      # 全量
)
print("全量 HR@100:", results['HR@100'], "NDCG@100:", results['NDCG@100'])