import sys, os, torch, joblib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import load_reviews, load_metadata, encode_and_split, build_item_features
from model_twotower import build_hard_negative_index, train_two_tower, create_score_fn, TwoTowerModel
from evaluate import hit_rate_and_ndcg

# 加载数据
reviews_df = load_reviews('data/CDs_and_Vinyl.csv.gz')
meta_df = load_metadata('data/meta_CDs_and_Vinyl.jsonl')
data = encode_and_split(reviews_df)

# 物品特征
item_feat_tensor, feature_dim = build_item_features(data['train_df'], meta_df, data['item_enc'])

# 硬负样本索引
hard_neg_index = build_hard_negative_index(data['train_df'], meta_df, data['item_enc'])

# 训练
model = train_two_tower(data, item_feat_tensor, feature_dim, epochs=20, hard_neg_index=hard_neg_index)

# 保存模型
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/tt_model_hard.pth')
joblib.dump(data['user_enc'], 'models/user_enc.pkl')
joblib.dump(data['item_enc'], 'models/item_enc.pkl')
joblib.dump(data['train_binary'], 'models/train_binary.pkl')
print("Model saved.")

# 快速评估（100-neg）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
score_fn = create_score_fn(model, item_feat_tensor, data, device)
results = hit_rate_and_ndcg(score_fn, data['test_df'], data['n_items'], K=10, user_seen=data['user_seen'])
print("100-neg HR@10:", results['HR@10'])