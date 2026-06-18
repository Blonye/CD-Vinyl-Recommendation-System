import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_reviews, load_metadata, encode_and_split, build_item_features
from evaluate import hit_rate_and_ndcg

# ---------- 1. 用户历史索引构建（供训练和评估使用）----------
def build_user_history(train_df):
    """返回 dict: user_id -> list of item_ids (训练集中所有交互，不区分顺序)"""
    return train_df.groupby('user_id')['item_id'].apply(list).to_dict()

# ---------- 2. Dataset（返回 user_id, pos_item_id, history_ids）----------
class PairDataset(Dataset):
    def __init__(self, train_df, user_history, max_hist=50):
        self.users = train_df['user_id'].values
        self.pos_items = train_df['item_id'].values
        self.user_history = user_history
        self.max_hist = max_hist

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = int(self.users[idx])
        pos = int(self.pos_items[idx])
        hist = self.user_history.get(u, [])
        # 移除当前正样本，避免信息泄露
        hist = [h for h in hist if h != pos]
        if len(hist) == 0:
            hist = [pos]  # fallback: 用自己作历史（其实不应发生，但避免空）
        # 随机采样或截断
        if len(hist) > self.max_hist:
            hist = np.random.choice(hist, self.max_hist, replace=False).tolist()
        # padding
        hist_len = len(hist)
        hist_padded = hist + [0] * (self.max_hist - hist_len)
        return (torch.tensor(u, dtype=torch.long),
                torch.tensor(pos, dtype=torch.long),
                torch.tensor(hist_padded, dtype=torch.long),
                torch.tensor(hist_len, dtype=torch.long))

# ---------- 3. 模型定义 ----------
class UserTower(nn.Module):
    def __init__(self, n_users, embed_dim=64):
        super().__init__()
        self.id_emb = nn.Embedding(n_users, embed_dim)
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
        self.id_emb = nn.Embedding(n_items, embed_dim)
        self.feat_mlp = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, embed_dim)
        )
        self.fusion = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, item_ids, item_feats):
        id_vec = self.id_emb(item_ids)
        feat_vec = self.feat_mlp(item_feats)
        fused = torch.cat([id_vec, feat_vec], dim=-1)
        return F.normalize(self.fusion(fused), dim=-1)

class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_items, feature_dim, embed_dim=64):
        super().__init__()
        self.user_tower = UserTower(n_users, embed_dim)
        self.item_tower = ItemTower(n_items, feature_dim, embed_dim)

    def encode_users(self, user_ids, hist_vecs):
        return self.user_tower(user_ids, hist_vecs)

    def encode_items(self, item_ids, item_feats):
        return self.item_tower(item_ids, item_feats)

# ---------- 4. InfoNCE Loss ----------
def info_nce_loss(user_vecs, item_vecs, temperature=0.07):
    logits = user_vecs @ item_vecs.T / temperature
    labels = torch.arange(len(user_vecs), device=user_vecs.device)
    return F.cross_entropy(logits, labels)

# ---------- 5. 训练函数 ----------
def train_two_tower(data, item_feat_tensor, feature_dim,
                    embed_dim=64, epochs=20, batch_size=1024, lr=1e-3,
                    temperature=0.07, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Two-Tower with InfoNCE + User History on {device}")

    n_users = data['n_users']
    n_items = data['n_items']
    item_feat_tensor = item_feat_tensor.to(device)

    model = TwoTowerModel(n_users, n_items, feature_dim, embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    user_history = build_user_history(data['train_df'])
    dataset = PairDataset(data['train_df'], user_history)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for user_ids, pos_ids, hist_ids, hist_lens in loader:
            user_ids = user_ids.to(device)
            pos_ids = pos_ids.to(device)
            hist_ids = hist_ids.to(device)
            hist_lens = hist_lens.to(device)

            # 计算历史 item 的 ID embedding 并做 masked mean pooling
            with torch.no_grad():
                # 获取历史物品的 ID embedding（item_tower.id_emb 是协同信号）
                hist_embs = model.item_tower.id_emb(hist_ids)  # (B, max_hist, D)
                mask = (torch.arange(hist_ids.size(1), device=device).unsqueeze(0) < hist_lens.unsqueeze(1)).float()
                hist_vecs = (hist_embs * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

            pos_feats = item_feat_tensor[pos_ids]  # (B, feature_dim)

            user_vecs = model.encode_users(user_ids, hist_vecs)
            item_vecs = model.encode_items(pos_ids, pos_feats)

            loss = info_nce_loss(user_vecs, item_vecs, temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f}")

    return model

# ---------- 6. 高效的评分函数（预计算用户历史向量缓存）----------
def create_score_fn(model, item_feat_tensor, data, device='cpu'):
    """
    返回一个 score_fn(user_id, item_ids) 函数。
    预先计算所有用户的历史向量缓存，避免重复计算。
    """
    model.eval()
    item_feat_tensor = item_feat_tensor.to(device)
    user_history = build_user_history(data['train_df'])

    # 预计算每个用户的平均历史向量（使用 item_tower 的 ID embedding）
    user_hist_vec_cache = {}
    with torch.no_grad():
        for uid, hist_list in user_history.items():
            if len(hist_list) == 0:
                # 冷启动：用零向量（或可训练的可学习向量，这里简单用零）
                hist_vec = torch.zeros(model.user_tower.hist_proj.in_features, device=device)
            else:
                # 取最多50个（与训练一致）
                hist_ids = torch.tensor(hist_list[:50], dtype=torch.long, device=device)
                hist_embs = model.item_tower.id_emb(hist_ids)  # (L, D)
                hist_vec = hist_embs.mean(dim=0)
            user_hist_vec_cache[uid] = hist_vec

    def score_fn(user_id, item_ids):
        with torch.no_grad():
            # 用户向量
            uid = user_id
            if uid in user_hist_vec_cache:
                hist_vec = user_hist_vec_cache[uid].unsqueeze(0)  # (1, D)
            else:
                # 未知用户（不在训练集中）：使用全局平均历史向量或零向量
                hist_vec = torch.zeros(model.user_tower.hist_proj.in_features, device=device).unsqueeze(0)
            u_ids = torch.tensor([uid], dtype=torch.long, device=device)
            u_vec = model.encode_users(u_ids, hist_vec)  # (1, D)

            # 物品向量
            i_ids = torch.tensor(item_ids, dtype=torch.long, device=device)
            i_feats = item_feat_tensor[i_ids]
            i_vecs = model.encode_items(i_ids, i_feats)   # (N, D)
            scores = (u_vec * i_vecs).sum(dim=-1).cpu().numpy()
            return scores
    return score_fn

# ---------- 7. 推荐函数 ----------
def recommend_two_tower(model, user_id, item_feat_tensor, train_binary,
                        item_enc, data, top_k=10, device='cpu'):
    model.eval()
    item_feat_tensor = item_feat_tensor.to(device)
    user_history = build_user_history(data['train_df'])

    with torch.no_grad():
        # 计算用户历史向量
        hist_list = user_history.get(user_id, [])
        if len(hist_list) > 0:
            hist_ids = torch.tensor(hist_list[:50], dtype=torch.long, device=device)
            hist_embs = model.item_tower.id_emb(hist_ids)
            hist_vec = hist_embs.mean(dim=0, keepdim=True)
        else:
            hist_vec = torch.zeros(model.user_tower.hist_proj.in_features, device=device).unsqueeze(0)

        u_ids = torch.tensor([user_id], dtype=torch.long, device=device)
        u_vec = model.encode_users(u_ids, hist_vec)

        n_items = item_feat_tensor.shape[0]
        all_ids = torch.arange(n_items, device=device)
        i_vecs = model.encode_items(all_ids, item_feat_tensor)
        scores = (u_vec * i_vecs).sum(dim=-1).cpu().numpy()

    seen = train_binary[user_id].indices
    scores[seen] = -np.inf
    top_ids = np.argsort(-scores)[:top_k]
    return item_enc.inverse_transform(top_ids)

# ---------- 8. 直接运行测试 ----------
if __name__ == '__main__':
    # 假设数据文件路径正确
    REVIEWS_PATH = '../data/CDs_and_Vinyl.csv.gz'
    METADATA_PATH = '../data/meta_CDs_and_Vinyl.jsonl'

    reviews_df = load_reviews(REVIEWS_PATH)
    meta_df = load_metadata(METADATA_PATH)
    data = encode_and_split(reviews_df)

    item_feat_tensor, feature_dim = build_item_features(
        data['train_df'], meta_df, data['item_enc']
    )

    model = train_two_tower(data, item_feat_tensor, feature_dim, epochs=20)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    score_fn = create_score_fn(model, item_feat_tensor, data, device)
    results = hit_rate_and_ndcg(
        score_fn, data['test_df'], data['n_items'],
        K=10, user_seen=data['user_seen']
    )
    print("Two-Tower results:", results)

    recs = recommend_two_tower(model, 0, item_feat_tensor,
                               data['train_binary'], data['item_enc'], data)
    print("Top-10 ASINs for user 0:", recs)