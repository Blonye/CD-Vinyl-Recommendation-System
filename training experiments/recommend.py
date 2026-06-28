"""
recommend.py — Two-Tower CD & Vinyl 交互式推荐演示
====================================================
从项目根目录运行：
    python recommend.py

启动时间：约 30 秒（预计算向量）
单次查询：< 0.1 秒（CPU）

特殊命令：
    'sample'  — 从数据集中随机取一个用户
    'exit'    — 退出
"""

import sys, os, json, time
import numpy as np
import torch
import joblib

# ── 路径设置 ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from model_twotower import TwoTowerModel

MODELS_DIR    = os.path.join(ROOT, 'models')
METADATA_PATH = os.path.join(ROOT, 'data', 'meta_CDs_and_Vinyl.jsonl')
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── 1. 加载 ASIN → 标题映射 ──────────────────────────────────────────────────
def load_title_map(path):
    """从 JSONL 元数据中读取 parent_asin → title 映射。"""
    print("Loading metadata titles...", end=' ', flush=True)
    title_map = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin  = d.get('parent_asin', '')
            title = d.get('title', '')
            if asin:
                title_map[asin] = title.strip() if title else '(no title)'
    print(f"done  ({len(title_map):,} entries)")
    return title_map


def get_item_title(asin):
    """给定 ASIN，返回商品标题；找不到则返回 ASIN 本身。"""
    return title_map.get(asin, asin)


# ── 2. 加载已保存的模型组件 ───────────────────────────────────────────────────
print("=" * 58)
print("   Two-Tower CD & Vinyl Recommendation System")
print("=" * 58)

title_map    = load_title_map(METADATA_PATH)

print("Loading encoders and model...", end=' ', flush=True)
user_enc         = joblib.load(os.path.join(MODELS_DIR, 'user_enc.pkl'))
item_enc         = joblib.load(os.path.join(MODELS_DIR, 'item_enc.pkl'))
train_binary     = joblib.load(os.path.join(MODELS_DIR, 'train_binary.pkl'))
item_feat_tensor = torch.load(os.path.join(MODELS_DIR, 'item_feat_tensor.pt'),
                               map_location='cpu')
feature_dim      = int(open(os.path.join(MODELS_DIR, 'feature_dim.txt')).read())

n_users = len(user_enc.classes_)
n_items = len(item_enc.classes_)

# 从权重推断 embed_dim，避免硬编码
_state    = torch.load(os.path.join(MODELS_DIR, 'tt_model.pth'), map_location='cpu')
embed_dim = _state['user_tower.id_emb.weight'].shape[1]

model = TwoTowerModel(n_users, n_items, feature_dim, embed_dim).to(DEVICE)
model.load_state_dict(_state)
model.eval()
del _state  # 释放内存

print(f"done  (users={n_users:,}  items={n_items:,}  "
      f"feat_dim={feature_dim}  embed_dim={embed_dim})")


# ── 3. 预计算所有物品向量 ALL_ITEM_VECS ───────────────────────────────────────
# 分 batch 计算避免 OOM；89k 物品约需 10-20 秒
print("Precomputing item vectors...", end=' ', flush=True)
t0, BATCH = time.time(), 4096
parts = []
with torch.no_grad():
    for start in range(0, n_items, BATCH):
        end   = min(start + BATCH, n_items)
        ids   = torch.arange(start, end, device=DEVICE)
        feats = item_feat_tensor[start:end].to(DEVICE)
        parts.append(model.item_tower(ids, feats).cpu())
ALL_ITEM_VECS = torch.cat(parts, dim=0)   # (n_items, D)，已 L2 归一化
print(f"done in {time.time()-t0:.1f}s  {tuple(ALL_ITEM_VECS.shape)}")


# ── 4. 预计算所有用户历史向量 ────────────────────────────────────────────────
# 稀疏矩阵乘法：train_binary (n_users×n_items) @ id_emb (n_items×D)
# 比逐用户循环快 ~100x，整体约 1-2 秒
print("Precomputing user history vectors...", end=' ', flush=True)
t0 = time.time()
with torch.no_grad():
    item_id_emb = model.item_tower.id_emb.weight.detach().cpu().numpy()  # (n_items, D)

row_sums       = np.array(train_binary.sum(axis=1)).flatten()             # (n_users,)
row_sums_safe  = np.maximum(row_sums, 1.0)
hist_sum       = train_binary.dot(item_id_emb)                            # (n_users, D)
USER_HIST_VECS = torch.tensor(
    hist_sum / row_sums_safe[:, None], dtype=torch.float32
)  # (n_users, D)
print(f"done in {time.time()-t0:.1f}s  {tuple(USER_HIST_VECS.shape)}")
print()


# ── 5. 推荐函数 ───────────────────────────────────────────────────────────────
def recommend(user_raw_id, top_k=10):
    """
    为给定用户返回 top_k 个推荐，格式为 [(asin, title), ...]。
    用户不存在时返回 None。
    """
    if user_raw_id not in user_enc.classes_:
        return None

    user_id = int(user_enc.transform([user_raw_id])[0])

    with torch.no_grad():
        hist_vec = USER_HIST_VECS[user_id].unsqueeze(0).to(DEVICE)     # (1, D)
        u_id_t   = torch.tensor([user_id], dtype=torch.long, device=DEVICE)
        u_vec    = model.encode_users(u_id_t, hist_vec).cpu()           # (1, D)

    # 点积打分：(1, D) × (D, n_items) → (n_items,)
    scores = (u_vec @ ALL_ITEM_VECS.T).squeeze(0).numpy()

    # 排除已购物品
    scores[train_binary[user_id].indices] = -np.inf

    top_ids   = np.argsort(-scores)[:top_k]
    top_asins = item_enc.inverse_transform(top_ids)
    return [(asin, get_item_title(asin)) for asin in top_asins]


def show_history(user_raw_id, n=3):
    """展示用户最多 n 条已购物品（辅助演示）。"""
    user_id  = int(user_enc.transform([user_raw_id])[0])
    seen_ids = train_binary[user_id].indices
    if len(seen_ids) == 0:
        return
    print(f"  Purchase history sample ({len(seen_ids)} total items):")
    for iid in seen_ids[:n]:
        asin  = item_enc.inverse_transform([iid])[0]
        title = get_item_title(asin)
        title_short = (title[:52] + '…') if len(title) > 52 else title
        print(f"    {asin}  {title_short}")
    print()


# ── 6. 交互式循环 ─────────────────────────────────────────────────────────────
print("Ready!  Enter a user ID to get Top-10 recommendations.")
print("Commands:  'sample' = random user   |   'exit' = quit\n")

while True:
    try:
        raw = input("User ID › ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break

    if not raw:
        continue

    if raw.lower() == 'exit':
        print("Goodbye!")
        break

    if raw.lower() == 'sample':
        raw = str(np.random.choice(user_enc.classes_))
        print(f"  → Random user: {raw}")

    t0   = time.time()
    recs = recommend(raw, top_k=10)
    ms   = (time.time() - t0) * 1000

    if recs is None:
        print(f"  [!] User '{raw}' not found in training data.\n")
        continue

    show_history(raw, n=3)

    print(f"  Top-10 Recommendations  ({ms:.1f} ms)\n")
    print(f"  {'#':>2}  {'ASIN':<12}  Title")
    print(f"  {'─'*2}  {'─'*12}  {'─'*50}")
    for i, (asin, title) in enumerate(recs, 1):
        title_disp = (title[:52] + '…') if len(title) > 52 else title
        print(f"  {i:>2}. {asin:<12}  {title_disp}")
    print()
