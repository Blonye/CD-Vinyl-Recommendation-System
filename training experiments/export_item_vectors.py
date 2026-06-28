"""
export_item_vectors.py — 导出 Two-Tower 物品向量
==================================================
从项目根目录运行：
    python export_item_vectors.py

输出文件：
    item_vectors.npy  — shape (n_items, embed_dim)，已 L2 归一化
    item_asins.npy    — shape (n_items,)，ASIN 字符串，顺序与向量一一对应

因为向量已经 L2 归一化，余弦相似度 = 点积：
    sim = item_vectors @ item_vectors[i]   # 物品 i 与所有物品的相似度

脚本末尾附带一个简单的相似商品查询示例。
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


# ── 1. 加载模型组件 ───────────────────────────────────────────────────────────
print("Loading model components...", end=' ', flush=True)

user_enc         = joblib.load(os.path.join(MODELS_DIR, 'user_enc.pkl'))
item_enc         = joblib.load(os.path.join(MODELS_DIR, 'item_enc.pkl'))
item_feat_tensor = torch.load(os.path.join(MODELS_DIR, 'item_feat_tensor.pt'),
                               map_location='cpu')
feature_dim      = int(open(os.path.join(MODELS_DIR, 'feature_dim.txt')).read())

n_users = len(user_enc.classes_)
n_items = len(item_enc.classes_)

_state    = torch.load(os.path.join(MODELS_DIR, 'tt_model.pth'), map_location='cpu')
embed_dim = _state['user_tower.id_emb.weight'].shape[1]

model = TwoTowerModel(n_users, n_items, feature_dim, embed_dim).to(DEVICE)
model.load_state_dict(_state)
model.eval()
del _state

print(f"done  (n_items={n_items:,}  feat_dim={feature_dim}  embed_dim={embed_dim})")


# ── 2. 分 batch 计算所有物品向量 ──────────────────────────────────────────────
print("Computing item vectors...", end=' ', flush=True)
t0, BATCH = time.time(), 4096
parts = []

with torch.no_grad():
    for start in range(0, n_items, BATCH):
        end   = min(start + BATCH, n_items)
        ids   = torch.arange(start, end, device=DEVICE)
        feats = item_feat_tensor[start:end].to(DEVICE)
        parts.append(model.item_tower(ids, feats).cpu().numpy())

item_vectors = np.vstack(parts).astype(np.float32)   # (n_items, D)
item_asins   = item_enc.classes_                       # (n_items,) ASIN 字符串数组

print(f"done in {time.time()-t0:.1f}s  shape={item_vectors.shape}")


# ── 3. 保存 ───────────────────────────────────────────────────────────────────
np.save('item_vectors.npy', item_vectors)
np.save('item_asins.npy',   item_asins)
print(f"Saved: item_vectors.npy  +  item_asins.npy")


# ── 4. 加载标题（用于可读的输出）────────────────────────────────────────────
print("Loading titles for sanity check...", end=' ', flush=True)
title_map = {}
with open(METADATA_PATH, 'r', encoding='utf-8') as f:
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
print("done")


# ── 5. 相似商品查询示例 ───────────────────────────────────────────────────────
def find_similar(query_asin, top_k=10):
    """
    给定一个 ASIN，返回最相似的 top_k 个商品。
    因为向量已 L2 归一化，余弦相似度 = 点积。
    """
    if query_asin not in title_map and query_asin not in set(item_asins):
        print(f"ASIN '{query_asin}' 不在物品集合中。")
        return

    # 找到该 ASIN 的内部 index
    asin_list = list(item_asins)
    if query_asin not in asin_list:
        print(f"ASIN '{query_asin}' 不在训练集物品中。")
        return

    idx       = asin_list.index(query_asin)
    query_vec = item_vectors[idx]                     # (D,)
    sim       = item_vectors @ query_vec               # (n_items,) 余弦相似度

    # 排除自身
    sim[idx] = -np.inf
    top_ids  = np.argsort(-sim)[:top_k]

    query_title = title_map.get(query_asin, query_asin)
    print(f"\nQuery: [{query_asin}]  {query_title[:60]}")
    print(f"Top-{top_k} similar items:\n")
    print(f"  {'#':>2}  {'ASIN':<12}  {'Sim':>6}  Title")
    print(f"  {'─'*2}  {'─'*12}  {'─'*6}  {'─'*50}")
    for rank, i in enumerate(top_ids, 1):
        asin  = item_asins[i]
        title = title_map.get(asin, '(no title)')
        title_disp = (title[:50] + '…') if len(title) > 50 else title
        print(f"  {rank:>2}. {asin:<12}  {sim[i]:>6.4f}  {title_disp}")


# ── 6. 自动运行两个示例 ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Sanity check — similar item examples")
print("=" * 60)

# 示例 1：数据集里的第一个物品
first_asin = item_asins[0]
find_similar(first_asin, top_k=5)

# 示例 2：数据集里的第 1000 个物品（更有代表性）
if len(item_asins) > 1000:
    find_similar(item_asins[1000], top_k=5)

find_similar('B0009WFELY', top_k=10)

print("\n" + "─" * 60)
print("要查询任意商品，在脚本末尾调用：")
print("    find_similar('B00XXXXXX', top_k=10)")
print("─" * 60)
