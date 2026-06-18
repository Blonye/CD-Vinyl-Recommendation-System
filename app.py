"""
app.py — Two-Tower CD & Vinyl 推荐系统 · Gradio 演示界面（含封面图）
=====================================================================
安装依赖（只需一次）：
    pip install gradio plotly

从项目根目录运行：
    python app.py
"""

import sys, os, json, time
import numpy as np
import torch
import joblib
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from model_twotower import TwoTowerModel

MODELS_DIR    = os.path.join(ROOT, 'models')
METADATA_PATH = os.path.join(ROOT, 'data', 'meta_CDs_and_Vinyl.jsonl')
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

PLACEHOLDER_IMG = (
    "https://via.placeholder.com/120x120/1a1a2e/ffffff?"
    "text=No+Image"
)


# ════════════════════════════════════════════════════════════════════════════════
# 启动加载
# ════════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  Loading Two-Tower Recommendation System...")
print("=" * 60)

# ── 1. 元数据（title + categories + image）────────────────────────────────────
print("  [1/5] Loading metadata...", end=' ', flush=True)
title_map = {}
cat_map   = {}
img_map   = {}   # ASIN → image URL

with open(METADATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue

        asin  = d.get('parent_asin', '') or d.get('asin', '')
        if not asin:
            continue

        title_map[asin] = (d.get('title', '') or '(no title)').strip()
        cats             = d.get('categories', [])
        cat_map[asin]    = ', '.join(cats[:3]) if isinstance(cats, list) else ''

        # ── 图片 URL（兼容 2023 新版和 2018 旧版字段名）──────────────────────
        url = ''
        imgs = d.get('images', [])
        if isinstance(imgs, list) and len(imgs) > 0:
            im = imgs[0]
            # 2023 格式: {"thumb":..., "large":..., "hi_res":..., "variant":...}
            url = (im.get('large', '')
                   or im.get('hi_res', '')
                   or im.get('thumb', ''))
        if not url:
            # 2018 旧版格式 fallback
            url = (d.get('imageURLHighRes', '')
                   or d.get('imageURL', ''))
        img_map[asin] = url or PLACEHOLDER_IMG

print(f"done  ({len(title_map):,} entries, "
      f"{sum(1 for v in img_map.values() if v != PLACEHOLDER_IMG):,} with images)")

# ── 2. 编码器 & 模型 ──────────────────────────────────────────────────────────
print("  [2/5] Loading encoders and model...", end=' ', flush=True)
user_enc         = joblib.load(os.path.join(MODELS_DIR, 'user_enc.pkl'))
item_enc         = joblib.load(os.path.join(MODELS_DIR, 'item_enc.pkl'))
train_binary     = joblib.load(os.path.join(MODELS_DIR, 'train_binary.pkl'))
item_feat_tensor = torch.load(os.path.join(MODELS_DIR, 'item_feat_tensor.pt'),
                               map_location='cpu')
feature_dim      = int(open(os.path.join(MODELS_DIR, 'feature_dim.txt')).read())
n_users = len(user_enc.classes_)
n_items = len(item_enc.classes_)

_state    = torch.load(os.path.join(MODELS_DIR, 'tt_model.pth'), map_location='cpu')
embed_dim = _state['user_tower.id_emb.weight'].shape[1]
model     = TwoTowerModel(n_users, n_items, feature_dim, embed_dim).to(DEVICE)
model.load_state_dict(_state)
model.eval()
del _state
print(f"done  (users={n_users:,}  items={n_items:,}  embed={embed_dim}d)")

# ── 3. 物品向量 ───────────────────────────────────────────────────────────────
print("  [3/5] Loading item vectors...", end=' ', flush=True)
npy_path = os.path.join(ROOT, 'item_vectors.npy')
if os.path.exists(npy_path):
    item_vectors = np.load(npy_path).astype(np.float32)
    item_asins   = np.load(os.path.join(ROOT, 'item_asins.npy'), allow_pickle=True)
    print(f"loaded  {item_vectors.shape}")
else:
    print("computing...", end=' ', flush=True)
    parts, BATCH = [], 4096
    with torch.no_grad():
        for s in range(0, n_items, BATCH):
            e     = min(s + BATCH, n_items)
            ids   = torch.arange(s, e, device=DEVICE)
            feats = item_feat_tensor[s:e].to(DEVICE)
            parts.append(model.item_tower(ids, feats).cpu().numpy())
    item_vectors = np.vstack(parts).astype(np.float32)
    item_asins   = item_enc.classes_
    np.save(npy_path, item_vectors)
    np.save(os.path.join(ROOT, 'item_asins.npy'), item_asins)
    print(f"done  {item_vectors.shape}")

ALL_ITEM_VECS = torch.tensor(item_vectors)
asin_to_idx   = {asin: i for i, asin in enumerate(item_asins)}

# ── 4. 用户历史向量 ───────────────────────────────────────────────────────────
print("  [4/5] Precomputing user history vectors...", end=' ', flush=True)
with torch.no_grad():
    id_emb_np = model.item_tower.id_emb.weight.detach().cpu().numpy()
row_sums       = np.array(train_binary.sum(axis=1)).flatten()
hist_sum       = train_binary.dot(id_emb_np)
USER_HIST_VECS = (hist_sum / np.maximum(row_sums, 1.0)[:, None]).astype(np.float32)
print(f"done  {USER_HIST_VECS.shape}")

print("  [5/5] All components ready.")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════════════════
# HTML 卡片渲染
# ════════════════════════════════════════════════════════════════════════════════

CARD_CSS = """
<style>
.card-grid {
    display: flex; flex-wrap: wrap; gap: 12px;
    padding: 8px 0;
}
.card {
    width: 140px; background: #ffffff;
    border-radius: 10px; overflow: hidden;
    box-shadow: 0 2px 10px rgba(0,0,0,0.12);
    font-family: sans-serif; color: #1a1a1a;
    transition: transform 0.15s, box-shadow 0.15s;
    border: 1px solid #e8e8e8;
}
.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 18px rgba(0,0,0,0.15);
}
.card img {
    width: 140px; height: 140px;
    object-fit: cover; display: block;
    background: #f0f0f0;
}
.card-body { padding: 7px 8px 9px; background: #ffffff; }
.card-rank {
    font-size: 10px; color: #999; margin-bottom: 2px; font-weight: 500;
}
.card-title {
    font-size: 11px; font-weight: 700;
    line-height: 1.35; max-height: 3em;
    overflow: hidden; display: -webkit-box;
    -webkit-line-clamp: 3; -webkit-box-orient: vertical;
    color: #111111;
}
.card-asin  { font-size: 9px; color: #999; margin-top: 4px; }
.card-score { font-size: 10px; color: #4a5cf6; margin-top: 2px; font-weight: 600; }
.card-cats  {
    font-size: 9px; color: #666; margin-top: 3px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.hist-card { background: #f0f7f0; border-color: #c3dfc3; }
.hist-card .card-body { background: #f0f7f0; }
.hist-card .card-title { color: #1a3d1a; }
.hist-card .card-rank  { color: #4a7c4a; }
</style>
"""

def make_cards(items, show_score=True, hist_style=False):
    """
    items: list of dict with keys asin, title, cats, score (optional), rank
    返回 HTML 字符串
    """
    extra_cls = "hist-card" if hist_style else ""
    cards_html = ""
    for it in items:
        img_url    = img_map.get(it['asin'], PLACEHOLDER_IMG)
        title_safe = it['title'].replace('<', '&lt;').replace('>', '&gt;')
        cats_safe  = it['cats'].replace('<', '&lt;').replace('>', '&gt;')
        score_html = (f'<div class="card-score">Score: {it["score"]}</div>'
                      if show_score and 'score' in it else '')
        cards_html += f"""
        <div class="card {extra_cls}">
            <img src="{img_url}"
                 onerror="this.src='{PLACEHOLDER_IMG}'"
                 loading="lazy" alt="{title_safe}">
            <div class="card-body">
                <div class="card-rank">#{it['rank']}</div>
                <div class="card-title">{title_safe}</div>
                <div class="card-asin">{it['asin']}</div>
                {score_html}
                <div class="card-cats">{cats_safe}</div>
            </div>
        </div>"""
    return CARD_CSS + f'<div class="card-grid">{cards_html}</div>'


# ════════════════════════════════════════════════════════════════════════════════
# 核心推荐 / 相似度函数
# ════════════════════════════════════════════════════════════════════════════════

def recommend_for_user(user_raw_id, top_k=10):
    user_raw_id = user_raw_id.strip()
    if not user_raw_id:
        return "<p>⚠️ Please enter a User ID.</p>", "<p></p>", ""

    if user_raw_id not in user_enc.classes_:
        return (f"<p>❌ User <b>{user_raw_id}</b> not found.</p>",
                "<p></p>", "")

    user_id = int(user_enc.transform([user_raw_id])[0])
    t0      = time.time()

    hist_vec = torch.tensor(USER_HIST_VECS[user_id]).unsqueeze(0).to(DEVICE)
    u_id_t   = torch.tensor([user_id], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        u_vec = model.encode_users(u_id_t, hist_vec).cpu()

    scores = (u_vec @ ALL_ITEM_VECS.T).squeeze(0).numpy()
    scores[train_binary[user_id].indices] = -np.inf
    top_ids   = np.argsort(-scores)[:top_k]
    top_asins = item_enc.inverse_transform(top_ids)
    ms        = (time.time() - t0) * 1000

    # 推荐卡片
    rec_items = [{
        'rank':  i + 1,
        'asin':  asin,
        'title': title_map.get(asin, '(no title)'),
        'cats':  cat_map.get(asin, ''),
        'score': f"{scores[top_ids[i]]:.4f}"
    } for i, asin in enumerate(top_asins)]
    recs_html = make_cards(rec_items, show_score=True)

    # 历史卡片
    seen_ids  = train_binary[user_id].indices
    hist_items = []
    for rank, iid in enumerate(seen_ids[:10], 1):
        asin = item_enc.inverse_transform([iid])[0]
        hist_items.append({
            'rank':  rank,
            'asin':  asin,
            'title': title_map.get(asin, '(no title)'),
            'cats':  cat_map.get(asin, ''),
        })
    hist_html = make_cards(hist_items, show_score=False, hist_style=True)

    status = (f"✅ &nbsp;User: <b>{user_raw_id}</b> &nbsp;|&nbsp; "
              f"Purchase history: <b>{len(seen_ids)}</b> items &nbsp;|&nbsp; "
              f"Query time: <b>{ms:.1f} ms</b>")
    return hist_html, recs_html, status


def random_user():
    uid = str(np.random.choice(user_enc.classes_))
    hist_html, recs_html, status = recommend_for_user(uid)
    return uid, hist_html, recs_html, status


def find_similar_items(query_asin, top_k=10):
    query_asin = query_asin.strip()
    if not query_asin:
        return "<p>⚠️ Please enter an ASIN.</p>", "<p></p>"

    if query_asin not in asin_to_idx:
        return f"<p>❌ ASIN <b>{query_asin}</b> not found.</p>", "<p></p>"

    idx       = asin_to_idx[query_asin]
    sim       = item_vectors @ item_vectors[idx]
    sim[idx]  = -np.inf
    top_ids   = np.argsort(-sim)[:top_k]

    q_title = title_map.get(query_asin, '(no title)')
    q_cats  = cat_map.get(query_asin, '')
    q_img   = img_map.get(query_asin, PLACEHOLDER_IMG)

    query_html = f"""
    {CARD_CSS}
    <div style="display:flex;align-items:flex-start;gap:16px;
                background:#1e1e2e;border-radius:10px;
                padding:14px;margin-bottom:12px">
        <img src="{q_img}"
             onerror="this.src='{PLACEHOLDER_IMG}'"
             style="width:100px;height:100px;object-fit:cover;
                    border-radius:8px;flex-shrink:0">
        <div>
            <div style="font-size:11px;color:#888">Query Item</div>
            <div style="font-size:15px;font-weight:700;
                        color:#fff;margin:4px 0">{q_title}</div>
            <div style="font-size:11px;color:#aaa">{query_asin}</div>
            <div style="font-size:11px;color:#aaa;margin-top:4px">{q_cats}</div>
        </div>
    </div>"""

    sim_items = [{
        'rank':  i + 1,
        'asin':  item_asins[j],
        'title': title_map.get(item_asins[j], '(no title)'),
        'cats':  cat_map.get(item_asins[j], ''),
        'score': f"{sim[j]:.4f}"
    } for i, j in enumerate(top_ids)]

    return query_html, make_cards(sim_items, show_score=True)


def random_item():
    candidates = [a for a in item_asins
                  if img_map.get(a, PLACEHOLDER_IMG) != PLACEHOLDER_IMG]
    asin = str(np.random.choice(candidates[:3000]))
    q_html, sim_html = find_similar_items(asin)
    return asin, q_html, sim_html


# ════════════════════════════════════════════════════════════════════════════════
# 评估指标图表
# ════════════════════════════════════════════════════════════════════════════════

def build_metrics_chart():
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Model Comparison (100-Negative Protocol)",
            "Evaluation Protocol Impact (Two-Tower)"
        ),
        horizontal_spacing=0.15
    )
    models_     = ["BPR", "Two-Tower"]
    hr_100neg   = [0.5905, 0.6227]
    ndcg_100neg = [0.3959, 0.4140]
    colors      = ["#636EFA", "#EF553B"]

    fig.add_trace(go.Bar(
        name="HR@10", x=models_, y=hr_100neg,
        marker_color=colors, text=[f"{v:.4f}" for v in hr_100neg],
        textposition='outside', offsetgroup=0
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        name="NDCG@10", x=models_, y=ndcg_100neg,
        marker_color=colors, text=[f"{v:.4f}" for v in ndcg_100neg],
        textposition='outside', offsetgroup=1,
        marker_pattern_shape="/", opacity=0.75
    ), row=1, col=1)

    protocols  = ["100-Neg<br>(Standard)", "Full Ranking<br>(Real-world)"]
    hr_proto   = [0.6227, 0.024]
    ndcg_proto = [0.4140, 0.0147]
    pc         = ["#EF553B", "#00CC96"]

    fig.add_trace(go.Bar(
        name="HR@10", x=protocols, y=hr_proto,
        marker_color=pc, text=[f"{v:.4f}" for v in hr_proto],
        textposition='outside', offsetgroup=0, showlegend=False
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        name="NDCG@10", x=protocols, y=ndcg_proto,
        marker_color=pc, text=[f"{v:.4f}" for v in ndcg_proto],
        textposition='outside', offsetgroup=1,
        marker_pattern_shape="/", opacity=0.75, showlegend=False
    ), row=1, col=2)

    fig.update_layout(
        height=420, barmode='group',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=13),
        legend=dict(orientation='h', yanchor='bottom', y=1.05, x=0),
        margin=dict(t=80, b=40, l=40, r=40)
    )
    fig.update_yaxes(range=[0, 0.78], gridcolor='rgba(128,128,128,0.2)')
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# Gradio 界面
# ════════════════════════════════════════════════════════════════════════════════

with gr.Blocks(
    title="CD & Vinyl Recommender",
    theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    css=".gradio-container{max-width:1150px!important}"
) as demo:

    gr.Markdown("""
    # 🎵 CD & Vinyl Recommendation System
    **Two-Tower Neural Model** · Amazon 5-core CDs & Vinyl 2023
    · 1.55M reviews · 123,876 users · 89,370 items
    """)

    with gr.Tabs():

        # ── Tab 1: User Recommendations ──────────────────────────────────────
        with gr.TabItem("🎧 User Recommendations"):
            gr.Markdown("Enter a User ID (or click **Random User**) to see "
                        "purchase history and personalised Top-10 recommendations.")
            with gr.Row():
                uid_input  = gr.Textbox(label="User ID",
                                        placeholder="e.g. AH7GZG4G2THFYNF5WKFREJPBLSVQ",
                                        scale=4)
                btn_rec    = gr.Button("🔍  Recommend", variant="primary", scale=1)
                btn_random = gr.Button("🎲  Random User", scale=1)

            status_md = gr.HTML("<p style='color:#aaa'>Enter a user ID and click Recommend.</p>")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 📦 Purchase History (up to 10)")
                    hist_out = gr.HTML()
                with gr.Column(scale=1):
                    gr.Markdown("#### ⭐ Top-10 Recommendations")
                    recs_out = gr.HTML()

            btn_rec.click(
                fn=recommend_for_user,
                inputs=[uid_input],
                outputs=[hist_out, recs_out, status_md]
            )
            btn_random.click(
                fn=random_user,
                inputs=[],
                outputs=[uid_input, hist_out, recs_out, status_md]
            )

        # ── Tab 2: Similar Items ──────────────────────────────────────────────
        with gr.TabItem("🔗 Similar Items"):
            gr.Markdown("Enter an item ASIN to find the most similar items "
                        "by cosine similarity of their Two-Tower embeddings.")
            with gr.Row():
                asin_input   = gr.Textbox(label="Item ASIN",
                                          placeholder="e.g. B000002ANG",
                                          scale=4)
                btn_sim      = gr.Button("🔍  Find Similar", variant="primary", scale=1)
                btn_rand_itm = gr.Button("🎲  Random Item", scale=1)

            query_out = gr.HTML()
            sim_out   = gr.HTML()

            btn_sim.click(
                fn=find_similar_items,
                inputs=[asin_input],
                outputs=[query_out, sim_out]
            )
            btn_rand_itm.click(
                fn=random_item,
                inputs=[],
                outputs=[asin_input, query_out, sim_out]
            )

        # ── Tab 3: Model Performance ──────────────────────────────────────────
        with gr.TabItem("📊 Model Performance"):
            gr.Markdown("""
### Evaluation Results

| Protocol | Description |
|---|---|
| **100-Negative** | Rank test item among 100 random negatives + 1 positive (standard) |
| **Full Ranking** | Rank test item among all 89,370 items (real-world) |

> The large gap between protocols reflects that random negatives are "easy" to
> distinguish. Full-ranking is a much harder and realistic setting
> (Krichene & Rendle, 2020).
            """)

            gr.Plot(value=build_metrics_chart())

            gr.Markdown("""
| Model | Protocol | HR@10 | NDCG@10 |
|---|---|---|---|
| BPR | 100-Negative | 0.5905 | 0.3959 |
| **Two-Tower** | **100-Negative** | **0.6227** | **0.4140** |
| Two-Tower | Full Ranking (1k sample) | 0.024 | 0.0147 |

Two-Tower outperforms BPR by **+5.4% HR@10**, by combining collaborative signals
(user/item ID embeddings) with content features (genre, artist, price)
and user history pooling via InfoNCE contrastive loss.
            """)

    gr.Markdown("---\n*Amazon Reviews 2023 · CDs & Vinyl · "
                "Two-Tower Model with InfoNCE Loss + History Pooling*")

if __name__ == '__main__':
    demo.launch(inbrowser=True)
