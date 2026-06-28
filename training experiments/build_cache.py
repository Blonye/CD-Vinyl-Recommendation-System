# 本地运行一次，生成缓存（新建 build_cache.py）
import json, pickle
from pathlib import Path

title_map, cat_map, img_map = {}, {}, {}
PLACEHOLDER = "https://via.placeholder.com/120x120/1a1a2e/ffffff?text=No+Image"

with open('data/meta_CDs_and_Vinyl.jsonl', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except:
            continue
        asin = d.get('parent_asin', '') or d.get('asin', '')
        if not asin:
            continue
        title_map[asin] = (d.get('title', '') or '(no title)').strip()
        cats = d.get('categories', [])
        cat_map[asin] = ', '.join(cats[:3]) if isinstance(cats, list) else ''

        url = ''
        imgs = d.get('images', [])
        if isinstance(imgs, list) and imgs:
            im = imgs[0]
            url = im.get('large', '') or im.get('hi_res', '') or im.get('thumb', '')
        if not url:
            url = d.get('imageURLHighRes', '') or d.get('imageURL', '')
        img_map[asin] = url or PLACEHOLDER

with open('models/meta_cache.pkl', 'wb') as f:
    pickle.dump({'title': title_map, 'cat': cat_map, 'img': img_map}, f)

print(f"Cache built: {len(title_map):,} entries")

import os
cache_size = os.path.getsize('models/meta_cache.pkl') / (1024*1024)
print(f"Cache file size: {cache_size:.2f} MB")
# 文件大小预计 50–80 MB，远小于原始 JSONL