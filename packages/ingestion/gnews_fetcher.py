"""
gnews_fetcher.py v2 — Discovery Layer (Metadata Collector)
==========================================================
Mengambil RSS Google News per-tokoh.
Fungsi: Mengumpulkan Judul, Tanggal, dan Snippet.
URL Google News tidak akan di-fetch oleh Enricher (karena terenkripsi).
Data ini akan diproses oleh NLP Worker di Tier 2 (Snippet Only).
"""

import os
import sys
import re
import time
import hashlib
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

try:
    import httpx
    from supabase import create_client, Client
except ImportError:
    print("[ERROR] pip install httpx supabase python-dotenv"); sys.exit(1)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SERVICE_KEY  = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

def get_client() -> Client:
    return create_client(SUPABASE_URL, SERVICE_KEY)

def extract_tag(raw: str, tag: str) -> str | None:
    patterns = [
        rf'<{tag}[^>]*><!\[CDATA\[([\s\S]*?)\]\]></{tag}>',
        rf'<{tag}[^>]*>([\s\S]*?)</{tag}>'
    ]
    for p in patterns:
        m = re.search(p, raw, re.IGNORECASE)
        if m and m.group(1).strip(): return m.group(1).strip()
    return None

def parse_rss(xml: str, config_name: str, entity_id: str) -> list[dict]:
    items = []
    matches = re.findall(r'<(?:item|entry)[^>]*>([\s\S]*?)<\/(?:item|entry)>', xml, re.IGNORECASE)
    
    for raw in matches:
        title = extract_tag(raw, 'title')
        link = extract_tag(raw, 'link') or extract_tag(raw, 'id')
        guid = extract_tag(raw, 'guid') or link
        desc = extract_tag(raw, 'description')
        pub_date_raw = extract_tag(raw, 'pubDate') or extract_tag(raw, 'published')
        
        if not title or not link: continue
        
        published_at = None
        if pub_date_raw:
            try:
                dt = datetime.strptime(pub_date_raw, '%a, %d %b %Y %H:%M:%S %Z')
                published_at = dt.isoformat()
            except ValueError:
                pass
                
        text_content = re.sub(r'<[^>]+>', '', desc or title).strip()
        
        items.append({
            "source": config_name,
            "source_id": guid,
            "title": title,
            "source_url": link, # URL GNews (akan di-bypass Enricher)
            "text": text_content, # Snippet pendek
            "text_hash": hashlib.sha256(title.encode()).hexdigest(),
            "metadata": {"configured_entity_id": entity_id},
            "published_at": published_at
        })
    return items

def insert_chunked(sb: Client, items: list[dict]) -> int:
    inserted = 0
    for i in range(0, len(items), 50):
        chunk = items[i:i + 50]
        try:
            res = sb.rpc("batch_insert_raw_texts", {"p_items": chunk}).execute()
            row = (res.data or [{}])[0]
            inserted += row.get("inserted_count", 0)
        except Exception as e:
            print(f"     [RPC_ERROR] {e}")
    return inserted

def main():
    sb = get_client()
    res = sb.table("scraping_configs") \
            .select("id, entity_id, config_name, url") \
            .eq("is_active", True) \
            .eq("source_type", "google_news_rss") \
            .execute()
            
    configs = res.data or []
    if not configs:
        print("[GNEWS] Tidak ada config Google News.")
        return

    print(f"[GNEWS] Mem-fetch {len(configs)} feed dengan jeda 3 detik...")
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    with httpx.Client(headers=headers, timeout=15) as client:
        for cfg in configs:
            print(f"  -> {cfg['config_name']}")
            try:
                r = client.get(cfg["url"])
                if r.status_code == 200:
                    items = parse_rss(r.text, cfg["config_name"], cfg.get("entity_id"))
                    if items:
                        ins_count = insert_chunked(sb, items)
                        print(f"     ✅ {ins_count} artikel baru dimasukkan.")
                else:
                    print(f"     ❌ HTTP {r.status_code}")
            except Exception as e:
                print(f"     [ERROR] {e}")
            time.sleep(3)

if __name__ == "__main__":
    main()