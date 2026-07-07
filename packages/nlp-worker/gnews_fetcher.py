"""
gnews_fetcher.py v2
===================
Khusus mem-fetch Google News RSS dengan jeda (delay) agar tidak kena block.
Dijalankan via GitHub Actions, menggantikan peran Supabase untuk GNews.

FIX v2:
  1. Support Atom Feed (<entry>) selain RSS (<item>).
  2. Parse published_at dari <pubDate>.
  3. Chunking insert (50 item per RPC) agar aman dari limit payload Supabase.
  4. Include text_hash dari title (anti-error constraint).
"""

import os
import sys
import re
import time
import hashlib
import argparse
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
    """Ekstrak isi tag XML, handle CDATA."""
    patterns = [
        rf'<{tag}[^>]*><!\[CDATA\[([\s\S]*?)\]\]></{tag}>',
        rf'<{tag}[^>]*>([\s\S]*?)</{tag}>'
    ]
    for p in patterns:
        m = re.search(p, raw, re.IGNORECASE)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return None

def parse_rss(xml: str, config_name: str, entity_id: str) -> list[dict]:
    items = []
    # FIX 1: Support <item> dan <entry>
    matches = re.findall(r'<(?:item|entry)[^>]*>([\s\S]*?)<\/(?:item|entry)>', xml, re.IGNORECASE)
    
    for raw in matches:
        title = extract_tag(raw, 'title')
        link = extract_tag(raw, 'link') or extract_tag(raw, 'id')
        guid = extract_tag(raw, 'guid') or link
        desc = extract_tag(raw, 'description')
        pub_date_raw = extract_tag(raw, 'pubDate') or extract_tag(raw, 'published')
        
        if not title or not link: continue
        
        # FIX 2: Parse published_at
        published_at = None
        if pub_date_raw:
            try:
                dt = datetime.strptime(pub_date_raw, '%a, %d %b %Y %H:%M:%S %Z')
                published_at = dt.isoformat()
            except ValueError:
                # Fallback jika format RFC822 bermasalah
                pass
                
        text_content = re.sub(r'<[^>]+>', '', desc or title).strip()
        
        items.append({
            "source": config_name,
            "source_id": guid,
            "title": title,
            "source_url": link,
            "text": text_content,
            "text_hash": hashlib.sha256(title.encode()).hexdigest(), # FIX: Tambah text_hash
            "metadata": {"configured_entity_id": entity_id},
            "published_at": published_at
        })
    return items

def insert_chunked(sb: Client, items: list[dict]) -> int:
    """Insert ke Supabase dengan chunk 50 item per RPC."""
    inserted = 0
    chunk_size = 50
    for i in range(0, len(items), chunk_size):
        chunk = items[i:i + chunk_size]
        try:
            res = sb.rpc("batch_insert_raw_texts", {"p_items": chunk}).execute()
            row = (res.data or [{}])[0]
            inserted += row.get("inserted_count", 0)
        except Exception as e:
            print(f"     [RPC_ERROR] {e}")
    return inserted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Batasi jumlah config yang diproses (0=semua)")
    args = parser.parse_args()

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

    if args.limit > 0:
        configs = configs[:args.limit]

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
                        # FIX 3: Insert secara chunked
                        ins_count = insert_chunked(sb, items)
                        print(f"     ✅ {ins_count} artikel baru dimasukkan (dari {len(items)} item).")
                    else:
                        print("     ℹ️ 0 artikel diparse.")
                else:
                    print(f"     ❌ HTTP {r.status_code}")
            except Exception as e:
                print(f"     [ERROR] {e}")
            
            time.sleep(3) # JEDA 3 DETIK AGAR TIDAK KENA BLOCK GOOGLE

    print("[GNEWS] Selesai.")

if __name__ == "__main__":
    main()