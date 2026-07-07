"""
gnews_fetcher.py
================
Khusus mem-fetch Google News RSS dengan jeda (delay) agar tidak kena block.
Dijalankan via GitHub Actions, menggantikan peran Supabase untuk GNews.
"""
import os
import sys
import time
import re
import argparse
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[4]
load_dotenv(ROOT_DIR / ".env")

try:
    import httpx
    from supabase import create_client, Client
except ImportError:
    print("[ERROR] pip install httpx supabase"); sys.exit(1)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SERVICE_KEY  = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

def get_client() -> Client:
    return create_client(SUPABASE_URL, SERVICE_KEY)

def parse_rss(xml: str, config_name: str, entity_id: str) -> list[dict]:
    items = []
    # Regex sederhana untuk RSS
    matches = re.findall(r'<item[^>]*>([\s\S]*?)<\/item>', xml, re.IGNORECASE)
    for raw in matches:
        def extract(tag):
            m = re.search(rf'<{tag}[^>]*>(?:<!\[CDATA\[)?([\s\S]*?)(?:\]\]>)?<\/{tag}>', raw, re.IGNORECASE)
            return m.group(1).strip() if m else None
            
        title = extract('title')
        link = extract('link')
        guid = extract('guid') or link
        desc = extract('description')
        
        if not title or not link: continue
        
        items.append({
            "source": config_name,
            "source_id": guid,
            "title": title,
            "source_url": link,
            "text": re.sub(r'<[^>]+>', '', desc or title), # Bersihkan HTML di deskripsi
            "metadata": {"configured_entity_id": entity_id},
            "published_at": None
        })
    return items

def main():
    sb = get_client()
    # Ambil HANYA google_news_rss
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
                        # Insert batch ke Supabase
                        sb.rpc("batch_insert_raw_texts", {"p_items": items}).execute()
                        print(f"     ✅ {len(items)} artikel dimasukkan.")
                else:
                    print(f"     ❌ HTTP {r.status_code}")
            except Exception as e:
                print(f"     [ERROR] {e}")
            
            time.sleep(3) # JEDA 3 DETIK AGAR TIDAK KENA BLOCK GOOGLE

    print("[GNEWS] Selesai.")

if __name__ == "__main__":
    main()