"""
ddg_fetcher.py — Primary URL Provider (DuckDuckGo News)
========================================================
Mengambil berita per-tokoh dari DuckDuckGo News HTML.
Mengembalikan URL media asli (Detik, Kompas, dll) TANPA enkripsi redirect.
Data ini akan di-enrich menjadi Full Article oleh Enricher (Tier 1).
"""

import os
import sys
import time
import hashlib
from pathlib import Path
from urllib.parse import unquote
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

try:
    import httpx
    from bs4 import BeautifulSoup
    from supabase import create_client, Client
except ImportError:
    print("[ERROR] pip install httpx beautifulsoup4 supabase python-dotenv"); sys.exit(1)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SERVICE_KEY  = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

def get_client() -> Client:
    return create_client(SUPABASE_URL, SERVICE_KEY)

def fetch_ddg_news(entity_name: str) -> list[dict]:
    """Scrape DuckDuckGo News HTML untuk URL media asli."""
    url = f"https://duckduckgo.com/html/?q={entity_name.replace(' ', '+')}&kl=id-id"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"}
    
    try:
        r = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
        if r.status_code != 200: return []
            
        soup = BeautifulSoup(r.text, 'html.parser')
        items = []
        
        for res in soup.find_all('a', class_='result__a'):
            href = res.get('href')
            title = res.get_text(strip=True)
            
            if 'uddg=' in href:
                actual_url = href.split('uddg=')[1].split('&')[0]
                actual_url = unquote(actual_url)
            else:
                actual_url = href
                
            if actual_url.startswith('http') and 'duckduckgo.com' not in actual_url:
                items.append({"title": title, "url": actual_url})
                
        return items[:15] # Ambil 15 berita teratas per tokoh
        
    except Exception:
        return []

def main():
    sb = get_client()
    res = sb.table("scraping_configs") \
            .select("id, entity_id, config_name") \
            .eq("is_active", True) \
            .eq("source_type", "google_news_rss") \
            .execute()
            
    configs = res.data or []
    print(f"[DDG] Mem-fetch {len(configs)} tokoh dari DuckDuckGo News...")

    for cfg in configs:
        entity_name = cfg["config_name"].replace("gnews_", "").replace("_", " ")
        print(f"  -> Mencari: {entity_name}")
        
        articles = fetch_ddg_news(entity_name)
        if not articles:
            print("     ℹ️ 0 artikel ditemukan.")
            time.sleep(3)
            continue
            
        items_to_insert = []
        for art in articles:
            source_id = hashlib.sha256(art["url"].encode()).hexdigest()[:32]
            
            items_to_insert.append({
                "source": "ddg_" + cfg["config_name"],
                "source_id": source_id,
                "title": art["title"],
                "source_url": art["url"], # INI URL MEDIA ASLI!
                "text": art["title"],
                "text_hash": hashlib.sha256(art["title"].encode()).hexdigest(),
                "metadata": {"configured_entity_id": cfg.get("entity_id")},
                "published_at": None
            })
            
        try:
            # Insert batch ke Supabase
            for i in range(0, len(items_to_insert), 50):
                chunk = items_to_insert[i:i+50]
                sb.rpc("batch_insert_raw_texts", {"p_items": chunk}).execute()
            print(f"     ✅ {len(items_to_insert)} artikel URL asli dimasukkan.")
        except Exception as e:
            print(f"     [ERROR] {e}")
            
        time.sleep(5) # Jeda anti ban

if __name__ == "__main__":
    main()