"""
ddg_fetcher.py v2 — Primary URL Provider (DuckDuckGo News)
============================================================
Mengambil berita per-tokoh dari DuckDuckGo News HTML.
Mengembalikan URL media asli (Detik, Kompas, dll) TANPA enkripsi redirect.

PERBAIKAN v2:
  1. MONOREPO READY: Import dari packages.shared.
  2. SAFE URL PARSING: Pakai urllib.parse.queryqs untuk ekstraksi URL (anti crash).
  3. SNIPPET EXTRACTION: Ambil cuplikan isi berita, bukan cuma judul.
  4. OBSERVABILITY: Terintegrasi dengan pipeline_logger.
  5. ANTI-BAN JITTER: Jeda acak 5-10 detik antar request.
"""

import os
import sys
import time
import random
import hashlib
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

try:
    import httpx
    from bs4 import BeautifulSoup
except ImportError:
    print("[ERROR] pip install httpx beautifulsoup4"); sys.exit(1)

# Import dari Monorepo Shared
try:
    from packages.shared.db_client import get_client
    from packages.shared.logger import start_run, finish_run
except ImportError as e:
    print(f"[ERROR] Gagal load shared modules: {e}")
    sys.exit(1)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

def fetch_ddg_news(entity_name: str) -> list[dict]:
    """Scrape DuckDuckGo News HTML untuk URL media asli & snippet."""
    url = f"https://duckduckgo.com/html/?q={entity_name.replace(' ', '+')}&kl=id-id"
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    
    try:
        r = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
        if r.status_code != 200: 
            print(f"     [HTTP {r.status_code}] Gagal fetch DDG.")
            return []
            
        soup = BeautifulSoup(r.text, 'html.parser')
        items = []
        
        # DDG menyimpan hasil di div.result
        for res in soup.find_all('div', class_='result'):
            link_tag = res.find('a', class_='result__a')
            snippet_tag = res.find('a', class_='result__snippet')
            
            if not link_tag: continue
            
            href = link_tag.get('href')
            title = link_tag.get_text(strip=True)
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
            
            # Ekstraksi URL Aman pakai urllib
            parsed = urlparse(href)
            qs = parse_qs(parsed.query)
            
            actual_url = None
            if 'uddg' in qs:
                actual_url = unquote(qs['uddg'][0])
            elif href.startswith('http') and 'duckduckgo.com' not in href:
                actual_url = href
                
            # Pastikan URL valid dan bukan internal DDG
            if actual_url and actual_url.startswith('http'):
                items.append({
                    "title": title,
                    "url": actual_url,
                    "snippet": snippet
                })
                
        return items[:15] # Ambil 15 berita teratas per tokoh
        
    except Exception as e:
        print(f"     [ERROR] Exception saat fetch DDG: {e}")
        return []

def main():
    sb = get_client()
    run_id = start_run("ddg_fetcher", "v2")
    
    # Ambil config yang aktif (hanya yang google_news_rss karena kita pakai nama tokohnya)
    res = sb.table("scraping_configs") \
            .select("id, entity_id, config_name") \
            .eq("is_active", True) \
            .eq("source_type", "google_news_rss") \
            .execute()
            
    configs = res.data or []
    print(f"[DDG] Mem-fetch {len(configs)} tokoh dari DuckDuckGo News...")
    
    total_inserted = 0
    total_failed = 0

    for cfg in configs:
        # Ubah "gnews_Joko_Widodo" -> "Joko Widodo"
        entity_name = cfg["config_name"].replace("gnews_", "").replace("_", " ")
        print(f"  -> Mencari: {entity_name}")
        
        articles = fetch_ddg_news(entity_name)
        
        if not articles:
            print("     ℹ️ 0 artikel ditemukan.")
            # Jeda anti-ban meski 0 hasil
            time.sleep(random.uniform(3, 6))
            continue
            
        items_to_insert = []
        for art in articles:
            # Hash dari URL karena URL DDG sudah di-resolve ke URL asli media
            source_id = hashlib.sha256(art["url"].encode()).hexdigest()[:32]
            # Text diisi snippet, kalau nggak ada pakai judul
            text_content = art["snippet"] if art["snippet"] else art["title"]
            
            items_to_insert.append({
                "source": "ddg_" + cfg["config_name"],
                "source_id": source_id,
                "title": art["title"],
                "source_url": art["url"], 
                "text": text_content, # ISI SNIPPET AGAR ENRICHER LEBIH RINGAN
                "text_hash": hashlib.sha256(art["title"].encode()).hexdigest(),
                "metadata": {
                    "configured_entity_id": cfg.get("entity_id"),
                    "fetcher_source": "ddg"
                },
                "published_at": None
            })
            
        try:
            # Insert batch ke Supabase (chunk 50)
            for i in range(0, len(items_to_insert), 50):
                chunk = items_to_insert[i:i+50]
                sb.rpc("batch_insert_raw_texts", {"p_items": chunk}).execute()
                
            print(f"     ✅ {len(items_to_insert)} artikel URL asli dimasukkan.")
            total_inserted += len(items_to_insert)
        except Exception as e:
            print(f"     [DB_ERROR] {e}")
            total_failed += len(items_to_insert)
            
        # Jeda acak 5-10 detik agar tidak kena ban DDG
        sleep_time = random.uniform(5, 10)
        time.sleep(sleep_time)
        
    print(f"\n[DDG] Selesai. Inserted: {total_inserted} | Failed: {total_failed}")
    finish_run(run_id, len(configs), total_inserted, total_failed)

if __name__ == "__main__":
    main()