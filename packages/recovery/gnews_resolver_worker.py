"""
gnews_resolver_worker.py v4 — Aggressive Protobuf & Deep Scrape
=================================================================
PERUBAAHAN v4:
  1. AGGRESSIVE PROTOBUF: Membuang seluruh prefix non-base64 (CBMi, AU_yqL, dll)
     dan mendecode payload secara brute-force untuk mencari URL.
  2. DEEP HTML SCRAPE: Men-scan seluruh tag <script> dan <a> di HTML mentah
     untuk mencari URL media yang tersembunyi.
  3. URL PATH TRICK: Mengubah /rss/articles/ menjadi /articles/ agar memicu
     redirect HTTP 301/302 yang kadang berhasil.
"""
import re
import sys
import base64
import random
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

try:
    import requests
    from trafilatura import extract as traf_extract
    from supabase import create_client, Client
except ImportError as e:
    print(f"[ERROR] {e}"); sys.exit(1)

from packages.shared.db_client import get_client
from packages.shared.logger import start_run, finish_run
from packages.shared import constants as pc

GNEWS_DOMAIN = "news.google.com"
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]
RESOLVER_VERSION = "v4_aggressive"
MAX_RETRIES = 3
MAX_WORKERS = 7

def decode_gnews_protobuf(url: str) -> tuple[str | None, str]:
    """Brute-force decode payload GNews untuk mencari URL."""
    match = re.search(r'/articles/(.*?)(\?|$)', url)
    if not match: return None, "failed_format"
    token = match.group(1)    
    token_std = token.replace('-', '+').replace('_', '/')    
    if len(token_std) > 4 and token_std[4] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/':
        test_token = token_std[4:]
        padding = 4 - (len(test_token) % 4)
        if padding != 4: test_token += '=' * padding
        
        try:
            decoded_bytes = base64.b64decode(test_token)
            decoded_str = decoded_bytes.decode('utf-8', errors='ignore')            
            url_match = re.search(r'(https?://[^\s\x00-\x1F"\'<>]+)', decoded_str)
            if url_match:
                clean_url = url_match.group(1).split('\\')[0] # Buang escape chars
                return clean_url, "resolved_protobuf_prefix_stripped"
        except Exception:
            pass            
    padding = 4 - (len(token_std) % 4)
    if padding != 4: token_std += '=' * padding
    try:
        decoded_bytes = base64.b64decode(token_std)
        decoded_str = decoded_bytes.decode('utf-8', errors='ignore')
        url_match = re.search(r'(https?://[^\s\x00-\x1F"\'<>]+)', decoded_str)
        if url_match:
            return url_match.group(1).split('\\')[0], "resolved_protobuf_raw"
    except Exception:
        pass
        
    return None, "failed_protobuf_decode"

def deep_html_scrape(html_text: str) -> str | None:
    """Scan mendalam HTML untuk mencari URL media yang tersembunyi."""
    # Cari di tag <a> biasa
    matches = re.findall(r'href="(https?://[^"]+)"', html_text)
    for m in matches:
        if not any(d in m for d in ['google.com', 'gstatic.com', 'googleapis.com', 'schema.org', 'youtube.com']):
            return m            
    matches = re.findall(r'"(https?://[^"]+)"', html_text)
    for m in matches:
        if not any(d in m for d in ['google.com', 'gstatic.com', 'googleapis.com', 'schema.org', 'youtube.com']):
            return m            
    match = re.search(r'data-n-au="([^"]+)"', html_text)
    if match and 'google.com' not in match.group(1):
        return match.group(1).replace('\\u003d', '=').replace('\\u0026', '&')
        
    return None

def resolve_via_http(url: str) -> tuple[str | None, str]:
    try:
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://news.google.com/"
        }
        
        if '/rss/articles/' in url:
            url = url.replace('/rss/articles/', '/articles/')    
        r = requests.get(url, headers=headers, timeout=10, allow_redirects=True)        
        # 1. Cek HTTP Redirect final
        if r.ok and GNEWS_DOMAIN not in r.url and 'google.com' not in r.url:
            return r.url, "resolved_http_redirect"
        if r.ok:
            # 2. Deep HTML Scrape
            scraped_url = deep_html_scrape(r.text)
            if scraped_url:
                return scraped_url, "resolved_deep_scrape"
                
        return None, f"failed_http_{r.status_code}"
    except requests.exceptions.Timeout:
        return None, "failed_timeout"
    except Exception:
        return None, "failed_http_exception"

def fetch_and_extract(url: str) -> tuple[str, str]:
    try:
        headers = {"User-Agent": random.choice(USER_AGENTS), "Accept-Language": "id-ID,id;q=0.9"}
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        if resp.ok:
            text = traf_extract(resp.text, include_comments=False, include_tables=False) or ""
            return text, "fetch_success" if len(text) >= 500 else "fetch_too_short"
        return "", f"fetch_http_{resp.status_code}"
    except Exception:
        return "", "fetch_exception"

def process_article(art: dict) -> dict:
    """Memproses 1 artikel (dijalankan di thread)."""
    url = art["source_url"]
    current_attempts = art.get("recovery_attempts", 0) + 1
    update_payload = {
        "id": art["id"],
        "recovery_attempts": current_attempts
    }
    # 1. Resolver Chain (Prioritas Protobuf Aggressive)
    resolved_url, resolve_status = decode_gnews_protobuf(url)
    if not resolved_url:
        resolved_url, resolve_status = resolve_via_http(url)
    if resolved_url:
        # 2. Fetch & Extract Content
        full_text, fetch_status = fetch_and_extract(resolved_url)
        
        if len(full_text) >= 500:
            current_meta = dict(art.get("metadata") or {})
            current_meta["is_snippet"] = False
            current_meta["resolved_url"] = resolved_url
            current_meta["resolver_method"] = resolve_status
            
            update_payload.update({
                "text": full_text,
                "status": pc.STATUS_ENRICHED,
                "content_type": "FULLTEXT",
                "metadata": current_meta,
                "recovery_status": "resolved",
                "content_hash": hashlib.sha256(full_text.encode()).hexdigest()
            })
            print("✅", end="", flush=True)
        else:
            update_payload["recovery_status"] = fetch_status
            print("❌", end="", flush=True)
    else:
        update_payload["recovery_status"] = resolve_status
        print("❌", end="", flush=True)
        
    return update_payload

def main(limit: int = 50, max_total: int = 0):
    sb = get_client()
    run_id = start_run("gnews_resolver_worker", RESOLVER_VERSION)
    
    print(f"[GNEWS_RECOVERY] Mencari artikel SNIPPET untuk dipulihkan (Max retries: {MAX_RETRIES})...")
    
    res = sb.table("raw_texts") \
            .select("id, source_url, text, metadata, recovery_attempts") \
            .eq("status", pc.STATUS_ENRICHED) \
            .eq("content_type", "SNIPPET") \
            .lt("recovery_attempts", MAX_RETRIES) \
            .limit(limit) \
            .execute()
            
    articles = res.data or []
    if not articles:
        print("[GNEWS_RECOVERY] Tidak ada artikel untuk dipulihkan.")
        finish_run(run_id, 0, 0, 0)
        return
        
    print(f"[GNEWS_RECOVERY] Memproses {len(articles)} artikel secara paralel ({MAX_WORKERS} threads)...")
    print("  Progress: ", end="")
    
    updates = []
    resolved_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_article, art): art for art in articles}
        for future in as_completed(futures):
            try:
                result = future.result()
                updates.append(result)
                if result.get("recovery_status") == "resolved":
                    resolved_count += 1
                else:
                    failed_count += 1
            except Exception:
                failed_count += 1
                print("💥", end="", flush=True)
                
    print("\n")
        
    if updates:
        try: sb.rpc("bulk_update_raw_texts", {"p_updates": updates}).execute()
        except Exception as e: print(f"[DB_ERROR] {e}")
            
    print(f"[GNEWS_RECOVERY] Selesai. Resolved: {resolved_count} | Failed: {failed_count}")
    finish_run(run_id, len(articles), resolved_count, failed_count)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    main(limit=args.limit)