"""
gnews_resolver_worker.py — Layer 2.4 (Content Recovery)
=========================================================
Tugas: Memulihkan konten GNews yang tertinggal sebagai Snippet.
Jika berhasil -> Ambil Full Text -> Ubah content_type='FULLTEXT' & status='enriched'.
Jika gagal -> Catat recovery_status & tambah recovery_attempts (Max 3).

Tidak mengembalikan ke 'pending' agar tidak mengulang Enrichment.
Langsung di-set 'enriched' agar masuk ke Validation Worker.
"""
import os
import re
import sys
import time
import base64
import random
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

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
RESOLVER_VERSION = "v1_recovery"
MAX_RETRIES = 3

def decode_gnews_base64(url: str) -> tuple[str | None, str]:
    match = re.search(r'/articles/(.*?)(\?|$)', url)
    if not match: return None, "failed_base64_format"
    token = match.group(1)
    if token.startswith('CBMi'): token = token[4:]
    try:
        padding = 4 - (len(token) % 4)
        if padding != 4: token += '=' * padding
        decoded_str = base64.urlsafe_b64decode(token).decode('utf-8', errors='ignore')
        url_match = re.search(r'(https?://[^\s\x00-\x1F]+)', decoded_str)
        if url_match: return url_match.group(1), "resolved_base64"
    except Exception:
        pass
    return None, "failed_base64_decode"

def resolve_via_http(url: str) -> tuple[str | None, str]:
    try:
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        r = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        if r.ok and GNEWS_DOMAIN not in r.url:
            return r.url, "resolved_http_redirect"
        if r.ok:
            match = re.search(r'<form[^>]+action="([^"]+)"', r.text, re.IGNORECASE)
            if match and 'google.com' not in match.group(1):
                return match.group(1).replace('&amp;', '&'), "resolved_form_action"
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

def main(limit: int = 50, max_total: int = 0):
    sb = get_client()
    run_id = start_run("gnews_resolver_worker", RESOLVER_VERSION)
    
    print(f"[GNEWS_RECOVERY] Mencari artikel SNIPPET untuk dipulihkan (Max retries: {MAX_RETRIES})...")
    
    # Ambil artikel yang enriched, SNIPPET, dan belum mencapai batas max retries
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
        
    print(f"[GNEWS_RECOVERY] Memproses {len(articles)} artikel...")
    
    resolved_count = 0
    failed_count = 0
    updates = []
    
    for art in articles:
        url = art["source_url"]
        current_attempts = art.get("recovery_attempts", 0) + 1
        
        # 1. Resolver Chain
        resolved_url, resolve_status = decode_gnews_base64(url)
        if not resolved_url:
            resolved_url, resolve_status = resolve_via_http(url)
            
        if resolved_url:
            # 2. Fetch & Extract Content
            full_text, fetch_status = fetch_and_extract(resolved_url)
            
            if len(full_text) >= 500:
                # BERHASIL DIPULIHKAN!
                current_meta = dict(art.get("metadata") or {})
                current_meta["is_snippet"] = False
                current_meta["resolved_url"] = resolved_url
                current_meta["resolver_method"] = resolve_status
                
                updates.append({
                    "id": art["id"],
                    "text": full_text,
                    "status": pc.STATUS_ENRICHED, # Tetap enriched, tapi sekarang FULLTEXT
                    "content_type": "FULLTEXT",   # Ubah ke FULLTEXT
                    "metadata": current_meta,
                    "recovery_attempts": current_attempts,
                    "recovery_status": "resolved",
                    "content_hash": hashlib.sha256(full_text.encode()).hexdigest()
                })
                resolved_count += 1
                print(f"  ✅ Resolved: {url[:40]}... -> {resolved_url[:40]}...")
            else:
                # Gagal fetch text
                updates.append({
                    "id": art["id"],
                    "recovery_attempts": current_attempts,
                    "recovery_status": fetch_status
                })
                failed_count += 1
        else:
            # Gagal resolve URL
            updates.append({
                "id": art["id"],
                "recovery_attempts": current_attempts,
                "recovery_status": resolve_status
            })
            failed_count += 1
            
        time.sleep(random.uniform(1, 2))
        
    if updates:
        try: sb.rpc("bulk_update_raw_texts", {"p_updates": updates}).execute()
        except Exception as e: print(f"[DB_ERROR] {e}")
            
    print(f"\n[GNEWS_RECOVERY] Selesai. Resolved: {resolved_count} | Failed: {failed_count}")
    finish_run(run_id, len(articles), resolved_count, failed_count)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    main(limit=args.limit)