"""
gnews_resolver_worker.py v6 — Cloud Bulletproof & Logic Fix
=================================================================
FIX v6:
  1. STATUS FAILED FIX: Semua artikel rejected/crash kini otomatis diset STATUS_FAILED
     agar tidak diambil terus menerus (mencegah infinite loop di Cloud).
  2. ADAPTIVE MAX LENGTH: Menaikkan batas MAX_ARTICLE_LENGTH ke 20000 (menerima long-form).
  3. SOUP REUSE: Mengirim objek soup ke fungsi JSON-LD agar tidak parse 2x (Hemat CPU).
  4. DEDUP GATE: Cek judul duplikat sebelum fetch HTTP agar tidak buang kuota Cloud.
  5. CONTINUOUS LOOP: Menggunakan while-loop agar bisa memproses banyak batch dalam 1 run.
"""

import re
import sys
import json
import base64
import random
import hashlib
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

try:
    import requests
    from trafilatura import extract as traf_extract
    from bs4 import BeautifulSoup
except ImportError as e:
    logger.error(f"Dependency missing: {e}"); sys.exit(1)

from packages.shared.db_client import get_client
from packages.shared.logger import start_run, finish_run
from packages.shared import constants as pc

GNEWS_DOMAIN = "news.google.com"
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]
RESOLVER_VERSION = "v6_cloud_bulletproof"
MAX_RETRIES = 3
MAX_WORKERS = 3  # Aman untuk 2 Core CPU GitHub Actions
MAX_ARTICLE_LENGTH = 20000  # FIX BUG 2: Terima long-form
MIN_ARTICLE_LENGTH = 500
MIN_PARAGRAPH_COUNT = 5
TITLE_MATCH_THRESHOLD = 0.15

def normalize_title(title: str) -> str:
    if not title: return ""
    title = title.lower().strip()
    title = re.sub(r'[\[\]\(\)\{\}"\':;,!?./]', '', title)
    title = re.sub(r'\s+', ' ', title)
    return title

def find_duplicate_titles(sb, rows: list) -> set:
    titles_to_check = [r.get("title") or "" for r in rows]
    titles_to_check = [t for t in titles_to_check if t]
    if not titles_to_check: return set()
    
    dup_titles = set()
    chunk_size = 50
    try:
        for i in range(0, len(titles_to_check), chunk_size):
            chunk = titles_to_check[i:i + chunk_size]
            res = sb.table("raw_texts") \
                    .select("title") \
                    .in_("title", chunk) \
                    .eq("content_type", "FULLTEXT") \
                    .execute()
            for row in (res.data or []):
                norm = normalize_title(row.get("title") or "")
                if norm: dup_titles.add(norm)
        return dup_titles
    except Exception as e:
        logger.warning(f"Gagal cek duplikat judul: {e}")
        return set()

def decode_gnews_protobuf(url: str) -> tuple[str | None, str]:
    match = re.search(r'/articles/(.*?)(\?|$)', url)
    if not match: return None, "failed_format"
    token = match.group(1).replace('-', '+').replace('_', '/')
    if len(token) > 4 and token[4] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/':
        test_token = token[4:]
        padding = 4 - (len(test_token) % 4)
        if padding != 4: test_token += '=' * padding
        try:
            decoded_str = base64.b64decode(test_token).decode('utf-8', errors='ignore')
            url_match = re.search(r'(https?://[^\s\x00-\x1F"\'<>]+)', decoded_str)
            if url_match: return url_match.group(1).split('\\')[0], "resolved_protobuf_prefix"
        except Exception: pass
    padding = 4 - (len(token) % 4)
    if padding != 4: token += '=' * padding
    try:
        decoded_str = base64.b64decode(token).decode('utf-8', errors='ignore')
        url_match = re.search(r'(https?://[^\s\x00-\x1F"\'<>]+)', decoded_str)
        if url_match: return url_match.group(1).split('\\')[0], "resolved_protobuf_raw"
    except Exception: pass
    return None, "failed_protobuf_decode"

def deep_html_scrape(html_text: str) -> str | None:
    matches = re.findall(r'href="(https?://[^"]+)"', html_text)
    for m in matches:
        if not any(d in m for d in ['google.com', 'gstatic.com', 'googleapis.com', 'schema.org', 'youtube.com']):
            return m
    return None

def resolve_via_http(url: str) -> tuple[str | None, str]:
    try:
        headers = {"User-Agent": random.choice(USER_AGENTS), "Referer": "https://news.google.com/"}
        if '/rss/articles/' in url: url = url.replace('/rss/articles/', '/articles/')
        r = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        if r.ok and GNEWS_DOMAIN not in r.url and 'google.com' not in r.url:
            return r.url, "resolved_http_redirect"
        if r.ok:
            scraped = deep_html_scrape(r.text)
            if scraped: return scraped, "resolved_deep_scrape"
        return None, "failed_google_loop"
    except: return None, "failed_http_exception"

def is_homepage_url(url: str) -> bool:
    if not url: return True
    path = urlparse(url).path
    return path in ("", "/", "/index.html", "/index.php")

# FIX BUG 3: Terima objek soup, jangan parse ulang
def extract_jsonld_article(soup: BeautifulSoup) -> str | None:
    try:
        scripts = soup.find_all("script", type="application/ld+json")
        for script in scripts:
            if not script.string: continue
            data = json.loads(script.string)
            items = data if isinstance(data, list) else [data]
            if "@graph" in items[0]: items = items[0]["@graph"]
            for item in items:
                if isinstance(item, dict) and item.get("@type") in ("NewsArticle", "Article"):
                    if item.get("articleBody"): return item["articleBody"]
    except: pass
    return None

def clean_boilerplate(text: str) -> str:
    if not text: return ""
    text = re.sub(r'(Baca Juga|Simak Juga)\s*:.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(Reporter|Editor|Penulis)\s*:\s*.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
    return re.sub(r'\n{3,}', '\n\n', text).strip()

def calculate_title_relevancy(title: str, text: str) -> float:
    if not title or not text: return 0.0
    title_words = set(re.findall(r'\b\w+\b', title.lower()))
    text_words = set(re.findall(r'\b\w+\b', text.lower()))
    if not title_words: return 0.0
    return sum(1 for w in title_words if w in text_words) / len(title_words)

def process_article(art: dict) -> dict:
    art_id = art["id"]
    url = art["source_url"]
    rss_title = art.get("title") or ""
    update_payload = {"id": art_id, "recovery_attempts": art.get("recovery_attempts", 0) + 1}
    
    resolved_url, resolve_status = decode_gnews_protobuf(url)
    if not resolved_url:
        resolved_url, resolve_status = resolve_via_http(url)
        
    if not resolved_url:
        update_payload["recovery_status"] = resolve_status
        # FIX BUG 1: Set status FAILED agar tidak infinite loop
        update_payload["status"] = pc.STATUS_FAILED
        logger.info(f"ID: {art_id[:8]} | Status: FAILED | Reason: {resolve_status}")
        return update_payload

    if is_homepage_url(resolved_url):
        update_payload["recovery_status"] = "rejected_homepage_redirect"
        update_payload["status"] = pc.STATUS_FAILED # FIX BUG 1
        logger.info(f"ID: {art_id[:8]} | Status: REJECTED | Reason: Homepage Redirect")
        return update_payload

    try:
        headers = {"User-Agent": random.choice(USER_AGENTS), "Accept-Language": "id-ID,id;q=0.9"}
        resp = requests.get(resolved_url, headers=headers, timeout=15, allow_redirects=True)
        if not resp.ok:
            update_payload["recovery_status"] = f"fetch_http_{resp.status_code}"
            update_payload["status"] = pc.STATUS_FAILED # FIX BUG 1
            logger.info(f"ID: {art_id[:8]} | Status: FAILED | Reason: HTTP {resp.status_code}")
            return update_payload
            
        html_text = resp.text
        soup = BeautifulSoup(html_text, "html.parser")
        
        if len(soup.find_all("p")) < MIN_PARAGRAPH_COUNT:
            update_payload["recovery_status"] = "rejected_low_paragraph_density"
            update_payload["status"] = pc.STATUS_FAILED # FIX BUG 1
            logger.info(f"ID: {art_id[:8]} | Status: REJECTED | Reason: Low Paragraph Density")
            return update_payload
            
        # FIX BUG 3: Lempar objek soup
        full_text = extract_jsonld_article(soup)
        if not full_text or len(full_text) < MIN_ARTICLE_LENGTH:
            full_text = traf_extract(html_text, include_comments=False, include_tables=False, favor_precision=True) or ""
            
        full_text = clean_boilerplate(full_text)
        
        if len(full_text) < MIN_ARTICLE_LENGTH:
            update_payload["recovery_status"] = "fetch_too_short"
            update_payload["status"] = pc.STATUS_FAILED # FIX BUG 1
            logger.info(f"ID: {art_id[:8]} | Status: FAILED | Reason: Text Too Short")
            return update_payload
            
        if len(full_text) > MAX_ARTICLE_LENGTH:
            update_payload["recovery_status"] = "rejected_section_page_too_long"
            update_payload["status"] = pc.STATUS_FAILED # FIX BUG 1
            logger.info(f"ID: {art_id[:8]} | Status: REJECTED | Reason: Section Page Leakage")
            return update_payload
            
        relevancy = calculate_title_relevancy(rss_title, full_text)
        if relevancy < TITLE_MATCH_THRESHOLD:
            update_payload["recovery_status"] = "rejected_title_mismatch"
            update_payload["status"] = pc.STATUS_FAILED # FIX BUG 1
            logger.info(f"ID: {art_id[:8]} | Status: REJECTED | Reason: Title Mismatch ({relevancy:.2f})")
            return update_payload
            
        current_meta = dict(art.get("metadata") or {})
        current_meta["is_snippet"] = False
        current_meta["resolved_url"] = resolved_url
        current_meta["resolver_method"] = resolve_status
        
        update_payload.update({
            "text": full_text, "status": pc.STATUS_ENRICHED, "content_type": "FULLTEXT",
            "metadata": current_meta, "recovery_status": "resolved",
            "content_hash": hashlib.sha256(full_text.encode()).hexdigest()
        })
        logger.info(f"ID: {art_id[:8]} | Status: RESOLVED | Len: {len(full_text)} | Rel: {relevancy:.2f}")
        return update_payload
        
    except Exception as e:
        update_payload["recovery_status"] = "fetch_exception"
        update_payload["status"] = pc.STATUS_FAILED # FIX BUG 1
        logger.error(f"ID: {art_id[:8]} | Status: CRASH | Error: {str(e)[:50]}")
        return update_payload

# FIX BUG 4: Tambahkan max_total dan while-loop
def main(limit: int = 50, max_total: int = 0):
    sb = get_client()
    run_id = start_run("gnews_resolver_worker", RESOLVER_VERSION)
    logger.info(f"Initializing Cloud GNews Resolver | Workers: {MAX_WORKERS} | Max: {'Unlimited' if max_total == 0 else max_total}")
    
    total_processed = 0
    total_resolved = 0
    total_failed = 0
    batch_num = 1

    while True:
        if max_total > 0 and total_processed >= max_total:
            logger.info(f"Max total ({max_total}) tercapai. Berhenti.")
            break

        current_limit = min(limit, max_total - total_processed) if max_total > 0 else limit
        
        try:
            time_filter = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            res = sb.table("raw_texts") \
                    .select("id, source_url, title, metadata, recovery_attempts") \
                    .eq("status", pc.STATUS_ENRICHED) \
                    .eq("content_type", "SNIPPET") \
                    .lt("recovery_attempts", MAX_RETRIES) \
                    .gte("ingested_at", time_filter) \
                    .limit(current_limit) \
                    .execute()
        except Exception as e:
            logger.warning(f"DB Query Error: {e}. Menunggu 10 detik...")
            time.sleep(10)
            continue
            
        articles = res.data or []
        if not articles:
            logger.info("Tidak ada artikel GNews (snippet) untuk dipulihkan.")
            break
            
        # FIX BUG 4: Early Deduplication Gate
        existing_titles = find_duplicate_titles(sb, articles)
        to_process = []
        skipped_dup = 0
        updates = []
        
        for art in articles:
            norm_title = normalize_title(art.get("title") or "")
            if norm_title and norm_title in existing_titles:
                skipped_dup += 1
                updates.append({
                    "id": art["id"],
                    "status": pc.STATUS_SKIPPED,
                    "recovery_attempts": art.get("recovery_attempts", 0) + 1,
                    "recovery_status": "skipped_duplicate_title",
                    "metadata": {**(art.get("metadata") or {}), "fail_reason": "duplicate_title_at_resolver"}
                })
            else:
                to_process.append(art)
                
        if skipped_dup > 0:
            logger.info(f"  [DEDUP] {skipped_dup} artikel duplikat dilewati tanpa fetch.")
            
        logger.info(f"Memproses {len(to_process)} artikel...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(process_article, art) for art in to_process]
            for future in as_completed(futures):
                result = future.result()
                updates.append(result)
                if result.get("recovery_status") == "resolved": total_resolved += 1
                else: total_failed += 1
                
        if updates:
            try:
                # Chunked RPC agar tidak kena payload limit
                for i in range(0, len(updates), 25):
                    chunk = updates[i:i+25]
                    sb.rpc("bulk_update_raw_texts", {"p_updates": chunk}).execute()
            except Exception as e: 
                logger.error(f"DB Bulk Update Error: {e}")
                
        total_processed += len(articles)
        batch_num += 1
        
        # Jeda rate limit
        sleep_time = random.uniform(2, 5)
        logger.info(f"Menunggu {sleep_time:.1f}s sebelum batch berikutnya...")
        time.sleep(sleep_time)
            
    logger.info(f"Eksekusi Selesai | Resolved: {total_resolved} | Failed/Rejected: {total_failed}")
    finish_run(run_id, total_processed, total_resolved, total_failed)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-total", type=int, default=0)
    args = parser.parse_args()
    main(limit=args.limit, max_total=args.max_total)