"""
gnews_resolver_worker.py v5 — Cloud Smart Filter & Clean Logging
=================================================================
PERUBAAHAN v5 (Cloud Optimized):
  1. CLEAN LOGGING: Menghapus emoji, format log terstruktur.
  2. SMART DB QUERY: Mengambil status 'failed' dan 'snippet' sekaligus.
  3. CONTENT VALIDATION: Menolak homepage, cek densitas <p>, dan title relevancy.
     Mencegah section leakage disimpan sebagai FULLTEXT.
  4. JSON-LD PRIORITY: Ekstraksi ringan sebelum Trafilatura.
"""
import os
import re
import sys
import json
import base64
import random
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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
RESOLVER_VERSION = "v5_cloud_smart"
MAX_RETRIES = 3
MAX_WORKERS = 4  # Cloud aman dengan 4 threads
MAX_ARTICLE_LENGTH = 8000
MIN_ARTICLE_LENGTH = 500
MIN_PARAGRAPH_COUNT = 5
TITLE_MATCH_THRESHOLD = 0.15

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

def extract_jsonld_article(html: str) -> str | None:
    try:
        soup = BeautifulSoup(html, "html.parser")
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
        logger.info(f"ID: {art_id[:8]} | Status: FAILED | Reason: {resolve_status}")
        return update_payload

    if is_homepage_url(resolved_url):
        update_payload["recovery_status"] = "rejected_homepage_redirect"
        # HAPUS: update_payload["status"] = pc.STATUS_FAILED
        logger.info(f"ID: {art_id[:8]} | Status: REJECTED | Reason: Homepage Redirect")
        return update_payload

    try:
        headers = {"User-Agent": random.choice(USER_AGENTS), "Accept-Language": "id-ID,id;q=0.9"}
        resp = requests.get(resolved_url, headers=headers, timeout=15, allow_redirects=True)
        if not resp.ok:
            update_payload["recovery_status"] = f"fetch_http_{resp.status_code}"
            logger.info(f"ID: {art_id[:8]} | Status: FAILED | Reason: HTTP {resp.status_code}")
            return update_payload
            
        html_text = resp.text
        soup = BeautifulSoup(html_text, "html.parser")
        
        if len(soup.find_all("p")) < MIN_PARAGRAPH_COUNT:
            update_payload["recovery_status"] = "rejected_low_paragraph_density"
            logger.info(f"ID: {art_id[:8]} | Status: REJECTED | Reason: Low Paragraph Density")
            return update_payload
            
        full_text = extract_jsonld_article(html_text)
        if not full_text or len(full_text) < MIN_ARTICLE_LENGTH:
            full_text = traf_extract(html_text, include_comments=False, include_tables=False, favor_precision=True) or ""
            
        full_text = clean_boilerplate(full_text)
        
        if len(full_text) < MIN_ARTICLE_LENGTH:
            update_payload["recovery_status"] = "fetch_too_short"
            logger.info(f"ID: {art_id[:8]} | Status: FAILED | Reason: Text Too Short")
            return update_payload
            
        if len(full_text) > MAX_ARTICLE_LENGTH:
            update_payload["recovery_status"] = "rejected_section_page_too_long"
            logger.info(f"ID: {art_id[:8]} | Status: REJECTED | Reason: Section Page Leakage")
            return update_payload
            
        relevancy = calculate_title_relevancy(rss_title, full_text)
        if relevancy < TITLE_MATCH_THRESHOLD:
            update_payload["recovery_status"] = "rejected_title_mismatch"
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
        logger.error(f"ID: {art_id[:8]} | Status: CRASH | Error: {str(e)[:50]}")
        return update_payload

def main(limit: int = 50):
    sb = get_client()
    run_id = start_run("gnews_resolver_worker", RESOLVER_VERSION)
    logger.info(f"Initializing Cloud GNews Resolver | Workers: {MAX_WORKERS}")
    
    # KEMBALI KE QUERY AWAL: Hanya ambil yang ENRICHED dan SNIPPET
    res = sb.table("raw_texts") \
            .select("id, source_url, title, metadata, recovery_attempts") \
            .eq("status", pc.STATUS_ENRICHED) \
            .eq("content_type", "SNIPPET") \
            .lt("recovery_attempts", MAX_RETRIES) \
            .limit(limit) \
            .execute()
            
    articles = res.data or []
    if not articles:
        logger.info("Tidak ada artikel GNews (snippet) untuk dipulihkan.")
        finish_run(run_id, 0, 0, 0); return
        
    logger.info(f"Memproses {len(articles)} artikel...")
    updates = []
    resolved_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(process_article, art) for art in articles]
        for future in as_completed(futures):
            result = future.result()
            updates.append(result)
            if result.get("recovery_status") == "resolved": resolved_count += 1
            else: failed_count += 1
                
    if updates:
        try: sb.rpc("bulk_update_raw_texts", {"p_updates": updates}).execute()
        except Exception as e: logger.error(f"DB Bulk Update Error: {e}")
            
    logger.info(f"Eksekusi Selesai | Resolved: {resolved_count} | Failed/Rejected: {failed_count}")
    finish_run(run_id, len(articles), resolved_count, failed_count)