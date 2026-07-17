"""
gnews_resolver.py v13 — Final Adaptive, Deduplication & Clean Logging
=============================================================================
PERUBAAHAN v13:
  1. EARLY DEDUPLICATION: Cek judul duplikat sebelum membuka Playwright (Hemat RAM/CPU).
  2. ADAPTIVE MAX LENGTH: Menaikkan batas MAX_ARTICLE_LENGTH ke 20000 (menerima long-form).
  3. EXPERT CONTENT FILTER: Menerapkan JSON-LD priority, Trafilatura favor_precision,
     Title Relevancy, dan Max/Min Length check. Membasmi sidebar/homepage leakage.
  4. CLEAN LOGGING: Menghapus emoji dan format dekoratif. Log terstruktur agar mudah dibaca.
  5. RESOURCE GUARD: Membatasi ukuran HTML (MAX 1.5MB) dan memanggil gc.collect() 
     untuk melindungi RAM 16GB lokal dari spike memori.
"""
import re
import sys
import gc
import json
import time
import asyncio
import argparse
import logging
import datetime
from pathlib import Path
from collections import Counter

sys.path.append(str(Path(__file__).resolve().parents[2]))
from devtools.common import get_supabase, build_text_hash

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError
    from trafilatura import extract as traf_extract
    from bs4 import BeautifulSoup
except ImportError as e:
    print(f"[ERROR] {e}\nPastikan: pip install playwright trafilatura beautifulsoup4")
    print("Dan jalankan: playwright install chromium")
    sys.exit(1)

from packages.shared import constants as pc

# Setup Clean Logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Expert Validation Config
MAX_HTML_SIZE_BYTES = 1500000  # 1.5 MB
MAX_ARTICLE_LENGTH = 20000     # Batas adaptif (menerima long-form journalism)
MIN_ARTICLE_LENGTH = 500
MIN_PARAGRAPH_COUNT = 5
TITLE_MATCH_THRESHOLD = 0.15

def normalize_title(title: str) -> str:
    """Normalisasi judul untuk deteksi duplikat (lowercase, hapus tanda baca)."""
    if not title: return ""
    title = title.lower().strip()
    title = re.sub(r'[\[\]\(\)\{\}"\':;,!?./]', '', title)
    title = re.sub(r'\s+', ' ', title)
    return title

def find_duplicate_titles(sb, rows: list) -> set:
    """Cek apakah ada artikel di DB dengan judul yang sama yang sudah berstatus FULLTEXT."""
    titles_to_check = [r.get("title") or "" for r in rows]
    titles_to_check = [t for t in titles_to_check if t]
    
    if not titles_to_check: return set()
    
    try:
        res = sb.table("raw_texts") \
                .select("title") \
                .in_("title", titles_to_check) \
                .eq("content_type", "FULLTEXT") \
                .execute()
                
        dup_titles = set()
        for row in (res.data or []):
            norm = normalize_title(row.get("title") or "")
            if norm: dup_titles.add(norm)
            
        return dup_titles
    except Exception as e:
        logger.warning(f"Gagal cek duplikat judul: {e}")
        return set()

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
    text = re.sub(r'(Baca Juga|Simak Juga|Berita Terkait)\s*:.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(Reporter|Editor|Penulis|Pewarta)\s*:\s*.*?(?=\n|$)', '', text, flags=re.IGNORECASE)
    return re.sub(r'\n{3,}', '\n\n', text).strip()

def calculate_title_relevancy(title: str, text: str) -> float:
    if not title or not text: return 0.0
    title_words = set(re.findall(r'\b\w+\b', title.lower()))
    text_words = set(re.findall(r'\b\w+\b', text.lower()))
    if not title_words: return 0.0
    return sum(1 for w in title_words if w in text_words) / len(title_words)

async def process_url(context, art: dict, semaphore: asyncio.Semaphore, timeout: int, stats: Counter) -> dict:
    art_id = art["id"]
    gnews_url = art["source_url"]
    rss_title = art.get("title") or ""
    current_attempts = art.get("recovery_attempts", 0) + 1
    start_time = time.perf_counter()
    
    if len(gnews_url) < 80:
        stats["invalid_url"] += 1
        return {"id": art_id, "recovery_attempts": pc.MAX_RECOVERY_RETRY, "recovery_status": "failed_truncated_url"}
        
    async with semaphore:
        page = await context.new_page()
        resolved_url = None
        html_content = None
        
        try:
            await page.goto(gnews_url, wait_until="domcontentloaded", timeout=timeout * 1000)
            try:
                await page.wait_for_url(lambda url: "google.com" not in url, timeout=timeout * 1000)
            except PlaywrightTimeoutError:
                stats["timeout"] += 1
            
            final_url = page.url
            if "google.com" not in final_url and "gstatic.com" not in final_url:
                resolved_url = final_url
                html_content = await page.content()
            else:
                stats["google_loop"] += 1
        except PlaywrightError:
            stats["http_error"] += 1
        except Exception:
            stats["http_error"] += 1
        finally:
            await page.close()
            
    if not resolved_url or not html_content:
        if not any(k in str(stats) for k in ["timeout", "google_loop", "http_error"]):
            stats["redirect_failed"] += 1
        logger.info(f"ID: {art_id[:8]} | Status: FAILED | Reason: Redirect/Network Error")
        return {"id": art_id, "recovery_attempts": current_attempts, "recovery_status": pc.RECOVERY_FAILED}

    # --- EXPERT VALIDATION ---
    html_size = len(html_content)
    if html_size > MAX_HTML_SIZE_BYTES:
        stats["html_too_large"] += 1
        logger.info(f"ID: {art_id[:8]} | Status: REJECTED | Reason: HTML Too Large ({html_size//1024}KB)")
        del html_content
        return {"id": art_id, "recovery_attempts": current_attempts, "recovery_status": "rejected_html_too_large", "status": pc.STATUS_FAILED}

    soup = BeautifulSoup(html_content, "html.parser")
    if len(soup.find_all("p")) < MIN_PARAGRAPH_COUNT:
        stats["low_density"] += 1
        logger.info(f"ID: {art_id[:8]} | Status: REJECTED | Reason: Low Paragraph Density | URL: {resolved_url}")
        del html_content, soup
        return {"id": art_id, "recovery_attempts": current_attempts, "recovery_status": "rejected_low_density", "status": pc.STATUS_FAILED}

    # 1. Try JSON-LD
    full_text = extract_jsonld_article(html_content)
    extraction_method = "jsonld"
    
    # 2. Fallback to Trafilatura Precision
    if not full_text or len(full_text) < MIN_ARTICLE_LENGTH:
        full_text = traf_extract(html_content, include_comments=False, include_tables=False, favor_precision=True) or ""
        extraction_method = "trafilatura"

    del html_content, soup
    gc.collect()

    # 3. Cleanup Boilerplate
    full_text = clean_boilerplate(full_text)

    if len(full_text) < MIN_ARTICLE_LENGTH:
        stats["text_too_short"] += 1
        logger.info(f"ID: {art_id[:8]} | Status: FAILED | Reason: Text Too Short ({len(full_text)} chars) | URL: {resolved_url}")
        return {"id": art_id, "recovery_attempts": current_attempts, "recovery_status": "fetch_too_short"}
        
    if len(full_text) > MAX_ARTICLE_LENGTH:
        stats["section_leakage"] += 1
        logger.info(f"ID: {art_id[:8]} | Status: REJECTED | Reason: Section Page Leakage ({len(full_text)} chars) | URL: {resolved_url}")
        return {"id": art_id, "recovery_attempts": current_attempts, "recovery_status": "rejected_section_page", "status": pc.STATUS_FAILED}

    # 4. Title Relevancy
    relevancy = calculate_title_relevancy(rss_title, full_text)
    if relevancy < TITLE_MATCH_THRESHOLD:
        stats["title_mismatch"] += 1
        logger.info(f"ID: {art_id[:8]} | Status: REJECTED | Reason: Title Mismatch ({relevancy:.2f}) | URL: {resolved_url}")
        return {"id": art_id, "recovery_attempts": current_attempts, "recovery_status": "rejected_title_mismatch", "status": pc.STATUS_FAILED}

    # SUCCESS
    elapsed = time.perf_counter() - start_time
    stats["resolved"] += 1
    stats["total_time"] += elapsed
    logger.info(f"ID: {art_id[:8]} | Status: RESOLVED | Method: {extraction_method} | Len: {len(full_text)} | Rel: {relevancy:.2f} | Time: {elapsed:.1f}s | URL: {resolved_url}")
    
    current_meta = dict(art.get("metadata") or {})
    current_meta["is_snippet"] = False
    current_meta["resolved_url"] = resolved_url
    current_meta["resolver_method"] = f"playwright_hybrid_v13_{extraction_method}"
    current_meta["content_relevancy"] = round(relevancy, 2)
    
    return {
        "id": art_id,
        "text": full_text,
        "status": pc.STATUS_ENRICHED,
        "content_type": "FULLTEXT",
        "metadata": current_meta,
        "recovery_attempts": current_attempts,
        "recovery_status": pc.RECOVERY_RESOLVED,
        "content_hash": build_text_hash(full_text)
    }

async def async_main(sb, args):
    total_processed = 0
    batch_num = 1
    global_start = time.perf_counter()
    stats = Counter()
    
    semaphore = asyncio.Semaphore(args.concurrency)

    logger.info(f"Initializing Local Playwright Resolver | Concurrency: {args.concurrency} | Timeout: {args.timeout}s | Visible: {args.visible}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=not args.visible,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="id-ID"
        )
        
        while True:
            if args.max_total > 0 and total_processed >= args.max_total:
                break
                
            logger.info(f"--- Batch {batch_num} (Total Processed: {total_processed}) ---")
            try:
                time_filter = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)).isoformat()
                
                res = sb.table("raw_texts") \
                    .select("id, source_url, title, metadata, recovery_attempts") \
                    .or_(f"status.eq.{pc.STATUS_FAILED},and(status.eq.{pc.STATUS_ENRICHED}.content_type.eq.SNIPPET)") \
                    .lt("recovery_attempts", pc.MAX_RECOVERY_RETRY) \
                    .gte("ingested_at", time_filter) \
                    .order("recovery_attempts", desc=False) \
                    .order("ingested_at", desc=False) \
                    .limit(args.limit) \
                    .execute()
            except Exception as e:
                logger.warning(f"DB Query Timeout/Error: {e}. Menunggu 10 detik sebelum retry...")
                await asyncio.sleep(10)
                continue
                    
            articles = res.data or []
            if not articles:
                logger.info("Tidak ada lagi artikel GNews (30 hari terakhir) untuk di-resolve.")
                break
                
            # --- EARLY DEDUPLICATION GATE ---
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
                logger.info(f"  [DEDUP] {skipped_dup} artikel duplikat dilewati tanpa Playwright.")
                
            # Jalankan Playwright hanya untuk artikel yang tidak duplikat
            tasks = [process_url(context, art, semaphore, args.timeout, stats) for art in to_process]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    stats["crash"] += 1
                    logger.error(f"Task Error: {str(result)[:50]}")
                else:
                    updates.append(result)
                        
            # --- CHUNKED RPC UPDATE (Agar pasti masuk DB) ---
            if updates:
                CHUNK_SIZE = 25
                try:
                    for i in range(0, len(updates), CHUNK_SIZE):
                        chunk = updates[i:i + CHUNK_SIZE]
                        sb.rpc("bulk_update_raw_texts", {"p_updates": chunk}).execute()
                except Exception as e: 
                    logger.error(f"DB Bulk Update Error: {e}")
                
            total_processed += len(articles)
            batch_num += 1
            
        await context.close()
        await browser.close()
        
    elapsed = time.perf_counter() - global_start
    total_resolved = stats["resolved"]
    
    logger.info("=" * 60)
    logger.info("RECOVERY STATISTICS REPORT")
    logger.info("=" * 60)
    logger.info(f"  Total Processed    : {total_processed}")
    logger.info(f"  Total Resolved     : {total_resolved}")
    if total_processed > 0:
        logger.info(f"  Success Rate       : {(total_resolved/total_processed*100):.1f}%")
    if total_resolved > 0:
        logger.info(f"  Average Resolve    : {(stats['total_time']/total_resolved):.2f}s")
    logger.info(f"  Total Exec Time    : {elapsed:.2f}s ({elapsed/60:.1f}m)")
    logger.info("-" * 60)
    logger.info("  Failure Breakdown  :")
    logger.info(f"    - Timeout        : {stats['timeout']}")
    logger.info(f"    - Redirect Fail  : {stats['redirect_failed']}")
    logger.info(f"    - HTTP Error     : {stats['http_error']}")
    logger.info(f"    - Google Loop    : {stats['google_loop']}")
    logger.info(f"    - HTML Too Large : {stats['html_too_large']}")
    logger.info(f"    - Low Density    : {stats['low_density']}")
    logger.info(f"    - Section Leakage: {stats['section_leakage']}")
    logger.info(f"    - Title Mismatch : {stats['title_mismatch']}")
    logger.info(f"    - Text Too Short : {stats['text_too_short']}")
    logger.info(f"    - Invalid URL    : {stats['invalid_url']}")
    logger.info(f"    - Task Crash     : {stats['crash']}")
    logger.info("=" * 60)

def main(args) -> None:
    sb = get_supabase()
    asyncio.run(async_main(sb, args))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local GNews Playwright Resolver (Expert)")
    parser.add_argument("--limit", type=int, default=20, help="Jumlah row per batch (default 20)")
    parser.add_argument("--max-total", type=int, default=0, help="Batas total proses (0 = unlimited)")
    parser.add_argument("--concurrency", type=int, default=3, help="Jumlah tab diproses bersamaan (default 3)")
    parser.add_argument("--timeout", type=int, default=15, help="Timeout tunggu redirect dalam detik (default 15)")
    parser.add_argument("--visible", action="store_true", help="Tampilkan browser di layar (headless=False)")
    parser.add_argument("--url", type=str, help="Test resolve 1 URL GNews secara manual")
    args = parser.parse_args()
    
    if args.url:
        logger.info("Mode Testing Manual (1 URL)")
        async def test_url():
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=False, args=['--disable-blink-features=AutomationControlled'])
                context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
                page = await context.new_page()
                try:
                    logger.info(f"  -> Membuka: {args.url}")
                    await page.goto(args.url, wait_until="domcontentloaded", timeout=20000)
                    
                    logger.info("  Menunggu redirect Google (max 15s)...")
                    resolved = None
                    try:
                        await page.wait_for_url(lambda url: "google.com" not in url, timeout=15000)
                        resolved = page.url
                    except: pass
                                
                    if resolved:
                        logger.info(f"  Hasil URL Final: {resolved}")
                        html = await page.content()
                        text = traf_extract(html, include_comments=False, include_tables=False) or ""
                        logger.info(f"  Panjang Teks: {len(text)} karakter")
                        logger.info(f"  Preview: {text[:200]}...")
                    else:
                        logger.warning("  Gagal keluar dari Google (Timeout 15s)")
                except Exception as e:
                    logger.error(f"  Error: {e}")
                finally:
                    await browser.close()
        asyncio.run(test_url())
    else:
        main(args)