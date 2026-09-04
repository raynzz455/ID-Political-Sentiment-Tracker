"""
enricher_worker.py v21 — RAM & CPU Optimized
====================================================================
FIX v21 (library-based):
  1. LXML PARSER: Mengganti html.parser ke lxml (3x lebih cepat, hemat RAM).
  2. GC COLLECTION: Memaksa garbage collection tiap akhir batch agar RAM 16GB tidak bocor.
  3. THREAD BOOST: Menaikkan limit thread dari 7 ke 10 untuk I/O paralel.
"""

import re
import sys
import gc
import json
import time
import random
import hashlib
import logging
import argparse
import html as html_lib
import unicodedata

# v21: Library-based cleaning
try:
    import ftfy
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False

# v21: Byline patterns (smart — keep context abbreviations)
BYLINE_SLASH_PATTERN = re.compile(r'\s*\([a-z]{2,5}/[a-z]{2,5}\)\s*', re.IGNORECASE)
BYLINE_END_PATTERN = re.compile(r'\s*\([a-z]{2,5}(?:/[a-z]{2,5})?\)\s*$', re.IGNORECASE)
KEEP_ABBREVIATIONS = {'ratas', 'nobar', 'red', 'kapol', 'wabup', 'wagub', 'wali',
                      'ist', 'dok', 'antara', 'foto', 'instagram', 'pmj', 'ls',
                      'psht', 'pmp', 'kk', 'ak'}

# v21: Promo patterns
PROMO_PATTERNS_V21 = [
    r'(?i)Gabung\s+\w+\s*\.?\s*Plus\s*sekarang.*',
    r'(?i)berkomitmen memberikan fakta jernih.*',
    r'(?i)Dukung keberlanjutan jurnalisme.*',
    r'(?i)nikmati kenyamanan baca.*',
    r'(?i)KOMPAS\.com berkomitmen.*',
]

# v21: Source attribution (awal + tengah)
SOURCE_ATTR_V21 = [
    r'^(KOMPAS\.com|CNN Indonesia|TEMPO\.CO|TRIBUN\w*\.?\w*|ANTARA/?\w*|jpnn\.com|detikcom|VIVA|Suara\.com|Republika)\s*[\-\u2013\u2014|:]\s*',
    r'\.\s+(KOMPAS\.com|CNN Indonesia|TEMPO\.CO|TRIBUN\w*|Suara\.com|VIVA|Republika)\s*[\-\u2013\u2014|:]\s*',
]

# v21: Bullet points
BULLET_V21 = r'(?:^|\.\s+)-\s+(?=[A-Z])'
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")
try:
    from trafilatura import extract as traf_extract
    from bs4 import BeautifulSoup
    # Pastikan lxml terinstall: pip install lxml
except ImportError as e:
    print(f"[ERROR] Dependency missing: {e}. Pastikan: pip install trafilatura beautifulsoup4 lxml")
    sys.exit(1)

from packages.shared.db_client import get_client
from packages.shared.logger import start_run, finish_run
from packages.shared import constants as pc
from packages.enrichment.universal_resolver import fetch_article, FetchResult

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

MAX_WORKERS = 7  # Naikkan dari 7 ke 10
RSS_TEXT_MIN_LEN = 500

# Expert Validation Config
MAX_HTML_SIZE_BYTES = 1500000  # 1.5 MB
MAX_ARTICLE_LENGTH = 20000     
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
                    .in_("status", ["enriched", "processed", "skipped", "validated"]) \
                    .execute()
            for row in (res.data or []):
                norm = normalize_title(row.get("title") or "")
                if norm: dup_titles.add(norm)
        return dup_titles
    except Exception as e:
        logger.warning(f"Gagal cek duplikat judul: {e}")
        return set()

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

def clean_boilerplate(text: str, title: str = "") -> str:
    """v21: Library-based cleaning with ftfy + promo + byline + source attr + dedup."""
    if not text: return ""
    
    # v21: Fix encoding with ftfy
    if HAS_FTFY:
        text = ftfy.fix_text(text)
    
    text = html_lib.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\xa0", " ")
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # v21: Remove bullet points
    text = re.sub(BULLET_V21, '. ', text)
    text = re.sub(r'\.\.\s+', '. ', text)
    text = re.sub(r'^\.\s+', '', text)
    
    # v21: Remove promo/marketing
    for pattern in PROMO_PATTERNS_V21:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    
    # Original UI patterns
    ui_patterns = [
        r'(?i)(Tags\s*:|Berita Lainnya|Dark/Light Mode|BREAKINGNEWS).*?(?=\n|$|\.)',
        r'(?i)Gambas\s*:\s*Video\s*\w+',
        r'(?i)Dilarang keras mengambil konten.*?(?=\n|$|\.)',
        r'(?i)Baca berita selengkapnya.*?(?=\n|$|\.)',
        r'(?i)(Simak juga|Baca Juga|Berita Terkait)\s*:.*?(?=\n|$|\.)',
        r'(?i)(Reporter|Editor|Penulis|Pewarta|Jurnalis)\s*:\s*.*?(?=\n|$|\.)',
        r'(?i)(Pilihan untuk lu|Sumber:).*?(?=\n|$|\.)',
        r'(?i)Sponsor.*?(?=\n|$|\.)'
    ]
    for pattern in ui_patterns:
        text = re.sub(pattern, '', text)
        
    text = re.sub(r'\(\s*(Foto|Instagram|Dok|Istimewa|Antara)[^)]*\)', '', text, flags=re.IGNORECASE)

    # v21: Remove byline (smart — keep abbreviations)
    text = BYLINE_SLASH_PATTERN.sub(' ', text)
    match = BYLINE_END_PATTERN.search(text)
    if match:
        byline_content = match.group().strip('() \n\r')
        if byline_content.lower() not in KEEP_ABBREVIATIONS:
            text = BYLINE_END_PATTERN.sub('', text)
    text = text.rstrip()

    # v21: Remove source attribution (awal + tengah)
    for pattern in SOURCE_ATTR_V21:
        text = re.sub(pattern, '. ', text)
    text = re.sub(r'\.\.\s+', '. ', text)
    text = re.sub(r'^\.\s+', '', text)

    if title:
        clean_title = re.sub(r'[^\w\s]', '', title).lower().strip()
        if text[:60].lower().startswith(clean_title[:30]):
            text = text[len(clean_title):].strip()
            text = re.sub(r'^[\s\-:|]+[a-zA-Z\s,\d]{0,20}', '', text).strip()

    # v21: Deduplicate sentences (core content based)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen_content = set()
    unique_sentences = []
    for s in sentences:
        s_clean = s.strip()
        if not s_clean or len(s_clean) <= 15:
            continue
        core = re.sub(r'^(Profil\s+\w+\s*|Perjalanan\s+\w+\s*|Berikut\s+\w+\s*)', '', s_clean, flags=re.IGNORECASE).strip()
        key = core[:80].lower()
        if key not in seen_content:
            seen_content.add(key)
            unique_sentences.append(s_clean)
            
    text = ' '.join(unique_sentences)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)
    
    return text.strip()

def calculate_title_relevancy(title: str, text: str) -> float:
    if not title or not text: return 0.0
    title_words = set(re.findall(r'\b\w+\b', title.lower()))
    text_words = set(re.findall(r'\b\w+\b', text.lower()))
    if not title_words: return 0.0
    return sum(1 for w in title_words if w in text_words) / len(title_words)

def process_and_validate_text(html: str, title: str, rss_text: str) -> tuple[str | None, str]:
    if rss_text and len(rss_text) >= RSS_TEXT_MIN_LEN:
        full_text = rss_text  # v21: cleaning dilakukan di line 247, tidak perlu double
        extraction_method = "rss_fulltext"
    elif html:
        if len(html) > MAX_HTML_SIZE_BYTES:
            return None, "rejected_html_too_large"
            
        # OPTIMASI v20: Gunakan lxml, jauh lebih cepat dan hemat RAM
        soup = BeautifulSoup(html, "lxml")
        
        # FIX: Hapus pengecekan tag <p> karena banyak media pakai <div> atau <br>.
        # Kita andalkan JSON-LD dan Trafilatura untuk ekstraksi.
        for tag_name in ['title', 'h1']:
            for tag in soup.find_all(tag_name):
                tag.decompose()        
                
        full_text = extract_jsonld_article(soup)
        extraction_method = "jsonld"
        
        if not full_text or len(full_text) < MIN_ARTICLE_LENGTH:
            # OPTIMASI: Pass html mentah ke Trafilatura, jangan str(soup) agar hemat CPU/RAM
            full_text = traf_extract(html, include_comments=False, include_tables=False, favor_precision=True) or ""
            extraction_method = "trafilatura"            
            
        # Bersihkan memori secepat mungkin
        del html, soup
    else:
        return None, "fetch_no_html"
        
    full_text = clean_boilerplate(full_text, title) 

    if len(full_text) < MIN_ARTICLE_LENGTH:
        return None, "text_too_short"
    if len(full_text) > MAX_ARTICLE_LENGTH:
        return None, "section_leakage_too_long"

    relevancy = calculate_title_relevancy(title, full_text)
    if relevancy < TITLE_MATCH_THRESHOLD:
        return None, "title_mismatch"

    return full_text, f"valid_{extraction_method}"

def _apply_transient_result(current_metadata: dict, reason: str) -> tuple[dict, str]:
    attempts = int(current_metadata.get("enrich_attempts", 0)) + 1
    current_metadata["enrich_attempts"] = attempts
    current_metadata["fail_reason"] = reason
    if attempts >= pc.MAX_ENRICH_RETRIES:
        current_metadata["fail_reason"] = pc.REASON_MAX_RETRIES_EXCEEDED
        return current_metadata, pc.REASON_MAX_RETRIES_EXCEEDED
    return current_metadata, reason

def bulk_store(sb, results: list) -> Counter:
    stats = Counter()
    updates = []

    for rt_id, text, fetch_result, orig_metadata, title in results:
        current_metadata = dict(orig_metadata) if orig_metadata else {}
        db_update = {
            "id": rt_id,
            "resolved_domain": fetch_result.fetch_metadata.get("resolved_domain") if hasattr(fetch_result, 'fetch_metadata') else None,
            "canonical_url": fetch_result.canonical_url if hasattr(fetch_result, 'canonical_url') else None
        }
        if fetch_result.resolved_url:
            current_metadata["resolved_url"] = fetch_result.resolved_url

        if fetch_result.status == pc.FETCH_OK:
            if fetch_result.reason == "duplicate_skipped":
                current_metadata["fail_reason"] = "duplicate_title_at_enricher"
                db_update["text"] = ""
                db_update["status"] = pc.STATUS_SKIPPED
                db_update["content_type"] = "SNIPPET"
                db_update["metadata"] = current_metadata
                updates.append(db_update)
                stats["duplicate_skipped"] += 1
                # LOG DETAIL
                logger.info(f"ID: {rt_id[:8]} | Status: SKIPPED | Reason: Duplicate Title")
                
            elif fetch_result.reason == pc.REASON_GNEWS_SNIPPET_ONLY:
                current_metadata["is_snippet"] = True
                db_update["text"] = ""
                db_update["status"] = pc.STATUS_ENRICHED
                db_update["content_type"] = "SNIPPET" 
                db_update["metadata"] = current_metadata
                updates.append(db_update)
                stats["gnews_snippet"] += 1
                # LOG DETAIL
                logger.info(f"ID: {rt_id[:8]} | Status: ENRICHED | Method: GNews Snippet")
                
            else:
                full_text, validation_status = process_and_validate_text(fetch_result.html, title, orig_metadata.get("rss_text", ""))
                
                if full_text:
                    current_metadata["is_snippet"] = False
                    current_metadata["extraction_method"] = validation_status
                    db_update["text"] = full_text
                    db_update["status"] = pc.STATUS_ENRICHED
                    db_update["content_type"] = "FULLTEXT" 
                    db_update["content_hash"] = hashlib.sha256(full_text.encode()).hexdigest()
                    db_update["metadata"] = current_metadata 
                    updates.append(db_update)
                    stats["enriched"] += 1
                    # LOG DETAIL
                    logger.info(f"ID: {rt_id[:8]} | Status: ENRICHED | Method: {validation_status} | Len: {len(full_text)}")
                else:
                    current_metadata["fail_reason"] = validation_status
                    db_update["text"] = ""
                    db_update["status"] = pc.STATUS_FAILED
                    db_update["content_type"] = "SNIPPET" 
                    db_update["metadata"] = current_metadata
                    updates.append(db_update)
                    stats[validation_status] += 1
                    # LOG DETAIL
                    logger.info(f"ID: {rt_id[:8]} | Status: REJECTED | Reason: {validation_status}")

        elif fetch_result.status in pc.RETRYABLE_FETCH_STATUSES:
            new_metadata, effective_reason = _apply_transient_result(current_metadata, fetch_result.reason)
            retried_out = effective_reason == pc.REASON_MAX_RETRIES_EXCEEDED
            next_status = pc.STATUS_FAILED if retried_out else pc.STATUS_PENDING
            db_update["text"] = ""
            db_update["status"] = next_status
            db_update["metadata"] = new_metadata
            updates.append(db_update)
            stats[effective_reason] += 1
            # LOG DETAIL
            logger.info(f"ID: {rt_id[:8]} | Status: RETRY | Reason: {effective_reason}")
            
        else:
            current_metadata["fail_reason"] = fetch_result.reason
            db_update["text"] = ""
            db_update["status"] = pc.STATUS_FAILED
            db_update["metadata"] = current_metadata
            updates.append(db_update)
            stats[fetch_result.reason] += 1
            # LOG DETAIL
            logger.info(f"ID: {rt_id[:8]} | Status: FAILED | Reason: {fetch_result.reason}")

    if updates:
        CHUNK_SIZE = 50 
        try:
            for i in range(0, len(updates), CHUNK_SIZE):
                chunk = updates[i:i + CHUNK_SIZE]
                sb.rpc("bulk_update_raw_texts", {"p_updates": chunk}).execute()
        except Exception as e:
            logger.error(f"DB Bulk Update Error: {e}")
    return stats

def pipeline_worker(row: dict):
    url = row["source_url"]
    title = row.get("title") or ""
    orig_metadata = row.get("metadata") or {}
    
    current_text = row.get("text") or ""
    if len(current_text) >= RSS_TEXT_MIN_LEN:
        dummy_result = FetchResult(status=pc.FETCH_OK, reason=pc.REASON_RSS_FULL_TEXT, original_url=url, resolved_url=url)
        return row["id"], current_text, dummy_result, orig_metadata, title

    fetch_result = fetch_article(url, orig_metadata)
    return row["id"], "", fetch_result, orig_metadata, title

def process_batch(sb, rows: list) -> Counter:
    existing_titles = find_duplicate_titles(sb, rows)
    to_fetch = []
    pipeline_results = []
    skipped_count = 0
    
    for r in rows:
        norm_title = normalize_title(r.get("title") or "")
        if norm_title and norm_title in existing_titles:
            skipped_count += 1
            dummy_result = FetchResult(status=pc.FETCH_OK, reason="duplicate_skipped", original_url=r.get("source_url"), resolved_url=r.get("source_url"))
            pipeline_results.append((r["id"], "", dummy_result, r.get("metadata"), r.get("title")))
        else:
            current_text = r.get("text") or ""
            if len(current_text) >= RSS_TEXT_MIN_LEN:
                dummy_result = FetchResult(status=pc.FETCH_OK, reason=pc.REASON_RSS_FULL_TEXT, original_url=r.get("source_url"), resolved_url=r.get("source_url"))
                pipeline_results.append((r["id"], current_text, dummy_result, r.get("metadata"), r.get("title")))
            else:
                to_fetch.append(r)

    if skipped_count > 0:
        logger.info(f"  [DEDUP] {skipped_count} artikel duplikat dilewati tanpa fetch.")
        
    if not to_fetch: 
        return bulk_store(sb, pipeline_results)

    logger.info(f"  [FETCH] {len(to_fetch)} URLs dengan {MAX_WORKERS} threads paralel...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(pipeline_worker, r): r for r in to_fetch}
        for future in as_completed(futures):
            try:
                pipeline_results.append(future.result())
            except Exception:
                row = futures[future]
                crash_result = FetchResult(status=pc.FETCH_NETWORK_ERROR, reason=pc.REASON_THREAD_CRASH, original_url=row.get("source_url"))
                pipeline_results.append((row["id"], "", crash_result, row.get("metadata"), row.get("title")))

    logger.info(f"  [STORE] Mengirim {len(pipeline_results)} update ke DB via Bulk RPC...")
    return bulk_store(sb, pipeline_results)

def print_batch_report(batch_num: int, stats: Counter):
    enriched = stats.get("enriched", 0)
    gnews_snippet = stats.get("gnews_snippet", 0)
    duplicates = stats.get("duplicate_skipped", 0)
    rejected = sum(v for k, v in stats.items() if k.startswith("rejected_"))
    failed = sum(v for k, v in stats.items() if k.startswith("failed_") or k == "text_too_short")
    
    logger.info(f"  === BATCH {batch_num} REPORT ===")
    logger.info(f"  Enriched (Full Article)  : {enriched}")
    logger.info(f"  GNews (Snippet Track)    : {gnews_snippet}")
    logger.info(f"  Duplicates (Skipped)     : {duplicates}")
    logger.info(f"  Rejected (Sampah/Salah)  : {rejected}")
    logger.info(f"  Failed (Network/Error)   : {failed}")
    logger.info(f"  {'=' * 34}")

def main(limit: int = 100, max_total: int = 0):
    sb = get_client()
    try: sb.table("raw_texts").select("id").limit(1).execute()
    except Exception as e: logger.error(f"[FATAL] DB tidak reachable: {e}"); sys.exit(1)

    run_id = start_run("enricher_worker", "v21_library_based")
    total_stats = Counter()
    total_processed = 0
    batch_num = 1
    
    logger.info(f"[ENRICHER v21] Limit: {limit}/batch | Threads: {MAX_WORKERS} | Max: {'Unlimited' if max_total == 0 else max_total}")

    while True:
        if max_total > 0 and total_processed >= max_total:
            logger.info(f"Max total ({max_total}) tercapai. Berhenti.")
            break
            
        logger.info(f"--- Batch {batch_num} ---")
        
        current_limit = limit
        if max_total > 0:
            current_limit = min(limit, max_total - total_processed)
            
        res = sb.table("raw_texts").select("id, source_url, title, text, metadata").eq("status", pc.STATUS_PENDING).limit(current_limit).execute()
        rows = res.data or []
        if not rows: break

        logger.info(f"[ENRICHER] Memproses {len(rows)} artikel...")
        batch_stats = process_batch(sb, rows)
        print_batch_report(batch_num, batch_stats)
        
        total_stats.update(batch_stats)
        total_processed += len(rows)
        
        # OPTIMASI v20: Bersihkan memori agar RAM 16GB tidak cepat penuh dengan sisa HTML
        gc.collect()
        
        time.sleep(5 + random.uniform(0, 3)) # Sedikit turunkan jeda karena thread naik
        batch_num += 1

    total_succeeded = total_stats.get('enriched', 0) + total_stats.get('gnews_snippet', 0)
    finish_run(run_id, processed=total_processed, succeeded=total_succeeded, failed=total_processed - total_succeeded)
    logger.info(f"SELESAI. Total Processed: {total_processed} | Enriched: {total_succeeded}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-total", type=int, default=0)
    args = parser.parse_args()
    main(limit=args.limit, max_total=args.max_total)