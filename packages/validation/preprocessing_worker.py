"""
preprocessing_worker.py v10 — I/O & CPU Optimized
=============================================================
FIX v10:
  1. THREADED NORMALIZATION: Menggunakan ThreadPoolExecutor untuk memparalelkan 
     Regex cleaning & Hashing per artikel (Memanfaatkan CPU multi-core).
  2. I/O BATCHING: Menaikkan chunk size untuk query Duplikat (50 -> 100) dan 
     RPC Update (25 -> 50) untuk mengurangi round-trip network ke Supabase.
  3. HEADLINE DE-GLUE & DOMAIN STRIP: Mempertahankan fix v9 yang akurat.
  4. GC COLLECTION: Menambah garbage collection agar RAM stabil.
"""

import time  
import re
import gc
import hashlib
import unicodedata
import logging
import argparse
import random
import html as html_lib
from datetime import datetime, timezone, timedelta  
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

from packages.shared.db_client import get_client
from packages.shared.logger import start_run, finish_run
from packages.shared import constants as pc

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

PIPELINE_VERSION = "v10_io_cpu_optimized"
CHUNK_SIZE = 50  # Naikkan dari 25 ke 50 untuk mengurangi Network I/O
MAX_WORKERS = 4  # Thread untuk Regex & Hashing paralel

# ─────────────────────────────────────────────────────────────
# MODULAR NORMALIZATION PIPELINE
# ─────────────────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    text = html_lib.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\xa0", " ")
    return text

def remove_urls_emails(text: str) -> tuple[str, int]:
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', ' ', text)
    return text, int(len(urls) + len(emails))

def strip_news_boilerplate_safe(text: str, title: str = "") -> str:    
    if title:
        title_words = re.findall(r'\w+', title)[:8]
        if title_words:
            pattern_title = r'\W*'.join(re.escape(w) for w in title_words)
            match = re.match(r'^\s*' + pattern_title, text, re.IGNORECASE)
            if match:
                text = text[match.end():].lstrip(" :-\n\"'")
    
    # Hapus nama domain yang nyangkut
    domain_pattern = r'^[\w\.\-]+\.(com|id|co|tv|news|net)\b\s*[\-\|:]*\s*'
    for _ in range(2):
        text = re.sub(domain_pattern, '', text, flags=re.IGNORECASE)                
    
    text = re.sub(r'<[^>]+>', ' ', text)    
    patterns = [
        r"(?i)(baca juga|simak juga|berita terkait)\s*:[^.\n]*\.?",
        r"(?i)(reporter|editor|penulis|pewarta|jurnalis)\s*:\s*[^.\n]*\.?",
        r"(?i)(berlangganan|iklan|advertisement|sponsor)\s*[^.\n]*\.?",
        r"(?i)(copyright|©|hak cipta)\s*[^.\n]*\.?",
        r"(?i)(scroll ke bawah|mau berita terbaru|pilihan untuk lu)\s*[^.\n]*\.?"
    ]
    for p in patterns:
        text = re.sub(p, '', text)        
    text = re.sub(r'\(\s*(Foto|Instagram|Dok|Istimewa|Antara)[^)]*\)', '', text, flags=re.IGNORECASE)
    return text.strip(" :-\n\"'")

def normalize_punctuation(text: str) -> str:
    text = text.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")
    text = text.replace('–', '-').replace('—', '-').replace('―', '-').replace('‒', '-')
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    return text

def normalize_whitespace(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    return re.sub(r'[ \t]+', ' ', text).strip()

def normalize_pipeline(text: str, title: str = "") -> tuple[str, dict]:
    stats = {"original_len": int(len(text)), "clean_len": 0}
    text = normalize_unicode(text)
    text, removed_count = remove_urls_emails(text)
    stats["urls_emails_removed"] = int(removed_count)    
    text = strip_news_boilerplate_safe(text, title)
    text = normalize_punctuation(text)
    text = normalize_whitespace(text)
    stats["clean_len"] = int(len(text))
    return text, stats

def process_single_article(art: dict) -> dict:
    """Worker function untuk ThreadPoolExecutor"""
    title = art.get("title") or ""
    clean_text, audit_stats = normalize_pipeline(art.get("text") or "", title)
    
    if not clean_text:
        content_hash = None
    else:
        content_hash = hashlib.sha256(clean_text.encode()).hexdigest()
        
    return {
        "id": art["id"],
        "text": clean_text,
        "hash": content_hash,
        "metadata": {**(art.get("metadata") or {}), "audit_stats": audit_stats},
        "orig_metadata": art.get("metadata") or {}
    }

# ─────────────────────────────────────────────────────────────
# MAIN WORKER
# ─────────────────────────────────────────────────────────────
def main(limit: int = 100, max_total: int = 0):
    sb = get_client()
    run_id = start_run("preprocessing_worker", PIPELINE_VERSION)
    
    total_processed = 0
    total_normalized = 0
    total_duplicates = 0
    batch_num = 1
    start_time = time.perf_counter()

    logger.info(f"[PREPROCESSOR v10] Limit: {limit}/batch | Max: {'Unlimited' if max_total == 0 else max_total}")
    
    while True:
        if max_total > 0 and total_processed >= max_total:
            logger.info(f"Max total ({max_total}) tercapai. Berhenti.")
            break
            
        logger.info(f"--- Batch {batch_num} ---")
        
        current_limit = limit
        if max_total > 0:
            current_limit = min(limit, max_total - total_processed)
        
        try:
            time_filter = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            res = sb.table("raw_texts") \
                    .select("id, title, text, metadata") \
                    .eq("status", pc.STATUS_VALIDATED) \
                    .or_(f"preprocessing_version.is.null,preprocessing_version.neq.{PIPELINE_VERSION}") \
                    .gte("ingested_at", time_filter) \
                    .limit(current_limit) \
                    .execute()
        except Exception as e:
            logger.warning(f"DB Query Timeout/Error: {e}. Menunggu 10 detik...")
            time.sleep(10)
            continue

        articles = res.data or []
        if not articles:
            logger.info("Tidak ada lagi artikel untuk diproses.")
            break
            
        updates = []
        stats = {"normalized": 0, "duplicates": 0}
        now_iso = datetime.now(timezone.utc).isoformat()
        
        # === OPTIMASI v10: THREADED NORMALIZATION ===
        # Paralelkan proses Regex dan Hashing ke 4 threads
        processed_items = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(process_single_article, art): art for art in articles}
            for future in as_completed(futures):
                try:
                    processed_items.append(future.result())
                except Exception as e:
                    logger.error(f"Preprocess thread crashed: {e}")

        # Kumpulkan hash yang tidak None untuk cek duplikat
        batch_hashes = [item["hash"] for item in processed_items if item["hash"]]
        
        # === OPTIMASI v10: I/O BATCHING (100 per query) ===
        db_hash_map = {}
        if batch_hashes:
            hash_chunk_size = 100  # Naikkan dari 50 ke 100
            for i in range(0, len(batch_hashes), hash_chunk_size):
                chunk = batch_hashes[i:i + hash_chunk_size]
                try:
                    dup_res = sb.table("raw_texts") \
                                .select("id, content_hash") \
                                .in_("content_hash", chunk) \
                                .execute()
                    for row in (dup_res.data or []):
                        db_hash_map[row["content_hash"]] = row["id"]
                except Exception as e:
                    logger.warning(f"Gagal cek duplikat hash: {e}")
                
        for item in processed_items:
            # Jika hash None (teks kosong)
            if not item["hash"]:
                updates.append({
                    "id": item["id"], 
                    "text": item["text"], 
                    "content_hash": None,
                    "preprocessed_at": now_iso,
                    "metadata": item["metadata"],
                    "preprocessing_version": PIPELINE_VERSION
                })
                stats["normalized"] += 1
                # LOG DETAIL
                logger.info(f"ID: {item['id'][:8]} | Status: NORMALIZED (Empty Text)")
                continue

            # Jika hash sudah ada di DB (duplikat)
            if item["hash"] in db_hash_map and db_hash_map[item["hash"]] != item["id"]:
                updates.append({
                    "id": item["id"], 
                    "status": pc.STATUS_SKIPPED, 
                    "metadata": {**item["orig_metadata"], "fail_reason": "duplicate_content"},
                    "preprocessed_at": now_iso,
                    "preprocessing_version": PIPELINE_VERSION,
                    "duplicate_of": db_hash_map[item["hash"]] 
                })
                stats["duplicates"] += 1
                # LOG DETAIL
                logger.info(f"ID: {item['id'][:8]} | Status: SKIPPED | Reason: Duplicate Content")
                continue
                
            # Jika bersih dan unik
            updates.append({
                "id": item["id"], 
                "text": item["text"], 
                "content_hash": item["hash"],
                "preprocessed_at": now_iso,
                "metadata": item["metadata"],
                "preprocessing_version": PIPELINE_VERSION
            })
            stats["normalized"] += 1
            # LOG DETAIL
            logger.info(f"ID: {item['id'][:8]} | Status: NORMALIZED | Clean Len: {item['metadata']['audit_stats']['clean_len']}")
            
        if updates:
            try:
                # Chunk size RPC dinaikkan ke 50
                for i in range(0, len(updates), CHUNK_SIZE):
                    chunk = updates[i:i + CHUNK_SIZE]
                    sb.rpc("bulk_update_raw_texts", {"p_updates": chunk}).execute()
            except Exception as e: 
                logger.error(f"DB Bulk Update Error: {e}")
                
        logger.info(f"Normalized: {stats['normalized']} | Duplicates: {stats['duplicates']}")
        
        total_processed += len(articles)
        total_normalized += stats["normalized"]
        total_duplicates += stats["duplicates"]
        batch_num += 1
        
        # Bersihkan memori
        gc.collect()
        
        sleep_time = random.uniform(1, 3)  # Turunkan jeda karena I/O sudah dioptimasi
        logger.info(f"Menunggu {sleep_time:.1f}s sebelum batch berikutnya...")
        time.sleep(sleep_time)
        
    elapsed = time.perf_counter() - start_time
    logger.info("=" * 50)
    logger.info("SELESAI (Preprocessing v10)")
    logger.info(f"  Total Processed : {total_processed}")
    logger.info(f"  Total Normalized: {total_normalized}")
    logger.info(f"  Total Duplicates: {total_duplicates}")
    logger.info(f"  Waktu Eksekusi  : {elapsed:.2f}s")
    logger.info("=" * 50)
    
    finish_run(run_id, total_processed, total_normalized, total_duplicates)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-total", type=int, default=0)
    args = parser.parse_args()
    main(limit=args.limit, max_total=args.max_total)