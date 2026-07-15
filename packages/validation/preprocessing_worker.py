"""
preprocessing_worker.py v4 — Modular Pipeline & Chunked RPC
=============================================================
FIX v4:
  1. CHUNKED RPC: Membagi payload bulk_update_raw_texts ke dalam chunk 50 baris.
     Mencegah error "JSON could not be generated" karena request body terlalu besar.
  2. TYPE CASTING: Memastikan semua nilai di audit_stats adalah int murni.
"""
import os
import re
import sys
import hashlib
import unicodedata
from datetime import datetime, timezone, time
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

try:
    from supabase import create_client, Client
except ImportError as e:
    print(f"[ERROR] {e}"); sys.exit(1)

from packages.shared.db_client import get_client
from packages.shared.logger import start_run, finish_run
from packages.shared import constants as pc

PIPELINE_VERSION = "v4_chunked"
CHUNK_SIZE = 50  # Bagi RPC menjadi 50 baris per request

# ─────────────────────────────────────────────────────────────
# MODULAR NORMALIZATION PIPELINE
# ─────────────────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\xa0", " ")

def remove_urls_emails(text: str) -> tuple[str, int]:
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', ' ', text)
    return text, int(len(urls) + len(emails))  # Cast to int

def strip_news_boilerplate(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    patterns = [
        r"baca juga:.*?(?=\n|$)", r"simak juga:.*?(?=\n|$)", r"berlangganan.*?(?=\n|$)",
        r"advertisement.*?(?=\n|$)", r"iklan.*?(?=\n|$)",
        r"reporter:.*?(?=\n|$)", r"editor:.*?(?=\n|$)", r"penulis:.*?(?=\n|$)",
        r"copyright.*?(?=\n|$)", r"©.*?(?=\n|$)"
    ]
    for p in patterns:
        text = re.sub(p, '', text, flags=re.IGNORECASE)
    return text

def normalize_punctuation(text: str) -> str:
    text = text.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")
    text = text.replace('–', '-').replace('—', '-').replace('―', '-').replace('‒', '-')
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    return text

def normalize_whitespace(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    return re.sub(r'[ \t]+', ' ', text).strip()

def normalize_pipeline(text: str) -> tuple[str, dict]:
    # Pastikan semua nilai adalah int murni (bukan numpy atau tipe lain)
    stats = {
        "original_len": int(len(text)), 
        "urls_emails_removed": 0, 
        "clean_len": 0
    }
    text = normalize_unicode(text)
    text, removed_count = remove_urls_emails(text)
    stats["urls_emails_removed"] = int(removed_count)
    text = strip_news_boilerplate(text)
    text = normalize_punctuation(text)
    text = normalize_whitespace(text)
    stats["clean_len"] = int(len(text))
    return text, stats

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

    print(f"[PREPROCESSOR] Limit: {limit}/batch | Max: {'Unlimited' if max_total == 0 else max_total}")
    
    while True:
        if max_total > 0 and total_processed >= max_total:
            break
            
        print(f"\n--- Batch {batch_num} ---")
        res = sb.table("raw_texts") \
                .select("id, text, metadata") \
                .eq("status", pc.STATUS_VALIDATED) \
                .or_(f"preprocessing_version.is.null,preprocessing_version.neq.{PIPELINE_VERSION}") \
                .limit(limit) \
                .execute()
                
        articles = res.data or []
        if not articles:
            print("[PREPROCESSOR] Tidak ada lagi artikel untuk diproses.")
            break
            
        updates = []
        stats = {"normalized": 0, "duplicates": 0}
        now_iso = datetime.now(timezone.utc).isoformat()
        
        processed_items = []
        batch_hashes = set()
        
        for art in articles:
            clean_text, audit_stats = normalize_pipeline(art.get("text") or "")
            content_hash = hashlib.sha256(clean_text.encode()).hexdigest()
            
            processed_items.append({
                "id": art["id"],
                "text": clean_text,
                "hash": content_hash,
                "metadata": {**(art.get("metadata") or {}), "audit_stats": audit_stats},
                "orig_metadata": art.get("metadata") or {}
            })
            batch_hashes.add(content_hash)
            
        db_hash_map = {}
        if batch_hashes:
            dup_res = sb.table("raw_texts") \
                        .select("id, content_hash") \
                        .in_("content_hash", list(batch_hashes)) \
                        .execute()
            for row in (dup_res.data or []):
                db_hash_map[row["content_hash"]] = row["id"]
                
        for item in processed_items:
            if item["hash"] in db_hash_map and db_hash_map[item["hash"]] != item["id"]:
                updates.append({
                    "id": item["id"], 
                    "status": pc.STATUS_SKIPPED, 
                    "metadata": {**item["orig_metadata"], "fail_reason": "duplicate_content"},
                    "preprocessed_at": now_iso,
                    "pipeline_version": PIPELINE_VERSION,
                    "duplicate_of": db_hash_map[item["hash"]] 
                })
                stats["duplicates"] += 1
                continue
                
            updates.append({
                "id": item["id"], 
                "text": item["text"], 
                "content_hash": item["hash"],
                "preprocessed_at": now_iso,
                "metadata": item["metadata"],
                "pipeline_version": PIPELINE_VERSION
            })
            stats["normalized"] += 1
            
        # CHUNKED RPC CALL
        if updates:
            try:
                for i in range(0, len(updates), CHUNK_SIZE):
                    chunk = updates[i:i + CHUNK_SIZE]
                    sb.rpc("bulk_update_raw_texts", {"p_updates": chunk}).execute()
            except Exception as e: 
                print(f"[DB_ERROR] {e}")
                
        print(f"[PREPROCESSOR] Normalized: {stats['normalized']} | Duplicates: {stats['duplicates']}")
        
        total_processed += len(articles)
        total_normalized += stats["normalized"]
        total_duplicates += stats["duplicates"]
        batch_num += 1
        
    elapsed = time.perf_counter() - start_time
    print(f"\n{'='*50}")
    print(f"SELESAI (Preprocessing)")
    print(f"  Total Processed : {total_processed}")
    print(f"  Total Normalized: {total_normalized}")
    print(f"  Total Duplicates: {total_duplicates}")
    print(f"  Waktu Eksekusi  : {elapsed:.2f}s")
    print(f"{'='*50}")
    finish_run(run_id, total_processed, total_normalized, total_duplicates)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-total", type=int, default=0)
    args = parser.parse_args()
    main(limit=args.limit, max_total=args.max_total)