"""
nlp_readiness_worker.py v7 — Safe Enqueue & Context Threshold Fix
====================================================================
FIX v7:
  1. SAFE ENQUEUE: Memasukkan ID ke PGMQ terlebih dahulu. Jika berhasil, baru
     update status DB menjadi 'queued'. Mencegah artikel hilang dari antrian.
  2. CONTEXT THRESHOLD FIX: Menurunkan MIN_CONTEXT_LEN dari 100 ke 50 agar
     konteks kalimat pendek yang padat (hasil Token Cap v14) tidak terbuang.
"""
import re
import time
import logging
import argparse
from datetime import datetime, timezone, timedelta

from packages.shared.db_client import get_client
from packages.shared.logger import start_run, finish_run
from packages.shared import constants as pc

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

READINESS_VERSION = "v7_safe_enqueue"
MIN_CONTEXT_LEN = 50  # Turunkan dari 100 ke 50
MIN_QUALITY_SCORE = 20
MIN_FULLTEXT_LEN = 150

def normalize_title(title: str) -> str:
    if not title: return ""
    title = title.lower().strip()
    title = re.sub(r'[\[\]\(\)\{\}"\':;,!?./]', '', title)
    title = re.sub(r'\s+', ' ', title)
    return title

def main(limit: int = 100, max_total: int = 0):
    sb = get_client()
    run_id = start_run("nlp_readiness_worker", READINESS_VERSION)
    
    total_processed = 0
    total_ready = 0
    total_rejected = 0
    total_duplicates = 0
    batch_num = 1

    logger.info(f"[NLP_READINESS v7] Limit: {limit}/batch | Max: {'Unlimited' if max_total == 0 else max_total}")

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
                    .not_.is_("context_extracted_at", "null") \
                    .is_("nlp_ready_at", "null") \
                    .gte("ingested_at", time_filter) \
                    .limit(current_limit) \
                    .execute()
        except Exception as e:
            logger.warning(f"DB Query Timeout/Error: {e}. Menunggu 10 detik...")
            time.sleep(10)
            continue
                
        articles = res.data or []
        if not articles:
            logger.info("Tidak ada artikel untuk di-readiness.")
            break
            
        art_ids = [a["id"] for a in articles]
        art_titles = [normalize_title(a.get("title") or "") for a in articles]
        
        # 1. BATCH QUERY: Cek duplikasi judul
        existing_titles = set()
        titles_to_check = [a.get("title") or "" for a in articles if a.get("title")]
        chunk_size = 50
        
        for i in range(0, len(titles_to_check), chunk_size):
            chunk = titles_to_check[i:i + chunk_size]
            try:
                dup_res = sb.table("raw_texts") \
                            .select("title") \
                            .in_("title", chunk) \
                            .not_.is_("nlp_ready_at", "null") \
                            .not_.in_("id", art_ids) \
                            .execute()
                for row in (dup_res.data or []):
                    existing_titles.add(normalize_title(row.get("title") or ""))
            except Exception as e:
                logger.warning(f"Gagal cek duplikat judul: {e}")

        # 2. BATCH QUERY: Ambil semua contexts
        try:
            ctx_res = sb.table("entity_contexts") \
                        .select("id, raw_text_id, context_text, metadata") \
                        .in_("raw_text_id", art_ids) \
                        .execute()
        except Exception as e:
            logger.warning(f"Gagal ambil contexts: {e}. Menunggu 5 detik...")
            time.sleep(5)
            continue
                    
        contexts_by_art = {}
        invalid_ctx_ids = []
        
        for ctx in (ctx_res.data or []):
            art_id = ctx["raw_text_id"]
            ctx_text = ctx.get("context_text") or ""
            meta = ctx.get("metadata") or {}
            quality_score = meta.get("quality_score", 0)
            
            if len(ctx_text) < MIN_CONTEXT_LEN or quality_score < MIN_QUALITY_SCORE:
                invalid_ctx_ids.append(ctx["id"])
            else:
                contexts_by_art.setdefault(art_id, []).append(ctx)
                
        if invalid_ctx_ids:
            try: sb.table("entity_contexts").delete().in_("id", invalid_ctx_ids).execute()
            except Exception as e: logger.error(f"Delete Context Error: {e}")
            
        # 3. KEPUTUSAN AKHIR NLP READINESS
        updates = []
        stats = {"ready": 0, "rejected": 0, "duplicate": 0}
        now_iso = datetime.now(timezone.utc).isoformat()
        
        for art, norm_title in zip(articles, art_titles):
            art_id = art["id"]
            metadata = art.get("metadata") or {}
            full_text = art.get("text") or ""
            
            # GATE 1: Cek Duplikat Judul
            if norm_title and norm_title in existing_titles:
                updates.append({
                    "id": art_id, "status": pc.STATUS_SKIPPED, 
                    "metadata": {**metadata, "fail_reason": "duplicate_title_at_gate"}
                })
                stats["duplicate"] += 1
                continue
                
            # GATE 2: Cek kelayakan teks utuh
            if len(full_text) < MIN_FULLTEXT_LEN:
                updates.append({
                    "id": art_id, "status": pc.STATUS_FAILED, 
                    "metadata": {**metadata, "fail_reason": "nlp_ready_fulltext_too_short"}
                })
                stats["rejected"] += 1
                continue
                
            valid_contexts = len(contexts_by_art.get(art_id, []))
            
            # GATE 3: Lolos jika ada context valid, ATAU teks utuh cukup panjang untuk fallback
            if valid_contexts > 0 or len(full_text) >= 500:
                
                # === SAFE ENQUEUE: Masuk PGMQ dulu, baru update DB ===
                try:
                    sb.rpc("enqueue_nlp_message", {"p_raw_text_id": art_id}).execute()
                    
                    # Jika PGMQ berhasil, baru masukkan ke list update sebagai 'queued'
                    updates.append({
                        "id": art_id, 
                        "status": pc.STATUS_QUEUED, 
                        "nlp_ready_at": now_iso,
                        "metadata": {**metadata, "nlp_readiness_version": READINESS_VERSION, "valid_ctx_count": valid_contexts}
                    })
                    stats["ready"] += 1
                    
                except Exception as e:
                    logger.error(f"Gagal enqueue PGMQ (ID: {art_id}): {e}")
                    updates.append({
                        "id": art_id, "status": pc.STATUS_FAILED, 
                        "metadata": {**metadata, "fail_reason": "pgmq_enqueue_failed"}
                    })
                    stats["rejected"] += 1
                    
            else:
                updates.append({
                    "id": art_id, "status": pc.STATUS_FAILED, 
                    "metadata": {**metadata, "fail_reason": "nlp_ready_no_valid_context"}
                })
                stats["rejected"] += 1
                
        # --- CHUNKED RPC UPDATE ---
        if updates:
            chunk_size = 25
            for i in range(0, len(updates), chunk_size):
                chunk = updates[i:i + chunk_size]
                try: 
                    sb.rpc("bulk_update_raw_texts", {"p_updates": chunk}).execute()
                except Exception as e: 
                    logger.error(f"RPC Error (bulk_update_raw_texts): {e}")
                
        logger.info(f"Ready: {stats['ready']} | Rejected: {stats['rejected']} | Duplicates: {stats['duplicate']} | Junk Deleted: {len(invalid_ctx_ids)}")
        
        total_processed += len(articles)
        total_ready += stats["ready"]
        total_rejected += stats["rejected"]
        total_duplicates += stats["duplicate"]
        batch_num += 1
        
    finish_run(run_id, total_processed, total_ready, total_rejected)
    logger.info(f"Total Duplicates Skipped: {total_duplicates}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-total", type=int, default=0)
    args = parser.parse_args()
    main(limit=args.limit, max_total=args.max_total)