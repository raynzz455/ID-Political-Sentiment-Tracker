"""
nlp_readiness_worker.py v5 — Final Gatekeeper & Title Deduplication
====================================================================
PERUBAAHAN v5:
  1. TITLE DEDUPLICATION: Mengecek apakah artikel dengan judul yang sama 
     sudah pernah masuk antrian NLP (queued/processed) sebelum memasukkannya ke PGMQ.
  2. NO N+1 QUERY: Ambil semua entity_contexts untuk 100 artikel sekali jalan.
  3. NO 1-by-1 DELETE: Kumpulkan ID context sampah, hapus sekali pakai IN().
  4. METADATA AWARE: Cek quality_score di metadata, bukan cuma panjang karakter.
"""
import os
import sys
import re
import logging
from datetime import datetime, timezone
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

# Setup Clean Logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

READINESS_VERSION = "v5_final_gate"
MIN_CONTEXT_LEN = 100
MIN_QUALITY_SCORE = 20
MIN_FULLTEXT_LEN = 150

def normalize_title(title: str) -> str:
    """Normalisasi judul untuk deteksi duplikat (lowercase, hapus tanda baca)."""
    if not title: return ""
    title = title.lower().strip()
    # Hapus tanda bata umum di judul berita
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

    logger.info(f"[NLP_READINESS v5] Limit: {limit}/batch | Max: {'Unlimited' if max_total == 0 else max_total}")

    while True:
        if max_total > 0 and total_processed >= max_total:
            break
            
        logger.info(f"--- Batch {batch_num} ---")
        res = sb.table("raw_texts") \
                .select("id, title, text, metadata") \
                .eq("status", pc.STATUS_VALIDATED) \
                .not_.is_("context_extracted_at", "null") \
                .is_("nlp_ready_at", "null") \
                .limit(limit) \
                .execute()
                
        articles = res.data or []
        if not articles:
            break
            
        art_ids = [a["id"] for a in articles]
        art_titles = [normalize_title(a.get("title") or "") for a in articles]
        
        # 1. BATCH QUERY: Cek duplikasi judul di DB (yang sudah queued/processed)
        existing_titles = set()
        try:
            # Cari artikel lain yang punya judul sama, tapi sudah pernah masuk NLP (nlp_ready_at tidak null)
            dup_res = sb.table("raw_texts") \
                        .select("title") \
                        .in_("title", [a.get("title") or "" for a in articles]) \
                        .not_.is_("nlp_ready_at", "null") \
                        .neq("id", art_ids[0]) \
                        .execute()
            for row in (dup_res.data or []):
                existing_titles.add(normalize_title(row.get("title") or ""))
        except Exception as e:
            logger.warning(f"Gagal cek duplikat judul: {e}")

        # 2. BATCH QUERY: Ambil semua contexts untuk 100 artikel sekaligus
        ctx_res = sb.table("entity_contexts") \
                    .select("id, raw_text_id, context_text, metadata") \
                    .in_("raw_text_id", art_ids) \
                    .execute()
                    
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
                logger.info(f"  [SKIPPED] ID: {art_id[:8]} | Reason: Duplicate Title")
                continue
                
            # GATE 2: Cek kelayakan teks utuh (Fallback National Index)
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
                updates.append({
                    "id": art_id, 
                    "nlp_ready_at": now_iso,
                    "metadata": {**metadata, "nlp_readiness_version": READINESS_VERSION, "valid_ctx_count": valid_contexts}
                })
                stats["ready"] += 1
            else:
                updates.append({
                    "id": art_id, "status": pc.STATUS_FAILED, 
                    "metadata": {**metadata, "fail_reason": "nlp_ready_no_valid_context"}
                })
                stats["rejected"] += 1
                
        if updates:
            try: sb.rpc("bulk_update_raw_texts", {"p_updates": updates}).execute()
            except Exception as e: logger.error(f"DB Error: {e}")
                
        logger.info(f"Ready: {stats['ready']} | Rejected: {stats['rejected']} | Duplicates: {stats['duplicate']} | Junk Deleted: {len(invalid_ctx_ids)}")
        
        total_processed += len(articles)
        total_ready += stats["ready"]
        total_rejected += stats["rejected"]
        total_duplicates += stats["duplicate"]
        batch_num += 1
        
    finish_run(run_id, total_processed, total_ready, total_rejected)
    logger.info(f"Total Duplicates Skipped: {total_duplicates}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-total", type=int, default=0)
    args = parser.parse_args()
    main(limit=args.limit, max_total=args.max_total)