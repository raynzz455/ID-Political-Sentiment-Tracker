"""
check_db_stats.py v3 — Detailed Audit & PGMQ Visibility
========================================================
FIX v3:
  1. PGMQ VISIBILITY: Memisahkan status 'queued' agar diketahui berapa
     artikel yang siap dimakan oleh NLP Worker.
  2. SKIP VS FAIL: Memisahkan 'skipped' (duplikat) dari 'failed' (error)
     agar angka penolakan tidak terlihat menakutkan dan akurat.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from collections import Counter

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

try:
    from supabase import create_client
except ImportError:
    print("[ERROR] pip install supabase"); sys.exit(1)

# Setup Clean Logging & Silence HTTPX Noise
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

def get_count(sb, table: str, column: str = None, value: str = None) -> int:
    """Helper untuk menghitung jumlah baris dengan filter opsional."""
    try:
        query = sb.table(table).select("id", count="exact")
        if column and value is not None:
            query = query.eq(column, value)
        res = query.limit(1).execute()
        return res.count if res.count else 0
    except Exception:
        return 0

def main():
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        logger.error("Set SUPABASE_URL & SUPABASE_SERVICE_ROLE_KEY di .env")
        sys.exit(1)
        
    sb = create_client(url, key)
    
    logger.info("=" * 50)
    logger.info("ID-SENTIMENT TRACKER: DETAILED HEALTH DASHBOARD")
    logger.info("=" * 50)
    
    # 1. VOLUME & STATUS BREAKDOWN
    logger.info("\n--- [ VOLUME & STATUS (Layer 1-4) ] ---")
    total = get_count(sb, "raw_texts")
    pending = get_count(sb, "raw_texts", "status", "pending")
    enriched = get_count(sb, "raw_texts", "status", "enriched")
    validated = get_count(sb, "raw_texts", "status", "validated")
    queued = get_count(sb, "raw_texts", "status", "queued")
    processed = get_count(sb, "raw_texts", "status", "processed")
    failed = get_count(sb, "raw_texts", "status", "failed")
    skipped = get_count(sb, "raw_texts", "status", "skipped")
    
    logger.info(f"  Total Articles : {total:>6}")
    logger.info(f"  - Pending      : {pending:>6}")
    logger.info(f"  - Enriched     : {enriched:>6}")
    logger.info(f"  - Validated    : {validated:>6}")
    logger.info(f"  - Queued (NLP) : {queued:>6}  <-- Siap diproses AI IndoBERT")
    logger.info(f"  - Processed    : {processed:>6}  <-- Sudah punya skor sentimen")
    logger.info(f"  - Skipped (Dup): {skipped:>6}  <-- Duplikat yang dibuang")
    logger.info(f"  - Failed (Err) : {failed:>6}  <-- Gagal karena error sistem")

    # 2. CONTENT QUALITY & ANOMALIES
    logger.info("\n--- [ CONTENT QUALITY ] ---")
    fulltext = get_count(sb, "raw_texts", "content_type", "FULLTEXT")
    snippet = get_count(sb, "raw_texts", "content_type", "SNIPPET")
    logger.info(f"  Fulltext Valid : {fulltext:>6}")
    logger.info(f"  Snippet (GNews): {snippet:>6}")
    
    # 3. ALASAN KEGAGALAN (Hanya yang Failed, bukan Skipped)
    try:
        fail_res = sb.table("raw_texts").select("metadata").eq("status", "failed").limit(1000).execute()
        reasons = Counter()
        for row in (fail_res.data or []):
            meta = row.get("metadata") or {}
            reason = meta.get("fail_reason", "unknown")
            reasons[reason] += 1
        
        if reasons:
            logger.info("\n--- [ TOP 5 ALASAN FAILED (Error Asli) ] ---")
            for reason, count in reasons.most_common(5):
                logger.info(f"  - {reason:30s}: {count}")
    except Exception:
        pass

    # 4. ENTITY & CONTEXT (Layer 3)
    logger.info("\n--- [ ENTITY & CONTEXT (Layer 3) ] ---")
    mentions = get_count(sb, "entity_mentions")
    contexts = get_count(sb, "entity_contexts")
    logger.info(f"  Entity Mentions: {mentions:>6}")
    logger.info(f"  Contexts Built : {contexts:>6}")

    # 5. SENTIMENT OUTPUT (Layer 4)
    logger.info("\n--- [ SENTIMENT OUTPUT (Layer 4) ] ---")
    total_sentiments = get_count(sb, "sentiment_scores")
    logger.info(f"  Total Scores   : {total_sentiments:>6}")
    
    if total_sentiments > 0:
        try:
            # Ambil sample untuk distribusi persentase
            sent_res = sb.table("sentiment_scores").select("label").limit(10000).execute()
            sent_dist = Counter(r["label"] for r in (sent_res.data or []))
            total_sample = sum(sent_dist.values())
            
            if total_sample > 0:
                pos = sent_dist.get('positive', 0)
                neg = sent_dist.get('negative', 0)
                neu = sent_dist.get('neutral', 0)
                logger.info(f"  - Positive     : {pos} ({(pos/total_sample*100):.1f}%)")
                logger.info(f"  - Negative     : {neg} ({(neg/total_sample*100):.1f}%)")
                logger.info(f"  - Neutral      : {neu} ({(neu/total_sample*100):.1f}%)")
        except Exception:
            pass

    # 6. PIPELINE DETAIL — Entity Resolution, Context, NLP Readiness
    logger.info("\n--- [ PIPELINE DETAIL (Layer 3.2 - 3.7) ] ---")
    
    # Entity Resolution stats
    ent_resolved = get_count(sb, "raw_texts", "entity_resolved_at", None)  # not null
    # Can't easily query "not null" with helper, use direct
    try:
        er = sb.table("raw_texts").select("id", count="exact").not_.is_("entity_resolved_at", "null").eq("status", "validated").limit(1).execute()
        ent_resolved = er.count
    except: ent_resolved = 0
    
    ctx_extracted = 0
    try:
        cr = sb.table("raw_texts").select("id", count="exact").not_.is_("context_extracted_at", "null").eq("status", "validated").limit(1).execute()
        ctx_extracted = cr.count
    except: pass
    
    nlp_ready = 0
    try:
        nr = sb.table("raw_texts").select("id", count="exact").not_.is_("nlp_ready_at", "null").limit(1).execute()
        nlp_ready = nr.count
    except: pass
    
    logger.info(f"  Entity Resolved : {ent_resolved:>6}")
    logger.info(f"  Context Built   : {ctx_extracted:>6}")
    logger.info(f"  NLP Ready       : {nlp_ready:>6}")
    
    # Entity Resolution version breakdown
    try:
        rv_res = sb.table("raw_texts").select("resolver_version").not_.is_("entity_resolved_at", "null").limit(500).execute()
        rv_dist = Counter(r.get("resolver_version", "unknown") for r in (rv_res.data or []))
        if rv_dist:
            logger.info(f"  Resolver Versions (sample 500):")
            for v, c in rv_dist.most_common(5):
                logger.info(f"    {v or 'unknown':30s}: {c}")
    except: pass
    
    # 7. CONTENT TYPE x STATUS CROSS-TAB
    logger.info("\n--- [ CONTENT TYPE x STATUS ] ---")
    for ct in ['FULLTEXT', 'SNIPPET']:
        for st in ['validated', 'queued', 'processed', 'failed', 'skipped']:
            try:
                r = sb.table("raw_texts").select("id", count="exact").eq("content_type", ct).eq("status", st).limit(1).execute()
                if r.count > 0:
                    logger.info(f"  {ct:8s} + {st:12s}: {r.count:>6}")
            except: pass
    
    # 8. DATASET EXPORT POTENTIAL
    logger.info("\n--- [ DATASET EXPORT POTENTIAL ] ---")
    # Articles with: context + sentiment → ready for dataset
    try:
        # Count contexts with sentiment
        ctx_total = get_count(sb, "entity_contexts")
        
        # Count scores (targeted + fallback)
        scores_targeted = get_count(sb, "sentiment_scores")  # all scores
        
        # Scores with entity_id
        scores_with_ent = 0
        try:
            se = sb.table("sentiment_scores").select("id", count="exact").not_.is_("entity_id", "null").limit(1).execute()
            scores_with_ent = se.count
        except: pass
        
        # Scores without entity_id (fallback)
        scores_fallback = scores_targeted - scores_with_ent
        
        logger.info(f"  Entity Contexts     : {ctx_total:>6}")
        logger.info(f"  Sentiment Scores    : {scores_targeted:>6}")
        logger.info(f"    - Targeted (ent)  : {scores_with_ent:>6}")
        logger.info(f"    - Fallback (gen)  : {scores_fallback:>6}")
        logger.info(f"  Est. Dataset Rows   : {min(ctx_total, scores_targeted):>6}  <-- Max possible export")
    except: pass
    
    # 9. GNEWS RESOLVER POTENTIAL
    logger.info("\n--- [ GNEWS RESOLVER POTENTIAL ] ---")
    snippet_pending = 0
    snippet_failed = 0
    snippet_skipped = 0
    try:
        r = sb.table("raw_texts").select("id", count="exact").eq("content_type", "SNIPPET").eq("status", "pending").limit(1).execute()
        snippet_pending = r.count
    except: pass
    try:
        r = sb.table("raw_texts").select("id", count="exact").eq("content_type", "SNIPPET").eq("status", "failed").limit(1).execute()
        snippet_failed = r.count
    except: pass
    try:
        r = sb.table("raw_texts").select("id", count="exact").eq("content_type", "SNIPPET").eq("status", "skipped").limit(1).execute()
        snippet_skipped = r.count
    except: pass
    
    logger.info(f"  Snippet Pending   : {snippet_pending:>6}  <-- Siap di-resolve")
    logger.info(f"  Snippet Failed    : {snippet_failed:>6}  <-- Bisa di-reset untuk retry")
    logger.info(f"  Snippet Skipped   : {snippet_skipped:>6}  <-- Bisa di-reset untuk retry")
    logger.info(f"  Total Resolvable  : {snippet_pending + snippet_failed + snippet_skipped:>6}")
    
    logger.info("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    main()