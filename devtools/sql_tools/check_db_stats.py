"""
check_db_status.py v2 — Detailed Audit & Clean Logging
========================================================
Cek kesehatan seluruh layer pipeline langsung dari terminal.
Usage: python -m devtools.sql_tools.check_db_stats
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
    logger.info("ID-SENTIMENT TRACKER: PIPELINE HEALTH DASHBOARD")
    logger.info("=" * 50)
    
    # 1. VOLUME & STATUS
    logger.info("\n--- [ VOLUME & STATUS (Layer 1-2) ] ---")
    total = get_count(sb, "raw_texts")
    pending = get_count(sb, "raw_texts", "status", "pending")
    enriched = get_count(sb, "raw_texts", "status", "enriched")
    validated = get_count(sb, "raw_texts", "status", "validated")
    processed = get_count(sb, "raw_texts", "status", "processed")
    failed = get_count(sb, "raw_texts", "status", "failed")
    skipped = get_count(sb, "raw_texts", "status", "skipped")
    
    logger.info(f"  Total Articles : {total:>6}")
    logger.info(f"  - Pending      : {pending:>6}")
    logger.info(f"  - Enriched     : {enriched:>6}")
    logger.info(f"  - Validated    : {validated:>6}")
    logger.info(f"  - Processed    : {processed:>6}")
    logger.info(f"  - Failed/Skip  : {failed + skipped:>6}")

    # 2. CONTENT QUALITY & ANOMALIES (Audit Adaptif)
    logger.info("\n--- [ CONTENT QUALITY & AUDIT ] ---")
    fulltext = get_count(sb, "raw_texts", "content_type", "FULLTEXT")
    snippet = get_count(sb, "raw_texts", "content_type", "SNIPPET")
    logger.info(f"  Fulltext       : {fulltext:>6}")
    logger.info(f"  Snippet (GNews): {snippet:>6}")
    
    # Cek Anomali: Section Leakage (Fulltext > 20000 chars)
    try:
        leak_res = sb.rpc("get_anomaly_count", {"p_type": "section_leakage"}).execute()
        leak_count = leak_res.data or 0
        if leak_count > 0:
            logger.warning(f"  [WARNING] Section Leakage (>20k chars): {leak_count} artikel (Perlu dibersihkan!)")
    except Exception:
        pass # Abaikan jika RPC belum ada

    # Cek Alasan Kegagalan (Ambil 1000 failed terakhir)
    try:
        fail_res = sb.table("raw_texts").select("metadata").eq("status", "failed").limit(1000).execute()
        reasons = Counter()
        for row in (fail_res.data or []):
            meta = row.get("metadata") or {}
            reason = meta.get("fail_reason", "unknown")
            reasons[reason] += 1
        
        if reasons:
            logger.info("\n--- [ TOP 5 ALASAN KEGAGALAN ] ---")
            for reason, count in reasons.most_common(5):
                logger.info(f"  - {reason:25s}: {count}")
    except Exception:
        pass

    # 3. ENTITY & CONTEXT (Layer 3)
    logger.info("\n--- [ ENTITY & CONTEXT (Layer 3) ] ---")
    mentions = get_count(sb, "entity_mentions")
    contexts = get_count(sb, "entity_contexts")
    logger.info(f"  Entity Mentions: {mentions:>6}")
    logger.info(f"  Contexts Built : {contexts:>6}")

    # 4. NLP & SENTIMENT (Layer 4)
    logger.info("\n--- [ SENTIMENT OUTPUT (Layer 4) ] ---")
    total_sentiments = get_count(sb, "sentiment_scores")
    logger.info(f"  Total Scores   : {total_sentiments:>6}")
    
    if total_sentiments > 0:
        try:
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

    logger.info("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    main()