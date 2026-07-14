"""
check_db_status.py — Pipeline Health Dashboard
================================================
Cek kesehatan seluruh layer pipeline langsung dari terminal.
Usage: python -m devtools.check_db_status
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from collections import Counter

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

try:
    from supabase import create_client
except ImportError:
    print("[ERROR] pip install supabase"); sys.exit(1)

def main():
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        print("[ERROR] Set SUPABASE_URL & SUPABASE_SERVICE_ROLE_KEY")
        sys.exit(1)
        
    sb = create_client(url, key)
    
    print("\n" + "="*50)
    print("📊 ID-SENTIMENT TRACKER: PIPELINE HEALTH DASHBOARD")
    print("="*50)
    
    # 1. Ambil statistik utama (1x RPC call)
    try:
        res = sb.rpc("get_db_stats").execute()
        stats = res.data or {}
    except Exception as e:
        print(f"[ERROR] Gagal mengambil stats: {e}")
        return

    if not stats:
        print("[!] Tidak ada data ditemukan.")
        return

    # 2. Ambil distribusi sentimen
    sent_dist = {}
    try:
        sent_res = sb.table("sentiment_scores") \
                       .select("label") \
                       .not_.is_("entity_id", "null") \
                       .execute()
        sent_dist = Counter(r["label"] for r in (sent_res.data or []))
    except:
        pass

    # 3. Format Output
    print("\n--- [ VOLUME & STATUS ] ---")
    print(f"  Total Articles : {stats.get('total_articles', 0):>6}")
    print(f"  - Pending      : {stats.get('status_pending', 0):>6}")
    print(f"  - Enriched     : {stats.get('status_enriched', 0):>6}")
    print(f"  - Validated    : {stats.get('status_validated', 0):>6}")
    print(f"  - Processed    : {stats.get('status_processed', 0):>6}")
    print(f"  - Failed/Skip  : {stats.get('status_failed', 0) + stats.get('status_skipped', 0):>6}")

    print("\n--- [ CONTENT TYPE & ENRICHMENT ] ---")
    print(f"  Fulltext       : {stats.get('type_fulltext', 0):>6}")
    print(f"  Snippet (GNews): {stats.get('type_snippet', 0):>6}")
    print(f"  Avg Text Length: {stats.get('avg_fulltext_len', 0):>6} chars")
    
    bad_snip = stats.get('bad_snippets', 0)
    if bad_snip > 0:
        print(f"  ⚠️ Bad Snippets : {bad_snip:>6} (Snippet > 500 chars, perlu cek logic!)")
    else:
        print(f"  ✅ Bad Snippets : {bad_snip:>6}")

    print("\n--- [ ENTITY & CONTEXT (Layer 3) ] ---")
    print(f"  Entity Mentions: {stats.get('total_mentions', 0):>6}")
    print(f"  Contexts Built : {stats.get('total_contexts', 0):>6}")

    print("\n--- [ NLP READINESS & QUEUE (Layer 3.7) ] ---")
    print(f"  NLP Ready      : {stats.get('nlp_ready', 0):>6}")
    print(f"  Queue (pgmq)   : {stats.get('queue_size', 0):>6}")

    print("\n--- [ SENTIMENT OUTPUT (Layer 4) ] ---")
    print(f"  Entity Scores  : {stats.get('total_entity_sentiments', 0):>6}")
    print(f"  Fallback Scores: {stats.get('total_fallback_sentiments', 0):>6}")
    
    if sent_dist:
        total_sent = sum(sent_dist.values())
        pos = sent_dist.get('positive', 0)
        neg = sent_dist.get('negative', 0)
        neu = sent_dist.get('neutral', 0)
        print(f"  - Positive     : {pos} ({(pos/total_sent*100):.1f}%)")
        print(f"  - Negative     : {neg} ({(neg/total_sent*100):.1f}%)")
        if neu > 0:
            print(f"  - Neutral      : {neu} ({(neu/total_sent*100):.1f}%)")
            
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()