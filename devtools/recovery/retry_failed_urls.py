"""
retry_failed_urls.py — Network Retry Tool (Refactored v3)
==========================================================
Berjalan terus-menerus (unlimited) seperti Enricher Worker hingga antrian habis.
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from devtools.common import get_supabase, setup_argparse

from packages.shared import constants as pc
from packages.enrichment.universal_resolver import fetch_article

def main(limit: int = 50, max_total: int = 0) -> None:
    sb = get_supabase()
    
    total_processed = 0
    total_success = 0
    batch_num = 1
    start_time = time.perf_counter()

    print(f"[RETRY] Mode: {'Unlimited' if max_total == 0 else f'Max {max_total}'} | Batch: {limit}")
    
    while True:
        if max_total > 0 and total_processed >= max_total:
            break
            
        print(f"\n--- Batch {batch_num} ---")
        res = sb.table("raw_texts") \
                .select("id, source_url, metadata") \
                .eq("status", pc.STATUS_FAILED) \
                .limit(limit * 2) \
                .execute()
                
        articles = []
        for r in (res.data or []):
            reason = (r.get("metadata") or {}).get("fail_reason", "")
            if reason in pc.RETRYABLE_FAILURES:
                articles.append(r)
                
        articles = articles[:limit]
        
        if not articles:
            print("[RETRY] Tidak ada lagi URL network error.")
            break
            
        updates = []
        batch_success = 0
        
        for art in articles:
            url = art["source_url"]
            print(f"  -> Retry: {url[:60]}...", end=" ")
            
            fetch_result = fetch_article(url)
            
            if fetch_result.status == pc.FETCH_OK:
                print("✅ HTTP 200 OK! Kembali ke PENDING.")
                current_meta = dict(art.get("metadata") or {})
                current_meta["fail_reason"] = None
                
                updates.append({
                    "id": art["id"],
                    "status": pc.STATUS_PENDING,
                    "metadata": current_meta
                })
                batch_success += 1
            else:
                print(f"❌ Masih gagal ({fetch_result.reason}).")
                
            time.sleep(pc.SLEEP_JITTER_SHORT)
            
        if updates:
            try: sb.rpc("bulk_update_raw_texts", {"p_updates": updates}).execute()
            except Exception as e: print(f"[DB_ERROR] {e}")
            
        total_processed += len(articles)
        total_success += batch_success
        batch_num += 1
        
    elapsed = time.perf_counter() - start_time
    print(f"\n{'='*55}")
    print(f"SELESAI (Retry URLs)")
    print(f"  Total Processed : {total_processed}")
    print(f"  Total Saved     : {total_success}")
    print(f"  Waktu Eksekusi  : {elapsed:.2f}s")
    print(f"{'='*55}")

if __name__ == "__main__":
    parser = setup_argparse("Network Retry Tool")
    parser.add_argument("--max-total", type=int, default=0, help="Batas total proses (0 = unlimited)")
    args = parser.parse_args()
    main(limit=args.limit, max_total=args.max_total)