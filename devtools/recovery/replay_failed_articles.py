"""
replay_failed_articles.py — Full Replay Enrichment (Refactored v3)
===================================================================
Berjalan terus-menerus (unlimited) seperti Enricher Worker hingga antrian habis.
"""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from devtools.common import get_supabase, setup_argparse, build_text_hash

import requests
from trafilatura import extract as traf_extract

from packages.shared import constants as pc
from packages.enrichment.universal_resolver import fetch_article

def extract_full_text(html: str) -> str:
    if not html: return ""
    return traf_extract(html, include_comments=False, include_tables=False) or ""

def main(limit: int = 20, max_total: int = 0) -> None:
    sb = get_supabase()
    
    total_processed = 0
    total_success = 0
    batch_num = 1
    start_time = time.perf_counter()

    print(f"[REPLAY] Mode: {'Unlimited' if max_total == 0 else f'Max {max_total}'} | Batch: {limit}")
    
    while True:
        if max_total > 0 and total_processed >= max_total:
            break
            
        print(f"\n--- Batch {batch_num} ---")
        res = sb.table("raw_texts") \
                .select("id, source_url, text, metadata") \
                .eq("status", pc.STATUS_FAILED) \
                .not_.like("source_url", "%news.google.com%") \
                .limit(limit) \
                .execute()
                
        articles = res.data or []
        if not articles:
            print("[REPLAY] Tidak ada lagi artikel untuk di-replay.")
            break
            
        updates = []
        batch_success = 0
        
        for art in articles:
            url = art["source_url"]
            print(f"  -> Re-fetching: {url[:60]}...", end=" ")
            
            fetch_result = fetch_article(url)
            
            if fetch_result.status == pc.FETCH_OK and fetch_result.html:
                text = extract_full_text(fetch_result.html)
                if text:
                    print(f"✅ Berhasil diekstrak!")
                    current_meta = dict(art.get("metadata") or {})
                    current_meta["fail_reason"] = None 
                    current_meta["resolver_method"] = "replay_success"
                    
                    updates.append({
                        "id": art["id"],
                        "text": text,
                        "status": pc.STATUS_ENRICHED,
                        "content_type": "FULLTEXT",
                        "metadata": current_meta,
                        "content_hash": build_text_hash(text)
                    })
                    batch_success += 1
                else:
                    print("❌ Gagal ekstrak (teks kosong).")
            else:
                print(f"❌ Gagal fetch ({fetch_result.reason}).")
                
            time.sleep(pc.SLEEP_JITTER_SHORT)
            
        if updates:
            try: sb.rpc("bulk_update_raw_texts", {"p_updates": updates}).execute()
            except Exception as e: print(f"[DB_ERROR] {e}")
            
        total_processed += len(articles)
        total_success += batch_success
        batch_num += 1
        
    elapsed = time.perf_counter() - start_time
    print(f"\n{'='*55}")
    print(f"SELESAI (Replay Articles)")
    print(f"  Total Processed : {total_processed}")
    print(f"  Total Saved     : {total_success}")
    print(f"  Waktu Eksekusi  : {elapsed:.2f}s")
    print(f"{'='*55}")

if __name__ == "__main__":
    parser = setup_argparse("Replay Failed Articles Tool")
    parser.add_argument("--max-total", type=int, default=0, help="Batas total proses (0 = unlimited)")
    args = parser.parse_args()
    main(limit=args.limit, max_total=args.max_total)