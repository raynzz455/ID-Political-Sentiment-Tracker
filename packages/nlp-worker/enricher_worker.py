"""
enricher_worker.py — Layer 2.5 (Enrichment Worker)
====================================================
Tugas: Memisahkan Network I/O (fetch URL) dari NLP Worker (AI Inference).

Cara Kerja:
  1. Ambil artikel di raw_texts dengan status='pending' & panjang teks < 500.
  2. Fetch URL aslinya menggunakan requests + User-Agent Chrome.
  3. Ekstrak full body menggunakan trafilatura.
  4. UPDATE raw_texts: Isi teks utuh & ubah status='enriched'.
  5. Jika URL mati (404) / gagal extract, ubah status='dead_link' agar tidak membebani antrian.

================================================
enricher_worker.py v13 — Pure Extraction with Detailed Observability
====================================================================
Menyediakan laporan distribusi kegagalan di akhir setiap batch.

Cara Jalankan (Lokal / GitHub Actions):
  python enricher_worker.py
  python enricher_worker.py --limit 100 --max-total 500
"""

import os
import sys
import time
import random
import argparse
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

try:
    from trafilatura import extract as traf_extract
    from supabase import create_client, Client
    from universal_resolver import fetch_article, FetchResult
except ImportError as e:
    print(f"[ERROR] Dependency missing: {e}")
    sys.exit(1)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SERVICE_KEY  = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
MAX_WORKERS = 7

def get_client() -> Client:
    return create_client(SUPABASE_URL, SERVICE_KEY)

def extract_text(html: str) -> str:
    if not html: return ""
    return traf_extract(html, include_comments=False, include_tables=False) or ""

def bulk_store(sb: Client, results: list) -> Counter:
    stats = Counter()
    updates = []

    for rt_id, text, fetch_result in results:
        status = fetch_result.status
        reason = fetch_result.reason
        
        if status == "ok":
            if reason == "gnews_snippet_only":
                # Khusus GNews: Teks tidak di-update (tetap pakai snippet RSS di DB).
                # Tapi status diubah jadi 'validated' agar langsung masuk antrian NLP (Tier 2).
                updates.append({"id": rt_id, "text": "", "status": "validated"})
                stats["gnews_snippet_validated"] += 1
            elif len(text) > 0:
                # URL media asli berhasil di-fetch dan diekstrak.
                updates.append({"id": rt_id, "text": text, "status": "enriched"})
                stats["enriched"] += 1
            else:
                # URL media asli berhasil di-fetch tapi trafilatura gagal ekstrak.
                updates.append({"id": rt_id, "text": "", "status": "extraction_failed"})
                stats["extract_empty"] += 1
        elif status in ["blocked", "timeout", "network_error"]:
            updates.append({"id": rt_id, "text": "", "status": status})
            stats[reason] += 1
        else:
            updates.append({"id": rt_id, "text": "", "status": "dead_link"})
            stats[reason] += 1

    if updates:
        try:
            sb.rpc("bulk_update_raw_texts", {"p_updates": updates}).execute()
        except Exception as e:
            print(f"    [BULK_DB_ERROR] {e}")
            
    return stats

def pipeline_worker(row):
    url = row["source_url"]
    
    # --- GNEWS BYPASS ---
    # Jika ini URL Google News, jangan coba resolve/fetch (karena terenkripsi & pasti gagal)
    if "news.google.com" in url:
        return row["id"], "", FetchResult(status="ok", reason="gnews_snippet_only", final_url=url)
    
    fetch_result = fetch_article(url)
    
    text = ""
    if fetch_result.status == "ok" and fetch_result.html:
        text = extract_text(fetch_result.html)
        
    return row["id"], text, fetch_result

def print_batch_report(batch_num: int, stats: Counter):
    """Mencetak laporan observabilitas yang ringkas dan informatif."""
    print(f"\n  📊 === BATCH {batch_num} REPORT ===")
    print(f"  ✅ Enriched (Full Art) : {stats.get('enriched', 0)}")
    print(f"  📝 GNews (Snippet Only): {stats.get('gnews_snippet_validated', 0)}")
    print(f"  ❌ Extract Empty       : {stats.get('extract_empty', 0)}")
    print(f"  ⏳ Timeout             : {stats.get('media_request_timeout', 0) + stats.get('gnews_request_timeout', 0)}")
    print(f"  🛑 Blocked (WAF/403)   : {stats.get('waf_cloudflare', 0) + stats.get('http_403', 0) + stats.get('http_429', 0)}")
    print(f"  💀 Dead Link (404)     : {stats.get('http_404', 0) + stats.get('gnews_resolve_failed', 0)}")
    print(f"  🌐 Network Error       : {stats.get('media_connection_error', 0)}")
    print(f"  ============================\n")

def process_batch(sb: Client, rows: list) -> Counter:
    to_fetch = []
    pipeline_results = []
    
    for r in rows:
        current_text = r.get("text") or ""
        if len(current_text) >= 500:
            pipeline_results.append((r["id"], current_text, FetchResult(status="ok", reason="rss_full_text")))
        else:
            to_fetch.append(r)

    if not to_fetch:
        return bulk_store(sb, pipeline_results)

    print(f"  [FETCH] {len(to_fetch)} URLs dengan {MAX_WORKERS} threads paralel...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(pipeline_worker, r): r for r in to_fetch}
        for future in as_completed(futures):
            try:
                pipeline_results.append(future.result())
            except Exception:
                row = futures[future]
                pipeline_results.append((row["id"], "", FetchResult(status="network_error", reason="thread_crash")))

    return bulk_store(sb, pipeline_results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-total", type=int, default=0)
    args = parser.parse_args()

    sb = get_client()
    total_stats = Counter()
    batch_num = 1

    print(f"[ENRICHER v13] Limit: {args.limit}/batch | Threads: {MAX_WORKERS}")

    while True:
        if args.max_total > 0 and sum(total_stats.values()) >= args.max_total:
            break

        print(f"--- Batch {batch_num} ---")
        res = sb.table("raw_texts") \
                .select("id, source_url, text") \
                .eq("status", "pending") \
                .limit(args.limit) \
                .execute()

        rows = res.data or []
        if not rows: break

        batch_stats = process_batch(sb, rows)
        print_batch_report(batch_num, batch_stats)
        
        total_stats.update(batch_stats)
        time.sleep(8 + random.uniform(0, 4))
        batch_num += 1

    print(f"\n{'='*55}")
    print(f"🏆 FINAL SUMMARY")
    print(f"{'='*55}")
    print(f"Total Enriched : {total_stats.get('enriched', 0)}")
    print(f"Total Failed   : {sum(v for k, v in total_stats.items() if k != 'enriched')}")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()