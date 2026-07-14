"""
enricher_worker.py v17 — Full Lineage & Pipeline Logging
============================================================
Tugas: Memisahkan Network I/O (fetch URL) dari NLP Worker.
"""

import os
import sys
import time
import random
import argparse
import hashlib
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

try:
    from trafilatura import extract as traf_extract
except ImportError as e:
    print(f"[ERROR] Dependency missing: {e}")
    sys.exit(1)

# IMPORT DARI MONOREPO SHARED & ENRICHMENT
from packages.shared.db_client import get_client
from packages.shared.logger import start_run, finish_run
from packages.shared import constants as pc
from packages.enrichment.universal_resolver import fetch_article, FetchResult

MAX_WORKERS = 7
RSS_TEXT_MIN_LEN = 500

def extract_text(html: str) -> str:
    if not html:
        return ""
    return traf_extract(html, include_comments=False, include_tables=False) or ""

def _apply_transient_result(current_metadata: dict, reason: str) -> tuple[dict, str]:
    attempts = int(current_metadata.get("enrich_attempts", 0)) + 1
    current_metadata["enrich_attempts"] = attempts
    current_metadata["fail_reason"] = reason
    if attempts >= pc.MAX_ENRICH_RETRIES:
        current_metadata["fail_reason"] = pc.REASON_MAX_RETRIES_EXCEEDED
        return current_metadata, pc.REASON_MAX_RETRIES_EXCEEDED
    return current_metadata, reason

def bulk_store(sb, results: list) -> Counter:
    stats = Counter()
    updates = []

    for rt_id, text, fetch_result, orig_metadata in results:
        current_metadata = dict(orig_metadata) if orig_metadata else {}
        db_update = {
            "id": rt_id,
            "metadata": current_metadata,
            "resolved_domain": fetch_result.fetch_metadata.get("resolved_domain") if hasattr(fetch_result, 'fetch_metadata') else None,
            "canonical_url": fetch_result.canonical_url if hasattr(fetch_result, 'canonical_url') else None
        }
        if fetch_result.resolved_url:
            current_metadata["resolved_url"] = fetch_result.resolved_url

        if fetch_result.status == pc.FETCH_OK:
            if fetch_result.reason == pc.REASON_GNEWS_SNIPPET_ONLY:
                current_metadata["is_snippet"] = True
                db_update["text"] = ""
                db_update["status"] = pc.STATUS_ENRICHED
                db_update["content_type"] = "SNIPPET" 
                updates.append(db_update)
                stats["gnews_snippet"] += 1
            elif len(text) >= 500: 
                current_metadata["is_snippet"] = False
                db_update["text"] = text
                db_update["status"] = pc.STATUS_ENRICHED
                db_update["content_type"] = "FULLTEXT" 
                db_update["content_hash"] = hashlib.sha256(text.encode()).hexdigest()
                updates.append(db_update)
                stats["enriched"] += 1
            else:
                current_metadata["fail_reason"] = "extract_too_short"
                current_metadata["content_type"] = "SNIPPET" 
                db_update["text"] = text
                db_update["status"] = pc.STATUS_FAILED
                db_update["content_type"] = "SNIPPET" 
                updates.append(db_update)
                stats["extract_too_short"] += 1

        elif fetch_result.status in pc.RETRYABLE_FETCH_STATUSES:
            new_metadata, effective_reason = _apply_transient_result(current_metadata, fetch_result.reason)
            retried_out = effective_reason == pc.REASON_MAX_RETRIES_EXCEEDED
            next_status = pc.STATUS_FAILED if retried_out else pc.STATUS_PENDING
            db_update["text"] = ""
            db_update["status"] = next_status
            db_update["metadata"] = new_metadata
            updates.append(db_update)
            stats[effective_reason] += 1

        else:
            current_metadata["fail_reason"] = fetch_result.reason
            db_update["text"] = ""
            db_update["status"] = pc.STATUS_FAILED
            db_update["metadata"] = current_metadata
            updates.append(db_update)
            stats[fetch_result.reason] += 1

    if updates:
        try: sb.rpc("bulk_update_raw_texts", {"p_updates": updates}).execute()
        except Exception as e: print(f"    [BULK_DB_ERROR] {e}")
    return stats

def pipeline_worker(row: dict):
    url = row["source_url"]
    orig_metadata = row.get("metadata") or {}
    fetch_result = fetch_article(url)
    text = extract_text(fetch_result.html) if (fetch_result.status == pc.FETCH_OK and fetch_result.html) else ""
    return row["id"], text, fetch_result, orig_metadata

def process_batch(sb, rows: list) -> Counter:
    to_fetch = []
    pipeline_results = []
    for r in rows:
        current_text = r.get("text") or ""
        if len(current_text) >= RSS_TEXT_MIN_LEN:
            dummy_result = FetchResult(status=pc.FETCH_OK, reason=pc.REASON_RSS_FULL_TEXT, original_url=r.get("source_url"), resolved_url=r.get("source_url"))
            pipeline_results.append((r["id"], current_text, dummy_result, r.get("metadata")))
        else:
            to_fetch.append(r)

    if not to_fetch: return bulk_store(sb, pipeline_results)

    print(f"  [FETCH] {len(to_fetch)} URLs dengan {MAX_WORKERS} threads paralel...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(pipeline_worker, r): r for r in to_fetch}
        for future in as_completed(futures):
            try:
                pipeline_results.append(future.result())
            except Exception:
                row = futures[future]
                crash_result = FetchResult(status=pc.FETCH_NETWORK_ERROR, reason=pc.REASON_THREAD_CRASH, original_url=row.get("source_url"))
                pipeline_results.append((row["id"], "", crash_result, row.get("metadata")))

    print(f"  [STORE] Mengirim {len(pipeline_results)} update ke DB via Bulk RPC...")
    return bulk_store(sb, pipeline_results)

def print_batch_report(batch_num: int, stats: Counter):
    enriched = stats.get("enriched", 0)
    gnews_snippet = stats.get("gnews_snippet", 0)
    other = Counter({k: v for k, v in stats.items() if k not in ("enriched", "gnews_snippet")})
    not_enriched_total = sum(other.values())
    by_category = Counter()
    for reason, count in other.items(): by_category[pc.categorize_reason(reason)] += count

    print(f"\n  === BATCH {batch_num} REPORT ===")
    print(f"  Enriched (Full Article)  : {enriched}")
    print(f"  GNews (Snippet Track)    : {gnews_snippet}")
    print(f"  Belum tuntas (total)     : {not_enriched_total}")
    print(f"  {'-' * 34}")
    print("  Breakdown per kategori:")
    for category, count in by_category.most_common(): print(f"    - {category:20s}: {count}")
    if other: print(f"  Detail reason mentah: {dict(other.most_common())}")
    print(f"  {'=' * 34}\n")

def main(limit: int = 100, max_total: int = 0):
    sb = get_client()
    try: sb.table("raw_texts").select("id").limit(1).execute()
    except Exception as e: print(f"[FATAL] DB tidak reachable: {e}"); sys.exit(1)

    run_id = start_run("enricher_worker", "v17")
    total_stats = Counter()
    batch_num = 1
    print(f"[ENRICHER v17] Limit: {limit}/batch | Threads: {MAX_WORKERS} | Max: {'Unlimited' if max_total == 0 else max_total}")

    while True:
        if max_total > 0 and sum(total_stats.values()) >= max_total: break
        print(f"\n--- Batch {batch_num} ---")
        res = sb.table("raw_texts").select("id, source_url, text, metadata").eq("status", pc.STATUS_PENDING).limit(limit).execute()
        rows = res.data or []
        if not rows: break

        print(f"[ENRICHER] Memproses {len(rows)} artikel...")
        batch_stats = process_batch(sb, rows)
        print_batch_report(batch_num, batch_stats)
        total_stats.update(batch_stats)
        time.sleep(8 + random.uniform(0, 4))
        batch_num += 1

    total_processed = sum(total_stats.values())
    total_succeeded = total_stats.get('enriched', 0) + total_stats.get('gnews_snippet', 0)
    finish_run(run_id, processed=total_processed, succeeded=total_succeeded, failed=total_processed - total_succeeded)
    print(f"\n{'=' * 55}\nSELESAI.\n  Enriched: {total_succeeded}\n{'=' * 55}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-total", type=int, default=0)
    args = parser.parse_args()
    main(limit=args.limit, max_total=args.max_total)