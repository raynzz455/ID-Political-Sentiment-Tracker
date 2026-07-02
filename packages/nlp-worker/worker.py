"""
worker.py — Production NLP Worker (Auto-Drain Queue)
=====================================================
Loop sederhana yang otomatis drain pgmq queue:
  dequeue batch → enrich body → match entity → predict → insert → ack
  Ulang sampai queue kosong, lalu sleep, lalu ulang lagi.

Bisa jalan di:
  - Laptop (PowerShell): python worker.py
  - Hugging Face Spaces / VPS: deploy sebagai background service

Ini BUKAN arsitektur baru — reuse semua logic dari cli_test.py.
Satu-satunya beda: while-loop + graceful shutdown.

Usage:
    python worker.py                          # default: batch=16, sleep=30s
    python worker.py --batch 32 --sleep 60    # custom
    python worker.py --once                   # drain sekali, lalu exit (test)

Env vars:
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
"""

import os
import sys
import time
import signal
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

# Reuse semua logic dari cli_test.py — jangan duplikasi
from cli_test import (
    get_client,
    load_entities,
    match_entities,
    enrich_if_needed,
    predict_sentiment,
)

try:
    from supabase import Client
except ImportError:
    print("[ERROR] pip install supabase")
    sys.exit(1)


# ============================================================
# Graceful shutdown (Ctrl+C / SIGTERM)
# ============================================================
RUNNING = True


def _signal_handler(signum, frame):
    global RUNNING
    print(f"\n[WORKER] Signal {signum} received — finishing current batch, then exit.")
    RUNNING = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ============================================================
# Process one batch
# ============================================================
def process_batch(sb: Client, entities: list, batch_size: int) -> dict:
    """
    Dequeue batch_size items, process masing-masing, return stats.
    Stats dict: {processed, skipped_no_entity, acked, errors, fetches}
    """
    stats = dict(processed=0, skipped_no_entity=0, acked=0, errors=0, fetches=0)

    # Dequeue (visibility timeout 600s = 10 min, aman untuk enrich + predict)
    try:
        res = sb.rpc("dequeue_nlp_batch", {"p_vt": 600, "p_qty": batch_size}).execute()
        items = res.data or []
    except Exception as e:
        print(f"[WORKER] Dequeue error: {e}")
        return stats

    if not items:
        return stats

    print(f"[WORKER] Dequeued {len(items)} items — processing...")

    for i, item in enumerate(items, 1):
        if not RUNNING:
            break

        text_raw = item.get("text", "") or ""
        title = item.get("title", "") or ""
        raw_id = item.get("raw_text_id")
        msg_id = item.get("msg_id")
        item_entity_id = item.get("entity_id")
        source_url = item.get("source_url") or ""

        # ── Enrich body (Lapis 2) ──
        try:
            combined = enrich_if_needed(item)
        except Exception as e:
            print(f"  [{i}] enrich error: {e}")
            combined = f"{title} {text_raw}".strip()

        was_fetched = len(combined) > len(f"{title} {text_raw}".strip()) + 20
        if was_fetched:
            stats["fetches"] += 1

        if not combined or len(combined) < 20:
            # Tetap ack item sampah (text kosong + fetch gagal) supaya queue drain
            try:
                sb.rpc("ack_nlp_message", {"p_msg_id": msg_id}).execute()
                stats["acked"] += 1
            except Exception:
                pass
            continue

        # ── Entity matching ──
        if item_entity_id:
            matched = [e for e in entities if e["id"] == item_entity_id]
        else:
            matched = match_entities(combined, title, entities)

        # ── Predict sentiment ──
        try:
            label, conf, scores = predict_sentiment(combined)
        except Exception as e:
            print(f"  [{i}] predict error: {e}")
            stats["errors"] += 1
            # Ack supaya tidak stuck (akan hilang dari queue)
            try:
                sb.rpc("ack_nlp_message", {"p_msg_id": msg_id}).execute()
                stats["acked"] += 1
            except Exception:
                pass
            continue

        # ── Insert scores ──
        if not matched:
            # General feed tanpa entity match — insert NULL untuk national index
            try:
                sb.rpc("insert_sentiment_score", {
                    "p_raw_text_id": raw_id,
                    "p_entity_id": None,
                    "p_label": label,
                    "p_neg": float(scores[0]),
                    "p_neu": float(scores[1]),
                    "p_pos": float(scores[2]),
                    "p_confidence": float(conf),
                    "p_model_version": "dummy-fallback-v1",
                }).execute()
            except Exception as e:
                print(f"  [{i}] insert NULL error: {e}")
                stats["errors"] += 1

            stats["skipped_no_entity"] += 1
        else:
            # Insert score untuk tiap matched entity
            for e in matched:
                try:
                    sb.rpc("insert_sentiment_score", {
                        "p_raw_text_id": raw_id,
                        "p_entity_id": e["id"],
                        "p_label": label,
                        "p_neg": float(scores[0]),
                        "p_neu": float(scores[1]),
                        "p_pos": float(scores[2]),
                        "p_confidence": float(conf),
                        "p_model_version": "dummy-v1",
                    }).execute()
                except Exception as ex:
                    print(f"  [{i}] insert error for {e.get('canonical_name')}: {ex}")
                    stats["errors"] += 1

            stats["processed"] += 1

        # ── Ack message ──
        try:
            sb.rpc("ack_nlp_message", {"p_msg_id": msg_id}).execute()
            stats["acked"] += 1
        except Exception as e:
            print(f"  [{i}] ack error: {e}")

    return stats


# ============================================================
# Main loop
# ============================================================
def run_worker(sb: Client, batch_size: int, sleep_seconds: int, once: bool):
    """Loop utama: drain queue, sleep, repeat."""
    global RUNNING

    print(f"\n{'='*60}")
    print(f"WORKER STARTED — batch={batch_size}, sleep={sleep_seconds}s, once={once}")
    print(f"{'='*60}")
    print(f"[Ctrl+C untuk graceful shutdown]\n")

    # Load entities sekali di awal
    entities = load_entities(sb)
    print(f"[WORKER] Loaded {len(entities)} active entities\n")

    cycle = 0
    total_all = dict(processed=0, skipped_no_entity=0, acked=0, errors=0, fetches=0)

    while RUNNING:
        cycle += 1
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"--- Cycle {cycle} @ {ts} ---")

        stats = process_batch(sb, entities, batch_size)

        # Akumulasi total
        for k in total_all:
            total_all[k] += stats[k]

        queue_empty = (stats["acked"] == 0 and stats["processed"] == 0 and
                       stats["skipped_no_entity"] == 0 and stats["errors"] == 0)

        if queue_empty:
            print(f"[WORKER] Queue empty — sleeping {sleep_seconds}s...")
        else:
            print(f"[WORKER] Cycle {cycle} done: "
                  f"processed={stats['processed']}, "
                  f"skipped={stats['skipped_no_entity']}, "
                  f"acked={stats['acked']}, "
                  f"errors={stats['errors']}, "
                  f"fetches={stats['fetches']}")

        if once:
            print(f"\n[WORKER] --once mode — exiting after 1 cycle.")
            break

        if RUNNING and not queue_empty:
            # Queue masih ada isinya, langsung cycle lagi (jangan sleep)
            continue

        if RUNNING:
            # Queue kosong, sleep sebentar
            for _ in range(sleep_seconds):
                if not RUNNING:
                    break
                time.sleep(1)

    # Summary
    print(f"\n{'='*60}")
    print(f"WORKER STOPPED — Total across {cycle} cycles:")
    print(f"  processed (entity matched): {total_all['processed']}")
    print(f"  skipped (no entity):        {total_all['skipped_no_entity']}")
    print(f"  acked:                      {total_all['acked']}")
    print(f"  errors:                     {total_all['errors']}")
    print(f"  fetches (Lapis 2):          {total_all['fetches']}")
    print(f"{'='*60}\n")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Production NLP Worker — auto-drain pgmq queue"
    )
    parser.add_argument("--batch", type=int, default=16,
                        help="Jumlah item per batch (default: 16)")
    parser.add_argument("--sleep", type=int, default=30,
                        help="Sleep detik saat queue kosong (default: 30)")
    python_parser = parser.add_argument(
        "--once", action="store_true",
        help="Drain 1 cycle lalu exit (test mode)"
    )
    args = parser.parse_args()

    sb = get_client()
    run_worker(sb, batch_size=args.batch, sleep_seconds=args.sleep, once=args.once)


if __name__ == "__main__":
    main()
