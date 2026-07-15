"""
nlp_worker.py v5 — Layer 4 (NLP Worker Targeted Sentiment)
============================================================
Kontrak:
  Input: status='queued' (artikel sudah punya entity_contexts)
  Output: sentiment_scores, status='processed'
  Dilarang: NER, Alias Matching, Context Extraction, Network Fetch.

PERUBAHAN v5:
  1. MONOREPO READY: Import dari packages.shared.
  2. TARGETED SENTIMENT: Membaca context_text dari entity_contexts, bukan teks utuh.
  3. CLEAN ARGUMENTS: Menerima parameter langsung dari main.py (orchestrator).
"""
import os
import sys
import time
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

try:
    from supabase import create_client, Client
except ImportError:
    print("[ERROR] pip install supabase"); sys.exit(1)

from packages.shared.db_client import get_client
from packages.shared.logger import start_run, finish_run
from packages.shared import constants as pc
from packages.nlp.sentiment_model import get_pipeline

MODEL_VERSION_FALLBACK = "indobert-fallback-v1"
MODEL_VERSION_GATED    = "indobert-ctx-relevancy-gated-v1"
NLP_VERSION = "v5_targeted"

def process_one(sb, pipeline, item: dict, stats: Counter) -> None:
    raw_id = item["raw_text_id"]
    title  = item.get("title") or ""
    text   = item.get("text") or ""
    
    # Guard: Jika teks utuh terlalu pendek, skip (untuk fallback national index)
    combined_text = f"{title} {text}".strip()
    if len(combined_text) < 50:
        stats["skipped_short"] += 1
        sb.rpc("ack_nlp_message", {"p_msg_id": item["msg_id"]}).execute()
        return

    # 1. FALLBACK NATIONAL INDEX (Document-level)
    fb = pipeline.predict_gated(text=combined_text, context=None)
    try:
        sb.rpc("insert_sentiment_score", {
            "p_raw_text_id": raw_id, "p_entity_id": None,
            "p_label": fb.label, "p_neg": float(fb.scores[0]),
            "p_neu": float(fb.scores[1]), "p_pos": float(fb.scores[2]),
            "p_confidence": float(fb.sentiment_confidence),
            "p_model_version": MODEL_VERSION_FALLBACK,
        }).execute()
        stats["fallback_inserted"] += 1
    except Exception:
        stats["fallback_error"] += 1

    # 2. TARGETED SENTIMENT (Entity-level via entity_contexts)
    ctx_res = sb.table("entity_contexts") \
                .select("entity_id, political_entities(canonical_name), context_text") \
                .eq("raw_text_id", raw_id) \
                .execute()
                
    contexts = ctx_res.data or []
    stats["contexts_found"] += len(contexts)

    for ctx in contexts:
        entity_id = ctx["entity_id"]
        entity_name = ctx["political_entities"]["canonical_name"]
        context_text = ctx["context_text"]
        
        # Step 5: Model Relevancy Gate
        try:
            result = pipeline.predict_gated(text=context_text, context=entity_name)
        except Exception:
            stats["gate_error"] += 1
            continue

        if not result.is_relevant:
            stats["gate_rejected"] += 1
            continue

        # Step 6: Binary Mapping (Hapus Netral untuk Termometer Digital)
        label = result.label
        scores = result.scores
        if label == "neutral":
            label = "positive" if scores[2] >= scores[0] else "negative"

        # Step 7: Insert Targeted Sentiment
        try:
            sb.rpc("insert_sentiment_score", {
                "p_raw_text_id": raw_id, "p_entity_id": entity_id,
                "p_label": label, "p_neg": float(scores[0]),
                "p_neu": float(scores[1]), "p_pos": float(scores[2]),
                "p_confidence": float(result.sentiment_confidence),
                "p_model_version": MODEL_VERSION_GATED,
            }).execute()
            stats["entity_inserted"] += 1
            stats[f"label_{label}"] += 1
        except Exception:
            stats["insert_error"] += 1

    # 4. Ack & Update Status to Processed
    try:
        sb.rpc("bulk_update_raw_texts", {
            "p_updates": [{"id": raw_id, "status": pc.STATUS_PROCESSED, "resolver_version": NLP_VERSION}]
        }).execute()
        sb.rpc("ack_nlp_message", {"p_msg_id": item["msg_id"]}).execute()
        stats["acked"] += 1
    except Exception:
        stats["ack_error"] += 1

def main(target: int = 300, batch_size: int = 30, run_all: bool = False):
    sb = get_client()
    run_id = start_run("nlp_worker", NLP_VERSION)
    
    print("Loading model (relevancy + sentiment + fallback)...")
    pipeline = get_pipeline()
    _ = pipeline.relevancy; _ = pipeline.sentiment; _ = pipeline.fallback
    print("Model siap.\n")

    stats = Counter()
    processed = 0
    start = time.time()

    print(f"{'='*70}\nDRAIN START (Targeted Sentiment Mode) — target={'ALL' if run_all else target}\n{'='*70}")

    while True:
        if not run_all and processed >= target: break

        remaining = (target - processed) if not run_all else batch_size
        qty = min(batch_size, remaining) if not run_all else batch_size
        qty = max(qty, 1)

        res = sb.rpc("dequeue_nlp_batch", {"p_vt": 300, "p_qty": qty}).execute()
        items = res.data or []

        if not items:
            print("\nQueue kosong. Drain selesai.")
            break

        for item in items:
            process_one(sb, pipeline, item, stats)
            processed += 1

    elapsed = time.time() - start
    print(f"\n{'='*70}\nRINGKASAN DRAIN")
    print(f"Total diproses          : {processed}")
    print(f"Waktu                   : {elapsed:.0f}s ({elapsed/60:.1f} menit)")
    print(f"Fallback inserted       : {stats['fallback_inserted']}")
    print(f"Contexts found          : {stats['contexts_found']}")
    print(f"Entity Match (Lolos)    : {stats['entity_inserted']}")
    print(f"Gate Rejected           : {stats['gate_rejected']}")
    print(f"Distribusi (Binary)     : Pos={stats['label_positive']} | Neg={stats['label_negative']}")
    print(f"{'='*70}")
    
    finish_run(run_id, processed, stats["entity_inserted"], stats.get("gate_error", 0))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Drain pgmq queue (NLP Worker v5)")
    parser.add_argument("--target", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    main(target=args.target, batch_size=args.batch_size, run_all=args.all)