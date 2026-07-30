"""
nlp_worker.py v14 — Clean Code & Decoupled Architecture
=================================================================
FIX v14:
  1. CLEAN CODE: Menghapus dead code (MAX_NLP_WORKERS) dan redundant import.
  2. DECOUPLED ARCHITECTURE: Fase 1 (Paralel GPU) & Fase 2 (Sekuensial DB).
  3. BATCH CONTEXT FETCH: Ambil semua context 1x per batch (Hemat HTTP request).
  4. ADAPTIVE THREADING: 8 threads di GPU, 1 thread di CPU (Anti OOM).
"""

import gc
import time
import logging
import torch
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

from packages.shared.db_client import get_client
from packages.shared.logger import start_run, finish_run
from packages.shared import constants as pc
from packages.nlp.sentiment_model import get_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

MODEL_VERSION_FALLBACK = "indobert-fallback-v1"
MODEL_VERSION_GATED    = "indobert-ctx-relevancy-gated-v1"
NLP_VERSION = "v14_decoupled_clean"

# Auto-Adaptif: 8 Threads di GPU (Colab), 1 Thread di CPU (GH Actions)
MAX_GPU_WORKERS = 8 if torch.cuda.is_available() else 1

def rpc_with_retry(sb, rpc_name: str, payload: dict, max_retries: int = 3) -> bool:
    for attempt in range(max_retries):
        try:
            sb.rpc(rpc_name, payload).execute()
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"RPC {rpc_name} gagal total setelah {max_retries} percobaan: {e}")
                return False
            logger.warning(f"RPC {rpc_name} gagal (Attempt {attempt+1}/{max_retries}). Retry dalam 2s... Error: {e}")
            time.sleep(2)
    return False

def check_db_health(sb) -> bool:
    required_rpcs = ["insert_sentiment_score", "dequeue_nlp_batch", "bulk_update_raw_texts", "ack_nlp_message"]
    for rpc in required_rpcs:
        try:
            res = sb.table("pg_proc").select("proname").eq("proname", rpc).limit(1).execute()
            if not res.data:
                logger.error(f"Health Check GAGAL: RPC '{rpc}' tidak ditemukan di database!")
                return False
        except Exception:
            pass 
    logger.info("Health Check RPC: OK")
    return True

def run_inference_only(pipeline, item: dict, contexts: list, stats: Counter) -> dict:
    """FASE 1: HANYA menjalankan GPU. Tidak ada panggilan Database di sini."""
    raw_id = item["raw_text_id"]
    title  = item.get("title") or ""
    text   = item.get("text") or ""
    combined_text = f"{title} {text}".strip()
    
    if len(combined_text) < 50:
        stats["skipped_short"] += 1
        return {"raw_id": raw_id, "msg_id": item["msg_id"], "is_skipped": True}

    fb_text = combined_text[:1500]
    fb = pipeline.predict_gated(text=fb_text, context=None)
    fb_payload = {
        "p_raw_text_id": raw_id, "p_entity_id": None,
        "p_label": fb.label, "p_neg": float(fb.scores[0]),
        "p_neu": float(fb.scores[1]), "p_pos": float(fb.scores[2]),
        "p_confidence": float(fb.sentiment_confidence),
        "p_aspect": "general", "p_model_version": MODEL_VERSION_FALLBACK,
    }

    targeted_payloads = []
    for ctx in contexts:
        entity_id = ctx["entity_id"]
        entity_name = ctx["political_entities"]["canonical_name"]
        context_text = ctx.get("context_text") or ""
        
        if len(context_text.strip()) < 10:
            stats["ctx_empty"] += 1
            continue

        result = pipeline.predict_gated(text=context_text, context=entity_name)
        if not result.is_relevant:
            stats["gate_rejected"] += 1
            continue

        targeted_payloads.append({
            "p_raw_text_id": raw_id, "p_entity_id": entity_id,
            "p_label": result.label, "p_neg": float(result.scores[0]),
            "p_neu": float(result.scores[1]), "p_pos": float(result.scores[2]),
            "p_confidence": float(result.sentiment_confidence),
            "p_aspect": entity_name, "p_model_version": MODEL_VERSION_GATED,
        })

    return {
        "raw_id": raw_id, "msg_id": item["msg_id"], "is_skipped": False,
        "fb_payload": fb_payload, "targeted_payloads": targeted_payloads
    }

def write_results_to_db(sb, res: dict, stats: Counter) -> None:
    """FASE 2: Menulis ke DB secara sekuensial (1 per 1) agar Supabase aman."""
    if res["is_skipped"]:
        rpc_with_retry(sb, "ack_nlp_message", {"p_msg_id": res["msg_id"]})
        return

    if rpc_with_retry(sb, "insert_sentiment_score", res["fb_payload"]):
        stats["fallback_inserted"] += 1

    for payload in res["targeted_payloads"]:
        if rpc_with_retry(sb, "insert_sentiment_score", payload):
            stats["entity_inserted"] += 1
            stats[f"label_{payload['p_label']}"] += 1

    update_payload = {"p_updates": [{"id": res["raw_id"], "status": str(pc.STATUS_PROCESSED), "pipeline_version": NLP_VERSION}]}
    if rpc_with_retry(sb, "bulk_update_raw_texts", update_payload) and rpc_with_retry(sb, "ack_nlp_message", {"p_msg_id": res["msg_id"]}):
        stats["acked"] += 1

def main(target: int = 500, batch_size: int = 100, run_all: bool = False):
    sb = get_client()
    if not check_db_health(sb): return

    run_id = start_run("nlp_worker", NLP_VERSION)
    print("Loading model...")
    pipeline = get_pipeline()
    _ = pipeline.relevancy; _ = pipeline.sentiment; _ = pipeline.fallback
    print("Model siap.\n")

    stats = Counter()
    processed = 0
    start = time.time()

    print(f"{'='*70}\nDRAIN START (Decoupled) — target={'ALL' if run_all else target} | GPU Threads: {MAX_GPU_WORKERS}\n{'='*70}")

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

        # 1. BATCH FETCH CONTEXTS
        batch_ids = [item["raw_text_id"] for item in items]
        try:
            ctx_res = sb.table("entity_contexts") \
                        .select("raw_text_id, entity_id, political_entities(canonical_name), context_text") \
                        .in_("raw_text_id", batch_ids).execute()
            contexts_data = ctx_res.data or []
        except Exception:
            contexts_data = []

        contexts_map = {}
        for ctx in contexts_data:
            contexts_map.setdefault(ctx["raw_text_id"], []).append(ctx)

        # 2. GPU INFERENCE (PARALEL AMAN)
        inference_results = []
        with ThreadPoolExecutor(max_workers=MAX_GPU_WORKERS) as pool:
            futures = []
            for item in items:
                ctxs = contexts_map.get(item["raw_text_id"], [])
                futures.append(pool.submit(run_inference_only, pipeline, item, ctxs, stats))
            
            for future in as_completed(futures):
                try:
                    res = future.result()
                    if res: inference_results.append(res)
                except Exception as e:
                    logger.error(f"GPU Inference crashed: {e}")

        # 3. SEQUENTIAL DB WRITE (AMAN DARI DISCONNECT)
        for res in inference_results:
            write_results_to_db(sb, res, stats)
            processed += 1
            
            if processed % 10 == 0:
                elapsed = time.time() - start
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"[PROGRESS] Total: {processed} | Speed: {rate:.1f} art/s | Pos={stats['label_positive']} Neg={stats['label_negative']} Neu={stats['label_neutral']}", flush=True)

        gc.collect()
        time.sleep(0.5)

    elapsed = time.time() - start
    print(f"\n{'='*70}\nRINGKASAN DRAIN")
    print(f"Total diproses          : {processed}")
    print(f"Waktu                   : {elapsed:.0f}s ({elapsed/60:.1f} menit)")
    print(f"Distribusi (Pure Label) : Pos={stats['label_positive']} | Neg={stats['label_negative']} | Neu={stats['label_neutral']}")
    print(f"{'='*70}")
    
    finish_run(run_id=run_id, processed=processed, succeeded=stats["acked"], failed=stats["ack_error"])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Drain pgmq queue (NLP Worker v14)")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    main(target=args.target, batch_size=args.batch_size, run_all=args.all)