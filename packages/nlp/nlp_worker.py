"""
nlp_worker.py v16 — LLM Hybrid Pipeline
========================================
TAHAP 3A: LLM Hybrid untuk akurasi 90%+

Flow:
  1. Model V2 predict (fast, 0.1s)
  2. confidence >= tau (0.70) → use model prediction
  3. confidence < tau → DEFER to LLM second-pass (accurate, 3s)
  4. Combine → 90%+ accuracy

Changes from v15:
  - Added llm_second_pass() function
  - DEFER cases now get LLM prediction instead of model prediction
  - Stats track model vs LLM predictions
  - Fallback (general) also uses LLM if DEFER

GitHub Actions compatible:
  - z-ai CLI for LLM calls (already installed in workflow)
  - No new deps
  - Per-article: ~0.1s (model) + ~3s (LLM for DEFER cases only)
"""
import gc, time, logging, torch, json, re, subprocess
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

MODEL_VERSION_FALLBACK = "indobert-fallback-v2-body-only"
MODEL_VERSION_GATED    = "indobert-ctx-relevancy-gated-v2"
MODEL_VERSION_LLM      = "llm-second-pass-v1"
NLP_VERSION = "v16_llm_hybrid"

MAX_GPU_WORKERS = 8 if torch.cuda.is_available() else 1
CONFIDENCE_TAU = 0.70  # defer predictions below this confidence

# ---------------------------------------------------------------------------
# LLM Second-Pass (for DEFER cases)
# ---------------------------------------------------------------------------
LLM_SYSTEM_PROMPT = """Anda adalah annotator ahli sentimen politik Indonesia.
Tentukan sentimen TERHADAP entitas (bukan YANG DIKATAKAN entitas).

- "positive": entitas dipuji/didukung/diprestasikan
- "neutral": laporan faktal, entitas sebagai pembicara
- "negative": entitas dikritik/dicela/divonis/dituduh

Output HANYA satu kata: positive, neutral, atau negative"""

def llm_second_pass(entity_name, context_text, model_label, model_conf):
    """LLM second-pass for DEFER cases. Returns (label, confidence)."""
    prompt = f"""Entitas: "{entity_name}"
Konteks: "{context_text[:400]}"
Model ML memprediksi: {model_label} (confidence: {model_conf:.1%}) — TIDAK YAKIN.
Tentukan label yang benar."""

    try:
        proc = subprocess.run(
            ["z-ai", "chat", "-p", prompt, "-s", LLM_SYSTEM_PROMPT],
            capture_output=True, text=True, timeout=30
        )
        if proc.returncode == 0:
            m = re.search(r'\{[\s\S]*"choices"[\s\S]*\}', proc.stdout)
            if m:
                env = json.loads(m.group(0))
                content = env["choices"][0]["message"]["content"].strip().lower()
                for label in ["positive", "neutral", "negative"]:
                    if label in content:
                        return label, 0.85
    except Exception as e:
        logger.warning(f"LLM second-pass failed: {e}")
    
    return model_label, model_conf  # fallback to model prediction

# ---------------------------------------------------------------------------
# RPC helpers
# ---------------------------------------------------------------------------
def rpc_with_retry(sb, rpc_name, payload, max_retries=3):
    for attempt in range(max_retries):
        try:
            sb.rpc(rpc_name, payload).execute()
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"RPC {rpc_name} gagal total: {e}")
                return False
            time.sleep(2)
    return False

def check_db_health(sb):
    required_rpcs = ["insert_sentiment_score", "dequeue_nlp_batch", "bulk_update_raw_texts", "ack_nlp_message"]
    for rpc in required_rpcs:
        try:
            res = sb.table("pg_proc").select("proname").eq("proname", rpc).limit(1).execute()
            if not res.data:
                logger.error(f"Health Check GAGAL: RPC '{rpc}' tidak ditemukan!")
                return False
        except:
            pass
    logger.info("Health Check RPC: OK")
    return True

# ---------------------------------------------------------------------------
# Inference (FASE 1 — GPU only)
# ---------------------------------------------------------------------------
def run_inference_only(pipeline, item, contexts, stats):
    """v16: Model predict → DEFER cases ke LLM second-pass."""
    raw_id = item["raw_text_id"]
    text = item.get("text") or ""
    combined_text = text.strip()  # body only (BUG B fix)

    if len(combined_text) < 50:
        stats["skipped_short"] += 1
        return {"raw_id": raw_id, "msg_id": item["msg_id"], "is_skipped": True}

    # Fallback (document-level)
    fb_text = combined_text[:1500]
    fb = pipeline.predict_gated(text=fb_text, context=None)
    fb_label = fb.label
    fb_conf = fb.sentiment_confidence or 0.0
    fb_scores = fb.scores
    fb_model_version = MODEL_VERSION_FALLBACK

    # v16: LLM hybrid for fallback DEFER
    if fb_conf < CONFIDENCE_TAU:
        llm_label, llm_conf = llm_second_pass("(general)", fb_text, fb_label, fb_conf)
        if llm_label != fb_label:
            fb_label = llm_label
            fb_conf = llm_conf
            fb_model_version = MODEL_VERSION_LLM
            stats["fallback_llm"] += 1
    else:
        stats["fallback_model"] += 1

    fb_payload = {
        "p_raw_text_id": raw_id, "p_entity_id": None,
        "p_label": fb_label, "p_neg": float(fb_scores[0]),
        "p_neu": float(fb_scores[1]), "p_pos": float(fb_scores[2]),
        "p_confidence": float(fb_conf),
        "p_aspect": "general", "p_model_version": fb_model_version,
    }

    # Entity-level (targeted)
    targeted_payloads = []
    for ctx in contexts:
        entity_id = ctx["entity_id"]
        entity_name = ctx["political_entities"]["canonical_name"]
        context_text = ctx.get("context_text") or ""
        metadata = ctx.get("metadata") or {}

        if len(context_text.strip()) < 10:
            stats["ctx_empty"] += 1
            continue

        # Relevancy pre-filter
        if metadata.get("is_relevant") is False:
            stats["relevancy_filtered"] += 1
            continue

        # Multi-mention aggregation
        all_spans = metadata.get("all_spans", [context_text])
        if len(all_spans) > 1:
            agg_scores = torch.zeros(3)
            total_w = 0.0
            for span_text in all_spans[:5]:
                if len(span_text.strip()) < 10: continue
                result = pipeline.predict_gated(text=span_text, context=entity_name)
                if not result.is_relevant:
                    stats["gate_rejected"] += 1
                    continue
                w = result.sentiment_confidence or 0.5
                agg_scores += w * torch.tensor(result.scores)
                total_w += w
                stats["spans_processed"] += 1
            if total_w == 0:
                continue
            agg_scores = (agg_scores / total_w).tolist()
            idx = max(range(3), key=lambda i: agg_scores[i])
            label = ["negative", "neutral", "positive"][idx]
            conf = agg_scores[idx]
            scores = tuple(agg_scores)
        else:
            result = pipeline.predict_gated(text=context_text, context=entity_name)
            if not result.is_relevant:
                stats["gate_rejected"] += 1
                continue
            label = result.label
            conf = result.sentiment_confidence
            scores = result.scores

        # v16: LLM hybrid for entity DEFER
        model_version = MODEL_VERSION_GATED
        if conf < CONFIDENCE_TAU:
            llm_label, llm_conf = llm_second_pass(entity_name, context_text, label, conf)
            if llm_label != label:
                label = llm_label
                conf = llm_conf
                model_version = MODEL_VERSION_LLM
                stats["entity_llm"] += 1
        else:
            stats["entity_model"] += 1

        targeted_payloads.append({
            "p_raw_text_id": raw_id, "p_entity_id": entity_id,
            "p_label": label, "p_neg": float(scores[0]),
            "p_neu": float(scores[1]), "p_pos": float(scores[2]),
            "p_confidence": float(conf),
            "p_aspect": entity_name, "p_model_version": model_version,
        })
        stats[f"label_{label}"] += 1

    return {
        "raw_id": raw_id, "msg_id": item["msg_id"], "is_skipped": False,
        "fb_payload": fb_payload, "targeted_payloads": targeted_payloads
    }

# ---------------------------------------------------------------------------
# DB Write (FASE 2 — sequential)
# ---------------------------------------------------------------------------
def write_results_to_db(sb, res, stats):
    if res["is_skipped"]:
        rpc_with_retry(sb, "ack_nlp_message", {"p_msg_id": res["msg_id"]})
        return

    if rpc_with_retry(sb, "insert_sentiment_score", res["fb_payload"]):
        stats["fallback_inserted"] += 1

    for payload in res["targeted_payloads"]:
        if rpc_with_retry(sb, "insert_sentiment_score", payload):
            stats["entity_inserted"] += 1

    update_payload = {"p_updates": [{"id": res["raw_id"], "status": str(pc.STATUS_PROCESSED), "pipeline_version": NLP_VERSION}]}
    if rpc_with_retry(sb, "bulk_update_raw_texts", update_payload) and rpc_with_retry(sb, "ack_nlp_message", {"p_msg_id": res["msg_id"]}):
        stats["acked"] += 1

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(target=500, batch_size=100, run_all=False):
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

    print(f"{'='*70}\nDRAIN START (v16 LLM Hybrid) — target={'ALL' if run_all else target} | GPU Threads: {MAX_GPU_WORKERS}\n{'='*70}")

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

        # BATCH FETCH CONTEXTS
        batch_ids = [item["raw_text_id"] for item in items]
        try:
            ctx_res = sb.table("entity_contexts") \
                        .select("raw_text_id, entity_id, political_entities(canonical_name), context_text, metadata") \
                        .in_("raw_text_id", batch_ids).execute()
            contexts_data = ctx_res.data or []
        except:
            contexts_data = []

        contexts_map = {}
        for ctx in contexts_data:
            contexts_map.setdefault(ctx["raw_text_id"], []).append(ctx)

        # GPU INFERENCE (PARALLEL)
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

        # SEQUENTIAL DB WRITE
        for res in inference_results:
            write_results_to_db(sb, res, stats)
            processed += 1
            if processed % 10 == 0:
                elapsed = time.time() - start
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"[PROGRESS] Total: {processed} | Speed: {rate:.1f} art/s | "
                      f"Pos={stats['label_positive']} Neg={stats['label_negative']} Neu={stats['label_neutral']} | "
                      f"Model={stats['entity_model']+stats['fallback_model']} LLM={stats['entity_llm']+stats['fallback_llm']}", flush=True)

        gc.collect()
        time.sleep(0.5)

    elapsed = time.time() - start
    print(f"\n{'='*70}\nRINGKASAN DRAIN (v16 LLM Hybrid)")
    print(f"Total diproses          : {processed}")
    print(f"Waktu                   : {elapsed:.0f}s ({elapsed/60:.1f} menit)")
    print(f"Distribusi (Pure Label) : Pos={stats['label_positive']} | Neg={stats['label_negative']} | Neu={stats['label_neutral']}")
    print(f"Model predictions        : {stats['entity_model']+stats['fallback_model']}")
    print(f"LLM second-pass          : {stats['entity_llm']+stats['fallback_llm']}")
    print(f"LLM rate                 : {(stats['entity_llm']+stats['fallback_llm'])/max(1,processed)*100:.1f}%")
    print(f"{'='*70}")

    finish_run(run_id=run_id, processed=processed, succeeded=stats["acked"], failed=stats.get("ack_error", 0))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Drain pgmq queue (NLP Worker v16 LLM Hybrid)")
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    main(target=args.target, batch_size=args.batch_size, run_all=args.all)
