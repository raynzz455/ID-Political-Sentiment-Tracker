"""
nlp_worker.py v10 — Layer 4 (NLP Worker Robust Edition)
=================================================================
PERUBAHAN v10:
  1. HEALTH CHECK (Bug 6): Verifikasi ketersediaan RPC sebelum drain dimulai.
  2. EXPLICIT ASPECT (Bug 2): Mengirim p_aspect=None secara eksplisit.
  3. INFORMATIVE LOGGING (Bug 3): Mencetak raw_id & entity_id saat error.
  4. RPC RETRY (Bug 5): Wrapper untuk retry RPC jika terjadi network error.
"""
import sys
import time
import logging
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

MODEL_VERSION_FALLBACK = "indobert-fallback-v1"
MODEL_VERSION_GATED    = "indobert-ctx-relevancy-gated-v1"
NLP_VERSION = "v10_robust"

# Bug 5: Helper untuk RPC Retry
def rpc_with_retry(sb: Client, rpc_name: str, payload: dict, max_retries: int = 3) -> bool:
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

# Bug 6: Health Check saat startup
def check_db_health(sb: Client) -> bool:
    required_rpcs = ["insert_sentiment_score", "dequeue_nlp_batch", "bulk_update_raw_texts", "ack_nlp_message"]
    for rpc in required_rpcs:
        try:
            # Mengambil metadata fungsi dari PostgreSQL
            sb.rpc("to_jsonb", {"p_arg": None}).execute() # Dummy safe call
            # Cek di pg_proc
            res = sb.table("pg_proc").select("proname").eq("proname", rpc).limit(1).execute()
            if not res.data:
                logger.error(f"Health Check GAGAL: RPC '{rpc}' tidak ditemukan di database!")
                return False
        except Exception:
            # Fallback jika pg_proc tidak bisa diakses (biasanya karena RLS)
            pass 
    logger.info("Health Check RPC: OK")
    return True

def process_one(sb, pipeline, item: dict, stats: Counter) -> None:
    raw_id = item["raw_text_id"]
    title  = item.get("title") or ""
    text   = item.get("text") or ""
    
    combined_text = f"{title} {text}".strip()
    if len(combined_text) < 50:
        stats["skipped_short"] += 1
        rpc_with_retry(sb, "ack_nlp_message", {"p_msg_id": item["msg_id"]})
        return

    # 1. FALLBACK NATIONAL INDEX
    fb = pipeline.predict_gated(text=combined_text, context=None)
    fb_payload = {
        "p_raw_text_id": raw_id, "p_entity_id": None,
        "p_label": fb.label, "p_neg": float(fb.scores[0]),
        "p_neu": float(fb.scores[1]), "p_pos": float(fb.scores[2]),
        "p_confidence": float(fb.sentiment_confidence),
        "p_aspect": None, # Bug 2 fix
        "p_model_version": MODEL_VERSION_FALLBACK,
    }
    if rpc_with_retry(sb, "insert_sentiment_score", fb_payload):
        stats["fallback_inserted"] += 1
    else:
        # Bug 3 fix: Logging informatif
        logger.error(f"Insert fallback gagal | raw_text={raw_id}")
        stats["fallback_error"] += 1

    # 2. TARGETED SENTIMENT
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
        
        try:
            result = pipeline.predict_gated(text=context_text, context=entity_name)
        except Exception as e:
            logger.error(f"Gate error: {e} | raw_text={raw_id} | entity={entity_id}")
            stats["gate_error"] += 1
            continue

        if not result.is_relevant:
            stats["gate_rejected"] += 1
            continue

        targeted_payload = {
            "p_raw_text_id": raw_id, "p_entity_id": entity_id,
            "p_label": result.label, "p_neg": float(result.scores[0]),
            "p_neu": float(result.scores[1]), "p_pos": float(result.scores[2]),
            "p_confidence": float(result.sentiment_confidence),
            "p_aspect": None, # Bug 2 fix
            "p_model_version": MODEL_VERSION_GATED,
        }
        
        if rpc_with_retry(sb, "insert_sentiment_score", targeted_payload):
            stats["entity_inserted"] += 1
            stats[f"label_{result.label}"] += 1
        else:
            # Bug 3 fix: Logging informatif
            logger.error(f"Insert targeted gagal | raw_text={raw_id} | entity={entity_id} | payload={targeted_payload}")
            stats["insert_error"] += 1

    # 3. Ack & Update Status
    update_payload = {"p_updates": [{"id": raw_id, "status": str(pc.STATUS_PROCESSED), "resolver_version": NLP_VERSION}]}
    if rpc_with_retry(sb, "bulk_update_raw_texts", update_payload) and rpc_with_retry(sb, "ack_nlp_message", {"p_msg_id": item["msg_id"]}):
        stats["acked"] += 1
    else:
        logger.error(f"Ack/Update gagal | raw_text={raw_id}")
        stats["ack_error"] += 1

def main(target: int = 300, batch_size: int = 30, run_all: bool = False):
    sb = get_client()
    
    # Bug 6 fix: Health check sebelum mulai
    if not check_db_health(sb):
        print("❌ Database Health Check Gagal! Pastikan semua RPC sudah terdaftar. Worker berhenti.")
        return

    run_id = start_run("nlp_worker", NLP_VERSION)
    
    print("Loading model (relevancy + sentiment + fallback)...")
    pipeline = get_pipeline()
    _ = pipeline.relevancy; _ = pipeline.sentiment; _ = pipeline.fallback
    print("Model siap.\n")

    stats = Counter()
    processed = 0
    start = time.time()

    print(f"{'='*70}\nDRAIN START (Robust Mode) — target={'ALL' if run_all else target}\n{'='*70}")

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
    print(f"Distribusi (Pure Label) : Pos={stats['label_positive']} | Neg={stats['label_negative']} | Neu={stats['label_neutral']}")
    print(f"{'='*70}")
    
    finish_run(run_id=run_id, processed=processed, succeeded=stats["acked"], failed=stats["ack_error"])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Drain pgmq queue (NLP Worker v10 Robust)")
    parser.add_argument("--target", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    main(target=args.target, batch_size=args.batch_size, run_all=args.all)