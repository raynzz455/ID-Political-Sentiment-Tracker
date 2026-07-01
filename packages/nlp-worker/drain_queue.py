"""
drain_queue.py
================
Drain pgmq queue dalam SATU process Python -- model di-load sekali,
dipakai berulang (beda dengan loop PowerShell yang spawn process baru
tiap kali, reload model 670MB berkali-kali).

INI BUKAN production daemon 24/7. Ini untuk fase pengumpulan data
ground truth -- berhenti sendiri begitu target tercapai atau queue habis.
Production worker (HF Spaces, polling otomatis) masih ditunda sampai
ground truth evaluation selesai.

Logic per artikel:
  1. SELALU hitung sentimen document-level (entity_id=NULL, untuk
     mv_national_monthly_summary / mv_national_yearly_summary)
  2. Cari entity candidate via alias matching (word-boundary)
  3. Untuk tiap candidate: cek relevancy gate
     - GAGAL relevancy -> skip, TIDAK insert apa pun untuk entity ini
     - LULUS relevancy  -> hitung sentimen context-conditioned, insert
  4. Ack message dari queue
  5. model_version di-tag eksplisit beda untuk tiap jenis insert

Usage:
    python drain_queue.py --target 300
    python drain_queue.py --target 500 --batch-size 30
    python drain_queue.py --all              # drain sampai queue benar2 habis

Env vars:
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
"""

import os
import re
import sys
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
from collections import Counter

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

try:
    from supabase import create_client, Client
except ImportError:
    print("[ERROR] pip install supabase")
    sys.exit(1)

from sentiment_model import get_pipeline

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SERVICE_KEY  = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
MIN_ALIAS_LEN = 4

MODEL_VERSION_FALLBACK = "indobert-fallback-v1"
MODEL_VERSION_GATED    = "indobert-ctx-relevancy-gated-v1"


def get_client() -> Client:
    if not SUPABASE_URL or not SERVICE_KEY:
        print("[ERROR] Set SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY")
        sys.exit(1)
    return create_client(SUPABASE_URL, SERVICE_KEY)


def load_entities(sb: Client) -> list[dict]:
    res = sb.table("political_entities") \
            .select("id, canonical_name, aliases") \
            .eq("is_active", True) \
            .execute()
    return res.data or []


def find_alias_candidates(title: str, text: str, entities: list[dict]) -> list[dict]:
    combined = f"{title or ''} {text or ''}".lower()
    matched, seen = [], set()
    for e in entities:
        if e["id"] in seen:
            continue
        for name in [e["canonical_name"]] + list(e.get("aliases") or []):
            if len(name) < MIN_ALIAS_LEN:
                continue
            if re.search(r'\b' + re.escape(name.lower()) + r'\b', combined):
                matched.append(e)
                seen.add(e["id"])
                break
    return matched


def process_one(sb, pipeline, entities, item: dict, stats: Counter) -> None:
    raw_id = item["raw_text_id"]
    title  = item.get("title") or ""
    text   = item.get("text") or ""

    # 1. SELALU: fallback document-level (national index)
    try:
        fb = pipeline.predict_gated(text=f"{title} {text}".strip(), context=None)
        sb.rpc("insert_sentiment_score", {
            "p_raw_text_id": raw_id,
            "p_entity_id": None,
            "p_label": fb.label,
            "p_neg": float(fb.scores[0]),
            "p_neu": float(fb.scores[1]),
            "p_pos": float(fb.scores[2]),
            "p_confidence": float(fb.sentiment_confidence),
            "p_model_version": MODEL_VERSION_FALLBACK,
        }).execute()
        stats["fallback_inserted"] += 1
    except Exception as e:
        stats["fallback_error"] += 1
        print(f"    [FALLBACK_ERROR] raw_id={raw_id}: {e}")

    # 2. Cari kandidat entity via alias
    candidates = find_alias_candidates(title, text, entities)
    stats["alias_candidates_total"] += len(candidates)

    # 3. Gate + sentiment per kandidat
    combined_text = f"{title} {text}".strip()
    for e in candidates:
        try:
            result = pipeline.predict_gated(text=combined_text, context=e["canonical_name"])
        except Exception as ex:
            stats["gate_error"] += 1
            print(f"    [GATE_ERROR] raw_id={raw_id} entity={e['canonical_name']}: {ex}")
            continue

        if not result.is_relevant:
            stats["gate_rejected"] += 1
            continue

        try:
            sb.rpc("insert_sentiment_score", {
                "p_raw_text_id": raw_id,
                "p_entity_id": e["id"],
                "p_label": result.label,
                "p_neg": float(result.scores[0]),
                "p_neu": float(result.scores[1]),
                "p_pos": float(result.scores[2]),
                "p_confidence": float(result.sentiment_confidence),
                "p_model_version": MODEL_VERSION_GATED,
            }).execute()
            stats["entity_inserted"] += 1
            stats[f"label_{result.label}"] += 1
        except Exception as ex:
            stats["insert_error"] += 1
            print(f"    [INSERT_ERROR] raw_id={raw_id} entity={e['canonical_name']}: {ex}")

    # 4. Ack
    try:
        sb.rpc("ack_nlp_message", {"p_msg_id": item["msg_id"]}).execute()
        stats["acked"] += 1
    except Exception as e:
        stats["ack_error"] += 1
        print(f"    [ACK_ERROR] msg_id={item['msg_id']}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Drain pgmq queue dalam satu process")
    parser.add_argument("--target", type=int, default=300,
                         help="Berhenti setelah memproses N artikel (default 300)")
    parser.add_argument("--batch-size", type=int, default=30,
                         help="Jumlah item per dequeue call (default 30)")
    parser.add_argument("--all", action="store_true",
                         help="Abaikan --target, drain sampai queue benar-benar habis")
    parser.add_argument("--progress-every", type=int, default=20,
                         help="Print progress tiap N artikel (default 20)")
    args = parser.parse_args()

    sb = get_client()
    entities = load_entities(sb)
    print(f"Loaded {len(entities)} entitas aktif")

    print("Loading model (relevancy + sentiment + fallback) -- sekali saja untuk seluruh drain ...")
    pipeline = get_pipeline()
    # Trigger load eksplisit sekarang (bukan nunggu lazy-load di tengah loop)
    _ = pipeline.relevancy
    _ = pipeline.sentiment
    _ = pipeline.fallback
    print("Model siap.\n")

    stats = Counter()
    processed = 0
    start = time.time()

    print(f"{'='*70}")
    print(f"DRAIN START — target={'ALL (sampai habis)' if args.all else args.target}")
    print(f"{'='*70}")

    while True:
        if not args.all and processed >= args.target:
            print(f"\nTarget {args.target} tercapai. Berhenti.")
            break

        remaining = (args.target - processed) if not args.all else args.batch_size
        qty = min(args.batch_size, remaining) if not args.all else args.batch_size
        qty = max(qty, 1)

        res = sb.rpc("dequeue_nlp_batch", {"p_vt": 300, "p_qty": qty}).execute()
        items = res.data or []

        if not items:
            print("\nQueue kosong. Drain selesai (tidak ada lagi yang bisa diproses).")
            break

        for item in items:
            process_one(sb, pipeline, entities, item, stats)
            processed += 1

            if processed % args.progress_every == 0:
                elapsed = time.time() - start
                rate = processed / elapsed if elapsed > 0 else 0
                print(
                    f"  [{processed} diproses] "
                    f"fallback={stats['fallback_inserted']} "
                    f"entity_match={stats['entity_inserted']} "
                    f"gate_reject={stats['gate_rejected']} "
                    f"| {rate:.2f} artikel/detik, {elapsed:.0f}s berlalu"
                )

            if not args.all and processed >= args.target:
                break

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print("RINGKASAN DRAIN")
    print(f"{'='*70}")
    print(f"Total diproses          : {processed}")
    print(f"Waktu                   : {elapsed:.0f}s ({elapsed/60:.1f} menit)")
    print(f"Fallback inserted       : {stats['fallback_inserted']} (national index, entity_id=NULL)")
    print(f"Alias candidates total  : {stats['alias_candidates_total']}")
    print(f"  -> Lulus gate, inserted : {stats['entity_inserted']}")
    print(f"  -> Ditolak gate         : {stats['gate_rejected']}")
    print(f"Distribusi label (yang lulus gate):")
    for label in ["negative", "neutral", "positive"]:
        print(f"  {label:10s}: {stats[f'label_{label}']}")
    print(f"Errors: fallback={stats['fallback_error']} gate={stats['gate_error']} "
          f"insert={stats['insert_error']} ack={stats['ack_error']}")
    print(f"{'='*70}")
    print("\nLangkah selanjutnya: export_sentiment_ground_truth.py + export_relevancy_review.py")


if __name__ == "__main__":
    main()
