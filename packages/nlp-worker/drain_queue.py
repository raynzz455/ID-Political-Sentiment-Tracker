"""
drain_queue.py v3 — Layer 4 (NLP Worker Two-Tier Edition)
==========================================================
Memproses antrian pgmq dengan memisahkan jalur inferensi (Two-Tier Processing) 
untuk mencegah bias antara artikel utuh (RSS/DDG) dan snippet pendek (GNews).

FIX v3:
  1. TIER 1 (Full Article > 500 char): 2-Stage Gate + Binary Mapping (Pos/Neg).
  2. TIER 2 (Snippet < 500 char): Fallback Model only, NO Binary Mapping (Pos/Neu/Neg).
  3. Guard Teks Sangat Pendek: Skip total jika < 50 char.
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
    print("[ERROR] pip install supabase"); sys.exit(1)

from sentiment_model import get_pipeline

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SERVICE_KEY  = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
MIN_ALIAS_LEN = 4

MODEL_VERSION_FALLBACK = "indobert-fallback-v1"
MODEL_VERSION_GATED    = "indobert-ctx-relevancy-gated-v1"

def get_client() -> Client:
    if not SUPABASE_URL or not SERVICE_KEY:
        print("[ERROR] Set SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY"); sys.exit(1)
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
        if e["id"] in seen: continue
        for name in [e["canonical_name"]] + list(e.get("aliases") or []):
            if len(name) < MIN_ALIAS_LEN: continue
            if re.search(r'\b' + re.escape(name.lower()) + r'\b', combined):
                matched.append(e)
                seen.add(e["id"])
                break
    return matched

def process_one(sb, pipeline, entities, item: dict, stats: Counter) -> None:
    raw_id = item["raw_text_id"]
    title  = item.get("title") or ""
    text   = item.get("text") or ""
    combined_text = f"{title} {text}".strip()

    # GUARD TEKS SANGAT PENDEK (Skip total)
    if len(combined_text) < 50:
        stats["skipped_short"] += 1
        sb.rpc("ack_nlp_message", {"p_msg_id": item["msg_id"]}).execute()
        return

    # 1. SELALU HITUNG FALLBACK (Untuk National Mood Index)
    try:
        fb = pipeline.predict_gated(text=combined_text, context=None)
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
        sb.rpc("ack_nlp_message", {"p_msg_id": item["msg_id"]}).execute()
        return

    # 2. CEK PANJANG TEKS UNTUK MENENTUKAN JALUR (TIER)
    metadata = item.get("metadata") or {}
    item_entity_id = metadata.get("configured_entity_id")

    if len(combined_text) >= 500:
        # TIER 1: FULL ARTICLE (DDG / RSS Native)
        stats["tier1_full_article"] += 1
        
        if item_entity_id:
            matched = [e for e in entities if e["id"] == item_entity_id]
        else:
            matched = find_alias_candidates(title, text, entities)

        stats["alias_candidates_total"] += len(matched)

        for e in matched:
            try:
                result = pipeline.predict_gated(text=combined_text, context=e["canonical_name"])
            except Exception:
                stats["gate_error"] += 1
                continue

            if not result.is_relevant:
                stats["gate_rejected"] += 1
                continue

            # BINARY MAPPING (Hapus Netral untuk Termometer Digital)
            label = result.label
            scores = result.scores
            if label == "neutral":
                label = "positive" if scores[2] >= scores[0] else "negative"

            try:
                sb.rpc("insert_sentiment_score", {
                    "p_raw_text_id": raw_id, "p_entity_id": e["id"],
                    "p_label": label, "p_neg": float(scores[0]),
                    "p_neu": float(scores[1]), "p_pos": float(scores[2]),
                    "p_confidence": float(result.sentiment_confidence),
                    "p_model_version": MODEL_VERSION_GATED,
                }).execute()
                stats["entity_inserted"] += 1
                stats[f"gated_label_{label}"] += 1
            except Exception:
                stats["insert_error"] += 1

    else:
        # TIER 2: SNIPPET ONLY (GNews)
        stats["tier2_snippet"] += 1
        
        # TIDAK ADA Binary Mapping & TIDAK ADA Relevancy Gate.
        # Simpan apa adanya (Pos/Neu/Neg) menggunakan Fallback score.
        if item_entity_id:
            try:
                sb.rpc("insert_sentiment_score", {
                    "p_raw_text_id": raw_id, "p_entity_id": item_entity_id,
                    "p_label": fb.label, # Bisa Positif, Netral, atau Negatif
                    "p_neg": float(fb.scores[0]), "p_neu": float(fb.scores[1]), "p_pos": float(fb.scores[2]),
                    "p_confidence": float(fb.sentiment_confidence),
                    "p_model_version": MODEL_VERSION_FALLBACK, # Tag sebagai fallback
                }).execute()
                stats["snippet_entity_inserted"] += 1
                stats[f"snippet_label_{fb.label}"] += 1
            except Exception:
                stats["insert_error"] += 1

    # 4. Ack
    try:
        sb.rpc("ack_nlp_message", {"p_msg_id": item["msg_id"]}).execute()
        stats["acked"] += 1
    except Exception:
        stats["ack_error"] += 1

def main():
    parser = argparse.ArgumentParser(description="Drain pgmq queue (NLP Worker v3)")
    parser.add_argument("--target", type=int, default=300, help="Berhenti setelah N artikel")
    parser.add_argument("--batch-size", type=int, default=30, help="Jumlah item per dequeue")
    parser.add_argument("--all", action="store_true", help="Drain sampai habis")
    args = parser.parse_args()

    sb = get_client()
    entities = load_entities(sb)
    print(f"Loaded {len(entities)} entitas aktif")

    print("Loading model (relevancy + sentiment + fallback)...")
    pipeline = get_pipeline()
    _ = pipeline.relevancy; _ = pipeline.sentiment; _ = pipeline.fallback
    print("Model siap.\n")

    stats = Counter()
    processed = 0
    start = time.time()

    print(f"{'='*70}\nDRAIN START (Two-Tier Mode) — target={'ALL' if args.all else args.target}\n{'='*70}")

    while True:
        if not args.all and processed >= args.target: break

        remaining = (args.target - processed) if not args.all else args.batch_size
        qty = min(args.batch_size, remaining) if not args.all else args.batch_size
        qty = max(qty, 1)

        res = sb.rpc("dequeue_nlp_batch", {"p_vt": 300, "p_qty": qty}).execute()
        items = res.data or []

        if not items:
            print("\nQueue kosong. Drain selesai.")
            break

        for item in items:
            process_one(sb, pipeline, entities, item, stats)
            processed += 1

            if processed % 20 == 0:
                elapsed = time.time() - start
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"  [{processed} diproses] T1={stats['tier1_full_article']} T2={stats['tier2_snippet']} | {rate:.2f} art/detik")

    elapsed = time.time() - start
    print(f"\n{'='*70}\nRINGKASAN DRAIN (Two-Tier)")
    print(f"Total diproses          : {processed}")
    print(f"Waktu                   : {elapsed:.0f}s ({elapsed/60:.1f} menit)")
    print(f"Tier 1 (Full Article)   : {stats['tier1_full_article']}")
    print(f"  -> Entity Match (Lolos): {stats['entity_inserted']}")
    print(f"  -> Gate Rejected       : {stats['gate_rejected']}")
    print(f"  -> Distribusi (Binary) : Pos={stats['gated_label_positive']} | Neg={stats['gated_label_negative']}")
    print(f"Tier 2 (Snippet GNews)  : {stats['tier2_snippet']}")
    print(f"  -> Snippet Inserted    : {stats['snippet_entity_inserted']}")
    print(f"  -> Distribusi (Asli)   : Pos={stats['snippet_label_positive']} | Neu={stats['snippet_label_neutral']} | Neg={stats['snippet_label_negative']}")
    print(f"Fallback (National Index): {stats['fallback_inserted']}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()