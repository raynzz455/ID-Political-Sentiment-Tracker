"""
ID-Sentiment CLI — NLP Testing Tool (Terminal)
================================================
Tujuan: Lihat data real dari queue, jalankan sentiment model, observe distribusi
        sebelum commit ke production pipeline.

Usage:
    python cli_test.py inspect          # Lihat isi queue tanpa proses
    python cli_test.py sample 10        # Proses 10 item, tampilkan hasil
    python cli_test.py batch 50         # Proses 50, tampilkan distribusi
    python cli_test.py single "teks"    # Test 1 teks manual
    python cli_test.py stats            # Lihat statistik DB (processed/pending)

Env vars (bisa lewat .env atau environment):
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
"""
import os
import sys
import time
import argparse
import re
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv
from sentiment_model import get_pipeline

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

try:
    from supabase import create_client, Client
except ImportError:
    print("[ERROR] pip install supabase")
    sys.exit(1)

# ============================================================
# Config
# ============================================================
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

def get_client() -> Client:
    if not SUPABASE_URL or not SERVICE_KEY:
        print("[ERROR] Set SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY env vars")
        sys.exit(1)
    return create_client(SUPABASE_URL, SERVICE_KEY)

# ============================================================
# Entity matching
# ============================================================
ENTITY_CACHE = None

def load_entities(sb: Client):
    global ENTITY_CACHE
    if ENTITY_CACHE is not None:
        return ENTITY_CACHE
    res = sb.table("political_entities") \
            .select("id, canonical_name, aliases, is_active") \
            .eq("is_active", True) \
            .execute()
    ENTITY_CACHE = res.data
    return ENTITY_CACHE

def match_entities(text: str, title: str, entities: list) -> list:
    combined = f"{title} {text}".lower()
    matched = []
    seen_ids = set()
    for e in entities:
        if e["id"] in seen_ids:
            continue
        all_names = [e["canonical_name"]] + list(e.get("aliases", []))
        for name in all_names:        
            if len(name) < 4:
                continue
            pattern = r'\b' + re.escape(name.lower()) + r'\b'
            if re.search(pattern, combined):
                matched.append(e)
                seen_ids.add(e["id"])
                break  
    return matched

# ============================================================
# Commands
# ============================================================
def cmd_inspect(sb: Client, args):
    res = sb.rpc("dequeue_nlp_batch", {"p_vt": 60, "p_qty": 5}).execute()
    items = res.data or []

    print(f"\n{'='*60}\nPEEK QUEUE (peek-only)\n{'='*60}\nItems returned: {len(items)}\n")

    for i, item in enumerate(items, 1):
        text = (item.get("text") or "")[:120]
        title = (item.get("title") or "(no title)")[:80]
        print(f"[{i}] source: {item.get('source', '?')}")
        print(f"    title:  {title}")
        print(f"    text:   {text}{'...' if len(item.get('text','')) > 120 else ''}")
        print()

def cmd_sample(sb: Client, args):
    start_time = time.perf_counter()
    
    n = args.count
    res = sb.rpc("dequeue_nlp_batch", {"p_vt": 120, "p_qty": n}).execute()
    items = res.data or []

    print(f"\n{'='*60}\nSAMPLE PROCESS (Two-Tier) — {n} items\n{'='*60}")

    if not items:
        print("Queue kosong.")
        return

    entities = load_entities(sb)
    pipeline = get_pipeline()

    processed = 0
    no_entity = 0
    skipped_short = 0

    for i, item in enumerate(items, 1):
        title = item.get("title", "")
        raw_id = item.get("raw_text_id")
        msg_id = item.get("msg_id")
        
        metadata = item.get("metadata") or {}
        item_entity_id = metadata.get("configured_entity_id")

        # NLP Worker TIDAK melakukan Enrichment. Teks diambil apa adanya dari DB.
        combined_text = f"{title} {item.get('text', '')}".strip()

        # GUARD TEKS SANGAT PENDEK
        if len(combined_text) < 50:
            print(f"[{i}/{len(items)}] {title[:70]}")
            print(f"       ❌ SKIP: Teks terlalu pendek ({len(combined_text)} chars).")
            skipped_short += 1
            sb.rpc("ack_nlp_message", {"p_msg_id": msg_id}).execute()
            continue

        # 1. SELALU HITUNG FALLBACK (National Index)
        fb = pipeline.predict_gated(text=combined_text, context=None)
        try:
            sb.rpc("insert_sentiment_score", {
                "p_raw_text_id": raw_id, "p_entity_id": None,
                "p_label": fb.label, "p_neg": float(fb.scores[0]),
                "p_neu": float(fb.scores[1]), "p_pos": float(fb.scores[2]),
                "p_confidence": float(fb.sentiment_confidence),
                "p_model_version": "indobert-fallback-v1"
            }).execute()
        except Exception:
            pass

        print(f"[{i}/{len(items)}] {title[:70]}")
        print(f"       text_len: {len(combined_text)} chars")

        # 2. CEK PANJANG TEKS UNTUK MENENTUKAN JALUR (TIER)
        if len(combined_text) >= 500:
            # TIER 1: FULL ARTICLE
            print(f"       [TIER 1] Processing Full Article...")
            if item_entity_id:
                matched = [e for e in entities if e["id"] == item_entity_id]
            else:
                matched = match_entities(combined_text, title, entities)

            if not matched:
                print(f"       → inserted fallback only (no entity match)")
                no_entity += 1
            else:
                for e in matched:
                    result = pipeline.predict_gated(text=combined_text, context=e["canonical_name"])
                    if not result.is_relevant:
                        print(f"       -> SKIP {e['canonical_name']}: tidak relevan (conf={result.relevancy_confidence:.3f})")
                        continue

                    # BINARY MAPPING (Pos vs Neg)
                    label = result.label
                    scores = result.scores
                    if label == "neutral":
                        label = "positive" if scores[2] >= scores[0] else "negative"

                    sb.rpc("insert_sentiment_score", {
                        "p_raw_text_id": raw_id, "p_entity_id": e["id"],
                        "p_label": label, "p_neg": float(scores[0]),
                        "p_neu": float(scores[1]), "p_pos": float(scores[2]),
                        "p_confidence": float(result.sentiment_confidence),
                        "p_model_version": "indobert-ctx-relevancy-gated-v1"
                    }).execute()
                    print(f"       -> inserted score for {e['canonical_name']} (label={label}, conf={result.sentiment_confidence:.3f})")
        else:
            # TIER 2: SNIPPET ONLY
            print(f"       [TIER 2] Processing Snippet (No Binary Mapping)...")
            if item_entity_id:
                sb.rpc("insert_sentiment_score", {
                    "p_raw_text_id": raw_id, "p_entity_id": item_entity_id,
                    "p_label": fb.label, # Bisa Positif, Netral, atau Negatif
                    "p_neg": float(fb.scores[0]), "p_neu": float(fb.scores[1]), "p_pos": float(fb.scores[2]),
                    "p_confidence": float(fb.sentiment_confidence),
                    "p_model_version": "indobert-fallback-v1"
                }).execute()
                print(f"       -> inserted snippet score for entity (label={fb.label})")
            else:
                no_entity += 1

        sb.rpc("ack_nlp_message", {"p_msg_id": msg_id}).execute()
        processed += 1
        print()

    elapsed = time.perf_counter() - start_time
    
    print(f"{'='*60}")
    print(f"SUMMARY: processed={processed}, no-entity={no_entity}, skipped_short={skipped_short}")
    print(f"⏱️ Waktu Eksekusi: {elapsed:.2f} detik ({elapsed/60:.2f} menit)")
    if processed > 0:
        print(f"🚀 Throughput: {processed/elapsed:.2f} artikel/detik")
    print(f"{'='*60}\n")

def cmd_batch(sb: Client, args):
    start_time = time.perf_counter()
    
    n = args.count
    res = sb.rpc("dequeue_nlp_batch", {"p_vt": 300, "p_qty": n}).execute()
    items = res.data or []

    print(f"\n{'='*60}\nBATCH PROCESS (Two-Tier) — {n} items\n{'='*60}")

    if not items:
        print("Queue kosong.")
        return

    entities = load_entities(sb)
    pipeline = get_pipeline()

    # Stat terpisah untuk Tier 1 dan Tier 2
    tier1_labels = Counter()
    tier2_labels = Counter()
    processed = 0
    no_entity = 0
    skipped_short = 0

    for item in items:
        title = item.get("title", "")
        raw_id = item.get("raw_text_id")
        msg_id = item.get("msg_id")
        
        metadata = item.get("metadata") or {}
        item_entity_id = metadata.get("configured_entity_id")

        combined_text = f"{title} {item.get('text', '')}".strip()

        if len(combined_text) < 50:
            skipped_short += 1
            sb.rpc("ack_nlp_message", {"p_msg_id": msg_id}).execute()
            continue

        # 1. FALLBACK (National Index)
        fb = pipeline.predict_gated(text=combined_text, context=None)
        try:
            sb.rpc("insert_sentiment_score", {
                "p_raw_text_id": raw_id, "p_entity_id": None,
                "p_label": fb.label, "p_neg": float(fb.scores[0]),
                "p_neu": float(fb.scores[1]), "p_pos": float(fb.scores[2]),
                "p_confidence": float(fb.sentiment_confidence),
                "p_model_version": "indobert-fallback-v1"
            }).execute()
        except Exception:
            pass

        # 2. TIER PROCESSING
        if len(combined_text) >= 500:
            # TIER 1
            if item_entity_id:
                matched = [e for e in entities if e["id"] == item_entity_id]
            else:
                matched = match_entities(combined_text, title, entities)

            if not matched:
                no_entity += 1
            else:
                for e in matched:
                    result = pipeline.predict_gated(text=combined_text, context=e["canonical_name"])
                    if not result.is_relevant: continue

                    label = result.label
                    scores = result.scores
                    if label == "neutral":
                        label = "positive" if scores[2] >= scores[0] else "negative"

                    tier1_labels[label] += 1

                    sb.rpc("insert_sentiment_score", {
                        "p_raw_text_id": raw_id, "p_entity_id": e["id"],
                        "p_label": label, "p_neg": float(scores[0]),
                        "p_neu": float(scores[1]), "p_pos": float(scores[2]),
                        "p_confidence": float(result.sentiment_confidence),
                        "p_model_version": "indobert-ctx-relevancy-gated-v1"
                    }).execute()
        else:
            # TIER 2
            if item_entity_id:
                tier2_labels[fb.label] += 1
                sb.rpc("insert_sentiment_score", {
                    "p_raw_text_id": raw_id, "p_entity_id": item_entity_id,
                    "p_label": fb.label, "p_neg": float(fb.scores[0]),
                    "p_neu": float(fb.scores[1]), "p_pos": float(fb.scores[2]),
                    "p_confidence": float(fb.sentiment_confidence),
                    "p_model_version": "indobert-fallback-v1"
                }).execute()
            else:
                no_entity += 1

        sb.rpc("ack_nlp_message", {"p_msg_id": msg_id}).execute()
        processed += 1

    elapsed = time.perf_counter() - start_time
    
    print(f"\nTotal items: {n}")
    print(f"Processed: {processed}")
    print(f"Skipped (Text < 50 char): {skipped_short}")
    
    print("\n--- TIER 1 (Full Article > 500 char) ---")
    print("Distribusi (Binary Forced):")
    total_t1 = sum(tier1_labels.values())
    for label in ["positive", "negative"]:
        c = tier1_labels.get(label, 0)
        pct = (c / total_t1) * 100 if total_t1 > 0 else 0.0
        print(f"  {label:10s} {c:4d} ({pct:5.1f}%)")
        
    print("\n--- TIER 2 (Snippet < 500 char) ---")
    print("Distribusi (Original Labels):")
    total_t2 = sum(tier2_labels.values())
    for label in ["positive", "neutral", "negative"]:
        c = tier2_labels.get(label, 0)
        pct = (c / total_t2) * 100 if total_t2 > 0 else 0.0
        print(f"  {label:10s} {c:4d} ({pct:5.1f}%)")
        
    print(f"\n⏱️ Waktu Eksekusi: {elapsed:.2f} detik ({elapsed/60:.2f} menit)")
    if processed > 0 and elapsed > 0:
        print(f"🚀 Throughput: {processed/elapsed:.2f} artikel/detik")
    print(f"{'='*60}\n")

def cmd_single(sb: Client, args):
    text = args.text
    entities = load_entities(sb)
    pipeline = get_pipeline()

    print(f"\n{'='*60}\nSINGLE TEST (IndoBERT)\n{'='*60}\nText: {text}\n")

    result = pipeline.predict_gated(text=text, context=None)
    matched = match_entities(text, "", entities)

    print(f"Label: {result.label}")
    print(f"Confidence: {result.sentiment_confidence:.3f}")
    print(f"Scores: neg={result.scores[0]:.3f}, neu={result.scores[1]:.3f}, pos={result.scores[2]:.3f}")
    print(f"Matched entities: {[m['canonical_name'] for m in matched]}")
    print(f"{'='*60}\n")

def cmd_stats(sb: Client, args):
    print(f"\n{'='*60}\nDB STATS\n{'='*60}")

    res = sb.table("raw_texts").select("status").execute()
    status_counts = Counter(r["status"] for r in res.data)
    print("\nraw_texts by status:")
    for status, c in status_counts.most_common():
        print(f"  {status:15s} {c:5d}")
    print(f"  {'TOTAL':15s} {len(res.data):5d}")

    res2 = sb.table("sentiment_scores").select("id", count="exact").execute()
    print(f"\nsentiment_scores total: {len(res2.data)}")

    print("\nTop tokoh di sentiment_scores (kalau ada):")
    res3 = sb.table("sentiment_scores") \
             .select("entity_id, political_entities(canonical_name)") \
             .limit(500) \
             .execute()
    entity_counter = Counter()
    for r in res3.data:
        pe = r.get("political_entities") or {}
        name = pe.get("canonical_name", "?")        
        entity_counter[name] += 1
    for name, c in entity_counter.most_common(10):
        print(f"  {name:30s} {c:5d}")

    print(f"\n{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description="ID-Sentiment CLI — NLP testing tool v5")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="Lihat isi queue tanpa proses")
    p_inspect.set_defaults(func=cmd_inspect)

    p_sample = sub.add_parser("sample", help="Proses N item, tampilkan hasil detail")
    p_sample.add_argument("count", type=int, help="jumlah item")
    p_sample.set_defaults(func=cmd_sample)

    p_batch = sub.add_parser("batch", help="Proses N item, tampilkan distribusi")
    p_batch.add_argument("count", type=int, help="jumlah item")
    p_batch.set_defaults(func=cmd_batch)

    p_single = sub.add_parser("single", help="Test 1 teks manual")
    p_single.add_argument("text", type=str, help="teks untuk dianalisis")
    p_single.set_defaults(func=cmd_single)

    p_stats = sub.add_parser("stats", help="Lihat statistik DB")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    sb = get_client()
    args.func(sb, args)

if __name__ == "__main__":
    main()