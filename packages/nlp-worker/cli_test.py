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
# Content enrichment (Lapis 2)
# ============================================================
try:
    import requests
    from trafilatura import extract as traf_extract
    FETCH_AVAILABLE = True
except ImportError:
    FETCH_AVAILABLE = False

def fetch_full_body(url: str, timeout: int = 15) -> str:
    if not FETCH_AVAILABLE or not url:
        return ""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
        }
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        if not resp.ok:
            return ""
        return traf_extract(resp.text, include_comments=False, include_tables=False) or ""
    except Exception:
        return ""

def enrich_if_needed(item: dict, min_len: int = 500) -> str:
    text = (item.get("text") or "").strip()
    title = (item.get("title") or "").strip()
    source_url = item.get("source_url") or ""

    if len(text) >= min_len:
        return f"{title} {text}".strip()

    if source_url and FETCH_AVAILABLE:
        full = fetch_full_body(source_url)
        if len(full) >= min_len:
            return f"{title} {full}".strip()

    return f"{title} {text}".strip()

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
    start_time = time.perf_counter() # TIMER MULAI
    
    n = args.count
    res = sb.rpc("dequeue_nlp_batch", {"p_vt": 120, "p_qty": n}).execute()
    items = res.data or []

    print(f"\n{'='*60}\nSAMPLE PROCESS — {n} items\n{'='*60}")

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

        combined = enrich_if_needed(item)
        fetched = len(combined) > len(f"{title} {item.get('text', '')}".strip()) + 20

        if len(combined) < 200:
            print(f"[{i}/{len(items)}] {title[:70]}")
            print(f"SKIP: Teks terlalu pendek ({len(combined)} chars). Gagal enrich.")
            skipped_short += 1
            sb.rpc("ack_nlp_message", {"p_msg_id": msg_id}).execute()
            continue

        if item_entity_id:
            matched = [e for e in entities if e["id"] == item_entity_id]
        else:
            matched = match_entities(combined, title, entities)

        print(f"[{i}/{len(items)}] {title[:70]}{' [fetched]' if fetched else ''}")
        print(f"       text_len: {len(combined)} chars")
        print(f"       matched: {len(matched)} tokoh")

        if not matched:
            fb = pipeline.predict_gated(text=combined, context=None)
            sb.rpc("insert_sentiment_score", {
                "p_raw_text_id": raw_id, "p_entity_id": None,
                "p_label": fb.label, "p_neg": float(fb.scores[0]),
                "p_neu": float(fb.scores[1]), "p_pos": float(fb.scores[2]),
                "p_confidence": float(fb.sentiment_confidence),
                "p_model_version": "indobert-fallback-v1"
            }).execute()
            print(f"       → inserted fallback score (entity_id=NULL, label={fb.label})")
            no_entity += 1
            sb.rpc("ack_nlp_message", {"p_msg_id": msg_id}).execute()
            processed += 1
            print()
            continue

        for e in matched:
            result = pipeline.predict_gated(text=combined, context=e["canonical_name"])
            if not result.is_relevant:
                print(f"       -> SKIP {e['canonical_name']}: tidak relevan (conf={result.relevancy_confidence:.3f})")
                continue

            # --- BINARY FORCED MAPPING ---
            label = result.label
            scores = result.scores
            
            if label == "neutral":
                if scores[2] >= scores[0]:
                    label = "positive"
                else:
                    label = "negative"
            # -----------------------------

            sb.rpc("insert_sentiment_score", {
                "p_raw_text_id": raw_id, "p_entity_id": e["id"],
                "p_label": label, "p_neg": float(scores[0]),
                "p_neu": float(scores[1]), "p_pos": float(scores[2]),
                "p_confidence": float(result.sentiment_confidence),
                "p_model_version": "indobert-ctx-relevancy-gated-v1"
            }).execute()
            print(f"       -> inserted score for {e['canonical_name']} (label={label}, conf={result.sentiment_confidence:.3f})")

        sb.rpc("ack_nlp_message", {"p_msg_id": msg_id}).execute()
        processed += 1
        print()

    elapsed = time.perf_counter() - start_time # TIMER BERHENTI
    
    print(f"{'='*60}")
    print(f"SUMMARY: processed={processed}, no-entity={no_entity}, skipped_short={skipped_short}")
    print(f"⏱️ Waktu Eksekusi: {elapsed:.2f} detik ({elapsed/60:.2f} menit)")
    if processed > 0:
        print(f"🚀 Throughput: {processed/elapsed:.2f} artikel/detik")
    print(f"{'='*60}\n")

def cmd_batch(sb: Client, args):
    start_time = time.perf_counter() # TIMER MULAI
    
    n = args.count
    res = sb.rpc("dequeue_nlp_batch", {"p_vt": 300, "p_qty": n}).execute()
    items = res.data or []

    print(f"\n{'='*60}\nBATCH PROCESS — {n} items (distribusi)\n{'='*60}")

    if not items:
        print("Queue kosong.")
        return

    entities = load_entities(sb)
    pipeline = get_pipeline()

    label_counts = Counter()
    entity_counts = Counter()
    processed = 0
    no_entity = 0
    skipped_short = 0

    for item in items:
        title = item.get("title", "")
        raw_id = item.get("raw_text_id")
        msg_id = item.get("msg_id")
        
        metadata = item.get("metadata") or {}
        item_entity_id = metadata.get("configured_entity_id")

        combined = enrich_if_needed(item)

        if len(combined) < 200:
            skipped_short += 1
            sb.rpc("ack_nlp_message", {"p_msg_id": msg_id}).execute()
            continue

        if item_entity_id:
            matched = [e for e in entities if e["id"] == item_entity_id]
        else:
            matched = match_entities(combined, title, entities)

        if not matched:
            fb = pipeline.predict_gated(text=combined, context=None)
            sb.rpc("insert_sentiment_score", {
                "p_raw_text_id": raw_id, "p_entity_id": None,
                "p_label": fb.label, "p_neg": float(fb.scores[0]),
                "p_neu": float(fb.scores[1]), "p_pos": float(fb.scores[2]),
                "p_confidence": float(fb.sentiment_confidence),
                "p_model_version": "indobert-fallback-v1"
            }).execute()
            no_entity += 1
            sb.rpc("ack_nlp_message", {"p_msg_id": msg_id}).execute()
            continue

        for e in matched:
            result = pipeline.predict_gated(text=combined, context=e["canonical_name"])
            if not result.is_relevant:
                continue

            # --- BINARY FORCED MAPPING ---
            label = result.label
            scores = result.scores
            
            if label == "neutral":
                if scores[2] >= scores[0]:
                    label = "positive"
                else:
                    label = "negative"
            # -----------------------------

            label_counts[label] += 1
            entity_counts[e["canonical_name"]] += 1

            sb.rpc("insert_sentiment_score", {
                "p_raw_text_id": raw_id, "p_entity_id": e["id"],
                "p_label": label, "p_neg": float(scores[0]),
                "p_neu": float(scores[1]), "p_pos": float(scores[2]),
                "p_confidence": float(result.sentiment_confidence),
                "p_model_version": "indobert-ctx-relevancy-gated-v1"
            }).execute()

        sb.rpc("ack_nlp_message", {"p_msg_id": msg_id}).execute()
        processed += 1

    elapsed = time.perf_counter() - start_time # TIMER BERHENTI
    
    total = len(items)
    total_pred = sum(label_counts.values())
    print(f"\nTotal items: {total}")
    print(f"Processed (entity matched + inserted): {processed}")
    print(f"Skipped (no entity match, fallback inserted): {no_entity}")
    print(f"Skipped (text too short <200 chars): {skipped_short}")
    print("\nSentiment distribution (Binary Forced):")
    for label in ["positive", "negative"]: # Netral dihapus dari display
        c = label_counts.get(label, 0)
        pct = (c / total_pred) * 100 if total_pred > 0 else 0.0
        print(f"  {label:10s} {c:4d} ({pct:5.1f}%)")
        
    print(f"\n Waktu Eksekusi: {elapsed:.2f} detik ({elapsed/60:.2f} menit)")
    if processed > 0 and elapsed > 0:
        print(f" Throughput: {processed/elapsed:.2f} artikel/detik")
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

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="ID-Sentiment CLI — NLP testing tool")
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