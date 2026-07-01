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
import json
import argparse
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from sentiment_model import get_pipeline

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")
# Lazy imports — beri pesan jelas kalau belum install
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
        print("        Atau buat file .env di folder ini")
        sys.exit(1)
    return create_client(SUPABASE_URL, SERVICE_KEY)


# ============================================================
# Model placeholder — INI YANG AKAN DIGANTI ONNX NANTI
# ============================================================
# Untuk CLI testing awal, pakai rule-based dummy.
# Begitu ONNX model siap, ganti body fungsi ini.
# Output format HARUS konsisten: (label, confidence, [neg, neu, pos])
# confidence = max(neg, neu, pos)

def predict_sentiment(text: str):
    """
    PLACEHOLDER — rule-based sentiment untuk testing struktur.
    Target: ganti dengan IndoBERT ONNX inference.

    Returns: (label: str, confidence: float, scores: tuple)
    """
    t = text.lower()

    # Kamus kata sederhana (Indonesia)
    positive_words = [
        "bagus", "baik", "hebat", "sukses", "dukung", "mendukung", "kompak",
        "positif", "berhasil", "prestasi", "menguntungkan", "cerdas", "juara",
        "memuaskan", "unggul", "maju", "stabil", "aman", "prosperous",
    ]
    negative_words = [
        "buruk", "gagal", "korupsi", "skandal", "korup", "kritisi", "kritik",
        "negatif", "turun", "rugi", "tertangkap", "diduga", "tersangka",
        "kasus", "pelanggaran", "terlibat", "didakwa", "salah", "kecewa",
        "rusak", "krisis", "konflik", "demo", "unjuk rasa", "menolak",
    ]

    pos_count = sum(1 for w in positive_words if w in t)
    neg_count = sum(1 for w in negative_words if w in t)

    # Skor dummy (akan diganti softmax model asli)
    if pos_count > neg_count:
        scores = (0.15, 0.20, 0.65)  # positive dominant
        label = "positive"
    elif neg_count > pos_count:
        scores = (0.65, 0.20, 0.15)  # negative dominant
        label = "negative"
    else:
        scores = (0.20, 0.60, 0.20)  # neutral dominant
        label = "neutral"

    confidence = max(scores)
    return label, confidence, scores


# ============================================================
# Entity matching (sama dengan yang akan dipakai worker production)
# ============================================================
ENTITY_CACHE = None


def load_entities(sb: Client):
    """Cache political_entities + aliases ke memori."""
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
    """Match teks + title ke tokoh via aliases (case-insensitive substring)."""
    # Gabungkan title + text karena RSS sering kirim body kosong
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
# Content enrichment (Lapis 2 — 2-stage pipeline)
# ============================================================
# Saat body text dari RSS kosong/pendek, follow source_url:
#   1. Google News redirect → artikel asli (detik/kompas/cnn/...)
#   2. trafilatura.extract() → main content bersih dari HTML
# Ini unlock full body 300-500 kata untuk akurasi sentiment.

try:
    import requests
    from trafilatura import extract as traf_extract
    FETCH_AVAILABLE = True
except ImportError:
    FETCH_AVAILABLE = False


def fetch_full_body(url: str, timeout: int = 10) -> str:
    """
    Follow source_url (gnews redirect) → scrape main content via trafilatura.
    Returns: full body text, atau string kosong kalau gagal.
    """
    if not FETCH_AVAILABLE:
        return ""
    if not url:
        return ""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ID-Sentiment-Tracker/1.0)"
        }
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        if not resp.ok:
            return ""
        # trafilatura auto-detect bahasa, extract main content, buang menu/ad/footer
        body = traf_extract(resp.text, include_comments=False, include_tables=False)
        return body or ""
    except Exception as e:
        print(f"       [fetch_full_body error] {type(e).__name__}: {e}")
        return ""


def enrich_if_needed(item: dict, min_len: int = 80) -> str:
    """
    Kembalikan text yang siap untuk NLP.
    Kalau body RSS kosong/pendek (< min_len) & ada source_url → fetch full body.
    Fallback ke title+text kalau fetch gagal.
    """
    text = (item.get("text") or "").strip()
    title = (item.get("title") or "").strip()
    source_url = item.get("source_url") or ""

    # Sudah cukup panjang → pakai langsung
    if len(text) >= min_len:
        return f"{title} {text}".strip()

    # Body pendek/kosong + ada URL → fetch full article
    if source_url:
        full = fetch_full_body(source_url)
        if len(full) >= min_len:
            return f"{title} {full}".strip()

    # Fallback terakhir: title + text apa adanya
    return f"{title} {text}".strip()


# ============================================================
# Commands
# ============================================================
def cmd_inspect(sb: Client, args):
    """Lihat isi queue tanpa memproses (peek-only)."""
    res = sb.rpc("dequeue_nlp_batch", {"p_vt": 60, "p_qty": 5}).execute()
    items = res.data or []

    print(f"\n{'='*60}")
    print(f"PEEK QUEUE (peek-only, tidak ack)")
    print(f"{'='*60}")
    print(f"Items returned: {len(items)}")
    print()

    for i, item in enumerate(items, 1):
        text = (item.get("text") or "")[:120]
        title = (item.get("title") or "(no title)")[:80]
        print(f"[{i}] source: {item.get('source', '?')}")
        print(f"    title:  {title}")
        print(f"    text:   {text}{'...' if len(item.get('text','')) > 120 else ''}")
        print()

    print("Catatan: 5 item ini sekarang invisible di queue selama 60s (vt).")
    print("         Mereka akan reappear otomatis kalau tidak di-ack.")
    print(f"{'='*60}\n")


def cmd_sample(sb: Client, args):
    """Proses N item, tampilkan hasil, AKAN ack + insert score."""
    n = args.count
    res = sb.rpc("dequeue_nlp_batch", {"p_vt": 120, "p_qty": n}).execute()
    items = res.data or []

    print(f"\n{'='*60}")
    print(f"SAMPLE PROCESS — {n} items")
    print(f"{'='*60}")

    if not items:
        print("Queue kosong. Jalankan ingestion dulu (curl edge function).")
        return

    entities = load_entities(sb)
    print(f"Entities loaded: {len(entities)} tokoh aktif\n")

    # Inisialisasi pipeline model asli
    pipeline = get_pipeline()

    processed = 0
    no_entity = 0

    for i, item in enumerate(items, 1):
        text = item.get("text", "")
        title = item.get("title", "")
        raw_id = item.get("raw_text_id")
        msg_id = item.get("msg_id")
        item_entity_id = item.get("entity_id")  # sudah di-set dari gnews feeds

        # LAPIS 2: enrich body kalau RSS kosong/pendek (follow source_url → scrape)
        combined = enrich_if_needed(item)
        fetched = len(combined) > len(f"{title} {text}".strip()) + 20

        # PRIORITAS 1: pakai entity_id dari queue (per-tokoh feeds sudah attribute)
        if item_entity_id:
            matched = [e for e in entities if e["id"] == item_entity_id]
        else:
            # PRIORITAS 2: general feed (entity_id NULL) → fallback text matching
            matched = match_entities(combined, title, entities)

        title_preview = (item.get("title") or "")[:70]
        text_preview = combined[:100]
        fetch_tag = " [fetched]" if fetched else ""
        print(f"[{i}/{len(items)}] {title_preview}{fetch_tag}")
        print(f"       text: {text_preview}{'...' if len(combined) > 100 else ''} ({len(combined)} chars)")
        print(f"       matched: {len(matched)} tokoh — {[m['canonical_name'] for m in matched][:3]}")

        if not matched:
            # Gunakan dummy/placeholder model default jika tidak ada entitas yang match
            # Tetap dimasukkan dengan entity_id = NULL untuk pipeline testing
            label, conf, scores = predict_sentiment(combined)
            sb.rpc("insert_sentiment_score", {
                "p_raw_text_id": raw_id,
                "p_entity_id": None,
                "p_label": label,
                "p_neg": float(scores[0]),
                "p_neu": float(scores[1]),
                "p_pos": float(scores[2]),
                "p_confidence": float(conf),
            }).execute()
            print(f"       → inserted score (no entity, entity_id=NULL)")
            no_entity += 1
            sb.rpc("ack_nlp_message", {"p_msg_id": msg_id}).execute()
            print(f"       ✓ acked msg_id={msg_id}")
            processed += 1
            print()
            continue

        # Loop per-entitas menggunakan model pipeline asli dengan gating relevansi
        for e in matched:
            result = pipeline.predict_gated(text=combined, context=e["canonical_name"])
            
            if not result.is_relevant:
                print(f"       -> SKIP {e['canonical_name']}: tidak relevan "
                      f"(confidence={result.relevancy_confidence:.3f})")
                continue   # JANGAN insert_sentiment_score untuk entity ini

            # Hanya insert kalau relevan
            sb.rpc("insert_sentiment_score", {
                "p_raw_text_id": raw_id,
                "p_entity_id": e["id"],
                "p_label": result.label,
                "p_neg": float(result.scores[0]),
                "p_neu": float(result.scores[1]),
                "p_pos": float(result.scores[2]),
                "p_confidence": float(result.sentiment_confidence),
            }).execute()
            print(f"       -> inserted score for {e['canonical_name']} "
                  f"(relevancy={result.relevancy_confidence:.3f})")

        # Ack message
        sb.rpc("ack_nlp_message", {"p_msg_id": msg_id}).execute()
        print(f"       ✓ acked msg_id={msg_id}")
        processed += 1
        print()

    print(f"{'='*60}")
    print(f"SUMMARY: processed={processed}, no-entity (NULL)={no_entity}")
    print(f"{'='*60}\n")


def cmd_batch(sb: Client, args):
    """Proses N item, tampilkan distribusi sentiment. Akan commit."""
    n = args.count
    res = sb.rpc("dequeue_nlp_batch", {"p_vt": 300, "p_qty": n}).execute()
    items = res.data or []

    print(f"\n{'='*60}")
    print(f"BATCH PROCESS — {n} items (distribusi)")
    print(f"{'='*60}")

    if not items:
        print("Queue kosong.")
        return

    entities = load_entities(sb)
    
    # Inisialisasi pipeline model asli
    pipeline = get_pipeline()

    label_counts = Counter()
    entity_counts = Counter()
    conf_buckets = Counter()
    processed = 0
    no_entity = 0
    conf_sum = 0.0
    total_predictions = 0

    for item in items:
        text = item.get("text", "")
        title = item.get("title", "")
        raw_id = item.get("raw_text_id")
        msg_id = item.get("msg_id")
        item_entity_id = item.get("entity_id")  # sudah di-set dari gnews feeds

        # LAPIS 2: enrich body kalau RSS kosong/pendek (follow source_url → scrape)
        combined = enrich_if_needed(item)

        # PRIORITAS 1: pakai entity_id dari queue (per-tokoh feeds sudah attribute)
        if item_entity_id:
            matched = [e for e in entities if e["id"] == item_entity_id]
        else:
            # PRIORITAS 2: general feed (entity_id NULL) → fallback text matching
            matched = match_entities(combined, title, entities)

        if not matched:
            no_entity += 1
            sb.rpc("ack_nlp_message", {"p_msg_id": msg_id}).execute()
            continue

        for e in matched:
            result = pipeline.predict_gated(text=combined, context=e["canonical_name"])
            
            if not result.is_relevant:
                continue   # Skip insert dan metrik statistik distribusi jika tidak relevan

            label = result.label
            conf = result.sentiment_confidence
            
            label_counts[label] += 1
            conf_sum += conf
            total_predictions += 1

            # Bucket confidence
            if conf < 0.5:
                conf_buckets["<0.5"] += 1
            elif conf < 0.7:
                conf_buckets["0.5-0.7"] += 1
            elif conf < 0.85:
                conf_buckets["0.7-0.85"] += 1
            else:
                conf_buckets[">=0.85"] += 1

            entity_counts[e["canonical_name"]] += 1
            sb.rpc("insert_sentiment_score", {
                "p_raw_text_id": raw_id,
                "p_entity_id": e["id"],
                "p_label": label,
                "p_neg": float(result.scores[0]),
                "p_neu": float(result.scores[1]),
                "p_pos": float(result.scores[2]),
                "p_confidence": float(conf),
            }).execute()

        sb.rpc("ack_nlp_message", {"p_msg_id": msg_id}).execute()
        processed += 1

    total = len(items)
    print(f"\nTotal items: {total}")
    print(f"Processed (entity matched + inserted): {processed}")
    print(f"Skipped (no entity match, acked only): {no_entity}")
    
    avg_conf = (conf_sum / total_predictions) if total_predictions > 0 else 0.0
    print(f"Avg confidence (relevant only): {avg_conf:.3f}")
    print()
    print("Sentiment distribution (relevant only):")
    for label in ["positive", "neutral", "negative"]:
        c = label_counts.get(label, 0)
        bar = "█" * int((c / total_predictions) * 40) if total_predictions > 0 else ""
        pct = (c / total_predictions) * 100 if total_predictions > 0 else 0.0
        print(f"  {label:10s} {c:3d} ({pct:5.1f}%) {bar}")
    print()
    print("Confidence buckets (relevant only):")
    for bucket in ["<0.5", "0.5-0.7", "0.7-0.85", ">=0.85"]:
        c = conf_buckets.get(bucket, 0)
        pct = (c / total_predictions) * 100 if total_predictions > 0 else 0.0
        print(f"  {bucket:10s} {c:3d} ({pct:5.1f}%)")
    print()
    print("Top mentioned entities (relevant only):")
    for name, c in entity_counts.most_common(10):
        print(f"  {name:30s} {c:3d} mentions")
    print(f"\n{'='*60}\n")


def cmd_single(sb: Client, args):
    """Test 1 teks manual."""
    text = args.text
    entities = load_entities(sb)

    print(f"\n{'='*60}")
    print(f"SINGLE TEST")
    print(f"{'='*60}")
    print(f"Text: {text}\n")

    label, conf, scores = predict_sentiment(text)
    matched = match_entities(text, "", entities)

    print(f"Label: {label}")
    print(f"Confidence: {conf:.3f}")
    print(f"Scores: neg={scores[0]:.3f}, neu={scores[1]:.3f}, pos={scores[2]:.3f}")
    print(f"Matched entities: {[m['canonical_name'] for m in matched]}")
    print(f"{'='*60}\n")


def cmd_stats(sb: Client, args):
    """Statistik DB: processed vs pending, queue depth."""
    print(f"\n{'='*60}")
    print(f"DB STATS")
    print(f"{'='*60}")

    # raw_texts status
    res = sb.table("raw_texts").select("status").execute()
    status_counts = Counter(r["status"] for r in res.data)
    print("\nraw_texts by status:")
    for status, c in status_counts.most_common():
        print(f"  {status:15s} {c:5d}")
    print(f"  {'TOTAL':15s} {len(res.data):5d}")

    # sentiment_scores count
    res2 = sb.table("sentiment_scores").select("id", count="exact").execute()
    print(f"\nsentiment_scores total: {len(res2.data)}")

    # Top entities by mention
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
    parser = argparse.ArgumentParser(
        description="ID-Sentiment CLI — NLP testing tool"
    )
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
    p_single.add_argument("--no-insert", action="store_true", help="jangan insert ke DB")
    p_single.set_defaults(func=cmd_single)

    p_stats = sub.add_parser("stats", help="Lihat statistik DB")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    sb = get_client()
    args.func(sb, args)


if __name__ == "__main__":
    main()