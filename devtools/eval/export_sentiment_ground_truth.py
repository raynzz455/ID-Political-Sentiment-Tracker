"""
export_sentiment_ground_truth.py v2
===================================
Export sample sentiment_scores (entity-specific, BUKAN NULL) ke CSV
untuk dilabeli manual. Ini untuk evaluasi STAGE 2 (sentiment) saja.

FIX v2:
  1. LAPIS 2 INTEGRATION: Ambil teks dari DB, jika < 500 char, fetch URL.
  2. URL EXPORT: Wajib bawa source_url & era_hint agar evaluator bisa baca 
     artikel aslinya di browser.
  3. EXCEL-SAFE: Pakai csv.QUOTE_ALL & utf-8-sig agar tidak pecah di Excel.
  4. CONTEXT AWARE: Menyertakan instruksi evaluasi spesifik per-tokoh.

Usage:
    python export_sentiment_ground_truth.py --n 150 --model-version indobert-ctx-relevancy-gated-v1
"""

import os
import sys
import csv
import re
import argparse
import random
from collections import defaultdict
from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
load_dotenv(ROOT_DIR / ".env")

try:
    from supabase import create_client, Client
except ImportError:
    print("[ERROR] pip install supabase")
    sys.exit(1)

try:
    import requests
    from trafilatura import extract as traf_extract
    FETCH_AVAILABLE = True
except ImportError:
    FETCH_AVAILABLE = False

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SERVICE_KEY  = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

def get_client() -> Client:
    if not SUPABASE_URL or not SERVICE_KEY:
        print("[ERROR] Set SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY")
        sys.exit(1)
    return create_client(SUPABASE_URL, SERVICE_KEY)

def strip_html(text: str) -> str:
    if not text: return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-z]+;', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def fetch_full_body(url: str, timeout: int = 15) -> str:
    if not FETCH_AVAILABLE or not url: return ""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        if not resp.ok: return ""
        return traf_extract(resp.text, include_comments=False, include_tables=False) or ""
    except Exception:
        return ""

def fetch_candidates(sb: Client, model_version: str | None) -> list[dict]:
    """Ambil sentiment_scores dan raw_texts secara terpisah lalu merge di Python."""
    
    # 1. Ambil sentiment_scores (dengan join ke political_entities yang punya FK)
    query = sb.table("sentiment_scores") \
              .select(
                  "id, raw_text_id, entity_id, label, confidence, "
                  "score_negative, score_neutral, score_positive, model_version, "
                  "political_entities(canonical_name)"
              ) \
              .not_.is_("entity_id", "null")

    if model_version:
        query = query.eq("model_version", model_version)

    res_scores = query.limit(2000).execute()
    scores_data = res_scores.data or []
    
    if not scores_data:
        return []

    # 2. Kumpulkan semua raw_text_id
    raw_text_ids = [r["raw_text_id"] for r in scores_data if r.get("raw_text_id")]
    
    # 3. Ambil raw_texts secara terpisah (karena tidak ada FK)
    raw_texts_map = {}
    if raw_text_ids:
        # PostgREST filter `in` butuh format string yang dipisah koma atau list
        res_raw = sb.table("raw_texts") \
                    .select("id, title, text, source_url, published_at, metadata") \
                    .in_("id", raw_text_ids) \
                    .execute()
        
        for rt in (res_raw.data or []):
            raw_texts_map[rt["id"]] = rt

    # 4. Merge hasilnya
    merged_rows = []
    for score in scores_data:
        rt_id = score.get("raw_text_id")
        raw_text_data = raw_texts_map.get(rt_id, {})
        
        # Bentuk ulang agar mirip dengan struktur nested select
        score["raw_texts"] = raw_text_data
        merged_rows.append(score)

    return merged_rows

def stratified_sample(rows: list[dict], n_per_class: int) -> list[dict]:
    by_label = defaultdict(list)
    for r in rows:
        by_label[r.get("label", "unknown")].append(r)

    sampled = []
    for label, items in by_label.items():
        random.shuffle(items)
        sampled.extend(items[:n_per_class])

    random.shuffle(sampled)
    return sampled

def main():
    parser = argparse.ArgumentParser(description="Export ground truth sentiment v2")
    parser.add_argument("--n", type=int, default=150, help="Total target sample")
    parser.add_argument("--model-version", type=str, default=None)
    parser.add_argument("--output", type=str, default="sentiment_ground_truth_TEMPLATE.csv")
    args = parser.parse_args()

    sb = get_client()
    print(f"Fetching sentiment_scores {f'(model_version={args.model_version})' if args.model_version else ''} ...")
    rows = fetch_candidates(sb, args.model_version)
    print(f"  -> {len(rows)} baris ditemukan di DB")

    if not rows:
        print("[ERROR] Tidak ada data. Jalankan batch processing dulu.")
        sys.exit(1)

    n_per_class = max(1, args.n // 3)
    sample = stratified_sample(rows, n_per_class)
    print(f"  -> {len(sample)} baris di-sample (stratified). Memulai enrichment teks...")

    rows_out = []
    for i, r in enumerate(sample, 1):
        entity = r.get("political_entities") or {}
        raw = r.get("raw_texts") or {}
        
        # Proses enrichment teks
        title = strip_html(raw.get("title") or "")
        text_db = strip_html(raw.get("text") or "")
        source_url = raw.get("source_url") or ""
        
        if len(text_db) < 500 and source_url:
            print(f"  [{i}/{len(sample)}] Enriching: {title[:50]}...")
            full_body = strip_html(fetch_full_body(source_url))
            if len(full_body) > len(text_db):
                text_db = full_body
                
        combined_text = f"{title} {text_db}".strip()
        
        # Ekstrak era
        pub_date = raw.get("published_at") or ""
        era_hint = f"Tahun {pub_date[:4]}" if pub_date and len(pub_date) >= 4 else "N/A"

        rows_out.append({
            "raw_text_id": r["raw_text_id"],
            "entity_name": entity.get("canonical_name", "?"),
            "title": title[:150],
            "enriched_text_preview": combined_text[:400], # Tampilkan 400 char
            "source_url": source_url,                     # Wajib untuk evaluasi manual
            "era_hint": era_hint,
            "predicted_label": r["label"],
            "predicted_confidence": f"{r['confidence']:.3f}",
            "model_version": r.get("model_version", "?"),
            "gold_label": "",   # ISI MANUAL: negative / neutral / positive
            "notes": ""
        })

    # Tulis ke CSV dengan format Excel-Safe (QUOTE_ALL & utf-8-sig)
    with open(args.output, "w", newline="", encoding="utf-8-sig", errors="replace") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nSelesai. File: {args.output}")
    print("LANGKAH SELANJUTNYA:")
    print("  1. Buka file CSV di Excel/Spreadsheet.")
    print("  2. Baca 'enriched_text_preview'. Jika bingung, KLIK 'source_url' untuk baca artikel aslinya.")
    print(f"  3. Fokus pada tokoh di kolom 'entity_name'. Apakah nada beritanya positif, netral, atau negatif TERHADAP tokoh tersebut?")
    print("  4. Isi 'gold_label': negative / neutral / positive")
    print("  5. Simpan ulang sebagai CSV (Pilih: CSV (Comma delimited) (*.csv)).")
    print(f"  6. Jalankan: python eval_metrics.py --sentiment {args.output}")

if __name__ == "__main__":
    main()