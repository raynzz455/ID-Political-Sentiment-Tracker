"""
export_sentiment_ground_truth.py
==================================
Export sample sentiment_scores (entity-specific, BUKAN NULL) ke CSV
untuk dilabeli manual. Ini untuk evaluasi STAGE 2 (sentiment) saja.
Stage 1 (relevancy) punya script terpisah: export_relevancy_review.py

Stratified sampling: ambil porsi seimbang dari tiap label (negative/
neutral/positive) supaya evaluasi precision/recall per kelas valid --
random sampling murni bisa under-represent kelas yang jarang.

PENTING: jalankan ini SETELAH kontaminasi dummy dibersihkan (lihat
diagnostic SQL) dan SETELAH re-run batch dengan model_version yang
sudah di-tag eksplisit.

Usage:
    python export_sentiment_ground_truth.py --n 200 --model-version indobert-ctx-relevancy-gated-v1

Output: sentiment_ground_truth_TEMPLATE.csv
    Kolom predicted_* sudah terisi dari DB.
    Kolom gold_label HARUS diisi manual oleh manusia (negative/neutral/positive).
    Kolom notes opsional untuk catatan kasus ambigu.
"""

import os
import sys
import csv
import argparse
import random
from collections import defaultdict

try:
    from supabase import create_client, Client
except ImportError:
    print("[ERROR] pip install supabase")
    sys.exit(1)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SERVICE_KEY  = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")


def get_client() -> Client:
    if not SUPABASE_URL or not SERVICE_KEY:
        print("[ERROR] Set SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY")
        sys.exit(1)
    return create_client(SUPABASE_URL, SERVICE_KEY)


def fetch_candidates(sb: Client, model_version: str | None) -> list[dict]:
    """Ambil sentiment_scores entity-specific (entity_id NOT NULL) join raw_texts untuk teks."""
    query = sb.table("sentiment_scores") \
              .select(
                  "id, raw_text_id, entity_id, label, confidence, "
                  "score_negative, score_neutral, score_positive, model_version, "
                  "political_entities(canonical_name), "
                  "raw_texts(title, text)"
              ) \
              .not_.is_("entity_id", "null")

    if model_version:
        query = query.eq("model_version", model_version)

    res = query.limit(2000).execute()
    return res.data or []


def stratified_sample(rows: list[dict], n_per_class: int) -> list[dict]:
    """Ambil maksimal n_per_class baris per label, random shuffle dulu."""
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
    parser = argparse.ArgumentParser(description="Export ground truth sample untuk sentiment stage")
    parser.add_argument("--n", type=int, default=200, help="Total target sample (dibagi rata per kelas)")
    parser.add_argument("--model-version", type=str, default=None,
                         help="Filter model_version tertentu (kosongkan untuk ambil semua)")
    parser.add_argument("--output", type=str, default="sentiment_ground_truth_TEMPLATE.csv")
    args = parser.parse_args()

    sb = get_client()
    print(f"Fetching sentiment_scores (entity-specific)"
          f"{f' model_version={args.model_version}' if args.model_version else ''} ...")

    rows = fetch_candidates(sb, args.model_version)
    print(f"  -> {len(rows)} baris ditemukan di DB")

    if not rows:
        print("[ERROR] Tidak ada data. Jalankan batch processing dulu, "
              "atau periksa filter --model-version.")
        sys.exit(1)

    n_per_class = max(1, args.n // 3)
    sample = stratified_sample(rows, n_per_class)
    print(f"  -> {len(sample)} baris di-sample (stratified, target {n_per_class}/kelas)")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "raw_text_id", "entity_id", "entity_name",
            "title", "text_preview",
            "predicted_label", "predicted_confidence",
            "score_negative", "score_neutral", "score_positive",
            "model_version",
            "gold_label",   # <-- ISI MANUAL: negative / neutral / positive
            "notes",        # <-- opsional: catatan kasus ambigu/sulit
        ])

        for r in sample:
            entity = r.get("political_entities") or {}
            raw = r.get("raw_texts") or {}
            text = (raw.get("text") or "")[:300]

            writer.writerow([
                r["raw_text_id"],
                r["entity_id"],
                entity.get("canonical_name", "?"),
                (raw.get("title") or "")[:150],
                text,
                r["label"],
                f"{r['confidence']:.3f}",
                f"{r['score_negative']:.3f}",
                f"{r['score_neutral']:.3f}",
                f"{r['score_positive']:.3f}",
                r.get("model_version", "?"),
                "",   # gold_label kosong, diisi manual
                "",   # notes kosong
            ])

    print(f"\nSelesai. File: {args.output}")
    print("LANGKAH SELANJUTNYA:")
    print("  1. Buka file ini di Excel/Google Sheets")
    print("  2. Baca kolom 'title' + 'text_preview', tentukan sentimen SEBENARNYA")
    print("     terhadap 'entity_name' yang disebut")
    print("  3. Isi kolom 'gold_label' dengan: negative / neutral / positive")
    print("  4. Simpan kembali sebagai CSV (format sama)")
    print("  5. Jalankan eval_metrics.py --sentiment <file_ini>")


if __name__ == "__main__":
    main()
