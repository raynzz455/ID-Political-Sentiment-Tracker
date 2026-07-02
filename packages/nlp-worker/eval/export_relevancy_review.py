"""
export_relevancy_review.py
=============================
Export sample untuk evaluasi STAGE 1 (relevancy gate) -- TERPISAH dari
evaluasi sentiment (stage 2). Ini penting karena laporan sebelumnya
(evaluasi Gemini) tidak pernah menunjukkan SATU PUN kasus di mana
relevancy gate benar-benar MENOLAK sesuatu -- jadi belum ada bukti nyata
gate ini bekerja, baru bukti sintetis dari test_sentiment_model.py.

Script ini SENGAJA reapply alias matching dari awal terhadap sample
raw_texts acak (bukan dari sentiment_scores yang sudah ke-filter),
supaya kita lihat SEMUA kandidat -- yang LULUS relevancy maupun yang
GAGAL -- dalam satu file untuk direview manusia.

Usage:
    python export_relevancy_review.py --n 300

Output: relevancy_ground_truth_TEMPLATE.csv
    Kolom gate_decision sudah terisi dari model (relevant/not_relevant).
    Kolom gold_relevant HARUS diisi manual: yes / no
"""

import os
import sys
import csv
import re
import argparse
import random
from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
load_dotenv(ROOT_DIR / ".env")
try:
    from supabase import create_client, Client
except ImportError:
    print("[ERROR] pip install supabase")
    sys.exit(1)

# Reuse pipeline yang sudah dibuat
from sentiment_model import get_pipeline

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SERVICE_KEY  = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
MIN_ALIAS_LEN = 4  # konsisten dengan fix word-boundary sebelumnya


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
    """
    Word-boundary alias matching (versi fixed, bukan substring naive).
    Return SEMUA entity yang alias-nya match -- ini kandidat MENTAH
    sebelum di-filter relevancy gate.
    """
    combined = f"{title or ''} {text or ''}".lower()
    matched = []
    seen_ids = set()

    for e in entities:
        if e["id"] in seen_ids:
            continue
        names = [e["canonical_name"]] + list(e.get("aliases") or [])
        for name in names:
            if len(name) < MIN_ALIAS_LEN:
                continue
            pattern = r'\b' + re.escape(name.lower()) + r'\b'
            if re.search(pattern, combined):
                matched.append(e)
                seen_ids.add(e["id"])
                break

    return matched


def fetch_random_raw_texts(sb: Client, n: int) -> list[dict]:
    """Ambil sample acak dari raw_texts yang punya text cukup panjang."""
    res = sb.table("raw_texts") \
            .select("id, title, text") \
            .not_.is_("text", "null") \
            .limit(2000) \
            .execute()

    rows = [r for r in (res.data or []) if len(r.get("text") or "") >= 20]
    random.shuffle(rows)
    return rows[:n]


def main():
    parser = argparse.ArgumentParser(description="Export ground truth sample untuk relevancy stage")
    parser.add_argument("--n", type=int, default=300,
                         help="Jumlah raw_texts acak untuk di-scan (default 300)")
    parser.add_argument("--max-candidates", type=int, default=250,
                         help="Batas atas baris output (relevancy gate butuh 1 inference/kandidat, "
                              "jaga supaya tidak terlalu lama)")
    parser.add_argument("--output", type=str, default="relevancy_ground_truth_TEMPLATE.csv")
    args = parser.parse_args()

    sb = get_client()
    entities = load_entities(sb)
    print(f"Loaded {len(entities)} entitas aktif")

    raw_texts = fetch_random_raw_texts(sb, args.n)
    print(f"Sampling {len(raw_texts)} raw_texts acak untuk di-scan alias matching ...")

    pipeline = get_pipeline()  # lazy load relevancy model saat dipanggil pertama

    rows_out = []
    for rt in raw_texts:
        candidates = find_alias_candidates(rt.get("title"), rt.get("text"), entities)
        if not candidates:
            continue

        text_for_gate = f"{rt.get('title') or ''} {rt.get('text') or ''}".strip()

        for entity in candidates:
            is_relevant, conf = pipeline.relevancy.check(entity["canonical_name"], text_for_gate)

            rows_out.append({
                "raw_text_id": rt["id"],
                "title": (rt.get("title") or "")[:150],
                "text_preview": (rt.get("text") or "")[:300],
                "entity_candidate": entity["canonical_name"],
                "relevancy_confidence": f"{conf:.4f}",
                "gate_decision": "relevant" if is_relevant else "not_relevant",
                "gold_relevant": "",   # <-- ISI MANUAL: yes / no
                "notes": "",
            })

            if len(rows_out) >= args.max_candidates:
                break
        if len(rows_out) >= args.max_candidates:
            break

    print(f"  -> {len(rows_out)} kandidat (alias-matched) ditemukan untuk direview")

    if not rows_out:
        print("[ERROR] Tidak ada kandidat ditemukan. Cek apakah raw_texts/aliases sudah benar.")
        sys.exit(1)

    # Hitung berapa yang gate tolak vs terima -- info awal, BUKAN pengganti ground truth
    n_relevant = sum(1 for r in rows_out if r["gate_decision"] == "relevant")
    n_not = len(rows_out) - n_relevant
    print(f"  -> Gate decision (BELUM divalidasi manusia): "
          f"{n_relevant} relevant, {n_not} not_relevant")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nSelesai. File: {args.output}")
    print("LANGKAH SELANJUTNYA:")
    print("  1. Buka file ini, baca tiap baris: apakah 'text_preview' BENAR tentang")
    print("     'entity_candidate' yang disebut, atau cuma kebetulan nama mirip")
    print("     (seperti kasus Kapolri 'Listyo Sigit Prabowo' vs Presiden Prabowo)?")
    print("  2. Isi kolom 'gold_relevant': yes / no")
    print("  3. PRIORITASKAN baris dengan gate_decision='not_relevant' -- ini paling")
    print("     penting diverifikasi karena buktikan/bantah klaim gate bekerja")
    print("  4. Jalankan eval_metrics.py --relevancy <file_ini>")


if __name__ == "__main__":
    main()
