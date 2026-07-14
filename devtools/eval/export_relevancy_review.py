"""
export_relevancy_review.py
=============================
Export sample untuk evaluasi STAGE 1 (relevancy gate) -- TERPISAH dari
evaluasi sentiment (stage 2).

FIX v2:
  1. LAPIS 2 INTEGRATION: Re-scan model TIDAK boleh pakai snippet RSS.
     Script sekarang WAJIB memanggil enrich_if_needed (trafilatura) supaya
     evaluasi gate 100% merepresentasikan kondisi production.
  2. CONTEXT EXPORT: Mengambil source_url & era_hint ke CSV. Evaluator
     (manusia) wajib membaca URL aslinya untuk menilai relevansi.
  3. API CONSISTENCY: Pakai pipeline.predict_gated() agar konsisten dengan
     drain_queue.py dan cli_test.py.

Usage:
    python export_relevancy_review.py --n 300

Output: relevancy_ground_truth_TEMPLATE.csv
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

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sentiment_model import get_pipeline
try:
    from supabase import create_client, Client
except ImportError:
    print("[ERROR] pip install supabase")
    sys.exit(1)

# Import dependency Lapis 2
try:
    import requests
    from trafilatura import extract as traf_extract
    FETCH_AVAILABLE = True
except ImportError:
    FETCH_AVAILABLE = False
    print("[WARNING] requests/trafilatura belum terinstall. Enrichment dimatikan.")

from sentiment_model import get_pipeline

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SERVICE_KEY  = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
MIN_ALIAS_LEN = 4

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
    matched, seen_ids = [], set()

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
def strip_html(text: str) -> str:
    """Hapus tag HTML, entity, dan newline berlebih yang bikin CSV pecah."""
    if not text:
        return ""
    import re
    # Hapus tag HTML seperti <p>, <br>, <div>
    text = re.sub(r'<[^>]+>', '', text)
    # Hapus HTML entities seperti &amp; &nbsp;
    text = re.sub(r'&[a-z]+;', ' ', text)
    # Ganti semua newline/tab dengan spasi tunggal (PENTING agar CSV tidak pecah baris)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def fetch_full_body(url: str, timeout: int = 15) -> str:
    if not FETCH_AVAILABLE or not url:
        return ""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        if not resp.ok:
            return ""
        return traf_extract(resp.text, include_comments=False, include_tables=False) or ""
    except Exception:
        return ""

def enrich_text(item: dict, min_len: int = 500) -> str:
    """Lapis 2: Jika teks < 500 char, fetch URL aslinya."""
    text = strip_html((item.get("text") or "").strip())
    title = strip_html((item.get("title") or "").strip())
    
    if len(text) >= min_len:
        return f"{title} {text}".strip()
        
    source_url = item.get("source_url") or ""
    if source_url:
        full = fetch_full_body(source_url)
        full = strip_html(full)  # PASTIKAN FULL BODY JUGA DI BERSIHKAN
        if len(full) >= min_len:
            return f"{title} {full}".strip()
            
    return f"{title} {text}".strip()

def fetch_random_raw_texts(sb: Client, n: int) -> list[dict]:
    """Ambil sample acak, pastikan bawa source_url dan metadata."""
    res = sb.table("raw_texts") \
            .select("id, title, text, source_url, metadata, published_at") \
            .not_.is_("text", "null") \
            .limit(2000) \
            .execute()
            
    rows = [r for r in (res.data or []) if len(r.get("text") or "") >= 20]
    random.shuffle(rows)
    return rows[:n]

def main():
    parser = argparse.ArgumentParser(description="Export ground truth sample untuk relevancy stage v2")
    parser.add_argument("--n", type=int, default=50,
                         help="Jumlah raw_texts acak untuk di-scan (default 50, naikkan jika perlu)")
    parser.add_argument("--max-candidates", type=int, default=50,
                         help="Batas atas baris output")
    parser.add_argument("--output", type=str, default="relevancy_ground_truth_TEMPLATE.csv")
    args = parser.parse_args()

    sb = get_client()
    entities = load_entities(sb)
    print(f"Loaded {len(entities)} entitas aktif")

    raw_texts = fetch_random_raw_texts(sb, args.n)
    print(f"Sampling {len(raw_texts)} raw_texts acak...")

    pipeline = get_pipeline()

    rows_out = []
    for i, rt in enumerate(raw_texts, 1):
        # 1. Cek dulu nama tokoh di teks mentah (snippet)
        candidates = find_alias_candidates(rt.get("title"), rt.get("text"), entities)
        if not candidates:
            continue

        print(f"  [{i}/{len(raw_texts)}] Enriching text for {len(candidates)} candidates...")
        
        # 2. ENRICHMENT WAJIB: Ambil full body sebelum dikirim ke model
        enriched_text = enrich_text(rt)
        
        # 3. Ekstrak metadata era_hint
               # 3. Ekstrak era_hint dari published_at (akurat untuk RSS & GDELT)
        pub_date = rt.get("published_at") or ""
        era_hint = "Tidak diketahui"
        
        if pub_date and len(pub_date) >= 4:
            # Format ISO: "2018-12-31T02:30:00+00:00" -> ambil 4 digit pertama
            era_hint = f"Tahun {pub_date[:4]}"
        else:
            # Fallback ke metadata GDELT jika published_at null
            metadata = rt.get("metadata") or {}
            gdelt_hint = metadata.get("gdelt_era_hint", "")
            if gdelt_hint:
                era_hint = gdelt_hint.replace("y", "Tahun ")

        # 4. Tembak model dengan teks yang sudah di-enrich
        for entity in candidates:
            result = pipeline.predict_gated(text=enriched_text, context=entity["canonical_name"])
            
            rows_out.append({
                "raw_text_id": rt["id"],
                "title": (rt.get("title") or "")[:150],
                "enriched_text_preview": enriched_text[:300], # Tampilkan 300 char dari full text
                "source_url": rt.get("source_url") or "",     # Wajib ada untuk evaluasi manual
                "era_hint": era_hint,                          # Konteks tahun/jabatan
                "entity_candidate": entity["canonical_name"],
                "relevancy_confidence": f"{result.relevancy_confidence:.4f}",
                "gate_decision": "relevant" if result.is_relevant else "not_relevant",
                "gold_relevant": "",   # <-- ISI MANUAL: yes / no
                "notes": "",
            })

            if len(rows_out) >= args.max_candidates:
                break
        if len(rows_out) >= args.max_candidates:
            break

    print(f"\n  -> {len(rows_out)} kandidat (alias-matched) siap dievaluasi")

    if not rows_out:
        print("[ERROR] Tidak ada kandidat ditemukan.")
        sys.exit(1)

    n_relevant = sum(1 for r in rows_out if r["gate_decision"] == "relevant")
    n_not = len(rows_out) - n_relevant
    print(f"  -> Gate decision (BELUM divalidasi manusia): {n_relevant} relevant, {n_not} not_relevant")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nSelesai. File: {args.output}")
    print("LANGKAH SELANJUTNYA:")
    print("  1. Buka file CSV, klik kolom 'source_url' untuk baca artikel aslinya.")
    print("  2. Cocokkan 'entity_candidate' dengan konteks artikel (perhatikan 'era_hint').")
    print("  3. Isi 'gold_relevant': yes (relevan) atau no (tidak relevan).")
    print("  4. Jalankan: python eval_metrics.py --relevancy relevancy_ground_truth_TEMPLATE.csv")

if __name__ == "__main__":
    main()