"""
enricher_worker.py — Layer 2.5 (Enrichment Worker)
====================================================
Tugas: Memisahkan Network I/O (fetch URL) dari NLP Worker (AI Inference).

Cara Kerja:
  1. Ambil artikel di raw_texts dengan status='pending' & panjang teks < 500.
  2. Fetch URL aslinya menggunakan requests + User-Agent Chrome.
  3. Ekstrak full body menggunakan trafilatura.
  4. UPDATE raw_texts: Isi teks utuh & ubah status='enriched'.
  5. Jika URL mati (404) / gagal extract, ubah status='dead_link' agar tidak membebani antrian.
"""
"""
enricher_worker.py v6 — Pure Network I/O
========================================
Tugas: Mengubah teks pendek menjadi full body article.
TIDAK ADA pre-filtering relevansi (biarkan NLP Worker yang menilai).
Aman dari Unique Constraint (tidak update text_hash).

Cara Jalankan:
  python enricher_worker.py (Mode backfill lokal)
  python enricher_worker.py --max-total 500 (Mode cloud/cron)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

try:
    import requests
    from trafilatura import extract as traf_extract
    from supabase import create_client, Client
except ImportError:
    print("[ERROR] pip install requests trafilatura supabase python-dotenv")
    sys.exit(1)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SERVICE_KEY  = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

def get_client() -> Client:
    return create_client(SUPABASE_URL, SERVICE_KEY)

def fetch_full_body(url: str) -> str:
    if not url: return ""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        if resp.ok:
            return traf_extract(resp.text, include_comments=False, include_tables=False) or ""
    except Exception:
        return ""
    return ""

def process_batch(sb: Client, rows: list):
    enriched_count = 0
    dead_count = 0
    
    to_fetch = []

    # 1. CEK TEKS DI DB DULU
    for r in rows:
        current_text = r.get("text") or ""
        
        # Jika ternyata RSS sudah ngasih teks utuh, langsung enriched tanpa fetch
        if len(current_text) >= 500:
            sb.table("raw_texts").update({"status": "enriched"}).eq("id", r["id"]).execute()
            enriched_count += 1
        else:
            # Kalau pendek, masukkan antrian untuk di-fetch
            to_fetch.append(r)

    if not to_fetch:
        return enriched_count, dead_count

    # 2. PARALLEL FETCH (Hanya yang teksnya pendek)
    print(f"  [PARALLEL] Fetching {len(to_fetch)} URLs dengan 10 threads...")
    results = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch_full_body, r["source_url"]): r for r in to_fetch}
        for future in as_completed(futures):
            row = futures[future]
            try:
                results[row["id"]] = future.result()
            except Exception:
                results[row["id"]] = ""

    # 3. DB UPDATE
    print(f"  [DB] Updating database...")
    for r in to_fetch:
        rt_id = r["id"]
        full_text = results.get(rt_id, "")
        
        if len(full_text) > 500:
            sb.table("raw_texts").update({
                "text": full_text,
                "status": "enriched"
            }).eq("id", rt_id).execute()
            enriched_count += 1
        else:
            # URL mati / gagal extract
            sb.table("raw_texts").update({"status": "dead_link"}).eq("id", rt_id).execute()
            dead_count += 1
            
    return enriched_count, dead_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200, help="Jumlah row per batch API (maks 500)")
    parser.add_argument("--max-total", type=int, default=0, help="Batasi total proses per run (0=unlimited)")
    args = parser.parse_args()

    sb = get_client()
    
    total_enriched = 0
    total_dead = 0
    batch_num = 1

    print(f"[ENRICHER] Limit per batch: {args.limit} | Max Total: {'Unlimited' if args.max_total == 0 else args.max_total}")
    
    while True:
        if args.max_total > 0 and (total_enriched + total_dead) >= args.max_total:
            print(f"\n[STOP] Mencapai batas --max-total ({args.max_total} artikel). Berhenti.")
            break

        print(f"\n--- Batch {batch_num} ---")
        res = sb.table("raw_texts") \
                .select("id, source_url, text") \
                .eq("status", "pending") \
                .limit(args.limit) \
                .execute()
                
        rows = res.data or []
        
        if not rows:
            print("[ENRICHER] Semua artikel pending sudah habis diproses!")
            break

        print(f"[ENRICHER] Memproses {len(rows)} artikel...")
        
        enr, dead = process_batch(sb, rows)
        total_enriched += enr
        total_dead += dead
        
        print(f"  -> Total Sementara: Enriched={total_enriched} | Dead={total_dead}")
        
        print("  [PAUSE] Jeda 5 detik untuk amankan API rate-limit...")
        time.sleep(5) # BUG FIXED: time module standar
        batch_num += 1

    print(f"\n{'='*50}")
    print(f"SELESAI TOTAL KESELURUHAN. Enriched: {total_enriched} | Dead: {total_dead}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()