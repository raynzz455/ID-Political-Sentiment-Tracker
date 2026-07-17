"""
audit_worker_failures.py — Evidence-Based Sampling Audit
==========================================================
Mengambil sampel acak dari berbagai kategori kegagalan (fail_reason)
untuk di-read secara manual. Tujuannya: Membuktikan apakah Enricher/Validation
salah, atau memang data sumbernya yang jelek.

Usage:
    python -m devtools.audit_worker_failures
"""
import os
import sys
import csv
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

try:
    from supabase import create_client
except ImportError:
    print("[ERROR] pip install supabase"); sys.exit(1)

def main():
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        print("[ERROR] Set SUPABASE_URL & SUPABASE_SERVICE_ROLE_KEY")
        sys.exit(1)
        
    sb = create_client(url, key)
    
    # Kategori yang ingin di-audit
    categories = [
        "extract_too_short",
        "low_quality_too_short",
        "low_quality_no_stopword",
        "low_quality",
        "max_retries_exceeded"
    ]
    
    sample_size = 10
    audit_results = []
    
    print(f"\n{'='*70}")
    print(f"🔍 STARTING AUDIT SAMPLING (Sample size: {sample_size} per category)")
    print(f"{'='*70}\n")
    
    for cat in categories:
        print(f"--- Category: {cat} ---")
        
        # Query Supabase: Cari yang gagal karena alasan ini, ambil acak
        # Kita filter yang BUKAN GNews agar fokus ke URL asli
        res = sb.table("raw_texts") \
                .select("id, source_url, title, text") \
                .eq("status", "failed") \
                .eq("metadata->>fail_reason", cat) \
                .not_.like("source_url", "%news.google.com%") \
                .limit(sample_size * 2) \
                .execute()
                
        rows = res.data or []
        if not rows:
            print("  (Tidak ada data untuk kategori ini)\n")
            continue
            
        # Ambil maksimal sample_size
        samples = rows[:sample_size]
        
        for i, s in enumerate(samples, 1):
            text_preview = (s.get("text") or "")[:300].replace("\n", " ")
            print(f"  [{i}] URL: {s['source_url'][:80]}")
            print(f"      Title: {s['title'][:80]}")
            print(f"      Text: {text_preview}...")
            print(f"      Length: {len(s.get('text') or '')} chars\n")
            
            # Simpan untuk laporan CSV
            audit_results.append({
                "category": cat,
                "id": s["id"],
                "url": s["source_url"],
                "title": s["title"],
                "text_length": len(s.get("text") or ""),
                "text_preview": text_preview
            })

    # Ekspor ke CSV agar mudah dibaca di Excel/Notepad
    csv_file = "audit_failures_report.csv"
    if audit_results:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=audit_results[0].keys())
            writer.writeheader()
            writer.writerows(audit_results)
        print(f"{'='*70}")
        print(f"✅ Audit selesai. Laporan lengkap disimpan di: {csv_file}")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    main()