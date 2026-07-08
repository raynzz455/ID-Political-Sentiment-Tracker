"""
validation_worker.py — Layer 2.6 (Quality Gate)
================================================
Mengeksekusi logika validate_article() secara terpisah.
Tugas: Ambil status='enriched' -> Cek Kualitas -> Update ke 'validated' atau 'validation_failed'.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

try:
    from supabase import create_client, Client
except ImportError as e:
    print(f"[ERROR] {e}"); sys.exit(1)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SERVICE_KEY  = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

ID_STOPWORDS = {'yang', 'dan', 'di', 'ke', 'untuk', 'dengan', 'ini', 'itu', 'atau', 'dari', 'pada', 'juga'}

def get_client() -> Client:
    return create_client(SUPABASE_URL, SERVICE_KEY)

def validate_article(text: str, title: str) -> bool:
    """Advanced Quality Gate."""
    if not text or not title: return False
    text_len = len(text)
    if text_len < 300: return False
    
    alpha_ratio = sum(1 for c in text if c.isalpha()) / text_len
    if alpha_ratio < 0.60: return False
    
    ws_ratio = text.count(' ') / text_len
    if ws_ratio < 0.10 or ws_ratio > 0.25: return False
    
    paragraphs = [p for p in text.split('\n') if len(p.strip()) > 20]
    if len(paragraphs) < 2: return False
    
    words = text.lower().split()
    if not any(w in ID_STOPWORDS for w in words): return False
    
    title_snippet = title[:20].lower().strip()
    if title_snippet:
        title_words = title_snippet.split()
        if len(title_words) >= 3:
            if not all(word in text[:500].lower() for word in title_words[:3]):
                return False
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    sb = get_client()
    print(f"[VALIDATOR] Mencari {args.limit} artikel 'enriched'...")

    res = sb.table("raw_texts") \
            .select("id, title, text") \
            .eq("status", "enriched") \
            .limit(args.limit) \
            .execute()

    rows = res.data or []
    if not rows:
        print("[VALIDATOR] Tidak ada artikel untuk divalidasi.")
        return

    updates = []
    valid_count = 0
    invalid_count = 0

    for r in rows:
        is_valid = validate_article(r.get("text") or "", r.get("title") or "")
        
        if is_valid:
            updates.append({"id": r["id"], "text": "", "status": "validated"})
            valid_count += 1
        else:
            updates.append({"id": r["id"], "text": "", "status": "validation_failed"})
            invalid_count += 1

    # Bulk Update DB
    if updates:
        try:
            # Catatan: text="" agar SQL function (COALESCE) tidak menimpa teks lama
            sb.rpc("bulk_update_raw_texts", {"p_updates": updates}).execute()
        except Exception as e:
            print(f"[BULK_DB_ERROR] {e}")

    print(f"{'='*50}")
    print(f"SELESAI. Valid: {valid_count} | Invalid: {invalid_count}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()