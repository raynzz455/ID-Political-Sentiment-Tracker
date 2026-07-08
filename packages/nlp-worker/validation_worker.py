"""
validation_worker.py v2 — Layer 2.6 (Quality Gate & Observability)
==================================================================
Modular validation dengan alasan kegagalan yang spesifik.
Mencegah infinite loop dengan menggunakan status eksplisit (bukan kembali ke enriched).

Status yang dikembalikan ke DB:
- 'validated' (Lolos)
- 'too_short', 'low_alpha', 'too_few_words', 'boilerplate'
- 'no_stopwords', 'bad_language', 'low_similarity', 'noise_page'
"""

import os
import re
import sys
import argparse
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

try:
    from supabase import create_client, Client
    from langdetect import detect, LangDetectException
except ImportError as e:
    print(f"[ERROR] Dependency missing: {e}. Pastikan pip install langdetect supabase")
    sys.exit(1)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SERVICE_KEY  = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

# Konfigurasi
MIN_ARTICLE_LENGTH = 300
MIN_WORD_COUNT = 70
MIN_UNIQUE_RATIO = 0.40
MIN_ALPHA_RATIO = 0.60

ID_STOPWORDS = {'yang', 'dan', 'di', 'ke', 'untuk', 'dengan', 'ini', 'itu', 'atau', 'dari', 'pada', 'juga'}
BAD_PATTERNS = ['login', 'sign in', 'captcha', 'subscribe', 'cookie', 'privacy policy', 'access denied']

def get_client() -> Client:
    return create_client(SUPABASE_URL, SERVICE_KEY)

# ─────────────────────────────────────────────────────────────
# MODULAR VALIDATION RULES
# ─────────────────────────────────────────────────────────────

def check_noise(text: str) -> str | None:
    lower_text = text[:1000].lower()
    for pattern in BAD_PATTERNS:
        if pattern in lower_text:
            return "noise_page"
    return None

def check_length(text: str) -> str | None:
    if len(text) < MIN_ARTICLE_LENGTH:
        return "too_short"
    return None

def check_alpha_ratio(text: str) -> str | None:
    alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
    if alpha_ratio < MIN_ALPHA_RATIO:
        return "low_alpha"
    return None

def check_word_count(text: str) -> str | None:
    words = text.split()
    if len(words) < MIN_WORD_COUNT:
        return "too_few_words"
    return None

def check_boilerplate(text: str) -> str | None:
    words = text.split()
    if len(words) == 0: return "empty"
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < MIN_UNIQUE_RATIO:
        return "boilerplate"
    return None

def check_stopwords(text: str) -> str | None:
    words = text.lower().split()
    if not any(w in ID_STOPWORDS for w in words):
        return "no_stopwords"
    return None

def check_language(text: str) -> str | None:
    try:
        lang = detect(text[:500]) # Cek bahasa dari 500 char pertama (hemat CPU)
        if lang != 'id':
            return "bad_language"
    except LangDetectException:
        return "language_error"
    return None

def check_title_similarity(title: str, text: str) -> str | None:
    # Pakai intersection kata (Jaccard sederhana)
    title_words = set(re.findall(r'\b\w+\b', title.lower()))
    text_words = set(re.findall(r'\b\w+\b', text.lower()))
    
    title_keywords = title_words - ID_STOPWORDS
    if not title_keywords: return None # Skip kalau judul cuma stopword
    
    match_count = sum(1 for w in title_keywords if w in text_words)
    if match_count < 2: # Minimal 2 keyword dari judul muncul di teks
        return "low_similarity"
    return None

def validate_article(text: str, title: str) -> str:
    """Orchestrator: Jalankan semua rule, return alasan gagal atau 'valid'."""
    if not text or not title:
        return "missing_data"

    # Urutkan dari cek paling ringan ke paling berat
    checks = [
        check_noise,
        check_length,
        check_alpha_ratio,
        check_word_count,
        check_boilerplate,
        check_stopwords,
        check_language,
        lambda t: check_title_similarity(title, t)
    ]

    for check in checks:
        result = check(text)
        if result:
            return result # Langsung return alasan kegagalan

    return "valid"

# ─────────────────────────────────────────────────────────────
# MAIN WORKER
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    sb = get_client()
    print(f"[VALIDATOR v2] Mencari {args.limit} artikel 'enriched'...")

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
    stats = Counter()

    for r in rows:
        validation_status = validate_article(r.get("text") or "", r.get("title") or "")
        
        if validation_status == "valid":
            updates.append({"id": r["id"], "text": "", "status": "validated"})
            stats["valid"] += 1
        else:
            # Masukkan alasan kegagalan sebagai status eksplisit
            updates.append({"id": r["id"], "text": "", "status": validation_status})
            stats[validation_status] += 1

    # Bulk Update DB
    if updates:
        try:
            sb.rpc("bulk_update_raw_texts", {"p_updates": updates}).execute()
        except Exception as e:
            print(f"[BULK_DB_ERROR] {e}")

    # Print Observability Report
    print(f"\n{'='*50}")
    print(f"VALIDATION REPORT")
    print(f"{'='*50}")
    print(f"Valid (Lolos)      : {stats.get('valid', 0)}")
    print(f"Failed (Dibuang)   : {sum(v for k, v in stats.items() if k != 'valid')}")
    print(f"{'-'*50}")
    print(f"Alasan Kegagalan:")
    for reason, count in stats.most_common():
        if reason != "valid" and count > 0:
            print(f"  - {reason:20s}: {count}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()