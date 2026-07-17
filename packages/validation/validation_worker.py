"""
validation_worker.py v12 — Expert Quality Scoring & Clean Logging
===================================================================
PERUBAAHAN v12:
  1. ADAPTIVE MAX LENGTH: Menaikkan batas MAX_ARTICLE_LENGTH ke 20000 (menerima long-form journalism).
  2. CLEAN LOGGING: Menghapus format dekoratif, menggunakan modul logging terstruktur.
  3. ANTI SECTION LEAKAGE: Menolak teks yang > 20000 karakter (halaman kategori/list).
  4. TITLE MATCH HARD REJECT: Menolak teks jika judul asli tidak cocok sama sekali (< 20% match),
     yang mengindikasikan salah redirect atau salah ekstraksi.
  5. Pure Quality Scoring (0-100). Tidak melakukan routing Headline/NLP.
  6. Tidak menghapus teks snippet GNews. Data tetap utuh untuk training/reprocessing.
  7. Mencatat pipeline_version untuk audit dan observability.
"""

import os
import re
import sys
import time
import random
import argparse
import logging
import datetime  
from collections import Counter
from pathlib import Path
from typing import NamedTuple
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

try:
    from langdetect import detect, LangDetectException, DetectorFactory
except ImportError as e:
    print(f"[ERROR] {e}. Pastikan pip install langdetect")
    sys.exit(1)

# IMPORT DARI MONOREPO SHARED
from packages.shared.db_client import get_client
from packages.shared.logger import start_run, finish_run
from packages.shared import constants as pc

# Setup Clean Logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DetectorFactory.seed = 0  # Fix langdetect non-determinism
PIPELINE_VERSION = "v12_validation"

ID_STOPWORDS = {"yang", "dan", "di", "ke", "untuk", "dengan", "ini", "itu", "atau", "dari", "pada", "juga"}
HARD_REJECT_WINDOW = 200
HARD_REJECT_PATTERNS = ["access denied", "enable javascript and cookies", "checking your browser", "attention required"]
SOFT_PENALTY_PATTERNS = ["captcha", "login", "sign in", "subscribe", "berlangganan", "cookie", "privacy policy", "all rights reserved"]
SOFT_PENALTY_PER_HIT = 8
QUALITY_THRESHOLD = 80

# Batas maksimal adaptif (menerima long-form, memblokir section leakage)
MAX_ARTICLE_LENGTH = 20000 

class QualityResult(NamedTuple):
    score: int
    reason: str

def calculate_quality_score(text: str, title: str) -> QualityResult:
    if not text or not title:
        return QualityResult(0, "empty_input")

    early_window = text[:HARD_REJECT_WINDOW].lower()
    if any(p in early_window for p in HARD_REJECT_PATTERNS):
        return QualityResult(0, "noise_page")

    # 1. CEGAH HALAMAN KATEGORI/TAG (Section Leakage)
    # Teks berita asli (long-form) bisa mencapai 15.000 karakter. 
    # Jika lebih dari 20.000, kemungkinan besar itu halaman list.
    if len(text) > MAX_ARTICLE_LENGTH:
        return QualityResult(0, "rejected_section_page")

    earned = 0
    max_possible = 0
    text_len = len(text)
    words = text.split()
    word_count = len(words)
    has_id_stopword = any(w in ID_STOPWORDS for w in words)

    max_possible += 25
    if text_len >= 1000: earned += 25
    elif text_len >= 500: earned += 15
    elif text_len >= 300: earned += 10

    max_possible += 25
    if word_count >= 150: earned += 25
    elif word_count >= 70: earned += 15
    elif word_count >= 40: earned += 5

    if word_count > 0:
        max_possible += 25
        if has_id_stopword: earned += 15
        try:
            if detect(text[:500]) == "id": earned += 10
        except LangDetectException:
            pass

    # 2. PERKUAT TITLE MATCH
    title_words = set(re.findall(r"\b\w+\b", title.lower())) - ID_STOPWORDS
    if title_words:
        max_possible += 25
        text_words = set(re.findall(r"\b\w+\b", text.lower()))
        match_ratio = sum(1 for w in title_words if w in text_words) / len(title_words)
        earned += int(match_ratio * 25)
        
        # Hard reject jika judul asli nyaris tidak cocok dengan teks (misal < 20% match)
        # Ini menangkap kasus salah redirect (misal ke halaman utama)
        if match_ratio < 0.2:
            return QualityResult(0, "rejected_title_mismatch")

    lower_full_text = text.lower()
    hits = sum(1 for p in SOFT_PENALTY_PATTERNS if p in lower_full_text)
    earned = max(0, earned - hits * SOFT_PENALTY_PER_HIT)

    score = int((earned / max_possible) * 100) if max_possible > 0 else 0
    score = min(score, 100)

    if score >= QUALITY_THRESHOLD:
        return QualityResult(score, "validated")
    if word_count < 70:
        return QualityResult(score, "low_quality_too_short")
    if not has_id_stopword:
        return QualityResult(score, "low_quality_no_stopword")
    return QualityResult(score, "low_quality")

def process_batch(sb, rows: list) -> Counter:
    stats = Counter()
    updates = []

    for r in rows:
        result = calculate_quality_score(r.get("text") or "", r.get("title") or "")
        current_metadata = dict(r.get("metadata") or {})

        if result.reason == "validated":
            if "content_type" not in current_metadata:
                current_metadata["content_type"] = "SNIPPET" if current_metadata.get("is_snippet") else "FULLTEXT"
            
            updates.append({
                "id": r["id"], 
                "status": pc.STATUS_VALIDATED, 
                "metadata": current_metadata,
                "pipeline_version": PIPELINE_VERSION 
            })
            stats["validated"] += 1
        else:
            current_metadata["fail_reason"] = result.reason
            current_metadata["quality_score"] = result.score
            updates.append({
                "id": r["id"], 
                "status": pc.STATUS_FAILED, 
                "metadata": current_metadata,
                "pipeline_version": PIPELINE_VERSION
            })
            stats["failed"] += 1
            stats[f"reason_{result.reason}"] += 1

    # --- PERBAIKAN: CHUNKED RPC CALL ---
    if updates:
        CHUNK_SIZE = 50
        try:
            for i in range(0, len(updates), CHUNK_SIZE):
                chunk = updates[i:i + CHUNK_SIZE]
                sb.rpc("bulk_update_raw_texts", {"p_updates": chunk}).execute()
        except Exception as e: 
            logger.error(f"DB Bulk Update Error: {e}")

    return stats

def print_batch_report(batch_num: int, stats: Counter):
    logger.info(f"--- BATCH {batch_num} REPORT ---")
    logger.info(f"  Validated : {stats.get('validated', 0)}")
    logger.info(f"  Failed    : {stats.get('failed', 0)}")
    
    reasons = {k: v for k, v in stats.items() if k.startswith("reason_") and v > 0}
    if reasons:
        for key, count in Counter(reasons).most_common():
            logger.info(f"  - {key.replace('reason_', ''):25s}: {count}")

def main(limit: int = 100, max_total: int = 0):
    sb = get_client()
    run_id = start_run("validation_worker", PIPELINE_VERSION)
    
    total_stats = Counter()
    batch_num = 1

    logger.info(f"[VALIDATOR v12] Limit: {limit}/batch | Max: {'Unlimited' if max_total == 0 else max_total}")

    while True:
        if max_total > 0 and sum(v for k, v in total_stats.items() if not k.startswith("reason_")) >= max_total:
            break

        try:
            time_filter = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)).isoformat()
            
            res = sb.table("raw_texts") \
                    .select("id, title, text, metadata") \
                    .eq("status", pc.STATUS_ENRICHED) \
                    .gte("ingested_at", time_filter) \
                    .limit(limit) \
                    .execute()
        except Exception as e:
            logger.warning(f"DB Query Timeout/Error: {e}. Menunggu 10 detik sebelum retry...")
            time.sleep(10)
            continue

        rows = res.data or []
        if not rows:
            logger.info("Tidak ada lagi artikel untuk divalidasi.")
            break

        logger.info(f"Scoring {len(rows)} artikel...")
        batch_stats = process_batch(sb, rows)
        print_batch_report(batch_num, batch_stats)
        total_stats.update(batch_stats)
        batch_num += 1
        time.sleep(2 + random.uniform(0, 2))

    total_processed = sum(v for k, v in total_stats.items() if not k.startswith("reason_"))
    total_succeeded = total_stats.get('validated', 0)
    total_failed = total_stats.get('failed', 0)
    
    finish_run(run_id, processed=total_processed, succeeded=total_succeeded, failed=total_failed)
    logger.info("Eksekusi Validation Selesai.")