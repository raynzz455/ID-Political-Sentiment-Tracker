"""
audit_fulltext_quality.py v2 — Local Tracker & Entity Check
=================================================================
Fungsi:
  1. READ-ONLY: Tidak mengubah data di database Supabase sama sekali.
  2. ENTITY CHECK: Mengecek apakah teks mengandung konteks tokoh politik.
  3. QUALITY AUDIT: Mengecek Section Leakage, Boilerplate Spam, & Title Causality.
  4. LOCAL TRACKER: Menyimpan & memperbarui hasil audit di file audit_results.csv.
"""
import re
import csv
import sys
import logging
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from packages.shared.db_client import get_client

# Setup Clean Logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Config
MAX_ARTICLE_LENGTH = 20000  
BOILERPLATE_THRESHOLD = 5
OUTPUT_CSV = ROOT_DIR / "devtools" / "audit_results.csv"

def load_existing_results() -> dict:
    """Memuat file CSV lokal agar bisa di-update, bukan ditimpa."""
    results = {}
    if OUTPUT_CSV.exists():
        try:
            with open(OUTPUT_CSV, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    results[row["id"]] = row
        except Exception as e:
            logger.warning(f"Gagal memuat CSV lama, akan membuat baru: {e}")
    return results

def save_results(results: dict):
    """Menyimpan dictionary hasil audit ke CSV lokal."""
    if not results: return
    fieldnames = ["id", "title", "text_length", "has_entity", "matched_entities", "is_valid", "reason", "audited_at"]
    with open(OUTPUT_CSV, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results.values():
            writer.writerow(row)
    logger.info(f"Hasil audit disimpan di: {OUTPUT_CSV}")

def get_entity_terms(sb) -> set:
    """Mengambil semua nama tokoh & alias dari database untuk dicek di teks."""
    terms = set()
    try:
        res = sb.table("political_entities").select("canonical_name, aliases").execute()
        for row in (res.data or []):
            if row.get("canonical_name"):
                terms.add(row["canonical_name"].lower())
            if row.get("aliases"):
                for alias in row["aliases"]:
                    terms.add(alias.lower())
    except Exception as e:
        logger.warning(f"Gagal mengambil entity dari DB: {e}")
    return terms

def check_entity_context(text: str, entity_terms: set) -> tuple[bool, str]:
    """Mencari apakah ada nama tokoh di dalam teks."""
    if not text or not entity_terms: return False, ""
    text_lower = text.lower()
    matched = [term for term in entity_terms if term in text_lower]
    return len(matched) > 0, "; ".join(matched[:3]) # Maksimal 3 tokoh untuk hemat space CSV

def check_title_causality(title: str, text: str) -> float:
    if not title or not text: return 0.0
    title_lower = title.lower()
    text_lower = text[:2000].lower()
    
    title_words = re.findall(r'\b\w+\b', title_lower)
    title_bigrams = set()
    for i in range(len(title_words) - 1):
        title_bigrams.add(f"{title_words[i]} {title_words[i+1]}")
    
    if not title_bigrams: return 0.0
    match_count = sum(1 for bg in title_bigrams if bg in text_lower)
    
    if match_count == 0:
        single_words = set(title_words) - {"yang", "dan", "di", "ke", "untuk", "dengan", "ini", "itu", "atau"}
        if not single_words: return 0.0
        single_match = sum(1 for w in single_words if w in text_lower)
        return (single_match / len(single_words)) * 0.5
    
    return match_count / len(title_bigrams)

def audit_text_quality(title: str, text: str) -> tuple[bool, str]:
    if not text: return False, "empty_text"
    if len(text) > MAX_ARTICLE_LENGTH: return False, "section_leakage_too_long"
        
    boilerplate_hits = len(re.findall(r'(Baca Juga|Simak Juga|Berita Terkait|Advertisement|Iklan)', text, flags=re.IGNORECASE))
    if boilerplate_hits > BOILERPLATE_THRESHOLD:
        return False, "boilerplate_spam"
        
    causality_score = check_title_causality(title, text)
    if causality_score < 0.15:
        return False, "title_causality_mismatch"
        
    return True, "valid"

def main():
    sb = get_client()
    from datetime import datetime
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info("Memulai Audit Kualitas FULLTEXT (Read-Only Mode)...")
    
    # 1. Cek total artikel FULLTEXT di database
    try:
        count_res = sb.table("raw_texts").select("id", count="exact").eq("content_type", "FULLTEXT").limit(1).execute()
        total_db = count_res.count or 0
    except Exception as e:
        logger.error(f"Gagal mengambil total count: {e}")
        total_db = 0
        
    logger.info(f"Total artikel FULLTEXT di DB: {total_db}")
    
    logger.info("Mengambil daftar tokoh dari database...")
    entity_terms = get_entity_terms(sb)
    logger.info(f"Ditemukan {len(entity_terms)} istilah tokoh untuk dicek.")
    
    local_results = load_existing_results()
    logger.info(f"Memuat {len(local_results)} hasil audit sebelumnya dari lokal.")
    
    total_audited = 0
    total_garbage = 0
    total_no_entity = 0
    offset = 0
    batch_size = 100
    
    try:
        while True:
            progress_pct = (total_audited / total_db * 100) if total_db > 0 else 0
            logger.info(f"--- Mengambil Batch (Offset: {offset}) | Progress: {total_audited}/{total_db} ({progress_pct:.1f}%) ---")
            
            res = sb.table("raw_texts") \
                    .select("id, title, text, metadata, ingested_at") \
                    .eq("content_type", "FULLTEXT") \
                    .order("ingested_at", desc=False) \
                    .range(offset, offset + batch_size - 1) \
                    .execute()
                    
            rows = res.data or []
            if not rows:
                logger.info("Tidak ada lagi FULLTEXT untuk diaudit.")
                break
                
            for r in rows:
                art_id = r["id"]
                title = r.get("title") or ""
                text = r.get("text") or ""
                
                # 1. Cek Konteks Tokoh
                has_entity, matched_str = check_entity_context(text, entity_terms)
                if not has_entity:
                    total_no_entity += 1
                    
                # 2. Cek Kualitas Teks
                is_valid, reason = audit_text_quality(title, text)
                
                if not is_valid:
                    total_garbage += 1
                    logger.info(f"  [GARBAGE] ID: {art_id[:8]} | Reason: {reason} | Entity: {matched_str or 'None'} | Title: {title[:30]}")
                elif not has_entity:
                    logger.info(f"  [NO ENTITY] ID: {art_id[:8]} | Teks valid tapi tidak menyebut tokoh politik.")
                else:
                    logger.info(f"  [VALID]    ID: {art_id[:8]} | Len: {len(text)} | Entity: {matched_str}")
                    
                # 3. Simpan/Update ke dictionary lokal
                local_results[art_id] = {
                    "id": art_id,
                    "title": title[:50],
                    "text_length": len(text),
                    "has_entity": has_entity,
                    "matched_entities": matched_str,
                    "is_valid": is_valid,
                    "reason": reason if not is_valid else "valid",
                    "audited_at": now_str
                }
                
            total_audited += len(rows)
            offset += batch_size
            
    except KeyboardInterrupt:
        logger.warning("Audit dihentikan secara manual (Ctrl+C). Menyimpan hasil yang sudah ada...")
    finally:
        # FIX AUTO-SAVE: Simpan hasil ke CSV baik saat selesai maupun saat di-stop paksa
        save_results(local_results)
        
    logger.info("=" * 50)
    logger.info("Audit Selesai.")
    logger.info(f"  Total Diaudit Sesi Ini : {total_audited}")
    logger.info(f"  Sampah Ditemukan       : {total_garbage}")
    logger.info(f"  Valid Tanpa Tokoh      : {total_no_entity}")
    logger.info(f"  Total Data di CSV      : {len(local_results)}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()