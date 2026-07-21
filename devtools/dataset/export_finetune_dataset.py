"""
export_finetune_dataset.py v7 — Excel Export & Full Article Text
=====================================================================
FIX v7:
  1. EXCEL EXPORT: Mengganti output dari CSV ke XLSX (Pandas + Openpyxl).
     Header otomatis Bold, kolom di-auto-fit, dan teks ter-wrap (mudah dibaca).
  2. FULL ARTICLE TEXT: Menambahkan kolom 'article_text' (isi asli hasil enricher)
     agar labeler manusia bisa membandingkan konteks pendek dengan artikel utuh.
  3. DEDUPLICATION FIX: Menggunakan (content_hash, entity_id) sebagai kunci duplikat.
  4. AUDIT REPORT: Mencetak statistik di setiap tahap filter.
"""

import sys
import math
import re
import logging
import argparse
import random
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")
sys.path.append(str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from packages.shared.db_client import get_client

# Import pandas dan openpyxl untuk menulis Excel
try:
    import pandas as pd
    from openpyxl.styles import Font, Alignment
except ImportError:
    logger.error("Dependency missing: pip install pandas openpyxl")
    sys.exit(1)

# Ubah ekstensi output menjadi .xlsx
OUTPUT_XLSX = ROOT_DIR / "devtools" / "dataset" / "finetune_dataset_V2.xlsx"

# Konfigurasi
MAX_PER_MEDIA = 500
MAX_PER_ENTITY = 300
MIN_ARTICLE_LEN = 300
MIN_CONTEXT_LEN = 50
MAX_BOILERPLATE_RATIO = 0.20

BOILERPLATE_RE = re.compile(r'(Baca Juga|Simak Juga|Berita Terkait|Advertisement|Ikuti Kami|Copyright|©|Reportase:|Jurnalis:|Editor:).*?(?=\n|$)', re.IGNORECASE)

def calculate_entropy(scores: list) -> float:
    try:
        return -sum(p * math.log(p) for p in scores if p > 1e-9)
    except:
        return 1.0

def main(limit: int = 10000):
    sb = get_client()
    logger.info("MEMULAI GOLD STANDARD CURATION (v7 Excel Export)...")
    
    audit = {
        "raw_contexts": 0, "missing_sentiment": 0, "article_short": 0,
        "boilerplate_fail": 0, "context_short": 0, "duplicate_pair": 0,
        "entity_limit": 0, "media_limit": 0
    }
    
    try:
        ctx_res = sb.table("entity_contexts") \
                .select(
                    "raw_text_id, entity_id, context_text, metadata, "
                    "raw_texts(source_url, resolved_domain, published_at, content_hash, text), "
                    "political_entities(canonical_name)"
                ) \
                .not_.is_("entity_id", "null") \
                .limit(limit) \
                .execute()
        raw_data = ctx_res.data or []
        audit["raw_contexts"] = len(raw_data)
        logger.info(f"Total kandidat context ditarik: {len(raw_data)}")
        
        if not raw_data: return
        
        rt_ids = list(set([r["raw_text_id"] for r in raw_data]))
        ss_data = []
        for i in range(0, len(rt_ids), 100):
            chunk = rt_ids[i:i+100]
            ss_res = sb.table("sentiment_scores") \
                        .select("raw_text_id, entity_id, label, confidence, score_negative, score_neutral, score_positive") \
                        .in_("raw_text_id", chunk) \
                        .not_.is_("entity_id", "null") \
                        .execute()
            ss_data.extend(ss_res.data or [])
            
        ss_map = {(s["raw_text_id"], s["entity_id"]): s for s in ss_data}
        
    except Exception as e:
        logger.error(f"Gagal query DB: {e}")
        return

    qualified_data = []
    
    for row in raw_data:
        rt = row.get("raw_texts")
        pe = row.get("political_entities")
        if not rt or not pe: continue
        
        ss = ss_map.get((row["raw_text_id"], row["entity_id"]))
        if not ss: 
            audit["missing_sentiment"] += 1
            continue
        
        full_text = rt.get("text") or ""
        ctx_text = row.get("context_text") or ""
        meta = row.get("metadata") or {}
        entity_name = pe.get("canonical_name") or "Unknown"
        
        if len(full_text) < MIN_ARTICLE_LEN: 
            audit["article_short"] += 1
            continue
        
        boilerplate_hits = len(BOILERPLATE_RE.findall(full_text))
        boilerplate_ratio = (boilerplate_hits * 50) / len(full_text)
        if boilerplate_ratio > MAX_BOILERPLATE_RATIO: 
            audit["boilerplate_fail"] += 1
            continue
        
        if len(ctx_text) < MIN_CONTEXT_LEN: 
            audit["context_short"] += 1
            continue
            
        scores = [ss.get("score_negative", 0), ss.get("score_neutral", 0), ss.get("score_positive", 0)]
        entropy = calculate_entropy(scores)
        
        ai_conf = ss.get("confidence", 0)
        ai_label = ss.get("label", "neutral")
        
        qualified_data.append({
            "raw_text_id": row["raw_text_id"],
            "entity_name": entity_name,
            "pseudo_label": ai_label,
            "ground_truth_label": "",
            "ai_confidence": round(ai_conf, 3),
            "entropy": round(entropy, 3),
            "media": rt.get("resolved_domain") or "unknown",
            "quality_score": meta.get("quality_score", 0),
            "context_text": ctx_text.replace("\n", " ").strip(),
            "article_text": full_text.replace("\n", " ").strip(), # TAMBAHAN: Teks utuh
            "content_hash": rt.get("content_hash"),
            "entity_id": row["entity_id"]
        })

    logger.info(f"Data yang lolos Quality Filter: {len(qualified_data)}")
    if not qualified_data: return

    # Balancing & Pair Deduplication
    media_counter = Counter()
    entity_counter = Counter()
    
    random.shuffle(qualified_data)
    
    final_dataset = []
    seen_pairs = set()
    
    for item in qualified_data:
        pair_key = (item["content_hash"], item["entity_id"])
        if pair_key in seen_pairs: 
            audit["duplicate_pair"] += 1
            continue
            
        media = item["media"]
        entity = item["entity_name"]
        
        if media_counter[media] >= MAX_PER_MEDIA: 
            audit["media_limit"] += 1
            continue
        if entity_counter[entity] >= MAX_PER_ENTITY: 
            audit["entity_limit"] += 1
            continue
        
        seen_pairs.add(pair_key)
        media_counter[media] += 1
        entity_counter[entity] += 1
        final_dataset.append(item)
        
    # EXPORT KE EXCEL (.xlsx)
    df = pd.DataFrame(final_dataset)
    
    try:
        with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Gold Dataset')
            
            # Ambil objek workbook dan worksheet untuk styling
            workbook = writer.book
            worksheet = writer.sheets['Gold Dataset']
            
            # 1. Buat Header Bold & Center
            header_font = Font(bold=True, color="FFFFFF")
            header_fill = Alignment(horizontal="center", vertical="center")
            for col in worksheet.iter_cols(1, worksheet.max_column, 1, 1):
                for cell in col:
                    cell.font = header_font
                    cell.alignment = header_fill
                    
            # 2. Atur Lebar Kolom Agar Mudah Dibaca
            col_widths = {
                'A': 35, # raw_text_id
                'B': 25, # entity_name
                'C': 15, # pseudo_label
                'D': 20, # ground_truth_label
                'E': 15, # ai_confidence
                'F': 15, # entropy
                'G': 20, # media
                'H': 15, # quality_score
                'I': 60, # context_text
                'J': 100,# article_text
                'K': 40, # content_hash
                'L': 35  # entity_id
            }
            for col_letter, width in col_widths.items():
                worksheet.column_dimensions[col_letter]. width = width
                
            # 3. Wrap Text untuk kolom context dan article
            wrap_alignment = Alignment(wrap_text=True, vertical='top')
            for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row, min_col=9, max_col=10): # Kolom I dan J
                for cell in row:
                    cell.alignment = wrap_alignment
                    
    except Exception as e:
        logger.error(f"Gagal menyimpan file Excel: {e}")
        return
        
    logger.info("=" * 60)
    logger.info("GOLD STANDARD CURATION SELESAI!")
    logger.info(f"Total Data Diekspor : {len(final_dataset)} baris")
    logger.info(f"File disimpan di    : {OUTPUT_XLSX}")
    logger.info("-" * 60)
    
    # PRINT AUDIT REPORT
    logger.info("=========== FILTER AUDIT REPORT ===========")
    logger.info(f"Raw Contexts               : {audit['raw_contexts']}")
    logger.info(f"Missing sentiment          : {audit['missing_sentiment']}")
    logger.info(f"Article too short          : {audit['article_short']}")
    logger.info(f"Boilerplate fail           : {audit['boilerplate_fail']}")
    logger.info(f"Context too short          : {audit['context_short']}")
    logger.info(f"Duplicate (hash, entity)   : {audit['duplicate_pair']}")
    logger.info(f"Entity limit reached       : {audit['entity_limit']}")
    logger.info(f"Media limit reached        : {audit['media_limit']}")
    logger.info(f"Final Export               : {len(final_dataset)}")
    logger.info("=" * 60)
    
    logger.info("Distribusi Media (Top 5):")
    for m, c in media_counter.most_common(5): logger.info(f"  {m:20s}: {c}")
    logger.info("Distribusi Entity (Top 5):")
    for e, c in entity_counter.most_common(5): logger.info(f"  {e:20s}: {c}")
    logger.info("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    main(limit=args.limit)