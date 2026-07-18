"""
export_finetune_dataset.py v5.1 — True Gold Standard Curation
=====================================================================
FIX v5.1:
  1. Buang filter context_version agar semua konteks (v6/v7/v8) bisa diekstrak.
  2. Handle NoneType pada media/entity untuk mencegah crash di log printing.
"""
import sys
import csv
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

OUTPUT_CSV = ROOT_DIR / "devtools" / "eval" / "finetune_dataset_gold.csv"

# Konfigurasi Balancing Metadata
# Konfigurasi Balancing Metadata (Longgarkan untuk dataset awal)
MAX_PER_MEDIA = 500  # Ubah dari 150 menjadi 500
MAX_PER_ENTITY = 300  # Ubah dari 100 menjadi 300
MIN_ARTICLE_LEN = 500
MIN_CONTEXT_LEN = 100

BOILERPLATE_RE = re.compile(r'(Baca Juga|Simak Juga|Berita Terkait|Advertisement|Ikuti Kami|Copyright|©|Reportase:|Jurnalis:|Editor:).*?(?=\n|$)', re.IGNORECASE)

def calculate_entropy(scores: list) -> float:
    try:
        return -sum(p * math.log(p) for p in scores if p > 1e-9)
    except:
        return 1.0

def main(limit: int = 10000):
    sb = get_client()
    logger.info("MEMULAI TRUE GOLD STANDARD CURATION (v5.1)...")
    
    try:
        # Query Context + Article + Entity (Tanpa filter version agar semua masuk)
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
        logger.info(f"Total kandidat context ditarik: {len(raw_data)}")
        
        if not raw_data: return
        
        # Query Sentiment Scores
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
        if not ss: continue
        
        full_text = rt.get("text") or ""
        ctx_text = row.get("context_text") or ""
        meta = row.get("metadata") or {}
        entity_name = pe.get("canonical_name") or "Unknown"
        
        if len(full_text) < MIN_ARTICLE_LEN: continue
        
        boilerplate_hits = len(BOILERPLATE_RE.findall(full_text))
        boilerplate_ratio = (boilerplate_hits * 50) / len(full_text)
        if boilerplate_ratio > 0.20: continue
        
        if len(ctx_text) < MIN_CONTEXT_LEN: continue
            
        scores = [ss.get("score_negative", 0), ss.get("score_neutral", 0), ss.get("score_positive", 0)]
        entropy = calculate_entropy(scores)
        
        ai_conf = ss.get("confidence", 0)
        ai_label = ss.get("label", "neutral")
        
        qualified_data.append({
            "raw_text_id": row["raw_text_id"],
            "entity_name": entity_name,
            "context_text": ctx_text.replace("\n", " ").strip(),
            "pseudo_label": ai_label,
            "ai_confidence": round(ai_conf, 3),
            "entropy": round(entropy, 3),
            "media": rt.get("resolved_domain") or "unknown", # Handle None
            "quality_score": meta.get("quality_score", 0),
            "content_hash": rt.get("content_hash"),
            "ground_truth_label": "" # Untuk diisi manusia
        })

    logger.info(f"Data yang lolos Quality Filter: {len(qualified_data)}")
    if not qualified_data: return

    # Balancing Metadata
    media_counter = Counter()
    entity_counter = Counter()
    
    random.shuffle(qualified_data)
    
    final_dataset = []
    seen_hashes = set()
    
    for item in qualified_data:
        if item["content_hash"] in seen_hashes: continue
            
        media = item["media"]
        entity = item["entity_name"]
        
        if media_counter[media] >= MAX_PER_MEDIA: continue
        if entity_counter[entity] >= MAX_PER_ENTITY: continue
        
        seen_hashes.add(item["content_hash"])
        media_counter[media] += 1
        entity_counter[entity] += 1
        final_dataset.append(item)
        
    # Export ke CSV
    fieldnames = list(final_dataset[0].keys())
    with open(OUTPUT_CSV, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(final_dataset)
        
    logger.info("=" * 60)
    logger.info("GOLD STANDARD CURATION SELESAI!")
    logger.info(f"Total Data Diekspor : {len(final_dataset)} baris")
    logger.info(f"File disimpan di    : {OUTPUT_CSV}")
    logger.info("-" * 60)
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