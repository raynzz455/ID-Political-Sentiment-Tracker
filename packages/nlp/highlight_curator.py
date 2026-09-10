"""
highlight_curator.py — Populate entity_highlights untuk dashboard
=================================================================
Pilih artikel terbaik per entity untuk ditampilkan di dashboard featured.

Logic:
  1. Query sentiment_scores WHERE confidence > 0.7 AND label != 'neutral'
  2. JOIN raw_texts untuk dapat title, source_url, image_url, published_at
  3. Top 5 per entity (sort by confidence DESC, prefer recent)
  4. Insert ke entity_highlights (delete old first — idempotent)

entity_highlights schema:
  entity_id, raw_text_id, polarity, title, source_url, source_name,
  image_url, label, confidence, score_positive, score_negative, published_at

Usage:
  python -m packages.nlp.highlight_curator
  python -m packages.nlp.highlight_curator --limit 20  # top 20 entities only
"""
from __future__ import annotations
import os
import sys
import logging
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
env_path = ROOT_DIR / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)

try:
    from supabase import create_client, Client
except ImportError:
    print("[ERROR] pip install supabase"); sys.exit(1)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Config
MIN_CONFIDENCE = 0.7          # hanya simpan high-confidence predictions
HIGHLIGHTS_PER_ENTITY = 5     # top 5 articles per entity
MAX_ENTITIES = 0              # 0 = all entities
DAYS_BACK = 30                # hanya artikel 30 hari terakhir


def get_client() -> Client:
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        raise Exception("SUPABASE_URL/KEY not set in .env")
    return create_client(url, key)


def curate_highlights(sb: Client, limit: int = 0) -> dict:
    """Populate entity_highlights with top articles per entity.

    Returns:
        {"entities_processed": N, "highlights_inserted": N, "failed": N}
    """
    logger.info("=" * 60)
    logger.info("  Highlight Curator — entity_highlights population")
    logger.info("=" * 60)

    # Get all entities that have sentiment_scores
    now = datetime.now(timezone.utc)
    days_ago = (now - timedelta(days=DAYS_BACK)).isoformat()

    # Query entities
    entity_res = sb.table("political_entities").select("id, canonical_name").execute()
    entities = entity_res.data or []

    if limit > 0:
        entities = entities[:limit]

    logger.info(f"Entities to process: {len(entities)}")
    logger.info(f"Min confidence: {MIN_CONFIDENCE}")
    logger.info(f"Highlights per entity: {HIGHLIGHTS_PER_ENTITY}")
    logger.info(f"Days back: {DAYS_BACK}")

    total_inserted = 0
    total_failed = 0
    entities_with_highlights = 0

    for i, entity in enumerate(entities, 1):
        entity_id = entity["id"]
        name = entity["canonical_name"]

        try:
            # Delete old highlights for this entity (idempotent)
            sb.table("entity_highlights").delete().eq("entity_id", entity_id).execute()

            # Query sentiment_scores for this entity (high confidence, non-neutral)
            # sentiment_scores is partitioned — query parent works
            scores_res = sb.table("sentiment_scores") \
                .select("raw_text_id, label, confidence, score_positive, score_negative, scored_at") \
                .eq("entity_id", entity_id) \
                .gte("confidence", MIN_CONFIDENCE) \
                .neq("label", "neutral") \
                .gte("scored_at", days_ago) \
                .order("confidence", desc=True) \
                .limit(HIGHLIGHTS_PER_ENTITY) \
                .execute()

            scores = scores_res.data or []

            if not scores:
                continue  # no high-confidence non-neutral articles

            # Get raw_texts data (title, source_url, image_url, published_at)
            raw_ids = [s["raw_text_id"] for s in scores]
            raw_res = sb.table("raw_texts") \
                .select("id, title, source_url, image_url, published_at") \
                .in_("id", raw_ids) \
                .execute()
            raw_map = {r["id"]: r for r in (raw_res.data or [])}

            # Build highlights
            highlights_to_insert = []
            for score in scores:
                raw = raw_map.get(score["raw_text_id"])
                if not raw:
                    continue

                # Determine polarity (only positive/negative for highlights)
                label = score["label"]
                if label not in ("positive", "negative"):
                    continue

                # Extract source name from URL
                source_url = raw.get("source_url") or ""
                source_name = ""
                if source_url:
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(source_url).netloc
                        source_name = domain.replace("www.", "")
                    except:
                        source_name = source_url[:50]

                highlights_to_insert.append({
                    "entity_id": entity_id,
                    "raw_text_id": score["raw_text_id"],
                    "polarity": label,  # 'positive' or 'negative'
                    "title": raw.get("title") or "",
                    "source_url": source_url,
                    "source_name": source_name,
                    "image_url": raw.get("image_url"),
                    "label": label,
                    "confidence": float(score["confidence"]),
                    "score_positive": float(score["score_positive"]),
                    "score_negative": float(score["score_negative"]),
                    "published_at": raw.get("published_at"),
                })

            if highlights_to_insert:
                # Insert highlights
                sb.table("entity_highlights").insert(highlights_to_insert).execute()
                total_inserted += len(highlights_to_insert)
                entities_with_highlights += 1

                if entities_with_highlights % 5 == 0:
                    logger.info(f"  Progress: {i}/{len(entities)} — {name}: {len(highlights_to_insert)} highlights")

        except Exception as e:
            logger.error(f"  ❌ Failed: {name} — {e}")
            total_failed += 1

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  HIGHLIGHT CURATION COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Entities processed:       {len(entities)}")
    logger.info(f"  Entities with highlights: {entities_with_highlights}")
    logger.info(f"  Highlights inserted:      {total_inserted}")
    logger.info(f"  Failed:                   {total_failed}")

    return {
        "entities_processed": len(entities),
        "highlights_inserted": total_inserted,
        "failed": total_failed,
    }


def main():
    ap = argparse.ArgumentParser(description="Curate entity highlights for dashboard")
    ap.add_argument("--limit", type=int, default=0, help="Max entities (0=all)")
    args = ap.parse_args()

    sb = get_client()
    curate_highlights(sb, limit=args.limit)


if __name__ == "__main__":
    main()
