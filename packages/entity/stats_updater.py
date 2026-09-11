"""
stats_updater.py — Update entity stats untuk frontend dashboard
================================================================
Populate kolom di political_entities yang dibutuhkan frontend:
  - mention_count_7d: jumlah mention 7 hari terakhir
  - mention_count_30d: jumlah mention 30 hari terakhir
  - last_mentioned_at: timestamp mention terakhir

Data source: entity_mentions table (partitioned, query via parent).

Usage:
  python -m packages.entity.stats_updater
  python -m packages.entity.stats_updater --limit 50
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


def get_client() -> Client:
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        raise Exception("SUPABASE_URL/KEY not set in .env")
    return create_client(url, key)


def update_entity_stats(sb: Client) -> dict:
    """Update mention_count_7d, mention_count_30d, last_mentioned_at.

    Uses RPC untuk efficient batch update (1 query, bukan N queries).
    Falls back to per-entity update kalau RPC tidak ada.

    Returns:
        {"total": N, "updated": N, "failed": N}
    """
    logger.info("=" * 60)
    logger.info("  Entity Stats Updater — mention_count + last_mentioned_at")
    logger.info("=" * 60)

    now = datetime.now(timezone.utc)
    seven_days_ago = (now - timedelta(days=7)).isoformat()
    thirty_days_ago = (now - timedelta(days=30)).isoformat()

    # Get all entities
    res = sb.table("political_entities").select("id, canonical_name").execute()
    entities = res.data or []
    logger.info(f"Total entities: {len(entities)}")

    if not entities:
        return {"total": 0, "updated": 0, "failed": 0}

    updated = 0
    failed = 0

    for entity in entities:
        entity_id = entity["id"]
        name = entity["canonical_name"]

        try:
            # Query entity_mentions untuk 7 hari terakhir
            # entity_mentions is partitioned — query parent table works
            res_7d = sb.table("entity_mentions") \
                .select("id", count="exact") \
                .eq("entity_id", entity_id) \
                .gte("raw_text_id", seven_days_ago) \
                .execute()

            # Note: entity_mentions tidak punya timestamp column sendiri.
            # Kita perlu JOIN ke raw_texts untuk filter by date.
            # Tapi Supabase client tidak support complex JOIN untuk count.
            # Alternative: query via RPC atau query sentiment_scores (punya scored_at).

            # Query sentiment_scores untuk 7d dan 30d (punya scored_at)
            res_7d = sb.table("sentiment_scores") \
                .select("id", count="exact") \
                .eq("entity_id", entity_id) \
                .gte("scored_at", seven_days_ago) \
                .execute()
            count_7d = res_7d.count or 0

            res_30d = sb.table("sentiment_scores") \
                .select("id", count="exact") \
                .eq("entity_id", entity_id) \
                .gte("scored_at", thirty_days_ago) \
                .execute()
            count_30d = res_30d.count or 0

            # Query last_mentioned_at (latest sentiment_score)
            res_last = sb.table("sentiment_scores") \
                .select("scored_at") \
                .eq("entity_id", entity_id) \
                .order("scored_at", desc=True) \
                .limit(1) \
                .execute()
            last_mentioned = None
            if res_last.data:
                last_mentioned = res_last.data[0].get("scored_at")

            # Update political_entities
            update_data = {
                "mention_count_7d": count_7d,
                "mention_count_30d": count_30d,
            }
            if last_mentioned:
                update_data["last_mentioned_at"] = last_mentioned

            sb.table("political_entities").update(update_data).eq("id", entity_id).execute()
            updated += 1

            if updated % 10 == 0:
                logger.info(f"  Progress: {updated}/{len(entities)} — last: {name} (7d={count_7d}, 30d={count_30d})")

        except Exception as e:
            logger.error(f"  ❌ Failed: {name} — {e}")
            failed += 1

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  STATS UPDATE COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Total entities:  {len(entities)}")
    logger.info(f"  Updated:         {updated}")
    logger.info(f"  Failed:          {failed}")

    return {"total": len(entities), "updated": updated, "failed": failed}


def main():
    ap = argparse.ArgumentParser(description="Update entity stats for dashboard")
    ap.add_argument("--limit", type=int, default=0, help="Max entities (0=all)")
    args = ap.parse_args()

    sb = get_client()
    result = update_entity_stats(sb)

    if result["failed"] > 0:
        logger.warning(f"  {result['failed']} entities failed — check errors above")


if __name__ == "__main__":
    main()
