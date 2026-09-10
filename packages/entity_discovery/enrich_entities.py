"""
enrich_entities.py — Auto-enrich political entities with Wikipedia data
=======================================================================
Untuk setiap entity di political_entities yang belum punya:
  - image_url: foto/gambar dari Wikipedia/Wikidata
  - bio: deskripsi singkat dari Wikipedia
  - party_affiliation: update kalau kosong
  - position: update kalau kosong
  - era: periode aktif

Cara kerja:
  1. Query political_entities WHERE image_url IS NULL OR bio IS NULL
  2. Search Wikipedia Indonesia: "entity_name"
  3. Extract: first paragraph (bio), first image (foto)
  4. Update DB

Library: wikipedia (Python Wikipedia API wrapper)
  pip install wikipedia

Usage:
  python -m packages.entity_discovery.enrich_entities
  python -m packages.entity_discovery.enrich_entities --limit 50
"""
from __future__ import annotations
import os
import sys
import time
import logging
import argparse
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

try:
    import wikipedia
    wikipedia.set_lang("id")  # Indonesian Wikipedia
except ImportError:
    print("[ERROR] pip install wikipedia"); sys.exit(1)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_client() -> Client:
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        raise Exception("SUPABASE_URL/KEY not set in .env")
    return create_client(url, key)


def search_wikipedia(entity_name: str) -> dict | None:
    """Search Wikipedia for entity, return bio + photo_url + summary.

    FIXED v2: Match actual DB schema:
      - photo_url (NOT image_url)
      - wikipedia_id_url (NOT wikipedia_url)
    """
    try:
        # Search Wikipedia
        search_results = wikipedia.search(entity_name, results=3)
        if not search_results:
            return None

        # Try exact match first, then first result
        page_title = None
        for result in search_results:
            if entity_name.lower() in result.lower():
                page_title = result
                break
        if not page_title:
            page_title = search_results[0]

        # Get page
        page = wikipedia.page(page_title, auto_suggest=False, redirect=True)

        # Extract first paragraph (bio)
        content = page.content
        first_para = content.split('\n\n')[0] if '\n\n' in content else content[:500]
        bio = first_para.strip().replace('\n', ' ')[:500]  # cap at 500 chars

        # Get first image (skip logos/icons) — stored as photo_url in DB
        photo_url = None
        for img in page.images:
            if any(skip in img.lower() for skip in ['.svg', '.png', 'logo', 'icon', 'commons-logo']):
                continue
            if '.jpg' in img.lower() or '.jpeg' in img.lower():
                photo_url = img
                break
        if not photo_url and page.images:
            photo_url = page.images[0]

        return {
            "bio": bio,
            "photo_url": photo_url,          # DB column: photo_url
            "wikipedia_id_url": page.url,    # DB column: wikipedia_id_url
        }
    except wikipedia.exceptions.DisambiguationError as e:
        if e.options:
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                return {
                    "bio": page.content.split('\n\n')[0][:500] if page.content else "",
                    "photo_url": page.images[0] if page.images else None,
                    "wikipedia_id_url": page.url,
                }
            except:
                pass
        return None
    except wikipedia.exceptions.PageError:
        return None
    except Exception as e:
        logger.debug(f"Wikipedia error for {entity_name}: {e}")
        return None


def enrich_entities(sb: Client, limit: int = 0) -> int:
    """Enrich entities that don't have photo_url or bio.

    FIXED v2: Match actual DB schema:
      - Query by photo_url IS NULL (NOT image_url)
      - Update photo_url, wikipedia_id_url, bio, enrichment_status
    """
    logger.info("=" * 60)
    logger.info("  Entity Enrichment — Wikipedia Auto-Fetch")
    logger.info("=" * 60)

    # Query entities that need enrichment
    # DB columns: photo_url, wikipedia_id_url, bio (new), enrichment_status (new)
    query = sb.table("political_entities").select(
        "id, canonical_name, aliases, entity_type, party_affiliation, position, era, "
        "photo_url, wikipedia_id_url, bio, enrichment_status"
    )

    res = query.execute()
    entities = res.data or []

    if not entities:
        logger.info("No entities found in DB.")
        return 0

    logger.info(f"Total entities: {len(entities)}")

    # Filter entities that need enrichment
    to_enrich = []
    for e in entities:
        has_photo = e.get("photo_url") or False
        has_bio = e.get("bio") or False
        status = e.get("enrichment_status") or "pending"
        if (not has_photo or not has_bio) and status != "enriched":
            to_enrich.append(e)

    if limit > 0:
        to_enrich = to_enrich[:limit]

    logger.info(f"Entities to enrich: {len(to_enrich)}")
    if not to_enrich:
        logger.info("All entities already enriched ✅")
        return 0

    enriched_count = 0
    failed_count = 0

    for i, entity in enumerate(to_enrich, 1):
        name = entity["canonical_name"]
        logger.info(f"[{i}/{len(to_enrich)}] Enriching: {name}")

        wiki_data = search_wikipedia(name)

        if wiki_data is None:
            # Try with aliases
            aliases = entity.get("aliases") or []
            for alias in aliases[:2]:
                wiki_data = search_wikipedia(alias)
                if wiki_data:
                    break

        if wiki_data is None:
            logger.warning(f"  ❌ No Wikipedia page found for {name}")
            # Update status to not_found
            try:
                sb.table("political_entities").update(
                    {"enrichment_status": "not_found"}
                ).eq("id", entity["id"]).execute()
            except:
                pass
            failed_count += 1
            time.sleep(0.5)
            continue

        # Update DB — use correct column names from schema
        update_data = {
            "enrichment_status": "enriched",
        }
        if wiki_data.get("bio"):
            update_data["bio"] = wiki_data["bio"]
        if wiki_data.get("photo_url"):
            update_data["photo_url"] = wiki_data["photo_url"]
        if wiki_data.get("wikipedia_id_url"):
            update_data["wikipedia_id_url"] = wiki_data["wikipedia_id_url"]

        try:
            sb.table("political_entities").update(
                update_data
            ).eq("id", entity["id"]).execute()
            logger.info(f"  ✅ Updated: bio={'✅' if 'bio' in update_data else '❌'}, "
                      f"photo={'✅' if 'photo_url' in update_data else '❌'}, "
                      f"wiki_url={'✅' if 'wikipedia_id_url' in update_data else '❌'}")
            enriched_count += 1
        except Exception as e:
            logger.error(f"  ❌ DB update failed: {e}")
            # Try updating just enrichment_status
            try:
                sb.table("political_entities").update(
                    {"enrichment_status": "failed"}
                ).eq("id", entity["id"]).execute()
            except:
                pass
            failed_count += 1

        time.sleep(1)  # rate limit Wikipedia API

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  ENRICHMENT COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Total entities:  {len(entities)}")
    logger.info(f"  Enriched:        {enriched_count}")
    logger.info(f"  Failed:          {failed_count}")
    logger.info(f"  Already OK:      {len(entities) - len(to_enrich)}")

    return enriched_count


def main():
    ap = argparse.ArgumentParser(description="Enrich political entities with Wikipedia data")
    ap.add_argument("--limit", type=int, default=0, help="Max entities to process (0=all)")
    args = ap.parse_args()

    sb = get_client()
    enrich_entities(sb, limit=args.limit)


if __name__ == "__main__":
    main()
