#!/usr/bin/env python3.13
"""
test_workers_quality.py
======================
Test script to verify quality of entity_resolution_worker v15 + context_worker v18.

Runs sample articles through both workers (in dry-run mode, no DB) and analyzes:
  1. Entity resolution: correct main entity? false positives? era/affiliation working?
  2. Context extraction: context_quality_score distribution? speaker_not_target cases?
     multi-mention spans? relevancy pre-filter?
  3. End-to-end: entity + context → does the output make sense?

Uses sample articles from dataset_v9.jsonl (already has human-labeled gold).

Usage:
  python test_workers_quality.py --samples 20
  python test_workers_quality.py --workers entity       # test entity worker only
  python test_workers_quality.py --workers context       # test context worker only
"""
import sys
import os
import json
import argparse
import random
from pathlib import Path
from collections import Counter

# Add repo to path
REPO = Path("/tmp/idpst_repo")
sys.path.insert(0, str(REPO))

# Import workers (this will load Stanza — takes ~10s)
print("Loading Stanza pipeline (this takes ~10s)...")
try:
    from packages.entity.entity_resolution_worker import (
        process_single_article_entity,
        load_caches as load_entity_caches,
        RESOLVER_VERSION as ENTITY_VERSION,
    )
    print(f"  entity_resolution_worker: {ENTITY_VERSION}")
except Exception as e:
    print(f"  FAILED to import entity worker: {e}")
    print("  (Stanza or DB connection required — running in limited mode)")
    ENTITY_VERSION = "import_failed"

try:
    from packages.context.context_worker import (
        process_single_article_context,
        CONTEXT_VERSION,
    )
    print(f"  context_worker: {CONTEXT_VERSION}")
except Exception as e:
    print(f"  FAILED to import context worker: {e}")
    CONTEXT_VERSION = "import_failed"


def load_samples(n=20):
    """Load sample articles from dataset_v9.jsonl."""
    ds_path = REPO / "finetuning" / "datasets" / "dataset_v9.jsonl"
    if not ds_path.exists():
        print(f"Dataset not found: {ds_path}")
        return []

    rows = [json.loads(l) for l in open(ds_path) if l.strip()]
    print(f"Loaded {len(rows)} rows from dataset_v9.jsonl")

    # Pick diverse samples: mix of positive/negative/neutral, different entities
    by_label = {"positive": [], "negative": [], "neutral": []}
    for r in rows:
        lab = r.get("gold_label", "neutral")
        if lab in by_label:
            by_label[lab].append(r)

    samples = []
    per_label = n // 3
    for label, items in by_label.items():
        random.seed(42)
        random.shuffle(items)
        samples.extend(items[:per_label])

    print(f"Selected {len(samples)} samples (balanced across labels)")
    return samples


def test_entity_worker(samples, sb=None):
    """Test entity_resolution_worker v15 on sample articles."""
    print(f"\n{'='*70}")
    print(f"TEST: entity_resolution_worker {ENTITY_VERSION}")
    print(f"{'='*70}")

    if ENTITY_VERSION == "import_failed":
        print("SKIP — entity worker import failed (Stanza/DB required)")
        return

    # Load caches (requires DB)
    if sb is None:
        try:
            from packages.shared.db_client import get_client
            sb = get_client()
        except Exception as e:
            print(f"DB connection failed: {e}")
            print("Cannot test entity worker without DB caches.")
            return

    print("Loading entity caches (era + affiliation)...")
    try:
        alias_map, entity_db_map, id_to_name, id_to_entity, regex_patterns = load_entity_caches(sb)
        print(f"  Loaded {len(regex_patterns)} patterns, {len(entity_db_map)} entities")
    except Exception as e:
        print(f"  Cache load failed: {e}")
        return

    # Run on samples
    stats = Counter()
    results = []

    for i, sample in enumerate(samples[:20]):  # cap at 20 for speed
        # Convert dataset row → article format
        art = {
            "id": sample.get("raw_text_id", f"sample_{i}"),
            "title": "",  # dataset doesn't have title
            "text": sample.get("article_text", sample.get("context_text", "")),
            "metadata": {},
            "ingested_month": "2024-08",
        }

        try:
            result = process_single_article_entity(
                art, alias_map, entity_db_map, id_to_name, id_to_entity, regex_patterns
            )
            if result:
                main_entity = result["mappings"][0] if result["mappings"] else None
                expected_entity = sample.get("entity_name", "")

                # Check if expected entity is in mappings
                found_entities = [m["entity_id"] for m in result["mappings"]]
                is_main_correct = False
                if main_entity:
                    main_name = id_to_name.get(main_entity["entity_id"], "")
                    is_main_correct = main_name.lower() == expected_entity.lower()

                stats["total"] += 1
                stats["entities_found"] += len(result["mappings"])
                stats["mentions_found"] += len(result["mentions"])
                if is_main_correct:
                    stats["main_correct"] += 1
                else:
                    stats["main_wrong"] += 1

                results.append({
                    "sample_id": i,
                    "expected": expected_entity,
                    "main_found": id_to_name.get(main_entity["entity_id"], "?") if main_entity else "NONE",
                    "is_main_correct": is_main_correct,
                    "n_entities": len(result["mappings"]),
                    "main_confidence": main_entity["confidence"] if main_entity else 0,
                    "main_resolver_source": main_entity["resolver_source"] if main_entity else "none",
                })
            else:
                stats["no_result"] += 1
                results.append({
                    "sample_id": i,
                    "expected": expected_entity,
                    "main_found": "NONE",
                    "is_main_correct": False,
                })
        except Exception as e:
            stats["error"] += 1
            print(f"  sample {i}: ERROR {e}")

    # Summary
    print(f"\n--- ENTITY WORKER RESULTS ---")
    print(f"  Total tested:        {stats['total']}")
    print(f"  Main entity correct: {stats['main_correct']} ({100*stats['main_correct']/max(1,stats['total']):.1f}%)")
    print(f"  Main entity wrong:   {stats['main_wrong']}")
    print(f"  Avg entities/article: {stats['entities_found']/max(1,stats['total']):.1f}")
    print(f"  Avg mentions/article: {stats['mentions_found']/max(1,stats['total']):.1f}")
    print(f"  No result:           {stats['no_result']}")
    print(f"  Errors:              {stats['error']}")

    # Show samples
    print(f"\n--- SAMPLE RESULTS (first 5) ---")
    for r in results[:5]:
        status = "✅" if r["is_main_correct"] else "❌"
        print(f"  {status} [{r['sample_id']}] expected='{r['expected'][:20]}' found='{r['main_found'][:20]}' conf={r.get('main_confidence',0):.2f}")

    return results


def test_context_worker(samples, sb=None):
    """Test context_worker v18 on sample articles."""
    print(f"\n{'='*70}")
    print(f"TEST: context_worker {CONTEXT_VERSION}")
    print(f"{'='*70}")

    if CONTEXT_VERSION == "import_failed":
        print("SKIP — context worker import failed (Stanza/DB required)")
        return

    # Build mentions_by_art from samples (simulate entity_mentions table)
    mentions_by_art = {}
    for i, sample in enumerate(samples[:20]):
        art_id = sample.get("raw_text_id", f"sample_{i}")
        entity_name = sample.get("entity_name", "")
        # Find entity in article text to get offset
        article = sample.get("article_text", sample.get("context_text", ""))
        offset = article.lower().find(entity_name.lower())
        if offset >= 0:
            mentions_by_art[art_id] = [{
                "entity_id": f"sample_entity_{i}",
                "start_offset": offset,
                "end_offset": offset + len(entity_name),
                "political_entities": {"canonical_name": entity_name},
            }]

    # Run on samples
    stats = Counter()
    results = []

    for i, sample in enumerate(samples[:20]):
        art_id = sample.get("raw_text_id", f"sample_{i}")
        art = {
            "id": art_id,
            "title": "",
            "text": sample.get("article_text", sample.get("context_text", "")),
            "ingested_month": "2024-08",
        }

        mentions = mentions_by_art.get(art_id, [])
        if not mentions:
            continue

        try:
            ctx_results = process_single_article_context(art, {art_id: mentions})
            if ctx_results:
                for ctx in ctx_results:
                    metadata = ctx.get("metadata", {})
                    stats["total"] += 1
                    stats["quality_scores"].append(metadata.get("quality_score", 0))
                    stats["relevancy_scores"].append(metadata.get("relevancy_score", 0))
                    stats["span_counts"].append(metadata.get("span_count", 1))

                    if metadata.get("has_sentiment_predicate"):
                        stats["has_sentiment"] += 1
                    if metadata.get("has_attribution"):
                        stats["has_attribution"] += 1
                    if metadata.get("is_relevant"):
                        stats["relevant"] += 1
                    else:
                        stats["not_relevant"] += 1

                    results.append({
                        "sample_id": i,
                        "entity": sample.get("entity_name", ""),
                        "context_len": len(ctx.get("context_text", "")),
                        "quality_score": metadata.get("quality_score", 0),
                        "relevancy_score": metadata.get("relevancy_score", 0),
                        "span_count": metadata.get("span_count", 1),
                        "is_relevant": metadata.get("is_relevant", True),
                        "has_sentiment": metadata.get("has_sentiment_predicate", False),
                        "has_attribution": metadata.get("has_attribution", False),
                        "context_preview": ctx.get("context_text", "")[:120] + "...",
                    })
            else:
                stats["no_result"] += 1
        except Exception as e:
            stats["error"] += 1
            print(f"  sample {i}: ERROR {e}")

    # Summary
    print(f"\n--- CONTEXT WORKER RESULTS ---")
    print(f"  Total contexts:      {stats['total']}")
    print(f"  Relevant (>=0.5):    {stats['relevant']} ({100*stats['relevant']/max(1,stats['total']):.1f}%)")
    print(f"  Not relevant (<0.5): {stats['not_relevant']}")
    print(f"  Has sentiment pred:  {stats['has_sentiment']}")
    print(f"  Has attribution:     {stats['has_attribution']} (speaker — should be neutral)")
    print(f"  Avg quality score:   {sum(stats['quality_scores'])/max(1,len(stats['quality_scores'])):.1f}")
    print(f"  Avg relevancy:       {sum(stats['relevancy_scores'])/max(1,len(stats['relevancy_scores'])):.3f}")
    print(f"  Avg span count:      {sum(stats['span_counts'])/max(1,len(stats['span_counts'])):.1f}")
    print(f"  No result:           {stats['no_result']}")
    print(f"  Errors:              {stats['error']}")

    # Show samples
    print(f"\n--- SAMPLE CONTEXTS (first 5) ---")
    for r in results[:5]:
        rel = "✅ REL" if r["is_relevant"] else "⚠️ LOW-REL"
        sent = "SENT" if r["has_sentiment"] else ("ATTR" if r["has_attribution"] else "----")
        print(f"  [{r['sample_id']}] {r['entity'][:18]:20s} Q={r['quality_score']:3d} R={r['relevancy_score']:.2f} "
              f"spans={r['span_count']} [{sent} {rel}]")
        print(f"    ctx: {r['context_preview']}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--workers", choices=["entity", "context", "both"], default="both")
    args = parser.parse_args()

    print("=" * 70)
    print("WORKER QUALITY TEST — entity_resolution v15 + context v18")
    print("=" * 70)

    samples = load_samples(args.samples)
    if not samples:
        print("No samples loaded. Exiting.")
        return

    # Get DB client
    sb = None
    try:
        from packages.shared.db_client import get_client
        sb = get_client()
        print(f"DB client: connected")
    except Exception as e:
        print(f"DB client: FAILED ({e})")
        print("Note: entity worker needs DB caches. Context worker can run without DB.")

    if args.workers in ("entity", "both"):
        test_entity_worker(samples, sb)

    if args.workers in ("context", "both"):
        test_context_worker(samples, sb)

    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
