#!/usr/bin/env python3.13
"""
test_workers_dynamic.py
======================
DYNAMIC test of entity_resolution_worker v15 + context_worker v18.

Runs ACTUAL workers (with Stanza) on sample articles from dataset_v9.jsonl.
No DB needed — builds mock entity cache from dataset's entity_names.

Tests:
  1. entity_resolution_worker: does it correctly identify main entity?
  2. context_worker: does it extract good context? quality_score, relevancy, spans?
  3. End-to-end: entity + context → is output usable for NLP layer?

Output: detailed report on actual worker behavior.
"""
import sys
import os
import json
import re
import time
import random
from pathlib import Path
from collections import Counter

# Add repo to path
REPO = Path("/tmp/idpst_repo")
sys.path.insert(0, str(REPO))

# Mock packages.shared.db_client and logger BEFORE importing workers
# (workers import these at module level)
print("Setting up mocks for DB client + logger...")

import types

# Mock packages.shared
shared_mod = types.ModuleType("packages.shared")
sys.modules["packages.shared"] = shared_mod

# Mock db_client
db_mod = types.ModuleType("packages.shared.db_client")
db_mod.get_client = lambda: None  # no-op
sys.modules["packages.shared.db_client"] = db_mod
shared_mod.db_client = db_mod

# Mock logger
log_mod = types.ModuleType("packages.shared.logger")
log_mod.start_run = lambda *a, **kw: "mock_run_id"
log_mod.finish_run = lambda *a, **kw: None
sys.modules["packages.shared.logger"] = log_mod
shared_mod.logger = log_mod

# Mock constants
const_mod = types.ModuleType("packages.shared.constants")
const_mod.STATUS_VALIDATED = "validated"
const_mod.STATUS_PROCESSED = "processed"
sys.modules["packages.shared.constants"] = const_mod
shared_mod.constants = const_mod

# Mock packages.nlp.sentiment_model (needed by context_worker for get_pipeline)
nlp_mod = types.ModuleType("packages.nlp")
sys.modules["packages.nlp"] = nlp_mod
sent_mod = types.ModuleType("packages.nlp.sentiment_model")
sent_mod.get_pipeline = lambda: None
sys.modules["packages.nlp.sentiment_model"] = sent_mod
nlp_mod.sentiment_model = sent_mod

print("  Mocks installed")

print("=" * 70)
print("DYNAMIC WORKER TEST — entity_resolution v15 + context v18")
print("=" * 70)
print()

# Suppress verbose logging
import logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("stanza").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load Stanza pipeline (needed by both workers)
print("[1/4] Loading Stanza pipeline (tokenize,pos,lemma,depparse)...")
t0 = time.time()
import stanza
try:
    NLP = stanza.Pipeline('id', processors='tokenize,pos,lemma,depparse',
                          verbose=False, use_gpu=False, batch_size=16)
    print(f"  Stanza loaded in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# Load samples
print("\n[2/4] Loading sample articles from dataset_v9.jsonl...")
ds_path = REPO / "finetuning" / "datasets" / "dataset_v9.jsonl"
rows = [json.loads(l) for l in open(ds_path) if l.strip()]
print(f"  Loaded {len(rows)} rows")

# Pick 12 diverse samples (mix of labels, different entities)
# IMPORTANT: only pick samples where entity IS in article_text
random.seed(42)
by_label = {"positive": [], "negative": [], "neutral": []}
for r in rows:
    lab = r.get("gold_label", "neutral")
    ent = r.get("entity_name", "")
    art = r.get("article_text", "")
    if lab in by_label and ent and art and ent.lower() in art.lower():
        by_label[lab].append(r)

samples = []
for label, items in by_label.items():
    random.shuffle(items)
    samples.extend(items[:4])  # 4 per label = 12 total

print(f"  Selected {len(samples)} samples (4 per label, entity in article)")

# Build mock entity cache (from all entity_names in dataset)
print("\n[3/4] Building mock entity cache...")
entity_names_set = set()
for r in rows:
    name = r.get("entity_name", "").strip()
    if name:
        entity_names_set.add(name)

# Mock cache: entity_name → entity_id (use index as ID)
entity_db_map = {}
alias_map = {}
id_to_name = {}
id_to_entity = {}
regex_patterns = []

for i, name in enumerate(sorted(entity_names_set)):
    ent_id = f"entity_{i:03d}"
    canon_lower = name.lower()
    entity_db_map[canon_lower] = ent_id
    id_to_name[ent_id] = name
    id_to_entity[ent_id] = {
        "name": name,
        "aliases": [],
        "entity_type": "other",
        "party": None,
        "position": None,
        "era": [],  # empty — era check becomes no-op
    }
    try:
        regex_patterns.append((re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE), canon_lower))
    except re.error:
        pass

print(f"  {len(entity_db_map)} entities, {len(regex_patterns)} patterns")

# Import worker functions
print("\n[4/4] Importing worker functions...")

# Monkey-patch the NLP in entity_resolution_worker
import packages.entity.entity_resolution_worker as entity_mod
entity_mod.NLP = NLP  # inject our loaded pipeline

from packages.entity.entity_resolution_worker import (
    process_single_article_entity,
    is_false_positive,
    check_semantic_role,
    check_era_compatibility,
    check_affiliation,
    RESOLVER_VERSION as ENTITY_VERSION,
)
print(f"  entity_resolution_worker: {ENTITY_VERSION}")

# Monkey-patch the NLP in context_worker
import packages.context.context_worker as ctx_mod
ctx_mod.NLP = NLP  # inject
# Disable relevancy model (no GPU/transformers) — fail-open
ctx_mod.get_relevancy_pipeline = lambda: None
ctx_mod.check_relevancy = lambda entity_name, context_text: 1.0  # fail-open

from packages.context.context_worker import (
    process_single_article_context,
    CONTEXT_VERSION,
)
print(f"  context_worker: {CONTEXT_VERSION}")

# ===== RUN TESTS =====
print(f"\n{'='*70}")
print("RUNNING TESTS ON 12 SAMPLES")
print(f"{'='*70}")

entity_results = []
context_results = []
entity_stats = Counter()
context_stats = Counter()

for i, sample in enumerate(samples):
    art_id = sample.get("raw_text_id", f"sample_{i}")
    entity_name = sample.get("entity_name", "")
    article_text = sample.get("article_text", "")
    expected_label = sample.get("gold_label", "neutral")

    print(f"\n--- Sample {i+1}/{len(samples)} ---")
    print(f"  Expected entity: {entity_name}")
    print(f"  Expected label: {expected_label}")
    print(f"  Article length: {len(article_text)} chars")

    # === TEST 1: Entity Resolution Worker v15 ===
    art = {
        "id": art_id,
        "title": "",  # dataset doesn't have separate title
        "text": article_text,
        "metadata": {},
        "ingested_month": "2024-08",
    }

    try:
        entity_result = process_single_article_entity(
            art, alias_map, entity_db_map, id_to_name, id_to_entity, regex_patterns
        )

        if entity_result and entity_result["mappings"]:
            main_mapping = entity_result["mappings"][0]
            main_ent_id = main_mapping["entity_id"]
            main_ent_name = id_to_name.get(main_ent_id, "?")
            main_conf = main_mapping["confidence"]
            is_correct = main_ent_name.lower() == entity_name.lower()

            # Check if expected entity is in any mapping (not just main)
            all_found_names = [id_to_name.get(m["entity_id"], "?") for m in entity_result["mappings"]]
            expected_in_mappings = entity_name.lower() in [n.lower() for n in all_found_names]

            entity_stats["total"] += 1
            entity_stats["entities_found"] += len(entity_result["mappings"])
            entity_stats["mentions_found"] += len(entity_result["mentions"])
            if is_correct:
                entity_stats["main_correct"] += 1
                print(f"  ✅ Entity main CORRECT: {main_ent_name} (conf={main_conf:.2f})")
            elif expected_in_mappings:
                entity_stats["main_wrong_but_found"] += 1
                print(f"  ⚠️ Entity found but not main: expected={entity_name}, main={main_ent_name}")
                print(f"     All found: {all_found_names}")
            else:
                entity_stats["main_wrong_not_found"] += 1
                print(f"  ❌ Entity NOT FOUND: expected={entity_name}, main={main_ent_name}")
                print(f"     All found: {all_found_names}")

            entity_results.append({
                "sample_id": i,
                "expected": entity_name,
                "expected_label": expected_label,
                "main_found": main_ent_name,
                "all_found": all_found_names,
                "is_main_correct": is_correct,
                "expected_in_mappings": expected_in_mappings,
                "confidence": main_conf,
                "n_entities": len(entity_result["mappings"]),
                "n_mentions": len(entity_result["mentions"]),
            })
        else:
            entity_stats["no_result"] += 1
            print(f"  ❌ No entity found")
            entity_results.append({
                "sample_id": i,
                "expected": entity_name,
                "main_found": "NONE",
                "is_main_correct": False,
            })
    except Exception as e:
        entity_stats["error"] += 1
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

    # === TEST 2: Context Worker v18 ===
    # Build mentions from expected entity (offset in article)
    offset = article_text.lower().find(entity_name.lower())
    if offset >= 0:
        mentions_by_art = {
            art_id: [{
                "entity_id": entity_db_map.get(entity_name.lower(), "unknown"),
                "start_offset": offset,
                "end_offset": offset + len(entity_name),
                "political_entities": {"canonical_name": entity_name},
            }]
        }

        try:
            ctx_results = process_single_article_context(art, mentions_by_art)
            if ctx_results:
                ctx = ctx_results[0]
                metadata = ctx.get("metadata", {})
                quality_score = metadata.get("quality_score", 0)
                has_sentiment = metadata.get("has_sentiment_predicate", False)
                has_attribution = metadata.get("has_attribution", False)
                is_main_actor = metadata.get("is_main_actor", False)
                span_count = metadata.get("span_count", 1)
                used_local_clause = metadata.get("used_local_clause", False)
                relevancy_score = metadata.get("relevancy_score", 1.0)  # fail-open

                context_stats["total"] += 1
                context_stats["quality_scores"].append(quality_score) if isinstance(context_stats["quality_scores"], list) else None
                if has_sentiment:
                    context_stats["has_sentiment"] += 1
                if has_attribution:
                    context_stats["has_attribution"] += 1
                if is_main_actor:
                    context_stats["is_main_actor"] += 1
                if used_local_clause:
                    context_stats["used_local_clause"] += 1
                context_stats["span_counts"].append(span_count) if isinstance(context_stats["span_counts"], list) else None

                ctx_preview = ctx.get("context_text", "")[:150] + "..." if len(ctx.get("context_text", "")) > 150 else ctx.get("context_text", "")

                # Check if context contains the entity
                entity_in_ctx = entity_name.lower() in ctx.get("context_text", "").lower()

                print(f"  Context: Q={quality_score} sent={has_sentiment} attr={has_attribution} "
                      f"main_actor={is_main_actor} spans={span_count} local_clause={used_local_clause}")
                print(f"  Entity in context: {'✅' if entity_in_ctx else '❌'}")
                print(f"  Context preview: {ctx_preview}")

                context_results.append({
                    "sample_id": i,
                    "entity": entity_name,
                    "quality_score": quality_score,
                    "has_sentiment": has_sentiment,
                    "has_attribution": has_attribution,
                    "is_main_actor": is_main_actor,
                    "span_count": span_count,
                    "used_local_clause": used_local_clause,
                    "entity_in_context": entity_in_ctx,
                    "context_len": len(ctx.get("context_text", "")),
                    "context_preview": ctx_preview,
                })
            else:
                context_stats["no_result"] += 1
                print(f"  ❌ No context extracted")
        except Exception as e:
            context_stats["error"] += 1
            print(f"  ❌ Context ERROR: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  ⚠️ Entity not found in article text — skipping context test")

# ===== SUMMARY =====
print(f"\n{'='*70}")
print("DYNAMIC TEST SUMMARY")
print(f"{'='*70}")

print(f"\n--- ENTITY RESOLUTION WORKER v15 ---")
print(f"  Total tested:           {entity_stats['total']}")
print(f"  Main entity CORRECT:    {entity_stats['main_correct']} ({100*entity_stats['main_correct']/max(1,entity_stats['total']):.1f}%)")
print(f"  Found but not main:     {entity_stats['main_wrong_but_found']}")
print(f"  Not found at all:       {entity_stats['main_wrong_not_found']}")
print(f"  No result:              {entity_stats['no_result']}")
print(f"  Errors:                 {entity_stats['error']}")
print(f"  Avg entities/article:   {entity_stats['entities_found']/max(1,entity_stats['total']):.1f}")
print(f"  Avg mentions/article:   {entity_stats['mentions_found']/max(1,entity_stats['total']):.1f}")

print(f"\n--- CONTEXT WORKER v18 ---")
total = context_stats['total']
print(f"  Total tested:           {total}")
print(f"  Has sentiment pred:     {context_stats['has_sentiment']} ({100*context_stats['has_sentiment']/max(1,total):.1f}%)")
print(f"  Has attribution:        {context_stats['has_attribution']} ({100*context_stats['has_attribution']/max(1,total):.1f}%) — speaker (should be neutral)")
print(f"  Is main actor:          {context_stats['is_main_actor']} ({100*context_stats['is_main_actor']/max(1,total):.1f}%)")
print(f"  Used local clause:     {context_stats['used_local_clause']} (crowded sentence handling)")
qs = context_stats.get('quality_scores', [])
if isinstance(qs, list) and qs:
    print(f"  Avg quality score:      {sum(qs)/len(qs):.1f}")
    print(f"  Min/Max quality:        {min(qs)}/{max(qs)}")
sc = context_stats.get('span_counts', [])
if isinstance(sc, list) and sc:
    print(f"  Avg span count:         {sum(sc)/len(sc):.1f}")

# Entity in context check
ctx_entity_found = sum(1 for r in context_results if r.get("entity_in_context"))
print(f"  Entity in context:      {ctx_entity_found}/{len(context_results)} ({100*ctx_entity_found/max(1,len(context_results)):.1f}%)")

# Save detailed results
out_path = "/home/z/my-project/finetuning/dynamic_test_results.json"
with open(out_path, "w") as f:
    json.dump({
        "entity_results": entity_results,
        "context_results": context_results,
        "entity_stats": dict(entity_stats),
        "context_stats": {k: v for k, v in context_stats.items() if not isinstance(v, list)},
    }, f, indent=2, ensure_ascii=False)
print(f"\nDetailed results saved: {out_path}")

print(f"\n{'='*70}")
print("VERDICT")
print(f"{'='*70}")
main_acc = 100*entity_stats['main_correct']/max(1,entity_stats['total'])
if main_acc >= 80:
    print(f"✅ Entity worker: GOOD ({main_acc:.1f}% main entity accuracy)")
elif main_acc >= 60:
    print(f"⚠️ Entity worker: MODERATE ({main_acc:.1f}% — needs improvement)")
else:
    print(f"❌ Entity worker: POOR ({main_acc:.1f}% — critical issues)")

ctx_sent_rate = 100*context_stats['has_sentiment']/max(1,total)
ctx_attr_rate = 100*context_stats['has_attribution']/max(1,total)
if ctx_sent_rate >= 30:
    print(f"✅ Context worker: GOOD ({ctx_sent_rate:.1f}% have sentiment predicates)")
elif ctx_sent_rate >= 15:
    print(f"⚠️ Context worker: MODERATE ({ctx_sent_rate:.1f}% sentiment — may miss cases)")
else:
    print(f"❌ Context worker: POOR ({ctx_sent_rate:.1f}% sentiment — verb sets too narrow?)")

if ctx_attr_rate > 50:
    print(f"⚠️ High attribution rate ({ctx_attr_rate:.1f}%) — many speakers, may cause neutral bias")
