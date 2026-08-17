"""
test_moe_workers.py — Test Entity Resolution + Context Extraction MoE
=====================================================================
Runs both MoE systems on sample articles from dataset_v9.jsonl.

Tests:
  1. Entity Resolution MoE: 5 experts detect entities in parallel
  2. Context Extraction MoE: 5 experts extract context per entity
  3. Compare with single-expert baseline (v15.1 + v19.1)

Usage:
  python finetuning/scripts/test_moe_workers.py --samples 10
  python finetuning/scripts/test_moe_workers.py --entity-only
  python finetuning/scripts/test_moe_workers.py --context-only
"""
import sys
import os
import json
import time
import random
import argparse
from pathlib import Path
from collections import Counter

REPO = Path("/tmp/idpst_repo_v2")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "patches"))

print("=" * 70)
print("MoE WORKERS TEST — Entity Resolution + Context Extraction")
print("=" * 70)

# Setup mocks for DB client (workers import these)
import types
shared_mod = types.ModuleType("packages.shared")
sys.modules["packages.shared"] = shared_mod
for mod_name, attrs in [
    ("packages.shared.db_client", {"get_client": lambda: None}),
    ("packages.shared.logger", {"start_run": lambda *a, **kw: "mock", "finish_run": lambda *a, **kw: None}),
    ("packages.shared.constants", {"STATUS_VALIDATED": "validated", "STATUS_PROCESSED": "processed"}),
]:
    m = types.ModuleType(mod_name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[mod_name] = m
setattr(shared_mod, mod_name.split('.')[-1], m)

import logging
logging.basicConfig(level=logging.WARNING)

# Load Stanza
print("\n[1/3] Loading Stanza pipeline...")
t0 = time.time()
import stanza
NLP = stanza.Pipeline('id', processors='tokenize,pos,lemma,depparse',
                      verbose=False, use_gpu=False, batch_size=16)
print(f"  Stanza loaded in {time.time()-t0:.1f}s")

# Load sample articles
print("\n[2/3] Loading samples...")
ds_path = REPO / "finetuning" / "datasets" / "dataset_v9.jsonl"
rows = [json.loads(l) for l in open(ds_path) if l.strip()]
print(f"  Loaded {len(rows)} rows")

random.seed(42)
# Pick samples where entity is in article_text
good_samples = []
for r in rows:
    ent = r.get("entity_name", "")
    art = r.get("article_text", "")
    if ent and art and ent.lower() in art.lower():
        good_samples.append(r)
samples = random.sample(good_samples, min(10, len(good_samples)))
print(f"  Selected {len(samples)} samples (entity in article)")

# Build entity cache
print("\n[3/3] Building entity cache...")
import re
entity_db_map = {}
alias_map = {}
id_to_name = {}
regex_patterns = []
for r in rows:
    name = r.get("entity_name", "").strip()
    if name and name not in id_to_name.values():
        ent_id = f"entity_{len(entity_db_map):03d}"
        canon_lower = name.lower()
        entity_db_map[canon_lower] = ent_id
        id_to_name[ent_id] = name
        try:
            regex_patterns.append((re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE), canon_lower))
        except re.error:
            pass
print(f"  {len(entity_db_map)} entities, {len(regex_patterns)} patterns")


def test_entity_moe(samples):
    """Test Entity Resolution MoE on samples."""
    print(f"\n{'='*70}")
    print("TEST 1: Entity Resolution MoE")
    print(f"{'='*70}")
    
    try:
        from entity_resolution_moe import EntityResolutionMoE, EntityRouter
        
        # Create MoE (disable DBpedia + embedding for offline test)
        moe = EntityResolutionMoE(
            entity_db_map=entity_db_map,
            alias_map=alias_map,
            id_to_name=id_to_name,
            regex_patterns=regex_patterns,
            stanza_nlp=NLP,
            enable_dbpedia=False,  # no internet in test
            enable_embedding=False,  # no model in test
            enable_spacy=False,  # no spacy installed
            parallel=True,
        )
        print(f"  MoE created with {len(moe.regex_expert and [1] or []) + (1 if moe.stanza_expert else 0)} experts active")
        print(f"  (DBpedia + embedding + spacy disabled for offline test)")
        
    except ImportError as e:
        print(f"  Cannot import MoE: {e}")
        return
    
    results = []
    stats = Counter()
    
    for i, sample in enumerate(samples):
        art_id = sample.get("raw_text_id", f"sample_{i}")
        entity_name = sample.get("entity_name", "")
        article = sample.get("article_text", "")
        
        print(f"\n  --- Sample {i+1}/{len(samples)} ---")
        print(f"  Expected: {entity_name}")
        
        try:
            t0 = time.time()
            result = moe.resolve(article)
            elapsed = time.time() - t0
            
            main = result.get("main_entity")
            if main:
                is_correct = main.entity_name.lower() == entity_name.lower()
                stats["total"] += 1
                if is_correct:
                    stats["correct"] += 1
                    print(f"  ✅ Main: {main.entity_name} (conf={main.confidence:.2f}, agreement={main.expert_agreement})")
                else:
                    stats["wrong"] += 1
                    print(f"  ❌ Main: {main.entity_name} (expected: {entity_name})")
                
                print(f"  Experts used: {result['experts_used']}")
                print(f"  Time: {elapsed*1000:.0f}ms")
                
                results.append({
                    "sample_id": i,
                    "expected": entity_name,
                    "main_found": main.entity_name,
                    "is_correct": is_correct,
                    "confidence": main.confidence,
                    "expert_agreement": main.expert_agreement,
                    "processing_time_ms": result["processing_time_ms"],
                    "experts_used": result["experts_used"],
                })
            else:
                stats["no_result"] += 1
                print(f"  ❌ No entity found")
                
        except Exception as e:
            stats["error"] += 1
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("ENTITY MoE SUMMARY")
    print(f"{'='*70}")
    print(f"  Total tested:        {stats['total']}")
    print(f"  Correct:             {stats['correct']} ({100*stats['correct']/max(1,stats['total']):.1f}%)")
    print(f"  Wrong:               {stats['wrong']}")
    print(f"  No result:           {stats['no_result']}")
    print(f"  Errors:              {stats['error']}")
    
    if results:
        avg_time = sum(r["processing_time_ms"] for r in results) / len(results)
        avg_agreement = sum(r["expert_agreement"] for r in results) / len(results)
        print(f"  Avg processing time: {avg_time:.0f}ms")
        print(f"  Avg expert agreement: {avg_agreement:.1f}")
    
    return results


def test_context_moe(samples):
    """Test Context Extraction MoE on samples."""
    print(f"\n{'='*70}")
    print("TEST 2: Context Extraction MoE")
    print(f"{'='*70}")
    
    try:
        from context_extraction_moe import ContextExtractionMoE
        
        moe = ContextExtractionMoE(
            stanza_nlp=NLP,
            enable_embedding=False,  # no model in test
            parallel=True,
        )
        print(f"  MoE created (embedding disabled for offline test)")
        
    except ImportError as e:
        print(f"  Cannot import MoE: {e}")
        return
    
    results = []
    stats = Counter()
    token_utils = []
    
    for i, sample in enumerate(samples):
        art_id = sample.get("raw_text_id", f"sample_{i}")
        entity_name = sample.get("entity_name", "")
        article = sample.get("article_text", "")
        
        offset = article.lower().find(entity_name.lower())
        if offset < 0:
            continue
        
        print(f"\n  --- Sample {i+1}/{len(samples)} ---")
        print(f"  Entity: {entity_name}")
        
        try:
            t0 = time.time()
            result = moe.extract(article, entity_name, offset)
            elapsed = time.time() - t0
            
            ctx_text = result.get("context_text", "")
            ctx_len = len(ctx_text)
            est_tokens = int(ctx_len / 3.5) + 7 + 3
            utilization = 100 * est_tokens / 256
            token_utils.append(utilization)
            
            stats["total"] += 1
            stats["quality_scores"] = result.get("quality_score", 0)
            
            print(f"  Quality: {result['quality_score']}")
            print(f"  Span count: {result['span_count']}")
            print(f"  Expert agreement: {result['expert_agreement']}")
            print(f"  Token utilization: {utilization:.1f}% ({est_tokens}/256)")
            print(f"  Time: {elapsed*1000:.0f}ms")
            print(f"  Context preview: {ctx_text[:150]}...")
            
            results.append({
                "sample_id": i,
                "entity": entity_name,
                "context_chars": ctx_len,
                "est_tokens": est_tokens,
                "token_utilization": utilization,
                "quality_score": result["quality_score"],
                "span_count": result["span_count"],
                "expert_agreement": result["expert_agreement"],
                "processing_time_ms": result["processing_time_ms"],
                "experts_used": result["experts_used"],
            })
            
        except Exception as e:
            stats["error"] += 1
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("CONTEXT MoE SUMMARY")
    print(f"{'='*70}")
    print(f"  Total tested:        {stats['total']}")
    print(f"  Errors:              {stats['error']}")
    
    if token_utils:
        avg_util = sum(token_utils) / len(token_utils)
        print(f"  Avg token utilization: {avg_util:.1f}%")
        print(f"  Min/Max utilization: {min(token_utils):.1f}% / {max(token_utils):.1f}%")
    
    if results:
        avg_quality = sum(r["quality_score"] for r in results) / len(results)
        avg_time = sum(r["processing_time_ms"] for r in results) / len(results)
        avg_agreement = sum(r["expert_agreement"] for r in results) / len(results)
        print(f"  Avg quality score: {avg_quality:.1f}")
        print(f"  Avg processing time: {avg_time:.0f}ms")
        print(f"  Avg expert agreement: {avg_agreement:.1f}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--entity-only", action="store_true")
    parser.add_argument("--context-only", action="store_true")
    args = parser.parse_args()
    
    n = args.samples
    samples_subset = samples[:n]
    
    if not args.context_only:
        entity_results = test_entity_moe(samples_subset)
    
    if not args.entity_only:
        context_results = test_context_moe(samples_subset)
    
    print(f"\n{'='*70}")
    print("MoE TEST COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
