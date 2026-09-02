#!/usr/bin/env python3
"""
================================================================
GOOGLE COLAB RUNNER v2 — ID-Political-Sentiment-Tracker
================================================================
Run this in Google Colab (with GPU) to test the CURRENT production workers
(entity v15.1, context v19.1, nlp v16) on live Supabase data.

REPLACES: old colab_runner.py which used inline patches v14.2/v18.1
NOW: imports directly from packages/ (always uses latest versions)

INSTRUCTIONS (copy-paste to Colab):
1. Clone repo: !git clone https://github.com/raynzz455/ID-Political-Sentiment-Tracker.git
2. %cd /content/ID-Political-Sentiment-Tracker
3. Set your Supabase env vars (cell 2)
4. Run all cells
5. Download output JSON files for analysis

OUTPUT FILES:
  - colab_entity_results.json    (entity resolution results)
  - colab_context_results.json   (context worker results — token utilization)
  - colab_nlp_results.json       (optional — nlp worker predictions)

PREREQUISITES:
  - Colab GPU enabled (Runtime → Change runtime type → T4 GPU)
  - Supabase service role key
"""

# ================================================================
# CELL 1: Setup (clone repo + install deps)
# ================================================================
# !git clone https://github.com/raynzz455/ID-Political-Sentiment-Tracker.git
# %cd /content/ID-Political-Sentiment-Tracker
# !pip install stanza torch transformers supabase peft --quiet
# !python -c "import stanza; stanza.download('id', processors='tokenize,pos,lemma,depparse')"

# ================================================================
# CELL 2: Set environment variables
# ================================================================
import os
os.environ["SUPABASE_URL"] = "https://bawvxtivogcuwvqdqoae.supabase.co"
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "YOUR_SERVICE_ROLE_KEY_HERE"  # ← paste your key

# v3 Finetuned model (optional — for testing finetuned sentiment)
# Set USE_FINETUNED_V3=1 if you have finetuned LoRA adapter ready
os.environ["USE_FINETUNED_V3"] = "0"  # "1" to enable v3, "0" for base model
os.environ["SENTIMENT_LORA_PATH"] = ""  # path to lora adapter
os.environ["SENTIMENT_TEMPERATURE"] = "1.3"

# ================================================================
# CELL 3: Configuration
# ================================================================
NUM_ARTICLES = 100      # how many articles to test (recommend 100-200)
BATCH_SAVE_EVERY = 10   # save progress every N articles (crash recovery)
OUTPUT_DIR = "/content"  # Colab default

# ================================================================
# CELL 4: Import production workers (always latest versions)
# ================================================================
import sys, json, time, gc, logging
from collections import Counter
from pathlib import Path

# Add repo to path
ROOT_DIR = Path("/content/ID-Political-Sentiment-Tracker")
sys.path.insert(0, str(ROOT_DIR))

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("stanza").setLevel(logging.WARNING)

# Import workers (these are the CURRENT production versions)
print("Loading production workers...")
from packages.entity.entity_resolution_worker import (
    process_single_article_entity,
    load_caches as load_entity_caches,
    RESOLVER_VERSION as ENTITY_VERSION,
)
print(f"  entity_resolution_worker: {ENTITY_VERSION}")

from packages.context.context_worker import (
    process_single_article_context,
    CONTEXT_VERSION,
)
print(f"  context_worker: {CONTEXT_VERSION}")

print(f"\n✅ Workers loaded — using LATEST production versions")

# ================================================================
# CELL 5: Connect to Supabase + fetch articles
# ================================================================
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

print(f"Fetching {NUM_ARTICLES} processed articles from Supabase...")
res = sb.table("raw_texts").select(
    "id, title, text, source_url, status, resolver_version"
).eq("status", "processed").limit(NUM_ARTICLES).execute()
articles = res.data
print(f"Got {len(articles)} articles")

# Fetch existing entity mappings + contexts for comparison
art_ids = [a["id"] for a in articles]
res_ctx = sb.table("entity_contexts").select(
    "raw_text_id, entity_id, context_text, metadata"
).in_("raw_text_id", art_ids).execute()
res_map = sb.table("article_entity_map").select(
    "raw_text_id, entity_id, is_main_entity, confidence"
).in_("raw_text_id", art_ids).execute()

# Fetch ALL entities + aliases
res_ent = sb.table("political_entities").select(
    "id, canonical_name, aliases, entity_type, party_affiliation, position"
).execute()
ent_full = {e["id"]: {
    "name": e["canonical_name"],
    "aliases": e.get("aliases") or [],
    "entity_type": e.get("entity_type"),
    "party": e.get("party_affiliation"),
    "position": e.get("position"),
    "era": [],  # era column may not exist in DB — defensive
} for e in res_ent.data}
ent_map = {e["id"]: e["canonical_name"] for e in res_ent.data}
print(f"Entities: {len(ent_full)} | Contexts: {len(res_ctx.data)} | Mappings: {len(res_map.data)}")

# Build lookup for comparison
prod_main = {}
for m in res_map.data:
    if m.get("is_main_entity"):
        prod_main[m["raw_text_id"]] = ent_map.get(m["entity_id"], "?")
prod_ctx = {}
for c in res_ctx.data:
    prod_ctx[(c["raw_text_id"], c["entity_id"])] = c

# ================================================================
# CELL 6: Build entity caches (era + affiliation)
# ================================================================
print("Building entity caches (era + affiliation for v15 validation)...")
import re

entity_db_map = {}
alias_map = {}
id_to_name = {}
id_to_entity = {}
regex_patterns = []

for ent_id, info in ent_full.items():
    canon = info["name"]
    canon_lower = canon.lower()
    entity_db_map[canon_lower] = ent_id
    id_to_name[ent_id] = canon
    id_to_entity[ent_id] = {
        "name": canon,
        "aliases": info["aliases"],
        "entity_type": info.get("entity_type"),
        "party": info.get("party"),
        "position": info.get("position"),
        "era": info.get("era", []),
    }
    try:
        regex_patterns.append((re.compile(r'\b' + re.escape(canon) + r'\b', re.IGNORECASE), canon_lower))
    except re.error:
        pass
    for alias in info["aliases"]:
        if len(alias) < 2:
            continue
        alias_lower = alias.lower()
        alias_map[alias_lower] = canon
        try:
            regex_patterns.append((re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE), alias_lower))
        except re.error:
            pass

print(f"Loaded {len(regex_patterns)} patterns, {len(entity_db_map)} entities")

# ================================================================
# CELL 7: RUN PIPELINE (entity + context workers)
# ================================================================
entity_results = []
context_results = []
stats = Counter()
stats_token_util = []  # separate list for token utilization
t0 = time.time()

for i, a in enumerate(articles):
    art_id = a["id"]

    # Run entity resolution worker (v15.1)
    art = {
        "id": art_id,
        "title": a.get("title", ""),
        "text": a.get("text", ""),
        "metadata": {},
        "ingested_month": "2024-08",
    }

    try:
        ent_result = process_single_article_entity(
            art, alias_map, entity_db_map, id_to_name, id_to_entity, regex_patterns
        )
    except Exception as e:
        print(f"  [{i+1}] entity error: {e}")
        ent_result = None

    prod_main_name = prod_main.get(art_id)

    # Run context worker (v19.1) for main entity
    ctx_result = None
    main_name = None
    if ent_result and ent_result["mappings"]:
        main_ent_id = ent_result["mappings"][0]["entity_id"]
        main_name = id_to_name.get(main_ent_id, "?")

        # Find entity offset in text for context extraction
        text = a.get("text", "")
        offset = text.lower().find(main_name.lower())
        if offset >= 0:
            mentions_by_art = {art_id: [{
                "entity_id": main_ent_id,
                "start_offset": offset,
                "end_offset": offset + len(main_name),
                "political_entities": {"canonical_name": main_name},
            }]}
            try:
                ctx_results_list = process_single_article_context(art, mentions_by_art)
                if ctx_results_list:
                    ctx_result = ctx_results_list[0]
            except Exception as e:
                print(f"  [{i+1}] context error: {e}")

    # Record entity result
    if ent_result and ent_result["mappings"]:
        main_mapping = ent_result["mappings"][0]
        agree_with_prod = (main_name == prod_main_name) if prod_main_name else None

        stats["entity_found"] += 1
        if agree_with_prod:
            stats["agree"] += 1
        elif prod_main_name:
            stats["disagree"] += 1

        entity_results.append({
            "art_id": art_id,
            "title": (a.get("title") or "")[:80],
            "prod_main": prod_main_name,
            "v15_main": main_name,
            "v15_confidence": main_mapping["confidence"],
            "v15_resolver_source": main_mapping["resolver_source"],
            "n_entities": len(ent_result["mappings"]),
            "n_mentions": len(ent_result["mentions"]),
            "agree_with_prod": agree_with_prod,
        })
    else:
        stats["entity_not_found"] += 1
        entity_results.append({
            "art_id": art_id,
            "title": (a.get("title") or "")[:80],
            "prod_main": prod_main_name,
            "v15_main": None,
        })

    # Record context result
    if ctx_result:
        metadata = ctx_result.get("metadata", {})
        ctx_text = ctx_result.get("context_text", "")
        # Token estimation (Indonesian: ~3.5 chars/token)
        est_tokens = int(len(ctx_text) / 3.5) + 7 + 3  # context + entity + special
        utilization = 100 * est_tokens / 256
        stats_token_util.append(utilization)

        stats["context_extracted"] += 1

        context_results.append({
            "art_id": art_id,
            "entity": main_name if ent_result else "?",
            "context_chars": len(ctx_text),
            "est_tokens": est_tokens,
            "token_utilization_pct": round(utilization, 1),
            "quality_score": metadata.get("quality_score", 0),
            "has_sentiment": metadata.get("has_sentiment_predicate", False),
            "has_attribution": metadata.get("has_attribution", False),
            "is_main_actor": metadata.get("is_main_actor", False),
            "span_count": metadata.get("span_count", 1),
            "relevancy_score": metadata.get("relevancy_score", 1.0),
            "is_relevant": metadata.get("is_relevant", True),
            "context_preview": ctx_text[:200] + "..." if len(ctx_text) > 200 else ctx_text,
        })

    gc.collect()

    if (i+1) % BATCH_SAVE_EVERY == 0:
        elapsed = time.time() - t0
        rate = (i+1) / elapsed
        print(f"  [{i+1}/{len(articles)}] {elapsed:.0f}s, {rate:.1f} art/s | "
              f"found={stats['entity_found']} agree={stats['agree']} "
              f"disagree={stats['disagree']} ctx={stats['context_extracted']}", flush=True)

        # Incremental save
        with open(f"{OUTPUT_DIR}/colab_entity_results.json", "w") as f:
            json.dump({"stats": dict(stats), "results": entity_results,
                       "total_time": time.time()-t0, "completed": i+1,
                       "worker_versions": {"entity": ENTITY_VERSION, "context": CONTEXT_VERSION}},
                      f, indent=2, ensure_ascii=False)
        with open(f"{OUTPUT_DIR}/colab_context_results.json", "w") as f:
            json.dump({"results": context_results, "completed": i+1,
                       "token_util_avg": sum(stats_token_util)/max(1,len(stats_token_util)),
                       "worker_versions": {"entity": ENTITY_VERSION, "context": CONTEXT_VERSION}},
                      f, indent=2, ensure_ascii=False)

# Final save
total_time = time.time() - t0
with open(f"{OUTPUT_DIR}/colab_entity_results.json", "w") as f:
    json.dump({"stats": dict(stats), "results": entity_results,
               "total_time": total_time, "completed": len(articles),
               "worker_versions": {"entity": ENTITY_VERSION, "context": CONTEXT_VERSION}},
              f, indent=2, ensure_ascii=False)
with open(f"{OUTPUT_DIR}/colab_context_results.json", "w") as f:
    json.dump({"results": context_results, "completed": len(articles),
               "token_util_avg": sum(stats_token_util)/max(1,len(stats_token_util)),
               "worker_versions": {"entity": ENTITY_VERSION, "context": CONTEXT_VERSION}},
              f, indent=2, ensure_ascii=False)

# ================================================================
# CELL 8: Summary
# ================================================================
print(f"\n{'='*60}")
print(f"DONE! {len(articles)} articles in {total_time:.0f}s ({total_time/len(articles):.1f}s/art)")
print(f"{'='*60}")
print(f"\nWorker versions tested:")
print(f"  entity_resolution_worker: {ENTITY_VERSION}")
print(f"  context_worker: {CONTEXT_VERSION}")
print(f"\nEntity stats:")
print(f"  Found: {stats['entity_found']}/{len(articles)}")
print(f"  Agree with production: {stats['agree']}")
print(f"  Disagree with production: {stats['disagree']}")
print(f"  Not found: {stats['entity_not_found']}")
print(f"\nContext stats:")
print(f"  Extracted: {stats['context_extracted']}/{len(articles)}")
if stats_token_util:
    print(f"  Avg token utilization: {sum(stats_token_util)/len(stats_token_util):.1f}%")
    print(f"  Min/Max utilization: {min(stats_token_util):.1f}% / {max(stats_token_util):.1f}%")
print(f"\nOutput files:")
print(f"  {OUTPUT_DIR}/colab_entity_results.json")
print(f"  {OUTPUT_DIR}/colab_context_results.json")
print(f"\nDownload these files for analysis!")

# ================================================================
# CELL 9 (optional): Test NLP Worker v16 with finetuned v3 model
# ================================================================
# Uncomment below to test full end-to-end NLP pipeline
#
# from packages.nlp.sentiment_model import get_pipeline
# from packages.nlp.nlp_worker import run_inference_only
#
# print("Loading sentiment pipeline (v3 if enabled, else base)...")
# pipeline = get_pipeline()
# print(f"  LoRA enabled: {pipeline.sentiment.uses_lora}")
# print(f"  Temperature: T={pipeline.sentiment.temperature}")
#
# nlp_results = []
# for cr in context_results[:20]:  # test first 20
#     item = {"raw_text_id": cr["art_id"], "title": "", "text": cr["context_preview"]}
#     contexts = [{"entity_id": "test", "political_entities": {"canonical_name": cr["entity"]},
#                  "context_text": cr["context_preview"], "metadata": {}}]
#     result = run_inference_only(pipeline, item, contexts, Counter())
#     if result and result.get("targeted_payloads"):
#         p = result["targeted_payloads"][0]
#         nlp_results.append({
#             "art_id": cr["art_id"], "entity": cr["entity"],
#             "label": p["p_label"], "confidence": p["p_confidence"],
#             "scores": [p["p_neg"], p["p_neu"], p["p_pos"]],
#         })
#
# with open(f"{OUTPUT_DIR}/colab_nlp_results.json", "w") as f:
#     json.dump(nlp_results, f, indent=2, ensure_ascii=False)
# print(f"Saved {OUTPUT_DIR}/colab_nlp_results.json ({len(nlp_results)} predictions)")
