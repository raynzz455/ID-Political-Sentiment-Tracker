#!/usr/bin/env python3.13
"""
build_dataset_v9.py
===================
Build FINAL clean dataset v9 — comprehensive preprocessing + dedup + merge.

Pipeline (7 tahap):
  1. LOAD semua datasets (v1, v2, v3, v5, v6, v7, enhanced) — ~6000 raw rows
  2. DEDUP by (raw_text_id, entity_name) — keep latest/most-verified version
  3. PREPROCESS:
     a. Strip bylines (JAKARTA-, KOMPAS.com-, etc.)
     b. Normalize whitespace
     c. Truncate article_text to 1500 chars
     d. Remove rows with empty context (<10 chars)
     e. Detect & flag corruption_stitch (context not in article)
  4. EXCLUDE bad flags (corruption_stitch, wrong_entity, background_only, llm_failed)
  5. EXCLUDE not_relevant (gold_relevancy == 'not_relevant')
  6. MERGE LLM-verified labels from v3 (llm_verified_v3.jsonl) + v8 (llm_verified_v8.jsonl)
  7. SAVE:
     - dataset_v9.jsonl           (final, all unique verified rows)
     - need_llm_verify_v9.json    (rows still needing verification)
     - dataset_v9_report.json     (statistics)

Output target: ≥1500 rows, 100% verified (or marked for LLM verify).
"""
import json
import re
import random
import hashlib
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE / "datasets"
SCRIPTS_DIR = BASE / "scripts"

# Input datasets (urut prioritas — paling baru/akurat dulu)
INPUT_FILES = [
    "dataset_v3.jsonl",         # 777 rows, 100% LLM-verified, conf 0.85 (HIGHEST PRIORITY)
    "dataset_v7.jsonl",         # 1101 rows, 100% LLM-verified, conf 0.55-0.69
    "dataset_v6.jsonl",         # 1280 rows, mixed
    "dataset_v5.jsonl",         # 1347 rows, mixed
    "dataset_enhanced.jsonl",   # 909 rows, mixed
    "dataset_v2.jsonl",         # 777 rows, superset v3
    "dataset_v1.jsonl",         # 909 rows, original
]

# LLM-verified label files (apply after merge)
VERIFIED_FILES = [
    "llm_verified_v3.jsonl",     # 149 labels (Task 17-19)
    "llm_verified_v8.jsonl",     # ongoing verification
]

OUTPUT_V9 = DATASETS_DIR / "dataset_v9.jsonl"
OUTPUT_NEED_VERIFY = DATASETS_DIR / "need_llm_verify_v9.json"
OUTPUT_REPORT = BASE / "dataset_v9_report.json"

TRUSTED_SOURCES = {"llm_verified", "llm_verified_v8", "llm_second_pass", "gold_human"}
BAD_FLAGS = {"corruption_stitch", "wrong_entity", "background_only", "llm_failed"}

# Byline patterns to strip from article_text
BYLINE_PATTERNS = [
    re.compile(r'^[A-Z]{4,}[-—–]\s*[A-Za-z\s.]+[-—–]\s*', re.MULTILINE),  # JAKARTA-News-
    re.compile(r'^[A-Z]{4,}[-—–]', re.MULTILINE),  # JAKARTA-
    re.compile(r'^(KOMPAS|TEMPO|CNN|DETIK|TRIBUN|ANTARA)[\w.]*\s*[-—–,.]\s*', re.MULTILINE),
    re.compile(r'^[A-Z]{2,20}\s*[-—–]\s*[A-Z][a-z]+\s*[-—–]\s*', re.MULTILINE),
]


def make_key(r):
    """Unique key: (raw_text_id, entity_name)."""
    return (r.get("raw_text_id", ""), r.get("entity_name", ""))


def make_context_hash(r):
    """Hash context_text for additional dedup."""
    ctx = (r.get("context_text") or "").strip().lower()
    return hashlib.md5(ctx.encode()).hexdigest()[:16]


def strip_bylines(text):
    """Remove news bylines from article text."""
    if not text:
        return text
    cleaned = text
    for pat in BYLINE_PATTERNS:
        cleaned = pat.sub("", cleaned, count=1)
    return cleaned.strip()


def normalize_whitespace(text):
    """Normalize whitespace to single spaces."""
    if not text:
        return text
    return re.sub(r'\s+', ' ', text).strip()


def is_corruption_stitch(context, article):
    """Detect if context is stitched/corrupted (not substring of article)."""
    if not context or not article:
        return False
    ctx_clean = normalize_whitespace(context).lower()
    art_clean = normalize_whitespace(article).lower()
    # Allow partial match (first 100 chars)
    if len(ctx_clean) > 100:
        return ctx_clean[:100] not in art_clean
    return ctx_clean not in art_clean


def preprocess_row(r):
    """Apply all preprocessing to a single row."""
    r = dict(r)
    # Strip bylines from article
    if r.get("article_text"):
        r["article_text"] = strip_bylines(r["article_text"])
        r["article_text"] = normalize_whitespace(r["article_text"])[:1500]
    # Normalize context
    if r.get("context_text"):
        r["context_text"] = normalize_whitespace(r["context_text"])
    # Detect corruption
    if not r.get("context_flag"):
        r["context_flag"] = "corruption_stitch" if is_corruption_stitch(
            r.get("context_text", ""), r.get("article_text", "")
        ) else "clean"
    return r


def main():
    print("=" * 70)
    print("BUILD DATASET v9 — comprehensive preprocessing + dedup")
    print("=" * 70)

    # ===== Step 1: Load all datasets =====
    print("\n[1/7] Loading datasets...")
    all_rows = {}  # key -> (row, source_priority)
    source_priority = {name: i for i, name in enumerate(INPUT_FILES)}
    stats = Counter()

    for fname in INPUT_FILES:
        path = DATASETS_DIR / fname
        if not path.exists():
            print(f"  skip (not found): {fname}")
            continue
        rows = [json.loads(l) for l in open(path) if l.strip()]
        added = 0
        for r in rows:
            k = make_key(r)
            priority = source_priority.get(fname, 99)
            if k not in all_rows or priority < all_rows[k][1]:
                # Preprocess
                r = preprocess_row(r)
                r["source_dataset"] = fname.replace(".jsonl", "")
                all_rows[k] = (r, priority)
                added += 1
        stats[fname] = added
        print(f"  {fname:30s}: {len(rows):4d} rows, +{added:4d} new/better")

    merged = {k: v[0] for k, v in all_rows.items()}
    print(f"\nTotal unique (raw_text_id, entity_name) pairs: {len(merged)}")

    # ===== Step 2: Apply LLM-verified labels =====
    print("\n[2/7] Applying LLM-verified labels...")
    verified_applied = 0
    for vfname in VERIFIED_FILES:
        vpath = BASE / vfname
        if not vpath.exists():
            print(f"  skip (not found): {vfname}")
            continue
        vrows = [json.loads(l) for l in open(vpath) if l.strip()]
        applied = 0
        for v in vrows:
            # v3 format: base_row_index; v8 format: raw_text_id + entity_name
            if "base_row_index" in v:
                # v3 format — match by row_index base
                key = None
                for k, r in merged.items():
                    ri = str(r.get("row_index", ""))
                    if ri and (ri == v["base_row_index"] or ri.startswith(v["base_row_index"] + "_aug_")):
                        if r.get("entity_name") == v.get("entity_name"):
                            key = k
                            break
            else:
                # v8 format
                key = (v.get("raw_text_id", ""), v.get("entity_name", ""))
            if key and key in merged:
                r = merged[key]
                r["prev_label"] = r.get("gold_label")
                r["prev_label_source"] = r.get("label_source")
                r["prev_label_confidence"] = r.get("label_confidence")
                r["gold_label"] = v["gold_label"]
                r["gold_relevancy"] = v.get("gold_relevancy", "relevant")
                r["label_source"] = v.get("label_source", "llm_verified")
                r["label_confidence"] = v.get("label_confidence", 0.85)
                r["verification_reasoning"] = v.get("reasoning", "")
                applied += 1
        verified_applied += applied
        print(f"  {vfname:30s}: {len(vrows):4d} verified labels, {applied:4d} applied")

    # ===== Step 3: Filter bad flags + not_relevant =====
    print("\n[3/7] Filtering bad flags + not_relevant...")
    excluded = Counter()
    final_rows = []
    for k, r in merged.items():
        flag = r.get("context_flag", "clean")
        if flag in BAD_FLAGS:
            excluded[flag] += 1
            continue
        if r.get("gold_relevancy") == "not_relevant":
            excluded["not_relevant"] += 1
            continue
        # Skip rows with empty context
        if len((r.get("context_text") or "").strip()) < 10:
            excluded["empty_context"] += 1
            continue
        final_rows.append(r)
    print(f"  Excluded: {dict(excluded)}")
    print(f"  Remaining: {len(final_rows)} rows")

    # ===== Step 4: Verify verification status =====
    print("\n[4/7] Checking verification status...")
    verified = 0
    need_verify = 0
    for r in final_rows:
        src = r.get("label_source", "")
        conf = r.get("label_confidence", 0)
        if src in TRUSTED_SOURCES or conf >= 0.7:
            verified += 1
        else:
            need_verify += 1
            r["needs_llm_verify"] = True
    print(f"  Already verified: {verified}")
    print(f"  Need LLM verify:  {need_verify}")

    # ===== Step 5: Dedup by context hash (remove near-duplicates) =====
    print("\n[5/7] Dedup by context hash...")
    seen_hashes = {}
    dedup_rows = []
    dup_count = 0
    for r in final_rows:
        h = make_context_hash(r)
        if h in seen_hashes:
            # Keep the one with higher confidence
            existing = seen_hashes[h]
            if r.get("label_confidence", 0) > existing.get("label_confidence", 0):
                # Replace
                dedup_rows = [r if (make_context_hash(x) == h) else x for x in dedup_rows]
                seen_hashes[h] = r
            dup_count += 1
        else:
            seen_hashes[h] = r
            dedup_rows.append(r)
    print(f"  Removed {dup_count} duplicate contexts")
    print(f"  After dedup: {len(dedup_rows)} rows")
    final_rows = dedup_rows

    # ===== Step 6: Save need-verify list =====
    need_verify_list = [
        {
            "raw_text_id": r["raw_text_id"],
            "entity_name": r["entity_name"],
            "context_text": r["context_text"],
            "pseudo_label": r.get("pseudo_label", ""),
            "current_label": r.get("gold_label", ""),
            "current_source": r.get("label_source", ""),
            "current_confidence": r.get("label_confidence", 0),
        }
        for r in final_rows if r.get("needs_llm_verify")
    ]
    with open(OUTPUT_NEED_VERIFY, "w") as f:
        json.dump(need_verify_list, f, ensure_ascii=False, indent=2)
    print(f"\n[6/7] Need-verify list: {OUTPUT_NEED_VERIFY} ({len(need_verify_list)} rows)")

    # ===== Step 7: Save final dataset =====
    random.seed(42)
    random.shuffle(final_rows)
    with open(OUTPUT_V9, "w") as f:
        for r in final_rows:
            # Remove transient fields
            r.pop("needs_llm_verify", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ===== Stats =====
    labels = Counter(r["gold_label"] for r in final_rows)
    sources = Counter(r.get("label_source", "?") for r in final_rows)
    conf_buckets = Counter()
    for r in final_rows:
        c = r.get("label_confidence", 0)
        if c >= 0.85: conf_buckets[">=0.85 (LLM/gold)"] += 1
        elif c >= 0.7: conf_buckets["0.70-0.84 (trusted)"] += 1
        elif c >= 0.55: conf_buckets["0.55-0.69 (low)"] += 1
        else: conf_buckets["<0.55 (unverified)"] += 1

    report = {
        "output": str(OUTPUT_V9),
        "total_rows": len(final_rows),
        "verified_rows": verified,
        "need_verify_rows": need_verify,
        "label_distribution": dict(labels),
        "label_source_distribution": dict(sources),
        "confidence_buckets": dict(conf_buckets),
        "excluded": dict(excluded),
        "duplicates_removed": dup_count,
    }
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[7/7] Final dataset: {OUTPUT_V9}")
    print(f"{'='*70}")
    print(f"DATASET v9 FINAL")
    print(f"{'='*70}")
    print(f"Total rows:    {len(final_rows)}")
    print(f"Verified:      {verified} ({100*verified/max(1,len(final_rows)):.1f}%)")
    print(f"Need verify:   {need_verify} ({100*need_verify/max(1,len(final_rows)):.1f}%)")
    print(f"\nLabel distribution:")
    for l in ["positive", "neutral", "negative"]:
        v = labels.get(l, 0)
        print(f"  {l:10s}: {v:4d} ({100*v/max(1,len(final_rows)):.1f}%)")
    print(f"\nLabel source:")
    for k, v in sources.most_common():
        print(f"  {k:30s}: {v:4d} ({100*v/max(1,len(final_rows)):.1f}%)")
    print(f"\nConfidence:")
    for k, v in conf_buckets.most_common():
        print(f"  {k:30s}: {v:4d} ({100*v/max(1,len(final_rows)):.1f}%)")
    print(f"\nReport: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
