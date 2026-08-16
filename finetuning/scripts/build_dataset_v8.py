#!/usr/bin/env python3.13
"""
build_dataset_v8.py
===================
Build final finetuning dataset v8 — target ≥1500 rows.

Strategi:
  1. MERGE semua unique rows dari dataset v3, v5, v6, v7, enhanced, v2 → ~1405 unique
  2. UPGRADE overlap rows (v3 ⊂ v7) — pakai label v3 (lebih baru, higher conf 0.85)
  3. UNVERIFIED rows → flag untuk LLM verify (788 rows)
  4. OVERSAMPLE minor class (negative) untuk balance — target 1500 rows
  5. EXCLUDE bad flags (corruption_stitch, wrong_entity, llm_failed, background_only)
  6. FILTER sentiment-relevant only (exclude not_relevant from LLM verification)

Output:
  finetuning/datasets/dataset_v8_merged.jsonl   (1405 unique, sebelum oversample)
  finetuning/datasets/need_llm_verify_v8.json   (list baris yang perlu LLM verify)
  finetuning/datasets/dataset_v8.jsonl          (final, ≥1500 rows setelah oversample)
"""
import json
import random
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE / "datasets"

# Input datasets (urut dari yang paling baru/akurat)
INPUT_FILES = [
    "dataset_v3.jsonl",      # 777 rows, 100% verified, conf 0.85 (prioritas)
    "dataset_v7.jsonl",      # 1101 rows, 100% verified, conf 0.55-0.69
    "dataset_v6.jsonl",     # 1280 rows, mixed
    "dataset_v5.jsonl",     # 1347 rows, mixed
    "dataset_enhanced.jsonl",  # 909 rows, mixed
    "dataset_v2.jsonl",     # 777 rows, superset of v3
]

OUTPUT_MERGED = DATASETS_DIR / "dataset_v8_merged.jsonl"
OUTPUT_NEED_VERIFY = DATASETS_DIR / "need_llm_verify_v8.json"
OUTPUT_FINAL = DATASETS_DIR / "dataset_v8.jsonl"

random.seed(42)

# Label sources yang dianggap TRUSTED (tidak perlu re-verify)
TRUSTED_SOURCES = {"llm_verified", "llm_second_pass", "gold_human"}
# Bad flags yang di-exclude dari training
BAD_FLAGS = {"corruption_stitch", "wrong_entity", "background_only", "llm_failed"}


def make_key(r):
    """Unique key: (raw_text_id, entity_name)."""
    return (r.get("raw_text_id", ""), r.get("entity_name", ""))


def main():
    print("=" * 70)
    print("BUILD DATASET v8 — target ≥1500 rows")
    print("=" * 70)

    # Step 1: Load semua datasets dengan prioritas
    merged = {}  # key -> row
    source_counts = Counter()

    for fname in INPUT_FILES:
        path = DATASETS_DIR / fname
        if not path.exists():
            print(f"  skip (not found): {fname}")
            continue
        rows = [json.loads(l) for l in open(path) if l.strip()]
        added = 0
        for r in rows:
            k = make_key(r)
            if k not in merged:
                # Normalize: ensure minimum required fields
                r_normalized = {
                    "raw_text_id": r.get("raw_text_id", ""),
                    "entity_name": r.get("entity_name", ""),
                    "context_text": r.get("context_text", ""),
                    "article_text": r.get("article_text", ""),
                    "pseudo_label": r.get("pseudo_label", ""),
                    "gold_label": r.get("gold_label", r.get("pseudo_label", "neutral")),
                    "gold_relevancy": r.get("gold_relevancy", "relevant"),
                    "label_source": r.get("label_source", "unknown"),
                    "label_confidence": r.get("label_confidence", 0.3),
                    "context_flag": r.get("context_flag", "clean"),
                    "reasoning": r.get("reasoning", ""),
                    "premise": r.get("premise", r.get("entity_name", "")),
                    "hypothesis": r.get("hypothesis", r.get("context_text", "")),
                    "source_url": r.get("source_url", ""),
                    "source_dataset": fname.replace(".jsonl", ""),
                }
                merged[k] = r_normalized
                added += 1
                source_counts[fname] += 1
        print(f"  {fname:30s}: {len(rows):4d} rows, +{added:4d} new unique")

    print(f"\nTotal unique rows after merge: {len(merged)}")

    # Step 2: Analyze verification status
    verified = 0
    need_verify = 0
    bad_flag_count = 0
    not_relevant = 0
    final_rows = []

    for k, r in merged.items():
        # Exclude bad flags
        if r.get("context_flag") in BAD_FLAGS:
            bad_flag_count += 1
            continue
        # Exclude not_relevant
        if r.get("gold_relevancy") == "not_relevant":
            not_relevant += 1
            continue
        # Check verification status
        src = r.get("label_source", "")
        conf = r.get("label_confidence", 0)
        if src in TRUSTED_SOURCES or conf >= 0.7:
            verified += 1
            final_rows.append(r)
        else:
            need_verify += 1
            # Mark for LLM verify
            r["needs_llm_verify"] = True
            final_rows.append(r)

    print(f"\nAfter exclude bad/not_relevant:")
    print(f"  Excluded bad flags:     {bad_flag_count}")
    print(f"  Excluded not_relevant:  {not_relevant}")
    print(f"  Remaining:              {len(final_rows)}")
    print(f"    Already verified:     {verified}")
    print(f"    Need LLM verify:      {need_verify}")

    # Step 3: Save merged dataset
    with open(OUTPUT_MERGED, "w") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nMerged dataset saved: {OUTPUT_MERGED} ({len(final_rows)} rows)")

    # Step 4: Save list of rows needing LLM verify
    need_verify_rows = [
        {
            "raw_text_id": r["raw_text_id"],
            "entity_name": r["entity_name"],
            "context_text": r["context_text"],
            "pseudo_label": r.get("pseudo_label", ""),
            "current_label": r["gold_label"],
            "current_source": r["label_source"],
            "current_confidence": r["label_confidence"],
        }
        for r in final_rows if r.get("needs_llm_verify")
    ]
    with open(OUTPUT_NEED_VERIFY, "w") as f:
        json.dump(need_verify_rows, f, ensure_ascii=False, indent=2)
    print(f"Need-verify list saved: {OUTPUT_NEED_VERIFY} ({len(need_verify_rows)} rows)")

    # Step 5: Class distribution before oversampling
    labels_before = Counter(r["gold_label"] for r in final_rows)
    print(f"\nClass distribution (before oversample):")
    for l in ["positive", "neutral", "negative"]:
        v = labels_before.get(l, 0)
        print(f"  {l:10s}: {v:4d} ({100*v/max(1,len(final_rows)):.1f}%)")

    # Step 6: Oversample to ≥1500 rows
    TARGET = 1500
    if len(final_rows) < TARGET:
        deficit = TARGET - len(final_rows)
        # Oversample minority classes (positive + negative) to balance + fill deficit
        minor_classes = ["positive", "negative"]
        neutral_count = labels_before.get("neutral", 0)
        # Target: make positive = negative = neutral/2, then fill rest
        target_minor = max(neutral_count // 2, labels_before.get("positive", 0), labels_before.get("negative", 0))

        added = 0
        for label in minor_classes:
            class_rows = [r for r in final_rows if r["gold_label"] == label]
            if not class_rows:
                continue
            current = len(class_rows)
            needed = max(0, target_minor - current)
            for i in range(needed):
                if added >= deficit:
                    break
                src_row = class_rows[i % len(class_rows)]
                aug = dict(src_row)
                aug["row_index"] = f"{src_row['raw_text_id'][:8]}_aug_v8_{i}"
                aug["label_source"] = "oversampled_v8"
                aug["label_confidence"] = src_row.get("label_confidence", 0.5) * 0.95
                final_rows.append(aug)
                added += 1
            if added >= deficit:
                break

        # If still short, duplicate neutral too
        if len(final_rows) < TARGET:
            deficit2 = TARGET - len(final_rows)
            neutral_rows = [r for r in final_rows if r["gold_label"] == "neutral"]
            for i in range(deficit2):
                src_row = neutral_rows[i % len(neutral_rows)]
                aug = dict(src_row)
                aug["row_index"] = f"{src_row['raw_text_id'][:8]}_aug_neu_{i}"
                aug["label_source"] = "oversampled_v8"
                aug["label_confidence"] = src_row.get("label_confidence", 0.5) * 0.95
                final_rows.append(aug)

    # Shuffle
    random.shuffle(final_rows)

    # Save final dataset
    with open(OUTPUT_FINAL, "w") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Final stats
    labels_after = Counter(r["gold_label"] for r in final_rows)
    sources_after = Counter(r.get("label_source", "?") for r in final_rows)
    conf_buckets = Counter()
    for r in final_rows:
        c = r.get("label_confidence", 0)
        if c >= 0.85: conf_buckets[">=0.85 (LLM/gold)"] += 1
        elif c >= 0.7: conf_buckets["0.70-0.84 (trusted)"] += 1
        elif c >= 0.55: conf_buckets["0.55-0.69 (low)"] += 1
        else: conf_buckets["<0.55 (unverified)"] += 1

    print(f"\n{'='*70}")
    print(f"DATASET v8 FINAL")
    print(f"{'='*70}")
    print(f"Total rows: {len(final_rows)}")
    print(f"\nLabel distribution:")
    for l in ["positive", "neutral", "negative"]:
        v = labels_after.get(l, 0)
        print(f"  {l:10s}: {v:4d} ({100*v/len(final_rows):.1f}%)")
    print(f"\nLabel source distribution:")
    for k, v in sources_after.most_common():
        print(f"  {k:30s}: {v:4d} ({100*v/len(final_rows):.1f}%)")
    print(f"\nConfidence buckets:")
    for k, v in conf_buckets.most_common():
        print(f"  {k:30s}: {v:4d} ({100*v/len(final_rows):.1f}%)")
    print(f"\nOutput: {OUTPUT_FINAL}")
    print(f"\nNext: run verify_dataset_v8.mjs to LLM-verify {need_verify} unverified rows")


if __name__ == "__main__":
    main()
