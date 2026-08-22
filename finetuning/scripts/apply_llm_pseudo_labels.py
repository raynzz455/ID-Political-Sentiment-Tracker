#!/usr/bin/env python3
"""
apply_llm_pseudo_labels.py
=========================
Apply LLM-verified pseudo-labels to the merged dataset, producing a final
dataset with gold labels for fine-tuning.

Input: dataset_merged_final.jsonl (2459 rows) + llm_verified_pseudo.jsonl
Output: dataset_v10_final.jsonl (2459 rows with LLM labels applied)
"""
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATASET_IN   = BASE / "datasets" / "dataset_merged_final.jsonl"
LLM_VERIFIED = BASE / "datasets" / "llm_verified_pseudo.jsonl"
OUTPUT       = BASE / "datasets" / "dataset_v10_final.jsonl"


def load_jsonl(path):
    if not path.exists(): return []
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    print("=" * 64)
    print("APPLY LLM-VERIFIED PSEUDO-LABELS TO MERGED DATASET")
    print("=" * 64)
    dataset = load_jsonl(DATASET_IN)
    print(f"Loaded dataset: {len(dataset)} rows")
    verified = load_jsonl(LLM_VERIFIED)
    print(f"Loaded LLM verified: {len(verified)} rows")

    vmap = {}
    for v in verified:
        key = (v.get("raw_text_id", ""), v.get("entity_name", ""))
        vmap[key] = v
    print(f"Unique verified keys: {len(vmap)}")

    applied = 0; flipped = 0
    for r in dataset:
        key = (r.get("raw_text_id", ""), r.get("entity_name", ""))
        if key in vmap:
            v = vmap[key]
            old_label = r.get("gold_label") or r.get("pseudo_label", "neutral")
            new_label = v["gold_label"]
            r["gold_label"] = new_label
            r["gold_relevancy"] = v.get("gold_relevancy", "relevant")
            r["entity_is_main_subject"] = v.get("entity_is_main_subject", True)
            r["verification_reasoning"] = v.get("reasoning", "")
            r["label_source"] = v.get("label_source", "llm_verified")
            r["label_confidence"] = v.get("label_confidence", 0.85)
            if old_label != new_label: flipped += 1
            applied += 1

    print(f"\nApplied: {applied} labels")
    print(f"Flipped: {flipped}")

    with open(OUTPUT, "w") as f:
        for r in dataset: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Written: {OUTPUT} ({len(dataset)} rows)")

    labels = Counter(r.get("gold_label") or r.get("pseudo_label", "neutral") for r in dataset)
    sources = Counter(r.get("label_source", "unknown") for r in dataset)
    print(f"\nLabel distribution:")
    for k, v in sorted(labels.items(), key=lambda x: -x[1]):
        print(f"  {k:15s}: {v:5d} ({v/len(dataset)*100:5.1f}%)")
    print(f"\nLabel sources:")
    for k, v in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {k:25s}: {v:5d}")
    verified_count = sum(1 for r in dataset if r.get("label_source", "").startswith("llm"))
    pseudo_count = sum(1 for r in dataset if r.get("label_source") in ("v6_pseudo", "v6_export"))
    print(f"\nVerified (LLM): {verified_count}")
    print(f"Pseudo only:    {pseudo_count}")


if __name__ == "__main__":
    main()
