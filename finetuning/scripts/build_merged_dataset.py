#!/usr/bin/env python3
"""
build_merged_dataset.py
=======================
Merge V6 + V9 + V3 datasets + apply LLM verified labels.
"""
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# Input files
V6_PATH = BASE / "devtools" / "dataset" / "V6" / "finetune_dataset_ai.jsonl"
V9_PATH = BASE / "finetuning" / "datasets" / "dataset_v9.jsonl"
V3_PATH = BASE / "finetuning" / "datasets" / "dataset_v3.jsonl"
LLM_VERIFIED = BASE / "finetuning" / "datasets" / "llm_verified_merged.jsonl"

OUTPUT = BASE / "finetuning" / "datasets" / "dataset_merged_final.jsonl"


def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    print("=" * 60)
    print("MERGE DATASETS + APPLY LLM LABELS")
    print("=" * 60)

    # Load all datasets
    v6 = load_jsonl(V6_PATH)
    v9 = load_jsonl(V9_PATH)
    v3 = load_jsonl(V3_PATH)
    print(f"V6: {len(v6)} rows")
    print(f"V9: {len(v9)} rows")
    print(f"V3: {len(v3)} rows")

    # Load LLM verified labels
    llm_labels = {}
    if LLM_VERIFIED.exists():
        for l in open(LLM_VERIFIED):
            if not l.strip(): continue
            v = json.loads(l)
            key = (v.get("raw_text_id", ""), v.get("entity_name", ""))
            llm_labels[key] = v
    print(f"LLM verified: {len(llm_labels)} labels")

    # Merge: dedup by (raw_text_id, entity_name)
    merged = {}
    
    # Add V6 first (newest, from production export)
    for r in v6:
        key = (r.get("raw_text_id", ""), r.get("entity_name", ""))
        if key not in merged:
            merged[key] = {
                "raw_text_id": r.get("raw_text_id", ""),
                "entity_name": r.get("entity_name", ""),
                "context_text": r.get("context_text", ""),
                "article_text": r.get("article_text", "")[:1500],
                "source_url": r.get("source_url", ""),
                "pseudo_label": r.get("pseudo_label", "neutral"),
                "gold_label": "",
                "label_source": "v6_export",
                "label_confidence": 0.5,
            }

    # Add V9 (has gold_label from LLM verify)
    for r in v9:
        key = (r.get("raw_text_id", ""), r.get("entity_name", ""))
        if key not in merged:
            merged[key] = {
                "raw_text_id": r.get("raw_text_id", ""),
                "entity_name": r.get("entity_name", ""),
                "context_text": r.get("context_text", ""),
                "article_text": r.get("article_text", "")[:1500],
                "source_url": r.get("source_url", ""),
                "pseudo_label": r.get("gold_label", r.get("pseudo_label", "neutral")),
                "gold_label": r.get("gold_label", ""),
                "label_source": r.get("label_source", "v9"),
                "label_confidence": r.get("label_confidence", 0.5),
            }

    # Add V3
    for r in v3:
        key = (r.get("raw_text_id", ""), r.get("entity_name", ""))
        if key not in merged:
            merged[key] = {
                "raw_text_id": r.get("raw_text_id", ""),
                "entity_name": r.get("entity_name", ""),
                "context_text": r.get("context_text", ""),
                "article_text": r.get("article_text", "")[:1500],
                "source_url": r.get("source_url", ""),
                "pseudo_label": r.get("gold_label", r.get("pseudo_label", "neutral")),
                "gold_label": r.get("gold_label", ""),
                "label_source": r.get("label_source", "v3"),
                "label_confidence": r.get("label_confidence", 0.5),
            }

    # Apply LLM verified labels
    llm_applied = 0
    for key, r in merged.items():
        if key in llm_labels:
            v = llm_labels[key]
            r["gold_label"] = v["gold_label"]
            r["gold_relevancy"] = v.get("gold_relevancy", "relevant")
            r["label_source"] = "llm_verified_merged"
            r["label_confidence"] = 0.85
            r["verification_reasoning"] = v.get("reasoning", "")
            llm_applied += 1

    final = list(merged.values())
    print(f"\nMerged: {len(final)} unique rows")
    print(f"LLM labels applied: {llm_applied}")

    # Stats
    has_gold = sum(1 for r in final if r["gold_label"])
    no_gold = len(final) - has_gold
    print(f"Has gold label: {has_gold}")
    print(f"Need labeling: {no_gold}")

    # Label distribution
    label_field = "gold_label" if has_gold > 0 else "pseudo_label"
    labs = Counter(r.get(label_field) or r.get("pseudo_label", "neutral") for r in final)
    print(f"\nLabel distribution:")
    for l in ["positive", "neutral", "negative"]:
        c = labs.get(l, 0)
        print(f"  {l:10s}: {c:4d} ({100*c/len(final):.1f}%)")

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved: {OUTPUT} ({len(final)} rows)")


if __name__ == "__main__":
    main()
