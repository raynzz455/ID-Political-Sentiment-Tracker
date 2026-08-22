#!/usr/bin/env python3
"""
apply_reverified_labels.py
=========================
Apply re-verified labels to gold standard + remove entity-not-main-subject rows.

Input: dataset_gold_standard.jsonl + reverified_labels.jsonl
Output: dataset_gold_standard_final.jsonl (production-ready)
"""
import json, sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "datasets" / "dataset_gold_standard.jsonl"
REVERIFIED = BASE / "datasets" / "reverified_labels.jsonl"
OUTPUT = BASE / "datasets" / "dataset_gold_standard_final.jsonl"


def main():
    if not DATASET.exists():
        print(f"ERROR: {DATASET} not found"); sys.exit(1)
    rows = [json.loads(l) for l in open(DATASET) if l.strip()]
    print(f"Input: {len(rows)} rows")

    reverified = {}
    if REVERIFIED.exists():
        for l in open(REVERIFIED):
            if l.strip():
                r = json.loads(l)
                reverified[r['row_index']] = r
    print(f"Re-verified labels: {len(reverified)}")

    gold = []
    removed = {"not_main_subject": 0}
    label_changes = Counter()

    for i, r in enumerate(rows):
        if i in reverified:
            rv = reverified[i]
            if not rv.get('is_main_subject', True):
                removed["not_main_subject"] += 1
                continue
            old_label = r['label']
            new_label = rv['gold_label']
            if old_label != new_label:
                label_changes[(old_label, new_label)] += 1
            r['label'] = new_label
            if rv.get('reverify_source') != 'reverify_failed':
                r['label_confidence'] = rv.get('reverify_confidence', 0.9)
                r['label_source'] = rv.get('reverify_source', 'llm_reverified')
                r['verification_reasoning'] = rv.get('reasoning', '')
            r['is_main_subject'] = True
        gold.append(r)

    with open(OUTPUT, "w") as f:
        for r in gold:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    labels = Counter(r['label'] for r in gold)
    sources = Counter(r.get('label_source', 'unknown') for r in gold)
    rels = Counter(r.get('gold_relevancy', 'unknown') for r in gold)

    print(f"\n{'='*64}")
    print(f"GOLD STANDARD FINAL")
    print(f"{'='*64}")
    print(f"Input:    {len(rows)} rows")
    print(f"Removed:  {sum(removed.values())} (entity NOT main subject)")
    print(f"Output:   {len(gold)} rows ({len(gold)/len(rows)*100:.1f}% kept)")
    print(f"\nLABEL CHANGES APPLIED:")
    for (o, n), c in label_changes.most_common():
        print(f"  {o:10s} -> {n:10s}: {c}")
    print(f"  Total flipped: {sum(label_changes.values())}")
    print(f"\nFINAL LABEL DISTRIBUTION:")
    for k, v in sorted(labels.items(), key=lambda x: -x[1]):
        print(f"  {k:10s}: {v:5d} ({v/len(gold)*100:5.1f}%)")
    imbalance = max(labels.values()) / min(labels.values())
    print(f"  Imbalance: {imbalance:.1f}x")
    print(f"\nLABEL SOURCES:")
    for k, v in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {k:25s}: {v:5d}")
    verified_count = sum(1 for r in gold if r.get('label_source', '').startswith('llm'))
    print(f"\nLLM-verified: {verified_count} ({verified_count/len(gold)*100:.1f}%)")
    high_conf = sum(1 for r in gold if r.get('label_confidence', 0.5) >= 0.85)
    print(f"High confidence (≥0.85): {high_conf} ({high_conf/len(gold)*100:.1f}%)")
    print(f"\nOutput: {OUTPUT} ({len(gold)} rows)")


if __name__ == "__main__":
    main()
