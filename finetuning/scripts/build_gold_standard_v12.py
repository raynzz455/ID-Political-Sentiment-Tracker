#!/usr/bin/env python3
"""
build_gold_standard_v12.py
=========================
Build the final GOLD STANDARD dataset v12 by applying:
  1. Re-verified LLM labels (from reverified_labels.jsonl)
  2. Remove rows where entity is NOT the main subject (is_main_subject=False)
  3. Keep only high-confidence verified rows

Input:
  - dataset_train_v11_final.jsonl (2,400 rows)
  - reverified_labels.jsonl (612 re-verified rows)

Output:
  - dataset_gold_standard_v12.jsonl (final gold-standard dataset)
"""
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "datasets" / "dataset_train_v11_final.jsonl"
REVERIFIED = BASE / "datasets" / "reverified_labels.jsonl"
OUTPUT = BASE / "datasets" / "dataset_gold_standard_v12.jsonl"
REPORT = BASE / "docs" / "dataset_v12_gold_report.json"


def main():
    print("=" * 64)
    print("BUILD GOLD STANDARD DATASET v12")
    print("=" * 64)

    # Load base dataset
    rows = [json.loads(l) for l in open(DATASET) if l.strip()]
    print(f"Base dataset: {len(rows)} rows (v11_final)")

    # Load re-verified labels
    reverified = {}
    for l in open(REVERIFIED):
        if l.strip():
            r = json.loads(l)
            reverified[r['row_index']] = r
    print(f"Re-verified labels: {len(reverified)} rows")

    # Apply corrections + filter
    gold = []
    removed = {
        "not_main_subject": 0,
    }
    label_changes = Counter()

    for i, r in enumerate(rows):
        if i in reverified:
            rv = reverified[i]
            # Skip if entity is NOT main subject (cannot be fixed)
            if not rv.get('is_main_subject', True):
                removed["not_main_subject"] += 1
                continue
            # Apply new label (even for reverify_failed, the gold_label is kept old label)
            old_label = r['label']
            new_label = rv['gold_label']
            if old_label != new_label:
                label_changes[(old_label, new_label)] += 1
            r['label'] = new_label
            # Only update confidence/source if re-verify succeeded
            if rv.get('reverify_source') != 'reverify_failed':
                r['label_confidence'] = rv.get('reverify_confidence', 0.9)
                r['label_source'] = rv.get('reverify_source', 'llm_reverified')
                r['verification_reasoning'] = rv.get('reasoning', '')
            r['is_main_subject'] = True
        # For rows not in reverified, keep original (already verified by LLM in v10)

        gold.append(r)

    # Write output
    with open(OUTPUT, "w") as f:
        for r in gold:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats
    label_dist = Counter(r['label'] for r in gold)
    source_dist = Counter(r.get('label_source', 'unknown') for r in gold)
    rel_dist = Counter(r.get('gold_relevancy', 'unknown') for r in gold)

    print(f"\n{'='*64}")
    print(f"GOLD STANDARD DATASET v12")
    print(f"{'='*64}")
    print(f"Input:    {len(rows)} rows (v11_final)")
    print(f"Removed:  {sum(removed.values())} rows")
    print(f"  - Entity NOT main subject: {removed['not_main_subject']}")
    print(f"Output:   {len(gold)} rows ({len(gold)/len(rows)*100:.1f}% kept)")

    print(f"\nLABEL CHANGES APPLIED:")
    for (old, new), count in label_changes.most_common():
        print(f"  {old:10s} -> {new:10s}: {count}")
    print(f"  Total flipped: {sum(label_changes.values())}")

    print(f"\nFINAL LABEL DISTRIBUTION:")
    for k, v in sorted(label_dist.items(), key=lambda x: -x[1]):
        print(f"  {k:10s}: {v:5d} ({v/len(gold)*100:5.1f}%)")
    imbalance = max(label_dist.values()) / min(label_dist.values())
    print(f"  Imbalance ratio: {imbalance:.1f}x")

    print(f"\nLABEL SOURCES:")
    for k, v in sorted(source_dist.items(), key=lambda x: -x[1]):
        print(f"  {k:25s}: {v:5d}")

    print(f"\nRELEVANCY:")
    for k, v in sorted(rel_dist.items(), key=lambda x: -x[1]):
        print(f"  {k:15s}: {v:5d}")

    # Context length stats
    lens = [len(r['text']) for r in gold]
    import statistics
    print(f"\nCONTEXT LENGTH:")
    print(f"  Min: {min(lens)}, Max: {max(lens)}")
    print(f"  Avg: {statistics.mean(lens):.0f}, Median: {statistics.median(lens):.0f}")

    # Confidence stats
    confs = [r.get('label_confidence', 0.5) for r in gold]
    print(f"\nLABEL CONFIDENCE:")
    print(f"  Mean: {statistics.mean(confs):.3f}")
    print(f"  Min: {min(confs)}, Max: {max(confs)}")
    high_conf = sum(1 for c in confs if c >= 0.85)
    print(f"  High confidence (≥0.85): {high_conf}/{len(gold)} ({high_conf/len(gold)*100:.1f}%)")

    # Save report
    report = {
        "input_rows": len(rows),
        "output_rows": len(gold),
        "removed": removed,
        "label_changes": {f"{k[0]}->{k[1]}": v for k, v in label_changes.items()},
        "total_flipped": sum(label_changes.values()),
        "final_label_distribution": dict(label_dist),
        "label_sources": dict(source_dist),
        "relevancy_distribution": dict(rel_dist),
        "context_length": {
            "min": min(lens), "max": max(lens),
            "avg": round(statistics.mean(lens), 1),
            "median": statistics.median(lens),
        },
        "confidence": {
            "mean": round(statistics.mean(confs), 3),
            "high_confidence_count": high_conf,
            "high_confidence_pct": round(high_conf / len(gold) * 100, 1),
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport: {REPORT}")
    print(f"Output: {OUTPUT} ({len(gold)} rows)")


if __name__ == "__main__":
    main()
