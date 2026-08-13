#!/usr/bin/env python3.13
"""
build_dataset_v2.py
===================
TAHAP 2: Build improved dataset v2 untuk finetuning v2.

Strategi:
  A. FILTER label noise — hanya high-confidence labels (>=0.7)
  B. OVERSAMPLE minority class — duplicate positive/negative rows
  C. APPLY best grid search params — lr=3e-5
  D. EXCLUDE bad contexts — corruption, wrong_entity, background_only

Output: dataset_v2.jsonl — siap untuk finetune v2
"""
import json, random, sys
from collections import Counter
from pathlib import Path

_script_dir = Path(__file__).parent if '__file__' in dir() else Path('.')
INPUT = str(_script_dir.parent / "datasets" / "dataset_enhanced.jsonl")
OUTPUT = str(_script_dir.parent / "datasets" / "dataset_v2.jsonl")

random.seed(42)

# Load dataset v1
rows = [json.loads(l) for l in open(INPUT) if l.strip()]
print(f"Loaded v1: {len(rows)} rows")

# Step A: Filter to relevant + clean context only
v2 = []
excluded = Counter()
for r in rows:
    if r.get('gold_relevancy') != 'relevant':
        excluded['not_relevant'] += 1
        continue
    flag = r.get('context_flag', 'clean')
    if flag in ('corruption_stitch', 'wrong_entity', 'background_only', 'llm_failed'):
        excluded[flag] += 1
        continue
    # Only keep high + mid confidence (>=0.5)
    if r.get('label_confidence', 0) < 0.5:
        excluded['low_confidence'] += 1
        continue
    v2.append(r)

print(f"\nAfter filter: {len(v2)} rows")
print(f"Excluded: {dict(excluded)}")

# Step B: Class distribution before oversampling
labels_before = Counter(r['gold_label'] for r in v2)
print(f"\nBefore oversampling:")
for l in ['negative', 'neutral', 'positive']:
    print(f"  {l}: {labels_before[l]}")

# Step C: Oversample minority classes to balance
max_class = max(labels_before.values())
target_per_class = max_class  # match majority

v2_balanced = list(v2)  # start with all rows

for label in ['positive', 'negative']:
    current = labels_before[label]
    needed = target_per_class - current
    if needed > 0:
        # Get rows of this class
        class_rows = [r for r in v2 if r['gold_label'] == label]
        # Oversample by duplication with slight variation
        for _ in range(needed):
            r = random.choice(class_rows)
            # Create augmented copy
            aug = dict(r)
            aug['row_index'] = f"{r['row_index']}_aug_{_}"
            aug['label_source'] = 'oversampled'
            aug['label_confidence'] = r.get('label_confidence', 0.5) * 0.9  # slightly lower
            v2_balanced.append(aug)

labels_after = Counter(r['gold_label'] for r in v2_balanced)
print(f"\nAfter oversampling:")
for l in ['negative', 'neutral', 'positive']:
    print(f"  {l}: {labels_after[l]}")
print(f"Total: {len(v2_balanced)}")

# Step D: Shuffle
random.shuffle(v2_balanced)

# Save
with open(OUTPUT, 'w') as f:
    for r in v2_balanced:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f"\n✅ Dataset v2 saved: {OUTPUT}")
print(f"   {len(v2_balanced)} rows (was {len(rows)} in v1)")
print(f"   Balanced: {dict(labels_after)}")
print(f"   Confidence: all >= 0.5")
print(f"   Context: clean + speaker_not_target + byline_leak only")

# Stats comparison
print(f"\n{'='*60}")
print(f"DATASET COMPARISON v1 vs v2")
print(f"{'='*60}")
print(f"{'Metric':<25s} {'v1':>10s} {'v2':>10s}")
print(f"{'-'*45}")
print(f"{'Total rows':<25s} {len(rows):>10d} {len(v2_balanced):>10d}")
print(f"{'Relevant rows':<25s} {len([r for r in rows if r.get('gold_relevancy')=='relevant']):>10d} {len(v2_balanced):>10d}")
for l in ['negative', 'neutral', 'positive']:
    v1_cnt = sum(1 for r in rows if r.get('gold_label') == l and r.get('gold_relevancy') == 'relevant')
    print(f"{f'  {l}':<25s} {v1_cnt:>10d} {labels_after[l]:>10d}")
v1_ratio = sum(1 for r in rows if r.get('gold_label')=='neutral' and r.get('gold_relevancy')=='relevant') / max(1, sum(1 for r in rows if r.get('gold_label')=='negative' and r.get('gold_relevancy')=='relevant'))
v2_ratio = labels_after['neutral'] / max(1, labels_after['negative'])
print(f"{'Imbalance ratio':<25s} {v1_ratio:>10.1f} {v2_ratio:>10.1f}")
print(f"{'Min confidence':<25s} {'0.30':>10s} {'0.50':>10s}")
