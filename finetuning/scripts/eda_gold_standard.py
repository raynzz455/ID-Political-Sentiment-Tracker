#!/usr/bin/env python3
"""EDA on gold standard final dataset."""
import json, numpy as np, re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "datasets" / "dataset_gold_standard_final.jsonl"

rows = [json.loads(l) for l in open(DATASET) if l.strip()]
print("=" * 64)
print(f"EDA: GOLD STANDARD FINAL — {len(rows)} rows")
print("=" * 64)

# 1. Label distribution
labels = Counter(r["label"] for r in rows)
print(f"\n1. LABEL DISTRIBUTION")
for k, v in sorted(labels.items(), key=lambda x: -x[1]):
    print(f"   {k:10s}: {v:5d} ({v/len(rows)*100:5.1f}%)")
print(f"   Imbalance: {max(labels.values())/min(labels.values()):.1f}x")

# 2. Entity coverage
entities = Counter(r["entity_name"] for r in rows)
print(f"\n2. ENTITY COVERAGE")
print(f"   Unique entities: {len(entities)}")
print(f"   Mean samples/entity: {np.mean(list(entities.values())):.1f}")
print(f"   Entities with ≥5 samples: {sum(1 for c in entities.values() if c >= 5)}")
print(f"   Top 10:")
for e, c in entities.most_common(10):
    print(f"     {e:30s}: {c}")

# 3. Text length
lens = [len(r["text"]) for r in rows]
words = [len(r["text"].split()) for r in rows]
tokens = [int(w*1.3) for w in words]
print(f"\n3. TEXT LENGTH")
print(f"   Chars: mean={np.mean(lens):.0f}, p95={int(np.percentile(lens,95))}, max={max(lens)}")
print(f"   Words: mean={np.mean(words):.0f}, p95={int(np.percentile(words,95))}")
print(f"   Tokens est: p95={int(np.percentile(tokens,95))} → MAX_SEQ_LENGTH=256 {'sufficient' if int(np.percentile(tokens,95))<=256 else 'need 384'}")

# 4. Source diversity
urls = [r.get("source_url","") for r in rows if r.get("source_url")]
domains = Counter(urlparse(u).netloc.replace("www.","") for u in urls if urlparse(u).netloc)
print(f"\n4. SOURCE DIVERSITY")
print(f"   Unique URLs: {len(set(urls))}")
print(f"   Unique domains: {len(domains)}")
print(f"   Top 5:")
for d, c in domains.most_common(5):
    print(f"     {d:30s}: {c}")

# 5. Label sources
sources = Counter(r.get("label_source","unknown") for r in rows)
print(f"\n5. LABEL SOURCES")
for k, v in sorted(sources.items(), key=lambda x: -x[1]):
    print(f"   {k:25s}: {v:5d}")
verified = sum(1 for r in rows if r.get('label_source','').startswith('llm'))
print(f"   LLM-verified total: {verified} ({verified/len(rows)*100:.1f}%)")

# 6. Confidence
confs = [r.get("label_confidence",0.5) for r in rows]
print(f"\n6. CONFIDENCE")
print(f"   Mean: {np.mean(confs):.3f}, Median: {np.median(confs):.3f}")
high = sum(1 for c in confs if c >= 0.85)
print(f"   High (≥0.85): {high} ({high/len(rows)*100:.1f}%)")

# 7. Match types
mt = Counter(r.get("match_type","none") for r in rows)
print(f"\n7. MATCH TYPES")
for k, v in mt.most_common():
    print(f"   {k:25s}: {v:5d}")

# 8. Relevancy
rels = Counter(r.get("gold_relevancy","unknown") for r in rows)
print(f"\n8. RELEVANCY")
for k, v in sorted(rels.items(), key=lambda x: -x[1]):
    print(f"   {k:15s}: {v:5d}")

# 9. Duplicates
text_counts = Counter(r["text"] for r in rows)
dupes = sum(1 for t, c in text_counts.items() if c > 1)
print(f"\n9. DUPLICATES")
print(f"   Unique texts: {len(text_counts)}")
print(f"   Duplicate texts: {dupes}")
if dupes > 0:
    print(f"   Top duplicates:")
    for t, c in text_counts.most_common(3):
        if c > 1: print(f"     [{c}x] \"{t[:80]}...\"")

# 10. Recommendations
print(f"\n10. RECOMMENDATIONS")
recs = []
imb = max(labels.values())/min(labels.values())
if imb > 5:
    recs.append(f"Imbalance {imb:.1f}x → oversample negative to ~400 (hyperparams_v4 already configured)")
if int(np.percentile(tokens,95)) > 256:
    recs.append(f"p95 tokens={int(np.percentile(tokens,95))} → set MAX_SEQ_LENGTH=384")
else:
    recs.append(f"p95 tokens={int(np.percentile(tokens,95))} → MAX_SEQ_LENGTH=256 sufficient (current config)")
if dupes > 0:
    recs.append(f"{dupes} duplicate texts → consider dedup before training")
for r in recs: print(f"   • {r}")
