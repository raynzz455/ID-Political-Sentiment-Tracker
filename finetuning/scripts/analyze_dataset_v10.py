#!/usr/bin/env python3
"""
analyze_dataset_v10.py
=====================
Dataset v10 quality analysis — runs in sandbox (no torch needed).
Produces a comprehensive report for fine-tuning v4 planning.

Output: finetuning/docs/dataset_v10_analysis.json
"""
import json
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "datasets" / "dataset_train_v10.jsonl"
FULL    = BASE / "datasets" / "dataset_v10_final.jsonl"
OUT     = BASE / "docs" / "dataset_v10_analysis.json"


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    print("=" * 64)
    print("DATASET v10 QUALITY ANALYSIS")
    print("=" * 64)

    rows = load_jsonl(DATASET)
    full = load_jsonl(FULL)
    print(f"Training dataset: {len(rows)} rows")
    print(f"Full dataset:     {len(full)} rows")

    report = {"training_rows": len(rows), "full_rows": len(full)}

    # 1. Label distribution
    labels = Counter(r["label"] for r in rows)
    report["label_distribution"] = dict(labels)
    report["label_percentages"] = {k: round(v / len(rows) * 100, 2) for k, v in labels.items()}
    print(f"\n1. Label distribution:")
    for k, v in sorted(labels.items(), key=lambda x: -x[1]):
        print(f"   {k:10s}: {v:5d} ({v/len(rows)*100:5.1f}%)")

    # Imbalance ratio
    max_cls = max(labels.values())
    min_cls = min(labels.values())
    report["imbalance_ratio"] = round(max_cls / min_cls, 2)
    print(f"   Imbalance ratio (max/min): {max_cls}/{min_cls} = {max_cls/min_cls:.1f}x")

    # 2. Text length statistics
    lengths = [len(r["text"]) for r in rows]
    word_counts = [len(r["text"].split()) for r in rows]
    report["text_length"] = {
        "chars_mean": round(np.mean(lengths), 1),
        "chars_median": int(np.median(lengths)),
        "chars_p95": int(np.percentile(lengths, 95)),
        "chars_max": max(lengths),
        "chars_min": min(lengths),
        "words_mean": round(np.mean(word_counts), 1),
        "words_median": int(np.median(word_counts)),
        "words_p95": int(np.percentile(word_counts, 95)),
    }
    print(f"\n2. Text length:")
    print(f"   Chars: mean={report['text_length']['chars_mean']}, median={report['text_length']['chars_median']}, p95={report['text_length']['chars_p95']}")
    print(f"   Words: mean={report['text_length']['words_mean']}, median={report['text_length']['words_median']}")

    # 3. Token estimate (rough: 1 word ≈ 1.3 tokens for Indonesian)
    token_est = [wc * 1.3 for wc in word_counts]
    report["token_estimate_p95"] = int(np.percentile(token_est, 95))
    report["token_estimate_p99"] = int(np.percentile(token_est, 99))
    print(f"\n3. Token estimate (1 word ≈ 1.3 tokens):")
    print(f"   p95: {report['token_estimate_p95']} tokens")
    print(f"   p99: {report['token_estimate_p99']} tokens")
    print(f"   Recommendation: MAX_SEQ_LENGTH = 256 covers p95, 384 covers p99")

    # 4. Entity coverage
    entities = Counter(r["entity_name"] for r in rows)
    report["unique_entities"] = len(entities)
    report["top_10_entities"] = dict(entities.most_common(10))
    report["entities_with_5plus"] = sum(1 for v in entities.values() if v >= 5)
    report["entities_with_1"] = sum(1 for v in entities.values() if v == 1)
    print(f"\n4. Entity coverage:")
    print(f"   Unique entities: {len(entities)}")
    print(f"   Entities with ≥5 samples: {report['entities_with_5plus']}")
    print(f"   Entities with 1 sample:   {report['entities_with_1']}")
    print(f"   Top 5:")
    for e, c in entities.most_common(5):
        print(f"     {e:30s}: {c}")

    # 5. Per-entity label distribution (check for entity-label bias)
    entity_labels = defaultdict(lambda: Counter())
    for r in rows:
        entity_labels[r["entity_name"]][r["label"]] += 1
    # Entities that are all one label (potential bias)
    biased = []
    for e, lc in entity_labels.items():
        total = sum(lc.values())
        if total >= 10:
            max_frac = max(lc.values()) / total
            if max_frac >= 0.95:
                biased.append({"entity": e, "total": total, "dominant_label": max(lc, key=lc.get), "fraction": round(max_frac, 3)})
    report["biased_entities"] = biased[:20]
    print(f"\n5. Entity-label bias (≥10 samples, ≥95% one label): {len(biased)} entities")

    # 6. Source quality
    sources = Counter(r.get("label_source", "unknown") for r in rows)
    report["label_sources"] = dict(sources)
    print(f"\n6. Label sources:")
    for k, v in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"   {k:25s}: {v}")

    # 7. Relevancy distribution (for relevancy task)
    rels = Counter(r.get("gold_relevancy", "unknown") for r in rows)
    report["relevancy_distribution"] = dict(rels)
    print(f"\n7. Relevancy distribution:")
    for k, v in sorted(rels.items(), key=lambda x: -x[1]):
        print(f"   {k:15s}: {v}")

    # 8. Confidence distribution
    confs = [r.get("label_confidence", 0.5) for r in rows]
    report["confidence"] = {
        "mean": round(np.mean(confs), 3),
        "min": min(confs),
        "max": max(confs),
    }
    print(f"\n8. Label confidence: mean={report['confidence']['mean']}, range=[{report['confidence']['min']}, {report['confidence']['max']}]")

    # 9. Source URL diversity
    urls = Counter(r.get("source_url", "") for r in rows if r.get("source_url"))
    domains = Counter()
    for u in urls:
        if "://" in u:
            d = u.split("://")[1].split("/")[0]
            domains[d] += urls[u]
    report["unique_domains"] = len(domains)
    report["top_10_domains"] = dict(domains.most_common(10))
    print(f"\n9. Source diversity:")
    print(f"   Unique URLs: {len(urls)}")
    print(f"   Unique domains: {len(domains)}")

    # 10. Recommendations for v4 fine-tuning
    recs = []
    # Class imbalance
    if report["imbalance_ratio"] > 5:
        recs.append(f"High imbalance ({report['imbalance_ratio']}x) → use focal_loss gamma=2.5 + class_weights + oversample negative to ~400")
    # Dataset size
    if len(rows) < 3000:
        recs.append(f"Dataset size {len(rows)} < 3000 → use LoRA r=64, dropout=0.2, strong augmentation (mixup alpha=0.3)")
    else:
        recs.append(f"Dataset size {len(rows)} adequate → LoRA r=64, dropout=0.15")
    # Seq length
    if report["token_estimate_p95"] > 256:
        recs.append(f"p95 tokens={report['token_estimate_p95']} > 256 → set MAX_SEQ_LENGTH=384 for better coverage")
    else:
        recs.append(f"p95 tokens={report['token_estimate_p95']} ≤ 256 → MAX_SEQ_LENGTH=256 is sufficient")
    # Entity bias
    if len(biased) > 5:
        recs.append(f"{len(biased)} biased entities → ensure K-fold splits stratify by entity+label, not just label")
    report["recommendations"] = recs
    print(f"\n10. Recommendations for v4:")
    for r in recs:
        print(f"   • {r}")

    # Write report
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {OUT}")


if __name__ == "__main__":
    main()
