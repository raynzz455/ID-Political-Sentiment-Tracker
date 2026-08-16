#!/usr/bin/env python3.13
"""
build_dataset_v3.py
===================
Build final finetuning-ready dataset v3 by merging:

  1. HIGH-CONFIDENCE rows from dataset_v2 (label_source already trusted:
     llm_verified, llm_second_pass, gold_human; OR label_confidence >= 0.7)
     → kept as-is.

  2. LOW-CONFIDENCE rows that we just LLM-verified via verify_dataset_v2.mjs
     (llm_verified_v3.jsonl) → label + confidence + source upgraded to
     llm_verified / 0.85. Propagated to ALL oversampled copies that share
     the same base row_index.

  3. LOW-CONFIDENCE rows that FAILED LLM verification (still rate-limited)
     → kept with original heuristic label, confidence DOWNGRADED to 0.45,
     label_source='llm_verify_pending' so finetune.py can down-weight them.

  4. Rows where gold_relevancy == 'not_relevant' (from LLM verification)
     → EXCLUDED from sentiment training set (still logged).

Output:
  finetuning/datasets/dataset_v3.jsonl  — final, clean, LLM-verified, balanced
"""
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
INPUT  = BASE / "datasets" / "dataset_v2.jsonl"
VERIFY = BASE / "llm_verified_v3.jsonl"
OUTPUT = BASE / "datasets" / "dataset_v3.jsonl"
REPORT = BASE / "dataset_v3_report.json"


def base_idx(ri):
    s = str(ri)
    return s.split("_aug_")[0] if "_aug_" in s else s


def main():
    # ---- Load dataset_v2 ----
    rows = [json.loads(l) for l in open(INPUT) if l.strip()]
    print(f"Loaded dataset_v2: {len(rows)} rows")

    # ---- Load LLM-verified labels ----
    verified = {}
    if VERIFY.exists():
        for l in open(VERIFY):
            if not l.strip(): continue
            v = json.loads(l)
            verified[v["base_row_index"]] = v
        print(f"Loaded LLM-verified: {len(verified)} base rows")
    else:
        print(f"WARNING: {VERIFY} not found — no LLM verification applied")

    # ---- Merge ----
    TRUSTED_SOURCES = {"llm_verified", "llm_second_pass", "gold_human"}
    stats = Counter()
    v3 = []

    for r in rows:
        b = base_idx(r.get("row_index", ""))
        src = r.get("label_source", "")
        conf = r.get("label_confidence", 0)

        if b in verified:
            # Apply LLM-verified label
            v = verified[b]
            r = dict(r)  # copy
            # Preserve original for audit
            r["prev_label"] = r.get("gold_label")
            r["prev_label_source"] = src
            r["prev_label_confidence"] = conf
            # Apply verified
            if v["gold_relevancy"] == "not_relevant":
                # Excluded from sentiment training — but keep row for audit
                r["gold_relevancy"] = "not_relevant"
                r["exclude_from_sentiment"] = True
                r["gold_label"] = v["gold_label"]
                r["label_source"] = "llm_verified"
                r["label_confidence"] = 0.85
                r["verification_reasoning"] = v.get("reasoning", "")
                stats["llm_verified_not_relevant"] += 1
            else:
                r["gold_label"] = v["gold_label"]
                r["gold_relevancy"] = v["gold_relevancy"]
                r["label_source"] = "llm_verified"
                r["label_confidence"] = 0.85
                r["verification_reasoning"] = v.get("reasoning", "")
                r["entity_is_main_subject"] = v.get("entity_is_main_subject", True)
                if r.get("prev_label") and r["prev_label"] != r["gold_label"]:
                    stats["label_flipped"] += 1
                else:
                    stats["label_confirmed"] += 1
            v3.append(r)
        elif src in TRUSTED_SOURCES or conf >= 0.7:
            # Already trusted — keep as-is
            v3.append(r)
            stats["already_trusted"] += 1
        else:
            # Low-confidence + not LLM-verified (rate-limited failure)
            # Keep heuristic label but downgrade confidence for down-weighting
            r = dict(r)
            r["prev_label"] = r.get("gold_label")
            r["prev_label_source"] = src
            r["prev_label_confidence"] = conf
            r["label_confidence"] = 0.45
            r["label_source"] = "llm_verify_pending"
            r["verification_reasoning"] = "LLM verification pending (rate-limited)"
            v3.append(r)
            stats["pending_low_confidence"] += 1

    # ---- Filter for sentiment training (relevant only) ----
    v3_sent = [r for r in v3 if r.get("gold_relevancy") == "relevant"]
    excluded_not_rel = len(v3) - len(v3_sent)

    # ---- Save ----
    with open(OUTPUT, "w") as f:
        for r in v3:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- Report ----
    final_labels = Counter(r["gold_label"] for r in v3_sent)
    final_sources = Counter(r["label_source"] for r in v3_sent)
    final_conf_buckets = Counter()
    for r in v3_sent:
        c = r.get("label_confidence", 0)
        if c >= 0.85: final_conf_buckets[">=0.85 (LLM/gold)"] += 1
        elif c >= 0.7: final_conf_buckets["0.70-0.84 (trusted)"] += 1
        elif c >= 0.55: final_conf_buckets["0.55-0.69 (heuristic)"] += 1
        else: final_conf_buckets["<0.55 (pending)"] += 1

    report = {
        "input": str(INPUT),
        "llm_verified_file": str(VERIFY),
        "output": str(OUTPUT),
        "total_rows_v2": len(rows),
        "total_rows_v3": len(v3),
        "sentiment_relevant_rows": len(v3_sent),
        "excluded_not_relevant": excluded_not_rel,
        "merge_stats": dict(stats),
        "final_label_distribution": dict(final_labels),
        "final_label_source_distribution": dict(final_sources),
        "final_confidence_buckets": dict(final_conf_buckets),
    }
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print("DATASET v3 BUILD COMPLETE")
    print("=" * 60)
    print(f"Input v2 rows:              {len(rows)}")
    print(f"LLM-verified base rows:     {len(verified)}")
    print(f"Output v3 rows:             {len(v3)}")
    print(f"  Sentiment-relevant:       {len(v3_sent)}")
    print(f"  Excluded (not_relevant):   {excluded_not_rel}")
    print()
    print("Merge stats:")
    for k, v in stats.most_common():
        print(f"  {k:35s}: {v}")
    print()
    print("Final sentiment label distribution:")
    for k in ["positive", "neutral", "negative"]:
        v = final_labels.get(k, 0)
        print(f"  {k:15s}: {v:4d} ({100*v/max(1,len(v3_sent)):.1f}%)")
    print()
    print("Final label source distribution:")
    for k, v in final_sources.most_common():
        print(f"  {k:35s}: {v:4d} ({100*v/max(1,len(v3_sent)):.1f}%)")
    print()
    print("Final confidence buckets:")
    for k, v in final_conf_buckets.most_common():
        print(f"  {k:30s}: {v:4d} ({100*v/max(1,len(v3_sent)):.1f}%)")
    print()
    print(f"Output: {OUTPUT}")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
