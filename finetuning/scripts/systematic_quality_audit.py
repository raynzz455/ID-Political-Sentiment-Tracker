#!/usr/bin/env python3
"""
systematic_quality_audit.py
===========================
Find suspicious labels in the gold standard dataset for re-verification.

Checks:
  1. Entity NOT main subject (entity only in "era X" context)
  2. Suspicious label (negative but positive context, etc.)
  3. Empty reasoning
  4. Low confidence (≤0.55)

Output: suspicious_rows.json (for reverify_suspicious.mjs)
"""
import json, re, sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "datasets" / "dataset_gold_standard.jsonl"
OUTPUT = BASE / "datasets" / "suspicious_rows.json"


def detect_entity_not_main(row):
    entity = row.get("entity_name", "").lower()
    text = row.get("text", "").lower()
    mt = row.get("match_type", "")
    if not entity or not text: return False, ""
    era_patterns = [
        r'era\s+' + re.escape(entity), r'zaman\s+' + re.escape(entity),
        r'masa\s+' + re.escape(entity), r'pemerintahan\s+' + re.escape(entity),
    ]
    for p in era_patterns:
        if re.search(p, text):
            mentions = text.count(entity)
            era_mentions = sum(1 for p in era_patterns if re.search(p, text))
            if mentions <= era_mentions + 1:
                return True, f"entity_only_in_era (mentions={mentions})"
    if mt in ("first_name", "last_name", "first_last"):
        pos = text.find(entity)
        if pos < 0:
            parts = entity.split()
            for p in parts:
                if p in text: pos = text.find(p); break
        if pos > 0 and pos / len(text) > 0.5:
            return True, f"entity_mentioned_late (pos={pos/len(text)*100:.0f}%)"
    return False, ""


def detect_suspicious_label(row):
    label = row.get("label", "")
    text = row.get("text", "").lower()
    conf = row.get("label_confidence", 0.5)
    reasoning = row.get("verification_reasoning", "")
    if conf <= 0.55 and not reasoning.strip():
        return True, f"low_conf_no_reasoning (conf={conf})"
    if label == "negative":
        for p in [r'\bmenang\b', r'berhasil\b', r'prestasi', r'penghargaan', r'dilantik\s+sebagai', r'juara']:
            if re.search(p, text): return True, f"negative_but_positive ({p})"
    if label == "positive":
        for p in [r'divonis', r'ditahan', r'tersangka', r'korupsi', r'dicopot', r'mundur\s+dari\s+jabatan']:
            if re.search(p, text): return True, f"positive_but_negative ({p})"
    if label == "negative" and conf <= 0.55:
        return True, f"negative_low_conf (conf={conf})"
    return False, ""


def main():
    if not DATASET.exists():
        print(f"ERROR: {DATASET} not found. Run build_gold_standard.py first."); sys.exit(1)
    rows = [json.loads(l) for l in open(DATASET) if l.strip()]
    print(f"Total rows: {len(rows)}")
    suspicious = []
    for i, r in enumerate(rows):
        is_susp, reason = detect_entity_not_main(r)
        if is_susp:
            suspicious.append({"row_index": i, "entity_name": r["entity_name"],
                "current_label": r["label"], "confidence": r.get("label_confidence", 0.5),
                "suspicion_reason": reason, "text": r.get("text", "")})
            continue
        is_susp, reason = detect_suspicious_label(r)
        if is_susp:
            suspicious.append({"row_index": i, "entity_name": r["entity_name"],
                "current_label": r["label"], "confidence": r.get("label_confidence", 0.5),
                "suspicion_reason": reason, "text": r.get("text", "")})
    print(f"Suspicious rows: {len(suspicious)}")
    by_reason = Counter(s["suspicion_reason"].split(" ")[0] for s in suspicious)
    print(f"By reason: {dict(by_reason)}")
    with open(OUTPUT, "w") as f:
        json.dump(suspicious, f, ensure_ascii=False, indent=2)
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
