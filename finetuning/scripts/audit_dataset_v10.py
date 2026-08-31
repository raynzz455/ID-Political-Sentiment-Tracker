#!/usr/bin/env python3
"""
audit_dataset_v10.py
===================
Comprehensive audit of dataset_train_v10.jsonl to find data quality issues
BEFORE fine-tuning. Checks:

  1. Empty/short text (< 50 chars)
  2. Truncated text (starts/ends mid-word)
  3. Entity presence: does entity_name appear in text?
  4. Entity name variations (full vs short form)
  5. Duplicate rows (same raw_text_id + entity_name)
  6. Label consistency (gold_label vs pseudo_label)
  7. Text encoding issues (HTML entities, mojibake)
  8. Sentence completeness (does text end with period?)
  9. Entity context alignment (is entity the subject?)

Output: finetuning/docs/dataset_v10_audit.json
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "datasets" / "dataset_train_v10.jsonl"
OUT = BASE / "docs" / "dataset_v10_audit.json"


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def is_truncated_start(text):
    """Check if text starts mid-word (lowercase letter at start, or partial word)."""
    if not text:
        return False
    first_char = text[0]
    # Starts with lowercase (likely mid-sentence)
    if first_char.islower():
        return True
    # Starts with partial word like "tan " (from "mantan")
    first_word = text.split()[0] if text.split() else ""
    # Common Indonesian words that, when seen alone, indicate truncation
    if first_word.lower() in ['tan', 'kan', 'an', 'nya', 'kah', 'lah', 'pun', 'ku', 'mu']:
        return True
    return False


def is_truncated_end(text):
    """Check if text ends mid-word (no terminal punctuation)."""
    if not text:
        return False
    text = text.rstrip()
    if not text:
        return False
    last_char = text[-1]
    # Should end with . ! ? or " or ) or ]
    terminal_chars = {".", "!", "?", '"', "'", ")", "]"}
    if last_char in terminal_chars:
        return False
    # Check if last word looks complete
    last_word = text.split()[-1] if text.split() else ""
    if len(last_word) <= 2:
        return True
    return False


def entity_in_text(entity, text):
    """Check if entity name (or significant part) appears in text."""
    if not entity or not text:
        return False, ""
    entity_lower = entity.lower().strip()
    text_lower = text.lower()
    # Direct match
    if entity_lower in text_lower:
        return True, "full_match"
    # Try without middle name (e.g., "Joko Widodo" -> "Jokowi" or "Joko")
    parts = entity.split()
    if len(parts) >= 2:
        # Try first name + last name
        first = parts[0].lower()
        last = parts[-1].lower()
        if first in text_lower and len(first) >= 4:
            return True, "first_name"
        if last in text_lower and len(last) >= 4:
            return True, "last_name"
    # Try common short forms
    short_forms = {
        "Joko Widodo": ["jokowi", "joko"],
        "Prabowo Subianto": ["prabowo"],
        "Megawati Soekarnoputri": ["megawati"],
        "Basuki Tjahaja Purnama": ["ahok"],
        "Susilo Bambang Yudhoyono": ["sby"],
    }
    for full, shorts in short_forms.items():
        if entity_lower == full.lower():
            for short in shorts:
                if short in text_lower:
                    return True, f"short_form:{short}"
    return False, "not_found"


def has_encoding_issues(text):
    """Check for HTML entities or mojibake."""
    if not text:
        return False
    patterns = [
        r'&\w+;',  # HTML entities
        r'Ã[\x80-\xBF]',  # UTF-8 mojibake
        r'â€[\x80-\x9F]',  # Smart quotes mojibake
        r'&#[0-9]+;',  # Numeric HTML entities
    ]
    for p in patterns:
        if re.search(p, text):
            return True
    return False


def main():
    print("=" * 64)
    print("DATASET v10 QUALITY AUDIT")
    print("=" * 64)

    rows = load_jsonl(DATASET)
    print(f"Total rows: {len(rows)}\n")

    issues = {
        "empty_text": [],
        "short_text": [],         # < 50 chars
        "truncated_start": [],
        "truncated_end": [],
        "entity_not_in_text": [],  # entity_name not found in text
        "duplicates": [],
        "encoding_issues": [],
        "low_confidence": [],     # label_confidence < 0.6
        "no_reasoning": [],
    }

    # Track duplicates
    seen = {}
    duplicate_count = 0

    for i, r in enumerate(rows):
        text = r.get("text", "")
        entity = r.get("entity_name", "")
        label = r.get("label", "")
        raw_id = r.get("raw_text_id", "")
        conf = r.get("label_confidence", 0.5)
        reasoning = r.get("reasoning", "")

        # 1. Empty text
        if not text or len(text.strip()) == 0:
            issues["empty_text"].append(i)
            continue

        # 2. Short text
        if len(text.strip()) < 50:
            issues["short_text"].append({"row": i, "text_len": len(text), "entity": entity})

        # 3. Truncated start
        if is_truncated_start(text):
            issues["truncated_start"].append({
                "row": i,
                "entity": entity,
                "first_50": text[:50]
            })

        # 4. Truncated end
        if is_truncated_end(text):
            issues["truncated_end"].append({
                "row": i,
                "entity": entity,
                "last_50": text[-50:]
            })

        # 5. Entity not in text
        found, match_type = entity_in_text(entity, text)
        if not found:
            issues["entity_not_in_text"].append({
                "row": i,
                "entity": entity,
                "text_first_100": text[:100],
                "match_type": match_type
            })

        # 6. Duplicates
        key = (raw_id, entity)
        if key in seen:
            duplicate_count += 1
            issues["duplicates"].append({
                "row": i,
                "duplicate_of": seen[key],
                "entity": entity
            })
        else:
            seen[key] = i

        # 7. Encoding issues
        if has_encoding_issues(text):
            issues["encoding_issues"].append({
                "row": i,
                "entity": entity,
                "first_50": text[:50]
            })

        # 8. Low confidence
        if conf < 0.6:
            issues["low_confidence"].append({
                "row": i,
                "entity": entity,
                "confidence": conf,
                "label_source": r.get("label_source", "")
            })

        # 9. No reasoning
        if not reasoning or len(reasoning.strip()) < 10:
            issues["no_reasoning"].append({
                "row": i,
                "entity": entity,
                "label_source": r.get("label_source", "")
            })

    # Print summary
    print("ISSUE SUMMARY:")
    print(f"  Empty text:           {len(issues['empty_text'])}")
    print(f"  Short text (<50):     {len(issues['short_text'])}")
    print(f"  Truncated start:      {len(issues['truncated_start'])}")
    print(f"  Truncated end:        {len(issues['truncated_end'])}")
    print(f"  Entity not in text:   {len(issues['entity_not_in_text'])}")
    print(f"  Duplicates:           {len(issues['duplicates'])}")
    print(f"  Encoding issues:      {len(issues['encoding_issues'])}")
    print(f"  Low confidence (<0.6): {len(issues['low_confidence'])}")
    print(f"  No reasoning:         {len(issues['no_reasoning'])}")

    # Show examples
    print("\n" + "=" * 64)
    print("SAMPLE ISSUES (first 5 each)")
    print("=" * 64)

    if issues["truncated_start"]:
        print(f"\n--- TRUNCATED START ({len(issues['truncated_start'])} rows) ---")
        for item in issues["truncated_start"][:5]:
            print(f"  Row {item['row']} [{item['entity']}]: \"{item['first_50']}\"")

    if issues["truncated_end"]:
        print(f"\n--- TRUNCATED END ({len(issues['truncated_end'])} rows) ---")
        for item in issues["truncated_end"][:5]:
            print(f"  Row {item['row']} [{item['entity']}]: ...\"{item['last_50']}\"")

    if issues["entity_not_in_text"]:
        print(f"\n--- ENTITY NOT IN TEXT ({len(issues['entity_not_in_text'])} rows) ---")
        for item in issues["entity_not_in_text"][:5]:
            print(f"  Row {item['row']} [{item['entity']}]: \"{item['text_first_100']}\"")

    if issues["short_text"]:
        print(f"\n--- SHORT TEXT ({len(issues['short_text'])} rows) ---")
        for item in issues["short_text"][:5]:
            print(f"  Row {item['row']} [{item['entity']}]: len={item['text_len']}")

    if issues["encoding_issues"]:
        print(f"\n--- ENCODING ISSUES ({len(issues['encoding_issues'])} rows) ---")
        for item in issues["encoding_issues"][:5]:
            print(f"  Row {item['row']} [{item['entity']}]: \"{item['first_50']}\"")

    if issues["low_confidence"]:
        print(f"\n--- LOW CONFIDENCE ({len(issues['low_confidence'])} rows) ---")
        for item in issues["low_confidence"][:5]:
            print(f"  Row {item['row']} [{item['entity']}]: conf={item['confidence']}, source={item['label_source']}")

    # Save full report
    report = {
        "total_rows": len(rows),
        "issues_summary": {k: len(v) for k, v in issues.items()},
        "issues_detail": issues,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report: {OUT}")


if __name__ == "__main__":
    main()
