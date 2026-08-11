#!/usr/bin/env python3
"""
build_enhanced_dataset.py
=========================
Merges ALL label sources into a single enhanced dataset that conforms to
dataset_schema.py:

  - gold_labels.jsonl       (27 human labels, confidence 1.0)
  - llm_labels.jsonl        (166 LLM second-pass labels, confidence 0.85)
  - heuristic relabel rules (308 background + 156 speaker + 5 corruption + 2 polarity)
  - pseudo_kept fallback    (remaining unverified rows, confidence 0.5/0.3)

Output: dataset_enhanced.jsonl — one row per original dataset row (909),
with ALL the new schema fields (entity correction, context quality, flags,
validation, sentence-pair format).

Also runs dataset_schema.validate_dataset() and prints the report.

Usage:
    python build_enhanced_dataset.py
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
from collections import Counter

HERE = Path(__file__).parent
RAW = HERE / "dataset.jsonl"
GOLD = HERE / "gold_labels.jsonl"
LLM = HERE / "llm_labels.jsonl"
OUT = HERE / "dataset_enhanced.jsonl"
REPORT = HERE / "enhanced_dataset_report.json"

sys.path.insert(0, str(HERE))
from dataset_schema import (
    validate_dataset, calculate_context_quality,
    SENTIMENT_LABELS, RELEVANCY_LABELS,
)
from relabel_dataset import (
    strip_bylines, is_context_in_article, BYLINE_IN_BODY_RE,
    detect_speaker_vs_target, detect_background_mention, detect_wrong_polarity,
    entity_aliases, NEG_CUES, POS_CUES,
)

# ---------------------------------------------------------------------------
# Load all sources
# ---------------------------------------------------------------------------
def load_jsonl(p):
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if l.strip()]

rows = load_jsonl(RAW)
gold = {g["row_index"]: g for g in load_jsonl(GOLD)}
llm  = {l["row_index"]: l for l in load_jsonl(LLM)}

# Load LLM-verified labels (from llm_verify_heuristics.py — 464 heuristic rows re-verified)
LLM_VERIFIED = HERE / "llm_verified_labels.jsonl"
llm_verified = {v["row_index"]: v for v in load_jsonl(LLM_VERIFIED)}
print(f"Raw: {len(rows)} | Gold: {len(gold)} | LLM: {len(llm)} | LLM-verified: {len(llm_verified)}")

# ---------------------------------------------------------------------------
# Alias detection — find what surface form of the entity appears in context
# ---------------------------------------------------------------------------
def find_aliases_in_context(entity_name, context):
    """Return list of surface forms of the entity found in context."""
    found = []
    cl = context.lower()
    # full canonical name
    if entity_name.lower() in cl:
        found.append(entity_name)
    # first name token
    parts = entity_name.split()
    if len(parts) >= 2:
        first = parts[0]
        if first.lower() in cl and len(first) >= 4:
            found.append(first)
        # last name token
        last = parts[-1]
        if last.lower() in cl and len(last) >= 4 and last != first:
            found.append(last)
    # known aliases from relabel_dataset
    for alias in entity_aliases.get(entity_name, set()):
        if alias.lower() in cl:
            found.append(alias)
    return list(dict.fromkeys(found))  # dedupe, preserve order

# ---------------------------------------------------------------------------
# Classify each row (mirrors llm_relabel.py classify_row)
# ---------------------------------------------------------------------------
def classify(idx, r):
    """Return (label_source, gold_label, gold_relevancy, confidence, defect, reasoning)."""
    # 1. gold human
    if idx in gold:
        g = gold[idx]
        return ("gold_human", g["gold_label"], g["gold_relevancy"], 1.0,
                g["defect_class"], g["reasoning"])
    # 1b. LLM-verified heuristic labels (from llm_verify_heuristics.py — highest priority after gold)
    if idx in llm_verified:
        v = llm_verified[idx]
        if v.get("label_source") == "llm_verified":
            return ("llm_verified", v["gold_label"], v["gold_relevancy"], 0.85,
                    "verified", v.get("reasoning", "LLM-verified heuristic label."))
        elif v.get("label_source") == "llm_verify_failed":
            # API failed, keep heuristic label but mark as failed
            return ("llm_verify_failed", v["gold_label"], v["gold_relevancy"], 0.5,
                    "llm_verify_failed", v.get("reasoning", "LLM verify API failed."))
    # 2. LLM second-pass (only if successful)
    if idx in llm and llm[idx].get("label_source") == "llm_second_pass":
        l = llm[idx]
        return ("llm_second_pass", l["gold_label"], l["gold_relevancy"], 0.85,
                l.get("context_flag", "clean"), l.get("reasoning", ""))
    # 3. heuristics
    entity_lower = r["entity_name"].lower()
    ctx_raw = r["context_text"] or ""
    ctx_clean, _ = strip_bylines(ctx_raw)
    byline_in_body = bool(BYLINE_IN_BODY_RE.search(ctx_raw.strip()[60:]))
    headline_leak = bool(re.match(r"^Headline\s+", ctx_raw))
    if byline_in_body or headline_leak:
        return ("heuristic_corruption", "neutral", "not_relevant", 0.9,
                "corruption_stitch", "Byline detected inside body — stitched articles.")
    if detect_background_mention(ctx_clean, entity_lower):
        return ("heuristic_background", "neutral", "not_relevant", 0.7,
                "background_only", "Entity only a background/temporal anchor.")
    flip, conf = detect_wrong_polarity(ctx_clean, r["pseudo_label"])
    if flip is not None:
        return ("heuristic_polarity", flip, "relevant", conf,
                "clean", f"Polarity flip: strong opposite cues detected.")
    if detect_speaker_vs_target(ctx_clean, entity_lower):
        return ("heuristic_speaker", "neutral", "relevant", 0.7,
                "speaker_not_target", "Entity is the speaker, not the target.")
    # 4. LLM failed
    if idx in llm and llm[idx].get("label_source") == "llm_failed":
        return ("llm_failed", r["pseudo_label"], "relevant", 0.3,
                "llm_failed", "LLM second-pass could not label; kept pseudo.")
    # 5. pseudo kept (never attempted or no rule)
    return ("pseudo_kept", r["pseudo_label"], "relevant", 0.5,
            "clean", "Unverified — kept pseudo-label with low confidence.")

# ---------------------------------------------------------------------------
# Build enhanced rows
# ---------------------------------------------------------------------------
enhanced = []
for idx, r in enumerate(rows):
    entity = r["entity_name"]
    ctx_raw = r["context_text"] or ""
    article = r["article_text"] or ""

    # clean context
    ctx_clean, bylines_removed = strip_bylines(ctx_raw)

    # classify
    label_source, gold_label, gold_rel, conf, defect, reasoning = classify(idx, r)

    # entity detection
    aliases_found = find_aliases_in_context(entity, ctx_clean)
    entity_in_context = len(aliases_found) > 0

    # LLM-provided fields (if available)
    llm_row = llm.get(idx, {})
    entity_is_main_subject = llm_row.get("entity_is_main_subject", True)
    if label_source == "gold_human":
        entity_is_main_subject = (gold_rel == "relevant")
    entity_corrected = llm_row.get("entity_corrected")
    entity_correction_reason = ""
    if entity_corrected and entity_corrected != entity:
        entity_correction_reason = llm_row.get("reasoning", "")[:200]
    elif not entity_in_context and gold_rel == "not_relevant":
        # entity not in context + not relevant = likely wrong entity attribution
        entity_corrected = None  # can't determine without re-extraction
        entity_correction_reason = "Entity not found in context; may be misattributed."

    # context properties
    context_in_article = is_context_in_article(ctx_clean, article)
    article_truncated = len(article) >= 990  # export truncates at 1000

    # context flag (merge defect + LLM flag)
    # Map defect classes to valid context_flag values:
    #   wrong_polarity -> clean (context itself is fine, only label was flipped)
    #   corruption_stitch, background_only, speaker_not_target, llm_failed -> as-is
    context_flag = defect if defect in (
        "corruption_stitch", "background_only", "speaker_not_target",
        "llm_failed"
    ) else "clean"
    # if entity not in context, override to wrong_entity (unless already a worse flag)
    if not entity_in_context and context_flag in ("clean", "byline_leak"):
        context_flag = "wrong_entity"
    if bylines_removed and context_flag == "clean":
        context_flag = "byline_leak"
    if len(ctx_clean) < 60 and context_flag == "clean":
        context_flag = "too_short"
    if len(ctx_clean) > 1184 and context_flag == "clean":
        context_flag = "too_long"

    # needs reextract?
    needs_reextract = (
        context_flag == "corruption_stitch"
        or (not context_in_article and article_truncated and entity_in_context)
    )

    # context quality
    context_quality = calculate_context_quality(
        ctx_clean, entity_in_context, context_in_article,
        bylines_removed, context_flag,
    )

    # sentence-pair
    alias_hint = ""
    if aliases_found:
        short_forms = [a for a in aliases_found if a != entity][:2]
        if short_forms:
            alias_hint = f" ({', '.join(short_forms)})"
    premise = f"{entity}{alias_hint}"
    hypothesis = ctx_clean

    enhanced.append({
        "raw_text_id": r["raw_text_id"],
        "row_index": idx,
        "source_url": r.get("source_url", ""),
        "entity_name": entity,
        "entity_aliases_found": aliases_found,
        "entity_in_context": entity_in_context,
        "entity_is_main_subject": entity_is_main_subject,
        "entity_corrected": entity_corrected,
        "entity_correction_reason": entity_correction_reason,
        "context_text": ctx_clean,
        "context_text_raw": ctx_raw,
        "context_quality": context_quality,
        "context_flag": context_flag,
        "context_in_article": context_in_article,
        "needs_reextract": needs_reextract,
        "context_len": len(ctx_clean),
        "article_text": article,
        "article_truncated": article_truncated,
        "pseudo_label": r["pseudo_label"],
        "gold_label": gold_label,
        "gold_relevancy": gold_rel,
        "label_source": label_source,
        "label_confidence": conf,
        "reasoning": reasoning,
        "premise": premise,
        "hypothesis": hypothesis,
        "bylines_removed": bylines_removed,
    })

# ---------------------------------------------------------------------------
# Write enhanced dataset
# ---------------------------------------------------------------------------
with open(OUT, "w") as f:
    for row in enhanced:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"\nWrote {len(enhanced)} enhanced rows -> {OUT}")

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
report = validate_dataset(enhanced)
print(f"\n{'='*60}\nVALIDATION REPORT\n{'='*60}")
print(f"  total rows    : {report['total']}")
print(f"  valid rows    : {report['valid']} ({report['validity_rate']*100:.1f}%)")
print(f"  invalid rows  : {report['invalid']}")
if report["violation_type_counts"]:
    print(f"\n  Violation types:")
    for k, v in report["violation_type_counts"].items():
        print(f"    {k}: {v}")
    print(f"\n  First 10 invalid rows:")
    for inv in report["first_10_invalid"]:
        print(f"    row {inv['row_index']}: {inv['violations']}")

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nENHANCED DATASET SUMMARY\n{'='*60}")

src_counts = Counter(r["label_source"] for r in enhanced)
print("\nLabel source breakdown:")
for s in ["gold_human", "llm_second_pass", "heuristic_background",
          "heuristic_speaker", "heuristic_corruption", "heuristic_polarity",
          "pseudo_kept", "llm_failed"]:
    c = src_counts.get(s, 0)
    print(f"  {s:25s} {c:4d}  ({c/len(enhanced)*100:.1f}%)")

well_labeled = sum(c for s, c in src_counts.items()
                   if s not in ("pseudo_kept", "llm_failed"))
print(f"\n  WELL-LABELED (verified): {well_labeled}/{len(enhanced)} "
      f"({well_labeled/len(enhanced)*100:.1f}%)")
print(f"  UNVERIFIED (pseudo):     {len(enhanced)-well_labeled}/{len(enhanced)} "
      f"({(len(enhanced)-well_labeled)/len(enhanced)*100:.1f}%)")

flag_counts = Counter(r["context_flag"] for r in enhanced)
print("\nContext flag breakdown:")
for f, c in flag_counts.most_common():
    print(f"  {f:25s} {c:4d}  ({c/len(enhanced)*100:.1f}%)")

label_counts = Counter(r["gold_label"] for r in enhanced)
print("\nGold label distribution (all rows):")
for l in SENTIMENT_LABELS:
    c = label_counts.get(l, 0)
    print(f"  {l:10s} {c:4d}  ({c/len(enhanced)*100:.1f}%)")

rel_counts = Counter(r["gold_relevancy"] for r in enhanced)
print("\nGold relevancy distribution:")
for l in RELEVANCY_LABELS:
    c = rel_counts.get(l, 0)
    print(f"  {l:15s} {c:4d}  ({c/len(enhanced)*100:.1f}%)")

# sentiment dataset (relevant only)
sent_rows = [r for r in enhanced if r["gold_relevancy"] == "relevant"]
sent_counts = Counter(r["gold_label"] for r in sent_rows)
print(f"\nSentiment dataset (relevant only, {len(sent_rows)} rows):")
for l in SENTIMENT_LABELS:
    c = sent_counts.get(l, 0)
    print(f"  {l:10s} {c:4d}  ({c/len(sent_rows)*100:.1f}%)")

# entity issues
wrong_entity = [r for r in enhanced if not r["entity_in_context"]]
print(f"\nEntity issues:")
print(f"  entity NOT in context: {len(wrong_entity)}")
corrected = [r for r in enhanced if r["entity_corrected"]]
print(f"  entity_corrected suggested: {len(corrected)}")
needs_re = [r for r in enhanced if r["needs_reextract"]]
print(f"  needs_reextract: {len(needs_re)}")

# quality distribution
q_buckets = Counter()
for r in enhanced:
    q = r["context_quality"]
    if q < 0.4: q_buckets["0.0-0.4 (bad)"] += 1
    elif q < 0.6: q_buckets["0.4-0.6 (poor)"] += 1
    elif q < 0.8: q_buckets["0.6-0.8 (ok)"] += 1
    else: q_buckets["0.8-1.0 (good)"] += 1
print(f"\nContext quality distribution:")
for b in ["0.0-0.4 (bad)", "0.4-0.6 (poor)", "0.6-0.8 (ok)", "0.8-1.0 (good)"]:
    print(f"  {b:20s} {q_buckets.get(b,0):4d}")

# agreement
agree = sum(1 for r in enhanced if r["pseudo_label"] == r["gold_label"])
print(f"\nPseudo vs gold agreement: {agree}/{len(enhanced)} ({agree/len(enhanced)*100:.1f}%)")

# save report
full_report = {**report, "summary": {
    "total": len(enhanced),
    "label_sources": dict(src_counts),
    "context_flags": dict(flag_counts),
    "gold_labels": dict(label_counts),
    "gold_relevancy": dict(rel_counts),
    "sentiment_labels_relevant": dict(sent_counts),
    "well_labeled": well_labeled,
    "unverified": len(enhanced) - well_labeled,
    "entity_not_in_context": len(wrong_entity),
    "entity_corrected": len(corrected),
    "needs_reextract": len(needs_re),
    "pseudo_vs_gold_agreement": agree / len(enhanced),
    "quality_buckets": dict(q_buckets),
}}
with open(REPORT, "w") as f:
    json.dump(full_report, f, indent=2, ensure_ascii=False)
print(f"\nFull report -> {REPORT}")
