#!/usr/bin/env python3
"""
dataset_schema.py
=================
Defines the ENHANCED dataset schema for ID-Political-Sentiment-Tracker and
enforces a set of validation rules (invariants) that every row must satisfy.

This is the "better dataset rules" the user asked for: not just labels, but
structural columns that catch:
  - entity utama salah        (wrong_entity)
  - context ngawur            (corruption_stitch / byline_leak / truncated)
  - entity hanya latar        (background_only)
  - entity sebagai pembicara  (speaker_not_target)
  - context terlalu pendek    (too_short)

Schema fields per row:
  --- identity ---
  raw_text_id            : str   (original article id)
  row_index              : int   (position in original dataset.jsonl)
  source_url             : str

  --- entity (with correction support) ---
  entity_name            : str   (original canonical name from extractor)
  entity_aliases_found   : list  (surface forms actually found in context)
  entity_in_context      : bool  (entity or alias literally present?)
  entity_is_main_subject : bool  (entity is the SUBJECT of sentiment, not bg?)
  entity_corrected       : str|null  (corrected entity if original was wrong)
  entity_correction_reason : str      (why the correction)

  --- context (with quality + flag) ---
  context_text           : str   (cleaned: bylines stripped)
  context_text_raw       : str   (original, pre-cleaning — for audit)
  context_quality        : float (0.0-1.0 composite score)
  context_flag           : str   (see CONTEXT_FLAGS below)
  context_in_article     : bool  (is context a substring of article_text?)
  needs_reextract        : bool  (should the article be re-fetched?)
  context_len            : int   (char length of cleaned context)

  --- article ---
  article_text           : str   (truncated original, for reference)
  article_truncated      : bool  (was article_text cut at 1000 chars?)

  --- labels (final, authoritative) ---
  pseudo_label           : str   (original broken model prediction)
  gold_label             : str   (positive | neutral | negative)
  gold_relevancy         : str   (relevant | not_relevant)
  label_source           : str   (see LABEL_SOURCES below)
  label_confidence       : float (0.0-1.0)
  reasoning              : str   (why this label — human or LLM justification)

  --- sentence-pair (ready for the base model) ---
  premise                : str   (entity_name + alias hint)
  hypothesis             : str   (cleaned context_text)

  --- audit ---
  bylines_removed        : list  (byline fragments stripped during cleaning)

Context flags (context_flag):
  clean                  — no issues detected
  byline_leak            — journalist byline / dateline leaked into context
  corruption_stitch      — context stitched from different articles (byline in body)
  background_only        — entity only a temporal/background anchor
  speaker_not_target     — entity is the speaker, not the target of sentiment
  wrong_entity           — entity_name doesn't match who the context is about
  too_short              — context < 60 chars (too little signal)
  too_long               — context > 1184 chars (exceeds model capacity)
  llm_failed             — LLM second-pass could not label (kept pseudo)

Label sources (label_source):
  gold_human             — human-labeled in gold_labels.jsonl (confidence 1.0)
  llm_second_pass        — LLM-labeled with strict prompt (confidence 0.85)
  heuristic_speaker      — rule detected speaker_vs_target (confidence 0.7)
  heuristic_background   — rule detected background mention (confidence 0.7)
  heuristic_polarity     — rule detected wrong polarity (confidence 0.6)
  heuristic_corruption   — rule detected corruption (confidence 0.9)
  pseudo_kept            — unverified, still pseudo-label (confidence 0.5)
  llm_failed             — LLM failed, kept pseudo (confidence 0.3)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
SENTIMENT_LABELS = ["positive", "neutral", "negative"]
RELEVANCY_LABELS = ["relevant", "not_relevant"]

CONTEXT_FLAGS = [
    "clean",
    "byline_leak",
    "corruption_stitch",
    "background_only",
    "speaker_not_target",
    "wrong_entity",
    "too_short",
    "too_long",
    "llm_failed",
]

LABEL_SOURCES = [
    "gold_human",
    "llm_second_pass",
    "heuristic_speaker",
    "heuristic_background",
    "heuristic_polarity",
    "heuristic_corruption",
    "pseudo_kept",
    "llm_failed",
]

# ---------------------------------------------------------------------------
# Required fields + their types
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = {
    "raw_text_id": str,
    "row_index": int,
    "source_url": str,
    "entity_name": str,
    "entity_aliases_found": list,
    "entity_in_context": bool,
    "entity_is_main_subject": bool,
    "entity_corrected": (type(None), str),
    "entity_correction_reason": str,
    "context_text": str,
    "context_quality": float,
    "context_flag": str,
    "context_in_article": bool,
    "needs_reextract": bool,
    "context_len": int,
    "article_text": str,
    "article_truncated": bool,
    "pseudo_label": str,
    "gold_label": str,
    "gold_relevancy": str,
    "label_source": str,
    "label_confidence": float,
    "reasoning": str,
    "premise": str,
    "hypothesis": str,
    "bylines_removed": list,
}

# ---------------------------------------------------------------------------
# Validation rules (invariants)
# ---------------------------------------------------------------------------
def validate_row(row: dict) -> list[str]:
    """Return a list of violation messages. Empty list = valid row."""
    violations = []

    # 1. Required fields present + correct type
    for fname, ftype in REQUIRED_FIELDS.items():
        if fname not in row:
            violations.append(f"missing_field:{fname}")
            continue
        v = row[fname]
        if ftype == (type(None), str):
            if v is not None and not isinstance(v, str):
                violations.append(f"type_error:{fname}")
        elif not isinstance(v, ftype):
            violations.append(f"type_error:{fname}(expected {ftype.__name__})")

    if violations:
        return violations  # can't check further if fields missing

    # 2. Enum membership
    if row["gold_label"] not in SENTIMENT_LABELS:
        violations.append(f"invalid_gold_label:{row['gold_label']}")
    if row["gold_relevancy"] not in RELEVANCY_LABELS:
        violations.append(f"invalid_gold_relevancy:{row['gold_relevancy']}")
    if row["context_flag"] not in CONTEXT_FLAGS:
        violations.append(f"invalid_context_flag:{row['context_flag']}")
    if row["label_source"] not in LABEL_SOURCES:
        violations.append(f"invalid_label_source:{row['label_source']}")
    if row["pseudo_label"] not in SENTIMENT_LABELS:
        violations.append(f"invalid_pseudo_label:{row['pseudo_label']}")

    # 3. Cross-field invariants
    # 3a. If entity not in context AND no correction -> wrong_entity flag
    if not row["entity_in_context"] and row["entity_corrected"] is None:
        if row["context_flag"] == "clean":
            violations.append("entity_missing_but_flagged_clean")

    # 3b. If context_flag is corruption_stitch -> needs_reextract must be True
    if row["context_flag"] == "corruption_stitch" and not row["needs_reextract"]:
        violations.append("corruption_but_no_reextract")

    # 3c. If gold_relevancy == not_relevant -> gold_label should be neutral
    #     (a non-relevant context has no meaningful sentiment toward entity)
    if row["gold_relevancy"] == "not_relevant" and row["gold_label"] != "neutral":
        violations.append(f"not_relevant_but_label={row['gold_label']}")

    # 3d. label_confidence must be in [0, 1]
    if not (0.0 <= row["label_confidence"] <= 1.0):
        violations.append(f"confidence_out_of_range:{row['label_confidence']}")

    # 3e. context_quality must be in [0, 1]
    if not (0.0 <= row["context_quality"] <= 1.0):
        violations.append(f"quality_out_of_range:{row['context_quality']}")

    # 3f. premise must contain entity_name (or corrected entity)
    expected_entity = row["entity_corrected"] or row["entity_name"]
    if expected_entity not in row["premise"]:
        violations.append("premise_missing_entity_name")

    # 3g. hypothesis must equal context_text (sentence-pair consistency)
    if row["hypothesis"] != row["context_text"]:
        violations.append("hypothesis_neq_context_text")

    # 3h. if label_source is gold_human -> confidence must be 1.0
    if row["label_source"] == "gold_human" and row["label_confidence"] != 1.0:
        violations.append("gold_human_but_confidence_not_1")

    return violations


def validate_dataset(rows: list[dict]) -> dict:
    """Validate an entire dataset. Returns a summary report."""
    total = len(rows)
    all_violations = []
    valid = 0
    invalid = 0
    violation_counts = Counter()

    for i, row in enumerate(rows):
        v = validate_row(row)
        if v:
            invalid += 1
            all_violations.append({"row_index": row.get("row_index", i), "violations": v})
            for msg in v:
                # bucket the violation type
                vtype = msg.split(":")[0]
                violation_counts[vtype] += 1
        else:
            valid += 1

    return {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "validity_rate": round(valid / total, 4) if total else 0,
        "violation_type_counts": dict(violation_counts.most_common()),
        "first_10_invalid": all_violations[:10],
    }


# ---------------------------------------------------------------------------
# Context quality calculator (composite score)
# ---------------------------------------------------------------------------
def calculate_context_quality(
    context_text: str,
    entity_in_context: bool,
    context_in_article: bool,
    bylines_removed: list,
    context_flag: str,
) -> float:
    """Composite quality score in [0, 1].

    Components (each contributes 0.2):
      1. entity_present       — entity (or alias) literally in context
      2. no_bylines           — no byline fragments were stripped
      3. in_article           — context is a substring of article_text
      4. length_ok            — 60 <= len <= 1184
      5. flag_is_clean        — context_flag == 'clean' or 'speaker_not_target'
    """
    score = 0.0
    if entity_in_context:
        score += 0.2
    if not bylines_removed:
        score += 0.2
    if context_in_article:
        score += 0.2
    if 60 <= len(context_text) <= 1184:
        score += 0.2
    if context_flag in ("clean", "speaker_not_target"):
        score += 0.2
    return round(score, 2)
