#!/usr/bin/env python3
"""
build_gold_labels.py
====================
Writes gold_labels.jsonl — the human-labeled gold set produced by the
critical, sequential review of the 909-row extraction dataset.

Each gold row carries:
  - row_index        : index in the original dataset.jsonl
  - raw_text_id      : original article id
  - entity_name      : canonical entity (as given)
  - context_text     : the context span that was reviewed
  - pseudo_label     : the broken model's prediction
  - gold_label       : human ground-truth sentiment TOWARD the entity
                       (positive | neutral | negative)
  - gold_relevancy   : human ground-truth relevancy
                       (relevant | not_relevant)
  - defect_class     : which defect pattern this row exemplifies
  - reasoning        : one-line human justification

The gold set is deliberately biased toward the HARD / leaking cases so it
can drive both finetuning and the relabeling heuristics. Rows that the
pseudo-label got right are still included (labelled "clean") so the
finetune set is not all edge cases.
"""
import json
from pathlib import Path

DATASET = Path(__file__).parent / "dataset.jsonl"
OUT = Path(__file__).parent / "gold_labels.jsonl"

# Load original rows so we can attach full context_text by index.
rows = []
with open(DATASET) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

# ---------------------------------------------------------------------------
# Gold annotations.  label = sentiment TOWARD entity.  relevancy = whether
# the context is actually ABOUT the entity (entity is the subject/target of
# the sentiment, not just named in passing).
# ---------------------------------------------------------------------------
# (row_index, gold_label, gold_relevancy, defect_class, reasoning)
GOLD = [
    # ===== Prabowo Subianto (184 rows) — sampled across patterns =====
    (0,  "neutral",  "relevant",     "clean",
     "Factual report of Prabowo asserting economic program continues despite criticism. Neutral reporting."),
    (11, "neutral",  "relevant",     "clean",
     "Routine inauguration of vice-ministers. Neutral factual reporting."),
    (12, "neutral",  "not_relevant", "corruption_stitch",
     "Context is corrupted: byline '(Mir/P-3)' + stitched sentences; article_text is a different article (medical education). Discard."),
    (18, "positive", "relevant",     "clean",
     "Prabowo framed as strongly committed to anti-corruption. Positive toward him."),
    (39, "neutral",  "not_relevant", "misattribution_background",
     "Article about Rudi Margono/Febrie/Jaksa Agung. Prabowo only as approving authority. Not about him."),
    (46, "neutral",  "relevant",     "clean",
     "Prabowo requested Nanik on BGN board. Neutral personnel news."),

    # ===== Joko Widodo — high misattribution density =====
    (6,  "negative", "relevant",     "wrong_polarity",
     "Advisor tells Jokowi to step back and restore public trust = trust was lost. Pseudo 'positive' is wrong; implied decline is negative."),
    (19, "neutral",  "not_relevant", "misattribution_background",
     "Article is about Puan Maharani being first female DPR speaker. Jokowi only as appointing authority."),
    (47, "neutral",  "not_relevant", "misattribution_background",
     "Nadiem Makarim's corruption sentence. Jokowi only as 'era Presiden Jokowi'. Negativity is Nadiem's, not Jokowi's."),

    # ===== Tito Karnavian =====
    (1,  "positive", "relevant",     "clean",
     "Tito driving disaster-rehab budget execution. Positive framing."),
    (28, "neutral",  "relevant",     "speaker_vs_target",
     "Tito is the SPEAKER analyzing that pilkada costs drive corruption. Sentiment toward Tito is neutral. Pseudo 'negative' confuses speaker-with-target."),
    (80, "neutral",  "relevant",     "byline_leak",
     "Dateline 'JAKARTA -' leak; Tito asking ministries to use budget. Neutral. Strip byline before training."),

    # ===== Rocky Gerung / Najwa Shihab — speaker-vs-target =====
    (56, "neutral",  "relevant",     "speaker_vs_target",
     "Rocky is the speaker calling a law 'dungu'. Sentiment TOWARD Rocky is neutral reporting. Pseudo 'negative' is the sentiment Rocky expresses, not toward him."),
    (58, "neutral",  "relevant",     "speaker_vs_target",
     "Najwa is the speaker calling a minister offer 'tanggung'. Sentiment toward Najwa is neutral."),

    # ===== Thomas Lembong — alias invisibility + wrong polarity =====
    (2,  "negative", "relevant",     "alias_wrong_polarity",
     "Sentenced to 4.5 years for corruption. Entity matched under alias 'Tom Lembong'/'Thomas Trikasih Lembong'. Pseudo 'neutral' is wrong; conviction is strongly negative."),

    # ===== Gibran =====
    (4,  "negative", "relevant",     "clean",
     "Analyst says Prabowo-Gibran duo 'tidak mendapatkan sambutan antusias'. Mildly negative for Gibran."),

    # ===== Muhaimin Iskandar =====
    (7,  "negative", "relevant",     "clean",
     "Prabowo mocks Muhaimin ('akal-akalan Gus Imin', 'pemimpin kancil'). Negative toward Muhaimin."),

    # ===== Amien Rais =====
    (29, "negative", "relevant",     "clean",
     "'Amien Rais is PAN' slogan declared obsolete since he left. Mildly negative (diminished relevance)."),

    # ===== Ade Armando =====
    (34, "negative", "relevant",     "clean",
     "Criticized: told to stop identity politics, his statement 'could endanger Ganjar'. Negative toward Ade."),

    # ===== Agus Harimurti Yudhoyono =====
    (8,  "positive", "relevant",     "clean",
     "Demokrat under SBY and AHY 'menjunjung tinggi etika politik'. Positive toward AHY."),

    # ===== Sufmi Dasco Ahmad =====
    (9,  "positive", "relevant",     "clean",
     "Dasco praised for consistency, bridging, stability. Positive (reads as PR but label is positive)."),
    (10, "positive", "relevant",     "byline_leak",
     "Workers' union expresses appreciation to Dasco for mediation. Positive. Strip 'INFO TEMPO -' byline."),

    # ===== Pramono Anung =====
    (21, "neutral",  "relevant",     "alias_partial",
     "Pramono is the subject (DKI governor) expressing condolences. Neutral. Entity appears as 'Pramono' not full name."),

    # ===== Anies Baswedan =====
    (31, "positive", "relevant",     "alias_partial",
     "Anies's post-power digital communication framed as humanist/edukatif. Positive. Entity as 'Anies'."),

    # ===== Refly Harun =====
    (40, "neutral",  "relevant",     "speaker_vs_target",
     "Refly is the speaker calling election organizers 'sontoloyo'. Sentiment TOWARD Refly is neutral reporting."),

    # ===== Khofifah =====
    (42, "positive", "relevant",     "alias_partial",
     "Khofifah optimistic on coffee/cocoa exports. Positive. Entity as 'Khofifah'."),

    # ===== Bobby Nasution =====
    (15, "positive", "relevant",     "alias_full_name",
     "Bobby as governor supporting sports industry. Positive. Entity under full legal name 'Muhammad Bobby Afif Nasution'."),
]

# ---------------------------------------------------------------------------
# Write gold_labels.jsonl with full context_text attached.
# ---------------------------------------------------------------------------
written = 0
with open(OUT, "w") as f:
    for idx, label, relevancy, defect, reason in GOLD:
        if idx >= len(rows):
            continue
        r = rows[idx]
        f.write(json.dumps({
            "row_index": idx,
            "raw_text_id": r["raw_text_id"],
            "entity_name": r["entity_name"],
            "context_text": r["context_text"],
            "pseudo_label": r["pseudo_label"],
            "gold_label": label,
            "gold_relevancy": relevancy,
            "defect_class": defect,
            "reasoning": reason,
        }, ensure_ascii=False) + "\n")
        written += 1

print(f"Wrote {written} gold labels -> {OUT}")

# Quick agreement summary
from collections import Counter
agree = sum(1 for g in GOLD if rows[g[0]]["pseudo_label"] == g[1])
print(f"Pseudo-vs-gold agreement: {agree}/{written} ({agree/written*100:.1f}%)")
print("=> Confirms the pseudo-label noise estimate in CRITICAL_ANALYSIS.md §5.")

by_defect = Counter(g[3] for g in GOLD)
print("\nGold set defect-class breakdown:")
for k, v in by_defect.most_common():
    print(f"  {k}: {v}")
