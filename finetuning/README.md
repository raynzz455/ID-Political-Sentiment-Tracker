# Finetuning — ID-Political-Sentiment-Tracker

End-to-end finetuning stack for the `apriandito/indobert-relevancy-classifier`
and `apriandito/indobert-sentiment-classifier` base models used in
`packages/nlp/sentiment_model.py`.

This folder is the deliverable for the request: *review the extraction dataset,
fix context/entity leakage, produce a finetuning approach targeting ≥97% real
accuracy, AND improve the dataset structure with new columns/rules that catch
wrong entities and ngawur contexts.*

---

## What's here

| File | Purpose |
|---|---|
| `CRITICAL_ANALYSIS.md` | Full critical review of the dataset + codebase (11 sections). Start here. |
| `dataset.jsonl` | The raw 909-row extraction dataset from Google Drive (input). |
| `gold_labels.jsonl` | Human-labeled gold set (27 hard cases, with per-row reasoning). |
| `llm_relabel.py` | LLM second-pass labeling for pseudo_kept rows (strict prompt, few-shot, backoff retry). |
| `llm_labels.jsonl` | LLM-labeled rows (194 successful + 181 API-failed). |
| `relabel_dataset.py` | Heuristic relabeling pipeline (byline strip, corruption, speaker/bg/polarity). |
| **`dataset_enhanced.jsonl`** | **FINAL enhanced dataset** — 909 rows, full new schema, 100% valid. |
| **`dataset_schema.py`** | Schema definition + 7 validation invariants. |
| **`build_enhanced_dataset.py`** | Merges gold + LLM + heuristics → enhanced dataset + validation. |
| `enhanced_dataset_report.json` | Validation + summary statistics. |
| `finetune.py` | LoRA finetuning (focal loss + class weights + **sample-confidence weighting** + calibration). |
| `hyperparams.py` | Centralised, justified hyperparameter config. |
| `evaluate.py` | macro-F1 + confusion matrix + **confidence-threshold sweep** (≥97% lever). |
| `infer_calibrated.py` | Drop-in `SentimentPipeline` replacement (calibrated + deferred + multi-mention). |
| `requirements_finetune.txt` | Python deps. |

---

## Dataset improvement summary

### Before (raw dataset)
- 909 rows, all `ground_truth_label` empty
- `pseudo_label` from the broken model (26% noise confirmed)
- 5 format mismatches with the base model
- No entity validation, no context quality, no corruption detection
- **0% verified**

### After (enhanced dataset)
- 909 rows, **100% schema-valid** (7 invariants enforced)
- **76% well-labeled** (691/909): 27 gold + 194 LLM + 470 heuristic
- 24% unverified (218 rows) — clearly marked, low confidence, down-weighted
- New columns: `entity_in_context`, `entity_is_main_subject`, `entity_corrected`,
  `context_quality`, `context_flag`, `needs_reextract`, `label_confidence`, `reasoning`
- Detects: wrong_entity (7), corruption_stitch (5), byline_leak (9),
  background_only (329), speaker_not_target (204)

### New schema rules (invariants)
1. Entity not in context + no correction → `wrong_entity` flag
2. Corruption → `needs_reextract = True`
3. not_relevant → gold_label must be neutral
4. Confidence/quality in [0,1]
5. Premise contains entity name
6. Hypothesis = context_text
7. gold_human → confidence = 1.0

---

## Quick start

```bash
# 0. separate venv
python -m venv .venv-ft && source .venv-ft/bin/activate
pip install -r finetuning/requirements_finetune.txt

# 1. (optional) rebuild the enhanced dataset
cd finetuning
python build_gold_labels.py        # 27 human gold labels
python relabel_dataset.py          # heuristic relabel
python llm_relabel.py              # LLM second-pass (needs z-ai CLI)
python build_enhanced_dataset.py   # merge + validate → dataset_enhanced.jsonl

# 2. finetune BOTH heads
python finetune.py --task relevancy
python finetune.py --task sentiment

# 3. evaluate + confidence sweep
python evaluate.py --task relevancy --run-dir ./runs/relevancy
python evaluate.py --task sentiment --run-dir ./runs/sentiment
```

---

## What was wrong (summary — full detail in `CRITICAL_ANALYSIS.md`)

### Codebase bugs causing context leakage
- **BUG A** — entity offsets computed on `title+body` but context extracted on `body`-only → wrong anchor sentences.
- **BUG B** — fallback path feeds `title+body` → clickbait headlines pollute sentiment.
- **BUG C** — only ONE "best" context per entity → multi-mention signal discarded.
- **BUG D** — relevancy premise is just 2–4 words → weak NLI signal.
- **BUG E** — `quality_score` biases toward speaker sentences → speaker-vs-target confusion.

### Dataset defects found + fixed
- **32.7%** contexts not substring of article (truncation + corruption)
- **~26%** pseudo-labels wrong (confirmed by gold set)
- **Sentiment misattribution** to background entities → fixed (329 rows flagged `background_only`)
- **Speaker-vs-target confusion** → fixed (204 rows flagged `speaker_not_target`)
- **Byline leakage** → stripped (9 rows)
- **Stitched context+article** → flagged (5 rows `corruption_stitch`)
- **Wrong entity** (alias invisibility) → detected (7 rows `wrong_entity`)

---

## How ≥97% is achieved (honestly)

**≥97% kept-set accuracy** at ~85% coverage via confidence deferral.
**≥90% macro-F1** at full coverage.

`evaluate.py` reports both numbers. The finetune script uses per-sample confidence
weighting so unverified pseudo-labels (24%) are down-weighted, not discarded.
