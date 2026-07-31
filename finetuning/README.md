# Finetuning — ID-Political-Sentiment-Tracker

End-to-end finetuning stack for the `apriandito/indobert-relevancy-classifier`
and `apriandito/indobert-sentiment-classifier` base models used in
`packages/nlp/sentiment_model.py`.

This folder is the deliverable for the request: *review the extraction dataset,
fix context/entity leakage, and produce a finetuning approach that can hit
≥97% real accuracy with mathematical/statistical justification.*

---

## What's here

| File | Purpose |
|---|---|
| `CRITICAL_ANALYSIS.md` | Full critical review of the dataset + codebase. Start here. |
| `dataset.jsonl` | The raw 909-row extraction dataset from Google Drive (input). |
| `gold_labels.jsonl` | Human-labeled gold set (27 hard cases, with per-row reasoning). |
| `build_gold_labels.py` | Rebuilds `gold_labels.jsonl` from the critical review. |
| `relabel_dataset.py` | Cleans bylines, flags corruption, applies gold + heuristics → emits train-ready sentence-pair datasets. |
| `dataset_relevancy.jsonl` | `(entity_name, context_text) → relevant\|not_relevant` (909 rows, 65/35 split). |
| `dataset_sentiment.jsonl` | `(entity_name, context_text) → positive\|neutral\|negative` (593 relevant rows, 21/68/11 split). |
| `relabel_audit.json` | Full audit of every relabeling decision. |
| `hyperparams.py` | Centralised, justified hyperparameter config. |
| `finetune.py` | LoRA finetuning (focal loss + class weights + early stopping + calibration). |
| `evaluate.py` | macro-F1 / confusion matrix + **confidence-threshold sweep** (the ≥97% lever). |
| `infer_calibrated.py` | Drop-in replacement for `SentimentPipeline` (calibrated + deferred + multi-mention aggregation). Fixes BUGs B, C, D. |
| `requirements_finetune.txt` | Python deps (separate venv from the Next.js app). |

---

## Quick start

```bash
# 0. separate venv (the ML stack is heavy; do NOT pollute the Next.js app)
python -m venv .venv-ft && source .venv-ft/bin/activate
pip install -r finetuning/requirements_finetune.txt

# 1. (optional) rebuild the gold set + relabeled datasets
cd finetuning
python build_gold_labels.py
python relabel_dataset.py

# 2. finetune BOTH heads of the 2-stage pipeline
python finetune.py --task relevancy
python finetune.py --task sentiment

# 3. evaluate + confidence-threshold sweep
python evaluate.py --task relevancy --run-dir ./runs/relevancy
python evaluate.py --task sentiment --run-dir ./runs/sentiment
```

---

## What was wrong (summary — full detail in `CRITICAL_ANALYSIS.md`)

### Codebase bugs causing context leakage
- **BUG A** — entity offsets are computed on `title+body` but context is extracted on `body`-only → wrong anchor sentences.
- **BUG B** — fallback path feeds `title+body` to the document model → clickbait headlines pollute sentiment.
- **BUG C** — only ONE "best" context per entity per article is kept → multi-mention sentiment signal is discarded.
- **BUG D** — relevancy premise is just the 2–4-word entity name → weak NLI signal.
- **BUG E** — heuristic `quality_score` biases toward attribution (speaker) sentences → conflates "sentiment BY the entity" with "sentiment TOWARD the entity".

### Dataset defects found in critical sequential review
- **32.7%** of contexts are not an exact substring of `article_text` (article hard-truncated to 1000 chars in export + some genuine stitch corruption).
- **~26%** of pseudo-labels are wrong (confirmed by 27-row gold set: 74.1% pseudo-vs-gold agreement).
- **Sentiment misattribution** to background-mentioned entities (e.g. Jokowi named as "era Presiden Jokowi" in Nadiem's corruption article → labeled negative for Jokowi).
- **Speaker-vs-target confusion** (e.g. Rocky criticizing a law → labeled negative, but Rocky is the speaker).
- **Byline/metadata leakage** in ~3.5% of contexts (`(Mir/P-3)`, `JAKARTA -`, `INFO TEMPO -`).
- **Stitched context+article** (Row 12: context from Tempo Koperasi-Day article + article_text from a medical-education article).

### Format mismatches with the base model
1. Single `pseudo_label` but the pipeline is 2-stage (relevancy + sentiment) — no relevancy ground truth.
2. No sentence-pair structure — the model expects `tokenizer(entity_name, context_text)`.
3. Contexts up to 1184 chars exceed `MAX_SEQ_LENGTH=256` → silent truncation at inference.
4. `pseudo_label` is from the broken model → training on it propagates errors.
5. Class imbalance 58/27/15 → naive cross-entropy collapses to neutral.

---

## How ≥97% is achieved (honestly)

A flat 97% on 3-class Indonesian political sentiment with full coverage is **not**
honestly achievable from 909 rows with the current label noise. The honest target
is **≥97% kept-set accuracy** at ~85% coverage, via:

1. **Clean gold labels** that remove the ~26% label noise.
2. **Confidence-based deferral** — defer the ~12–15% lowest-confidence predictions
   to a human/LLM second pass. `evaluate.py` sweeps `tau` and reports the
   kept-accuracy vs coverage curve.
3. **Focal loss + class-balanced weights** to push decision-boundary confidence
   high on easy cases and force calibration on hard ones.

The identity is:

```
EffectiveAccuracy = |{x in K : pred(x)=y(x)}| / |K|      (kept-set accuracy)
Coverage          = |K| / N                                (fraction not deferred)
K = {x : max_softmax(p(x)) >= tau}
```

`evaluate.py` prints the full sweep and flags every `tau` that hits ≥97%. The
report states both numbers (kept-accuracy AND coverage) so the target is
verifiable, not asserted.

---

## Drop-in production replacement

`infer_calibrated.py` exports `get_pipeline()` with the **same `predict_gated(text, context)`
API** as `packages/nlp/sentiment_model.py`. To upgrade the production worker:

```python
# in packages/nlp/nlp_worker.py
# OLD:
#   from packages.nlp.sentiment_model import get_pipeline
# NEW:
from infer_calibrated import get_pipeline   # calibrated + deferred + multi-mention
```

It also adds `predict_gated_multi(entity_name, contexts)` for the upgraded worker
that passes ALL context spans per entity (fixes BUG C — the original kept only one).

The `GatedResult` gains a `deferred: bool` field so the worker can route
low-confidence predictions to a human/LLM second pass instead of emitting a
noisy label.
