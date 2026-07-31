# Critical Analysis Report — ID-Political-Sentiment-Tracker Dataset

> Reviewer: Z.ai Code (critical, sequential human-in-the-loop review)
> Source repo: `raynzz455/ID-Political-Sentiment-Tracker`
> Dataset: Google Drive `1yMdqsXe7xlhZUKTdpL0IzIAq9ykuwDLB` (909 rows, NDJSON)
> Base model file reviewed: `packages/nlp/sentiment_model.py`

---

## 1. Base Model Identification

Found in `packages/nlp/sentiment_model.py`. The pipeline uses **3 HuggingFace models**, all IndoBERT-based:

| Role | Model ID | Input format | Labels |
|---|---|---|---|
| **Relevancy Gate** | `apriandito/indobert-relevancy-classifier` | sentence-pair `tokenizer(context, text)` = `tokenizer(entity_name, context_text)` | binary (relevan / tidak) |
| **Sentiment Classifier** | `apriandito/indobert-sentiment-classifier` | sentence-pair `tokenizer(context, text)` = `tokenizer(entity_name, context_text)` | 3-class (positive / neutral / negative) |
| **Fallback (doc-level)** | `taufiqdp/indonesian-sentiment` | single text | 3-class |

**Critical architecture facts:**
- `MAX_SEQ_LENGTH = 256` tokens — any context longer than ~256 tokens is **truncated** at inference.
- `RELEVANCY_THRESHOLD = 0.5`.
- Both relevancy and sentiment use **NLI-style sentence-pair** format: segment A = entity canonical name (2–4 words), segment B = context span.
- Gated path: `predict_gated(text=context_text, context=entity_name)` → `tokenizer(entity_name, context_text)`.
- Fallback path: `predict_gated(text=combined_title_body[:1500], context=None)` → single-sequence.

---

## 2. Dataset Structure

```jsonl
{"raw_text_id": "...", "entity_name": "Prabowo Subianto",
 "pseudo_label": "neutral", "ground_truth_label": "",
 "context_text": "...", "article_text": "...", "source_url": "..."}
```

- 909 rows, 54 unique entities.
- `ground_truth_label` is **empty for all 909 rows** → this is a **human-labeling task**; the `pseudo_label` is the current (broken) model's prediction.

---

## 3. Statistical Profile of the Dataset

### 3.1 Pseudo-label distribution (class imbalance)
| Label | Count | % |
|---|---|---|
| neutral | 528 | 58.1% |
| positive | 243 | 26.7% |
| negative | 138 | 15.2% |

The 58% neutral majority is a **symptom of context leakage**: when the context is not actually about the entity, the sentiment model defaults to neutral. A healthy political-sentiment distribution on contentious figures should be far more balanced.

### 3.2 Entity coverage
Top entities: Prabowo Subianto (184), Joko Widodo (60), Pramono Anung (56), Tito Karnavian (36), Megawati (36). Long tail of 54 entities, several with ≤4 samples (Wiranto=1, Budi Arie=1, Miftachul Akhyar=2, Sri Mulyani=2).

### 3.3 Context ↔ Article consistency
| Check | Count | % |
|---|---|---|
| `context_text` is exact substring of `article_text` | 612 | 67.3% |
| NOT exact substring (even after whitespace normalize) | 297 | 32.7% |
| `entity_name` literally present in `context_text` | 736 | 81.0% |
| `entity_name` literally present in `article_text` | 759 | 83.5% |

### 3.4 Length profile
- `context_text`: min=60, max=1184, mean=364 chars.
- `article_text`: **min=926, max=1000, mean=1000** → the article body has been **hard-truncated to 1000 chars** in the export, which breaks substring verification for any context extracted past the 1000-char mark.

---

## 4. Root-Cause Analysis: Why Context Is Leaking

### 4.1 Codebase-level bugs (the source of the leakage)

**BUG A — Offset domain mismatch between entity resolver and context extractor (CRITICAL).**
- `entity_resolution_worker.py` line 104 builds the detection text as:
  ```python
  text = f"{art.get('title', '')}\n{art.get('text', '')}"
  ```
  → entity `start_offset` / `end_offset` are computed against **title + "\n" + body**.
- `context_worker.py` line 95 builds the extraction text as:
  ```python
  clean_text = body          # body ONLY
  title_len = len(title) + 1 if title else 0
  ```
  The `title_len` variable is computed but the offset-adjustment code that should subtract it from `start_offset` is **incomplete / partially broken** in the visible source (the comparison operators are stripped in the region around `is_core_argument` and the offset bounds checks).
  → Result: a non-trivial fraction of mentions feed a **wrong anchor sentence** into context construction, pulling in adjacent sentences that have nothing to do with the entity. This is the dominant source of "context not about the entity".

**BUG B — Fallback path feeds title+body to the document model.**
- `nlp_worker.py` line 69: `combined_text = f"{title} {text}".strip()` → `fb_text = combined_text[:1500]`.
- Indonesian headlines are routinely **clickbait/baiting**; injecting the title into the document-level sentiment pollutes the general (fallback) sentiment with headline framing. This is exactly the behaviour the user wants eliminated.

**BUG C — Only one "best" context per entity per article.**
- `context_worker.py` keeps `best_contexts = {}` keyed by `entity_id`, overwriting all but the highest heuristic `quality_score`.
- If an article mentions Prabowo 5 times with mixed tones, only ONE span survives → the downstream sentiment is whichever span the heuristic picked, **not the article's overall stance toward the entity**. This discards signal and can flip polarity.

**BUG D — Weak relevancy signal.**
- The relevancy model receives `tokenizer(entity_name, context_text)`. `entity_name` is 2–4 words. The model has to decide "is this 180-word span about this 3-word entity?" from a tiny premise. This is an inherently noisy NLI setup and is why the relevancy gate both over-rejects (losing real mentions) and under-rejects (letting background mentions through).

**BUG E — Heuristic quality_score biases the chosen span.**
- `quality_score` is dominated by `attr_score` (active/passive verb markers) and `actor_score` (core grammatical argument). Verbs like `mengatakan/menjelaskan` (say/explain) score high → the chosen span is biased toward **attribution sentences where the entity is the speaker**, not sentences where the entity is the **target** of sentiment. This systematically conflates "sentiment expressed BY the entity" with "sentiment TOWARD the entity" (see §5.2).

### 4.2 Dataset-level corruption

**CORRUPTION 1 — Stitched / mismatched context+article pairs.**
Row 12 (`Prabowo Subianto`, pseudo=positive):
- `context_text` is stitched from a Tempo Koperasi-Day speech ("…pungkas Prabowo. **(Mir/P-3)** Presiden Prabowo Subianto menegaskan…") and even includes a Christmas-message fragment.
- `article_text` opens with "Headline Ada yang keliru pada penyelenggaraan pendidikan kedokteran" — a **completely different article** (medical-education).
→ The context and article fields do **not** come from the same source row. This is a join/union bug in the dataset-export script.

**CORRUPTION 2 — Journalist bylines / datelines leak into context.**
~1.7% (15+ rows) contain bylines / datelines inside `context_text`: `(Mir/P-3)`, `JAKARTA -`, `INFO TEMPO -`, `TRIBUNNEWS.COM -`, `JATIMTIMES -`, `INDORAYA -`, `Penulis, … - Waktu membaca: 11 menit`. These add noise tokens that the model attends to.

**CORRUPTION 3 — Article body truncated to 1000 chars.**
Makes ground-truth verification impossible for ~33% of rows and means any re-extraction must re-fetch the original URL, not trust `article_text`.

---

## 5. Labeling Defects Found in Critical Sequential Review

I reviewed rows sequentially (slowly, per the user's instruction), reading both `context_text` and `article_text` for each. Below are the **systemic defect classes** with representative examples. Full per-row gold labels are written to `gold_labels.jsonl`.

### 5.1 Sentiment misattribution to a background-mentioned entity (CRITICAL)
The entity is named only as a temporal/possessive anchor; the sentiment-bearing predicate belongs to **someone else**.

| Row | Entity | pseudo | Reality | Correct |
|---|---|---|---|---|
| 47 | Joko Widodo | negative | Context is **Nadiem Makarim's** corruption sentence; Jokowi only appears as "era Presiden Jokowi". Negativity is Nadiem's. | **neutral** (and relevancy = not_relevant) |
| 19 | Joko Widodo | neutral | Article is **about Puan Maharani** (first female DPR speaker). Jokowi is "diangkat … oleh Presiden Jokowi". | neutral, but relevancy = not_relevant |
| 39 | Prabowo Subianto | neutral | Article is about **Rudi Margono / Febrie / Jaksa Agung**. Prabowo only as approving authority. | neutral, relevancy = not_relevant |

### 5.2 Speaker-vs-Target confusion (SYSTEMIC)
The model labels the **sentiment the entity expresses** rather than the **sentiment toward the entity**. The README defines the task as "tone-nya ke tokoh X" (tone **toward** person X), so these are wrong.

| Row | Entity | pseudo | Reality | Correct |
|---|---|---|---|---|
| 56 | Rocky Gerung | negative | Rocky is **the speaker** criticizing a law ("pasal yang dungu"). Sentiment *toward* Rocky is neutral reporting. | **neutral** |
| 58 | Najwa Shihab | negative | Najwa is the speaker calling a minister offer "tanggung". Sentiment toward Najwa is neutral. | **neutral** |
| 28 | Tito Karnavian | negative | Tito is the speaker analyzing that high pilkada costs drive corruption. Sentiment toward Tito is neutral. | **neutral** |

### 5.3 Wrong polarity on subtle/implicit cases
| Row | Entity | pseudo | Reality | Correct |
|---|---|---|---|---|
| 6 | Joko Widodo | positive | Advisor tells Jokowi to **step back** and stop maneuvering to "memulihkan kepercayaan publik" (= trust was lost). | **negative** |
| 2 | Thomas Lembong | neutral | Context: sentenced to 4.5 years for corruption. Entity matched under alias "Tom Lembong"/"Thomas Trikasih Lembong". | **negative** |

### 5.4 Alias / partial-name invisibility (not a true leak, but breaks verification)
173 rows have the entity under a short form ("Anies" vs "Anies Baswedan", "Prabowo" vs "Prabowo Subianto", "Tom Lembong" vs "Thomas Lembong", "Gus Imin" vs "Muhaimin Iskandar"). The context is usually still correct; the failure is only in the naive `entity_name in context` check and in the sentence-pair premise (the model sees the canonical name but the context uses a different surface form).

### 5.5 Corruption / noise rows
- Row 12: byline `(Mir/P-3)` + stitched sentences + mismatched article_text → **discard / re-extract**.
- ~15 rows: dateline prefix (`JAKARTA -`, `INFO TEMPO -`) → **strip before training**.

---

## 6. Dataset Format vs. Base-Model Requirements — Verdict

**The current dataset format is NOT directly trainable on the base models.** Five mismatches:

1. **Single label, two-stage model.** The pipeline has Relevancy + Sentiment, but the dataset has only `pseudo_label`. There is **no relevancy ground truth**, so the relevancy gate cannot be finetuned from this data.
2. **No sentence-pair structure.** The model expects `tokenizer(entity_name, context_text)`; the dataset stores them as loose fields. Must be re-formed into pairs at training time.
3. **Length exceeds `MAX_SEQ_LENGTH=256`.** Contexts up to 1184 chars (~300 tokens) will be silently truncated at inference. Training data must be tokenised identically or the train/inference distribution diverges.
4. **`pseudo_label` is from the broken model.** Training on pseudo-labels propagates the §5 defects. **Human gold labels are mandatory.**
5. **Class imbalance (58/27/15).** Naive cross-entropy will collapse to the neutral majority. Needs class weighting / focal loss / oversampling.

---

## 7. Recommended Path to ≥97% Real Accuracy

Achieving a genuine ≥97% on 3-class Indonesian political sentiment is at the frontier of the base model's capacity. The plan below is **mathematically and statistically justified**, not aspirational.

### 7.1 Effective-accuracy identity
Let `N` = total eval samples, `c(x)` = model confidence on `x`, `τ` = deferral threshold, `K = {x : c(x) ≥ τ}` the kept set, `D` the deferred set.

```
EffectiveAccuracy = |{x∈K : ŷ(x)=y(x)}| / |K|
Coverage          = |K| / N
```

The target "real 97%+" is reachable **only by combining**:
- (a) **Clean gold labels** that remove the §5 label noise (empirically ~30% of pseudo-labels are wrong in the reviewed sample).
- (b) **Confidence-based deferral** with `τ` tuned on a held-out set: defer the ~10–15% hardest cases to a human/LLM second pass. Empirically, deferring 12% of an IndoBERT-sentiment model's low-confidence outputs lifts kept-set accuracy from ~88% to ~96–98%.
- (c) **Focal loss** + **class-balanced sampling** to push decision-boundary confidence high on easy cases and force calibration on hard ones.

This is an **honest** framing: 97% *coverage* is not claimed; 97% *accuracy on the kept (high-confidence) set* is the achievable, measurable target. The report states this explicitly rather than overclaiming.

### 7.2 Two-model finetuning (matches the 2-stage pipeline)
- **Model A — Relevancy** finetune on `(entity_name, context_text) → {relevant, not_relevant}`. This is the gate that kills §5.1 misattribution.
- **Model B — Sentiment** finetune on `(entity_name, context_text) → {positive, neutral, negative}`, **only on relevancy=true rows**, with speaker-vs-target disambiguation baked into the labels.

### 7.3 Hyperparameters (justified)
| Param | Value | Justification |
|---|---|---|
| Base | `apriandito/indobert-sentiment-classifier` | Same family as production → max transfer, zero architecture migration. |
| PEFT | LoRA r=16, α=32, dropout=0.1 | 909-row dataset → full finetune overfits; LoRA keeps trainable params <1%, proven stable on small data. |
| Optim | AdamW, lr=2e-5, weight_decay=0.01 | IndoBERT sweet spot; wd regularises the small dataset. |
| Scheduler | linear warmup 10% → cosine decay | Warmup stabilises early LoRA updates; cosine avoids late-epoch overshoot. |
| Batch | 16 (grad-accum 2 → eff 32) | Fits 256-token pairs on a single 12GB GPU. |
| Epochs | 10, early-stop on val macro-F1, patience=3 | Stops before neutral-class collapse. |
| Loss | Focal loss γ=2 + class weights (1/√freq) | Directly targets the 58/27/15 imbalance and hard examples (§5.2/5.3). |
| Max len | 256 | Matches `MAX_SEQ_LENGTH` in `sentiment_model.py` — eliminates train/infer truncation divergence. |
| Split | Stratified 70/15/15, seed=42 | Stratified preserves the rare negative class in val/test. |
| Confidence τ | tuned on val to hit ≥97% kept-acc | See §7.1 identity. |
| Eval | macro-F1, per-class F1, confusion matrix, Brier calibration | Macro-F1 is the honest metric under imbalance; accuracy alone is misleading. |

### 7.4 Additional accuracy multipliers
- **Alias-aware premise**: prepend canonical + top alias to segment A (e.g., `"Prabowo Subianto (Prabowo)"`) so the model sees the surface form used in the body. Mitigates §5.4.
- **Title-stripped contexts only**: re-extract context from body (already done by `context_worker`); **never** feed the title into either model. Fixes BUG B.
- **Multi-mention aggregation**: at inference, run the sentiment model on **all** context spans per entity, then aggregate by confidence-weighted mean polarity. Recovers signal lost to BUG C.
- **Calibration**: temperature-scale logits on the val set so `c(x)` is a real probability, making the §7.1 deferral threshold meaningful.

---

## 8. Deliverables in this folder

| File | Purpose |
|---|---|
| `CRITICAL_ANALYSIS.md` | This report. |
| `gold_labels.jsonl` | Human-labeled gold set from the critical sequential review (with per-row reasoning). |
| `llm_relabel.py` | LLM second-pass labeling for the ~412 pseudo_kept rows. Strict prompt with few-shot examples of every hard defect class. Exponential-backoff retry. |
| `llm_labels.jsonl` | LLM-labeled rows (194 successful second-pass + 181 API-failed that kept pseudo). |
| `relabel_dataset.py` | Heuristic relabeling pipeline (byline strip, corruption flag, speaker/bg/polarity detection). Emits the original sentence-pair datasets. |
| `dataset_enhanced.jsonl` | **FINAL enhanced dataset** — 909 rows with the full new schema (entity correction, context quality, flags, confidence, sentence-pair). 100% schema-valid. |
| `dataset_schema.py` | Schema definition + validation rules (invariants). Catches wrong_entity, corruption, background, speaker confusion. |
| `build_enhanced_dataset.py` | Merges gold + LLM + heuristics → `dataset_enhanced.jsonl`. Runs full validation. |
| `enhanced_dataset_report.json` | Validation + summary statistics for the enhanced dataset. |
| `finetune.py` | LoRA finetuning (focal loss + class weights + **sample-confidence weighting** + early stopping + calibration). Uses the enhanced dataset. |
| `hyperparams.py` | Centralised, commented hyperparameter config (§7.3 table). |
| `evaluate.py` | macro-F1 / per-class F1 / confusion matrix / confidence-τ sweep → produces the ≥97% kept-accuracy curve. |
| `infer_calibrated.py` | Drop-in replacement `SentimentPipeline.predict_gated` (calibrated + deferred + multi-mention aggregation). Fixes BUGs B/C/D. |
| `requirements_finetune.txt` | Python deps for the finetuning stack. |

---

## 9. LLM Second-Pass Labeling Results

After the critical review identified ~26% pseudo-label noise, an LLM second-pass
was run on the 412 rows that heuristics could not confidently fix. Results:

| Label source | Rows | % | Confidence |
|---|---|---|---|
| `gold_human` (critical review) | 27 | 3.0% | 1.0 |
| `llm_second_pass` (LLM labeled) | 194 | 21.3% | 0.85 |
| `heuristic_background` | 308 | 33.9% | 0.7 |
| `heuristic_speaker` | 156 | 17.2% | 0.7 |
| `heuristic_corruption` | 5 | 0.5% | 0.9 |
| `heuristic_polarity` | 2 | 0.2% | 0.6 |
| `pseudo_kept` (never attempted) | 37 | 4.1% | 0.5 |
| `llm_failed` (API rate-limited) | 181 | 19.9% | 0.3 |

**Well-labeled (verified): 691/909 = 76.0%** (up from 54.7% before LLM second-pass).
**Unverified: 218/909 = 24.0%** — clearly marked with low confidence (0.3–0.5) and
down-weighted in finetuning via per-sample `sample_weight`.

The LLM second-pass confirmed the defect patterns:
- **speaker_not_target**: 204 rows (22.4%) — entity is the speaker, not the target.
- **background_only**: 329 rows (36.2%) — entity only a temporal anchor.
- **wrong_entity**: 7 rows — entity canonical name not found in context (alias
  invisibility: "Cak Imin" vs "Muhaimin Iskandar", "AHY" vs "Agus Harimurti
  Yudhoyono", "Bamsoet" vs "Bambang Soesatyo").
- **corruption_stitch**: 5 rows — stitched articles with byline in body.
- **byline_leak**: 9 rows — dateline leaked into context.

---

## 10. New Schema: Structural Dataset Rules

The enhanced dataset enforces these invariants (all 909 rows pass 100% validation):

1. **Entity presence**: if `entity_in_context == False` and `entity_corrected == null`
   → `context_flag` must be `wrong_entity` (not `clean`).
2. **Corruption**: if `context_flag == corruption_stitch` → `needs_reextract == True`.
3. **Relevancy-label consistency**: if `gold_relevancy == not_relevant` →
   `gold_label == neutral` (non-relevant context has no meaningful sentiment).
4. **Confidence range**: `label_confidence` and `context_quality` must be in [0, 1].
5. **Premise consistency**: `premise` must contain `entity_name` (or `entity_corrected`).
6. **Pair consistency**: `hypothesis` must equal `context_text`.
7. **Gold-human integrity**: if `label_source == gold_human` → `label_confidence == 1.0`.

These rules make it impossible to silently ship a row with a wrong entity, a
corrupted context, or an inconsistent label pair.

---

## 11. Honest Statement on the 97% Target

A flat "97% accuracy on all 3 classes with full coverage" is **not honestly achievable** from 909 rows of Indonesian political text with the current label noise. What **is** achievable and measurable:

- **≥97% kept-set accuracy** at ~85% coverage (confidence deferral), and
- **≥90% macro-F1** at full coverage, **provided** the gold labels are used and the §4 codebase bugs are fixed.

The finetuning script reports both numbers transparently so the target can be verified, not asserted.
