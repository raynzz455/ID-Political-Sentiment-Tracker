---
Task ID: 1
Agent: Z.ai Code (main)
Task: Review ID-Political-Sentiment-Tracker extraction dataset (Google Drive), identify context/entity leakage, find the base model in sentiment_model.py, critically relabel wrong contexts, and produce a finetuning approach targeting ≥97% real accuracy with mathematical/statistical justification.

Work Log:
- Read GitHub README of raynzz455/ID-Political-Sentiment-Tracker via z-ai page_reader + curl. Understood the 6-layer pipeline (Ingestion → Enrichment → Validation → Entity/Context → NLP → DB → Dashboard) and the 2-stage gated sentiment architecture.
- Located packages/nlp/sentiment_model.py and downloaded all key sources (sentiment_model.py, nlp_worker.py, context_worker.py, entity_resolution_worker.py, nlp_readiness_worker.py) via curl.
- Identified the 3 base models: apriandito/indobert-relevancy-classifier, apriandito/indobert-sentiment-classifier, taufiqdp/indonesian-sentiment. Both relevancy and sentiment use NLI-style sentence-pair tokenizer(entity_name, context_text) with MAX_SEQ_LENGTH=256.
- Downloaded the Google Drive dataset (file id 1yMdqsXe7xlhZUKTdpL0IzIAq9ykuwDLB) → 909 rows NDJSON. Verified fields: raw_text_id, entity_name, pseudo_label, ground_truth_label (ALL EMPTY), context_text, article_text, source_url.
- Statistical profiling: pseudo-label distribution 58.1% neutral / 26.7% positive / 15.2% negative (neutral inflation = leakage symptom). 32.7% of contexts not an exact substring of article_text (article hard-truncated to 1000 chars in export). 19% of contexts don't literally contain the entity name (alias/partial-name issue).
- Critical sequential review of ~40 rows across leakage patterns. Identified 5 systemic defect classes: (1) sentiment misattribution to background-mentioned entities, (2) speaker-vs-target confusion, (3) wrong polarity on subtle cases, (4) alias/partial-name invisibility, (5) byline/metadata leakage + stitched context+article corruption.
- Identified 5 codebase bugs causing the leakage (BUG A: offset domain mismatch title+body vs body-only; BUG B: fallback feeds title+body; BUG C: only one context per entity kept; BUG D: weak relevancy premise; BUG E: quality_score biases toward speaker sentences).
- Built gold_labels.jsonl with 27 human-labeled hard cases (74.1% pseudo-vs-gold agreement, confirming ~26% label noise).
- Built relabel_dataset.py: strips bylines, flags corruption (5 real cases), applies gold overrides, runs conservative heuristics (speaker_vs_target, background detection, wrong-polarity flip). Outputs dataset_relevancy.jsonl (909 rows, 65/35 split) and dataset_sentiment.jsonl (593 relevant rows, 21/68/11 split).
- Designed finetuning approach with mathematical justification: LoRA (r=16, α=32) on Q/K/V/dense, focal loss γ=2 + class-balanced weights 1/√freq, stratified 70/15/15 split, early stopping on val macro-F1, temperature scaling calibration, confidence-threshold deferral (the ≥97% kept-accuracy lever).
- Implemented finetune.py (FocalLossTrainer with class weights + EarlyStoppingCallback + temperature calibration), evaluate.py (macro-F1 + confusion matrix + confidence-threshold sweep reporting kept-accuracy vs coverage), infer_calibrated.py (drop-in CalibratedSentimentPipeline replacement fixing BUGs B/C/D + multi-mention aggregation + deferred flag), hyperparams.py (centralised justified config), requirements_finetune.txt.
- Syntax-checked all 6 Python files — all compile.

Stage Summary:
- This is a DATA SCIENCE / ML task, NOT website development. The webDevReview cron rule does not apply.
- Base models confirmed: apriandito/indobert-relevancy-classifier + apriandito/indobert-sentiment-classifier (sentence-pair NLI format, MAX_SEQ_LENGTH=256).
- Dataset verdict: the current format is NOT directly trainable — 5 format mismatches documented (single label vs 2-stage model, no sentence-pair structure, length exceeds 256 tokens, pseudo_label from broken model, class imbalance 58/27/15).
- Deliverables in /home/z/my-project/finetuning/: CRITICAL_ANALYSIS.md (237 lines), README.md, gold_labels.jsonl (27 rows), dataset_relevancy.jsonl (909 rows), dataset_sentiment.jsonl (593 rows), relabel_dataset.py, build_gold_labels.py, finetune.py, evaluate.py, infer_calibrated.py, hyperparams.py, requirements_finetune.txt, relabel_audit.json.
- Honest accuracy target: ≥97% KEPT-set accuracy at ~85% coverage (via confidence deferral), NOT 97% full-coverage. ≥90% macro-F1 at full coverage is the realistic full-coverage target. evaluate.py reports both transparently.
- The user's concern about title-clickbait is validated: context_worker already excludes the title (good), BUT the fallback path (nlp_worker.py line 69) feeds title+body to the document model (BUG B). infer_calibrated.py fixes this.
- The context leakage root cause is BUG A (offset mismatch) + BUG C (single-context retention) + BUG E (speaker-biased quality_score), not the title.
- Unresolved / next-phase: (1) actually run finetune.py on a GPU to produce the LoRA adapters and the real evaluation numbers — the scripts are ready but require a GPU + the ML deps installed; (2) expand the gold set beyond 27 rows for higher-confidence heuristics; (3) fix BUG A in the production context_worker.py (offset adjustment code is partially broken).
