# Finetuning — ID-Political-Sentiment-Tracker

Optimized LoRA finetuning for `apriandito/indobert-sentiment-classifier` and
`apriandito/indobert-relevancy-classifier` base models.

## Structure

```
finetuning/
├── datasets/
│   ├── dataset_enhanced.jsonl     # FINAL dataset (909 rows, all labeled)
│   ├── dataset.jsonl              # Raw extraction dataset
│   ├── gold_labels.jsonl          # 27 human-labeled gold cases
│   ├── llm_labels.jsonl           # 375 LLM second-pass labels
│   └── llm_verified_labels.jsonl  # LLM-verified heuristic labels
├── scripts/
│   ├── finetune.py                # LoRA finetuning (focal loss + class weights)
│   ├── evaluate.py                # Evaluation + confidence threshold sweep
│   ├── upload_huggingface.py      # Upload merged model to HuggingFace
│   ├── build_enhanced_dataset.py  # Build enhanced dataset from all sources
│   ├── relabel_dataset.py         # Heuristic relabeling pipeline
│   ├── build_gold_labels.py       # Build gold human labels
│   ├── llm_verify_all.py          # LLM verification (Python CLI)
│   ├── llm_verify_sdk.mjs         # LLM verification (Node.js SDK)
│   ├── infer_calibrated.py        # Drop-in replacement for SentimentPipeline
│   ├── dataset_schema.py          # Schema validation (7 invariants)
│   └── hyperparams.py             # Base hyperparameters
├── configs/
│   └── hyperparams_optimized.py   # OPTIMIZED hyperparameters (r=32, SWA, etc.)
└── README.md
```

## Quick Start

```bash
# 1. Install deps
pip install torch transformers peft scikit-learn numpy huggingface_hub

# 2. Finetune (GPU recommended, ~10 min)
python scripts/finetune.py --task sentiment

# 3. Evaluate
python scripts/evaluate.py --task sentiment --run-dir ./runs/sentiment

# 4. Upload to HuggingFace
export HF_TOKEN=your_hf_token
python scripts/upload_huggingface.py --task sentiment --hf-token $HF_TOKEN
```

## Dataset Stats

| Metric | Value |
|--------|-------|
| Total rows | 909 |
| All labeled | YES |
| Label distribution | neutral 66%, positive 18%, negative 16% |
| High confidence (>=0.85) | 38% |
| Medium confidence (0.6-0.84) | 46% |
| Low confidence (<0.6) | 16% |

## Optimized Hyperparameters

| Param | Value | Justification |
|-------|-------|---------------|
| LoRA r | 32 | More capacity for 3-class sentiment |
| LoRA alpha | 64 | Scaling = 2.0 |
| Learning rate | 1.5e-5 | Stable with r=32 |
| Focal gamma | 2.5 | Stronger hard-example focus |
| Label smoothing | 0.05 | Prevents overconfidence |
| SWA | Enabled | Flatter optimum, better generalization |
| Effective batch | 64 | Stable gradients |
| Epochs | 15 | Early stop patience=4 |
| Temperature | 1.5 | Calibrated probabilities |
| Confidence tau | 0.80 | 97% kept-accuracy target |

## Target Metrics

| Metric | Target | How measured |
|--------|--------|--------------|
| macro-F1 (full) | >=0.90 | Held-out test set (137 rows) |
| Per-class F1 | >=0.85 | Each class independently |
| ECE | <=0.10 | Expected Calibration Error |
| Kept-set accuracy | >=0.97 | At tau=0.80, coverage ~80% |

## HuggingFace Upload

After finetuning, upload merged model:

```bash
export HF_TOKEN=hf_your_token_here
python scripts/upload_huggingface.py --task sentiment
python scripts/upload_huggingface.py --task relevancy
```

Models will be available at:
- `raynzz455/id-political-sentiment-sentiment-v1`
- `raynzz455/id-political-sentiment-relevancy-v1`
