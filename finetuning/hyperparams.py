"""
hyperparams.py
==============
Centralised, commented hyperparameter config for finetuning the
ID-Political-Sentiment-Tracker base models.

Every value below carries a one-line justification rooted in either:
  - the base-model architecture (apriandito/indobert-*-classifier, MAX_SEQ_LENGTH=256),
  - the dataset statistics (909 rows, 54 entities, 65/35 relevancy split,
    21/68/11 sentiment split), or
  - established small-data finetuning theory.

See CRITICAL_ANALYSIS.md §7 for the full derivation.
"""
from __future__ import annotations
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Base model identifiers (from packages/nlp/sentiment_model.py)
# ---------------------------------------------------------------------------
RELEVANCY_BASE = "apriandito/indobert-relevancy-classifier"
SENTIMENT_BASE = "apriandito/indobert-sentiment-classifier"
FALLBACK_BASE  = "taufiqdp/indonesian-sentiment"

# ---------------------------------------------------------------------------
# Tokenisation — MUST match production sentiment_model.py
# ---------------------------------------------------------------------------
MAX_SEQ_LENGTH = 256      # = sentiment_model.MAX_SEQ_LENGTH. Eliminates
                          # train/inference truncation divergence.
PAIR_FORMAT    = True     # tokenizer(premise, hypothesis) — NLI-style pair,
                          # matching _forward_pair in sentiment_model.py.

# ---------------------------------------------------------------------------
# PEFT (LoRA) — full finetune on 909 rows overfits; LoRA keeps <1% trainable.
# ---------------------------------------------------------------------------
@dataclass
class LoRAConfig:
    r: int = 16            # rank. 16 is the sweet spot for BERT-base on small
                           # data: enough capacity to learn domain signal,
                           # small enough to avoid overfitting.
    alpha: int = 32        # scaling = alpha/r = 2.0. Standard init (Hu et al. 2022).
    dropout: float = 0.1   # regularisation on LoRA weights.
    bias: str = "none"
    task_type = "SEQ_CLS"
    target_modules = ["query", "key", "value", "dense"]
                           # attention + FFN projections. Targeting only Q/K/V
                           # underfits; adding `dense` recovers sentiment nuance.

LORA = LoRAConfig()

# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------
LEARNING_RATE = 2e-5       # IndoBERT sweet spot. Higher (5e-5) destabilises
                           # LoRA on <1k rows; lower (1e-5) underfits.
WEIGHT_DECAY  = 0.01       # AdamW L2. Regularises the small dataset.
ADAM_EPSILON  = 1e-8
ADAM_BETA1    = 0.9
ADAM_BETA2    = 0.999      # default; small datasets don't benefit from tuning.
MAX_GRAD_NORM = 1.0        # gradient clipping — essential for LoRA stability.

# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
WARMUP_RATIO  = 0.1        # 10% of steps linear warmup. Stabilises early
                           # LoRA updates (random init at start).
SCHEDULER     = "cosine"   # cosine decay. Smoother than linear; avoids
                           # late-epoch overshoot that hurts macro-F1.

# ---------------------------------------------------------------------------
# Batch / epochs
# ---------------------------------------------------------------------------
BATCH_SIZE        = 16     # per-device. 256-token pairs fit on 12GB GPU.
GRAD_ACCUM_STEPS  = 2      # effective batch = 32. Reduces gradient noise.
NUM_EPOCHS        = 10
EARLY_STOP_PATIENCE = 3    # stop when val macro-F1 stops improving for 3 eps.
                            # Prevents neutral-class collapse.

# ---------------------------------------------------------------------------
# Loss — Focal + class-balanced weights
# ---------------------------------------------------------------------------
FOCAL_GAMMA   = 2.0        # Standard focal-loss focusing param. Down-weights
                           # easy examples (the 68% neutral majority) so the
                           # model actually learns positive/negative.
# Class weights computed as 1/sqrt(freq) — effective-number-of-samples
# reweighting (Cui et al. 2019). Computed at runtime from the train split.

# ---------------------------------------------------------------------------
# Split — stratified to preserve the rare negative class in val/test.
# ---------------------------------------------------------------------------
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
TRAIN_SPLIT = 1.0 - VAL_SPLIT - TEST_SPLIT   # 0.70
SEED        = 42

# ---------------------------------------------------------------------------
# Label maps (must match sentiment_model.py LABEL_NORMALIZE_MAP)
# ---------------------------------------------------------------------------
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
RELEVANCY_LABELS = ["not_relevant", "relevant"]

# ---------------------------------------------------------------------------
# Calibration + confidence-based deferral (the ≥97% kept-accuracy lever)
# ---------------------------------------------------------------------------
TEMPERATURE     = 1.5      # temperature scaling on logits (Guo et al. 2017).
                           # Tuned on val set to make softmax a real prob.
CONFIDENCE_TAU  = 0.75     # defer predictions with max-prob < tau to a
                           # human/LLM second pass. Tuned on val to hit
                           # ≥97% accuracy on the kept set.
                           # See evaluate.py::confidence_threshold_sweep().

# ---------------------------------------------------------------------------
# Output dirs
# ---------------------------------------------------------------------------
OUT_DIR_RELEVANCY = "./runs/relevancy"
OUT_DIR_SENTIMENT = "./runs/sentiment"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
DETERMINISTIC = True
FP16          = True       # mixed precision. 2x speed, no accuracy loss on BERT.
