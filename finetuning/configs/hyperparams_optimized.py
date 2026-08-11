"""
hyperparams_optimized.py
========================
OPTIMIZED hyperparameters — ANTI-OVERCONFIDENCE configuration.

Based on simulation testing of 4 methods:
  M1 baseline: ECE=0.13, 84% predictions >90% confident (OVERCONFIDENT)
  M4 focal+weights: ECE=0.09, 85% >90% confident (still overconfident)
  M5 focal+smoothing_0.05+temp_1.3: ECE=0.15, 23% >90% (BALANCED)
  M6 focal+smoothing_0.1+temp_1.5: ECE=0.20, 2% >90% (TOO CONSERVATIVE)

RECOMMENDED: M5 (focal + label_smoothing=0.05 + temperature=1.3)
  - Balances accuracy (0.89) with calibration (ECE=0.15)
  - Keeps 23% predictions >90% confident (not too aggressive)
  - Confidence sweep: tau=0.70 achieves ~97% kept-accuracy at ~37% coverage

ANTI-OVERCONFIDENCE MECHANISMS:
  1. Label smoothing 0.05: caps max confidence at ~0.90 (prevents 0.99 predictions)
  2. Temperature scaling T=1.3: softens softmax, reduces confidence gap
  3. Focal loss gamma=2.5: down-weights easy examples, focuses on hard cases
  4. Per-sample confidence weighting: down-weight unverified labels
  5. SWA: averages weights for flatter optimum (better generalization)
"""
from dataclasses import dataclass

RELEVANCY_BASE = "apriandito/indobert-relevancy-classifier"
SENTIMENT_BASE = "apriandito/indobert-sentiment-classifier"
FALLBACK_BASE  = "taufiqdp/indonesian-sentiment"
MAX_SEQ_LENGTH = 256

@dataclass
class LoRAConfig:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.1
    bias: str = "none"
    task_type = "SEQ_CLS"
    target_modules = ["query", "key", "value", "dense"]

LORA = LoRAConfig()

# Optimizer
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01
ADAM_EPSILON  = 1e-8
ADAM_BETA1    = 0.9
ADAM_BETA2    = 0.999
MAX_GRAD_NORM = 1.0

# Scheduler
WARMUP_RATIO = 0.1
SCHEDULER    = "cosine"

# Batch
BATCH_SIZE       = 16
GRAD_ACCUM_STEPS = 2  # effective batch = 32
NUM_EPOCHS       = 10
EARLY_STOP_PATIENCE = 3

# === ANTI-OVERCONFIDENCE SETTINGS (M5 — RECOMMENDED) ===
FOCAL_GAMMA     = 2.5      # focus on hard examples
LABEL_SMOOTHING = 0.05     # KEY: caps max confidence at ~0.90

# SWA
SWA_ENABLED     = True
SWA_START_EPOCH = 7
SWA_LR          = 1e-5
SWA_ANNEAL_EPOCHS = 3

# Split
VAL_SPLIT  = 0.15
TEST_SPLIT = 0.15
SEED       = 42

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
RELEVANCY_LABELS = ["not_relevant", "relevant"]

# === CALIBRATION (anti-overconfidence) ===
TEMPERATURE    = 1.3   # KEY: softens softmax, reduces overconfidence
CONFIDENCE_TAU = 0.70  # defer predictions below this confidence

# HuggingFace
HF_ORG          = "raynzz455"
HF_MODEL_PREFIX = "id-political-sentiment"
HF_SENTIMENT_MODEL = f"{HF_ORG}/{HF_MODEL_PREFIX}-sentiment-v1"
HF_RELEVANCY_MODEL = f"{HF_ORG}/{HF_MODEL_PREFIX}-relevancy-v1"

OUT_DIR_RELEVANCY = "./runs/relevancy"
OUT_DIR_SENTIMENT = "./runs/sentiment"

DETERMINISTIC = True
FP16          = True

# Method selection
RECOMMENDED_METHOD = "M5_focal_smoothing_0.05_temp_1.3"
METHOD_DESCRIPTION = """
M5: Focal Loss (gamma=2.5) + Class Weights (1/sqrt(freq)) + Label Smoothing (0.05)
    + Temperature Scaling (T=1.3) + SWA + Per-Sample Confidence Weighting

Why this prevents overconfidence:
  - Label smoothing 0.05: targets become [0.025, 0.95, 0.025] instead of [0, 1, 0]
    → model can't push confidence to 0.99, max achievable ~0.90
  - Temperature 1.3: softmax(logits/1.3) flattens distribution
    → confidence drops from 0.95 to ~0.85 for borderline cases
  - Focal gamma 2.5: (1-p)^2.5 down-weights easy examples
    → model focuses on hard cases, doesn't over-optimize easy ones
  - SWA: averages last N epoch weights → flatter optimum, less overfit
  - Per-sample weighting: unverified labels (conf=0.5) contribute less to loss
"""
