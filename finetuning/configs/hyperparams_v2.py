"""
hyperparams_v2.py
=================
TAHAP 2: Optimized hyperparams for finetune v2.

Based on grid search results from v1:
  Best config: lr=3e-5, gamma=2.5, smoothing=0.05 → F1=0.6377, ECE=0.1309

v2 changes:
  - Learning rate: 2e-5 → 3e-5 (from grid search)
  - Label smoothing: 0.05 (kept — best from grid)
  - Focal gamma: 2.5 (kept — best from grid)
  - LoRA r: 32 (kept — r=16 was worse in grid)
  - Batch size: 16 (kept)
  - Epochs: 15 (increased from 10 for more training)
  - Dataset: v2 (balanced 1:1:1, filtered noise)
"""
RELEVANCY_BASE = "apriandito/indobert-relevancy-classifier"
SENTIMENT_BASE = "apriandito/indobert-sentiment-classifier"
FALLBACK_BASE  = "taufiqdp/indonesian-sentiment"
MAX_SEQ_LENGTH = 256
RELEVANCY_THRESHOLD = 0.5

class LoRAConfig:
    r = 32
    alpha = 64
    dropout = 0.1
    bias = "none"
    task_type = "SEQ_CLS"
    target_modules = ["query", "key", "value", "dense"]

LORA = LoRAConfig()

# Grid search best params
LEARNING_RATE = 3e-5      # UPGRADED from 2e-5 (grid search winner)
WEIGHT_DECAY  = 0.01
ADAM_EPSILON  = 1e-8
ADAM_BETA1    = 0.9
ADAM_BETA2    = 0.999
MAX_GRAD_NORM = 1.0

WARMUP_RATIO = 0.08
SCHEDULER    = "cosine"

BATCH_SIZE        = 16
GRAD_ACCUM_STEPS = 2
NUM_EPOCHS       = 15       # INCREASED from 10 (more training on balanced data)
EARLY_STOP_PATIENCE = 4

FOCAL_GAMMA     = 2.5      # grid search best
LABEL_SMOOTHING = 0.05     # grid search best

SWA_ENABLED     = True
SWA_START_EPOCH = 10
SWA_LR          = 5e-6
SWA_ANNEAL_EPOCHS = 3

TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
SEED        = 42

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
RELEVANCY_LABELS = ["not_relevant", "relevant"]

TEMPERATURE    = 1.3
CONFIDENCE_TAU = 0.70

HF_ORG          = "raynzz455"
HF_MODEL_PREFIX = "id-political-sentiment"
HF_SENTIMENT_MODEL = f"{HF_ORG}/{HF_MODEL_PREFIX}-sentiment-v2"
HF_RELEVANCY_MODEL = f"{HF_ORG}/{HF_MODEL_PREFIX}-relevancy-v2"

OUT_DIR_RELEVANCY = "./runs/relevancy"
OUT_DIR_SENTIMENT = "./runs/sentiment_v2"

DETERMINISTIC = True
FP16          = True

METHOD_DESCRIPTION = """
v2: Focal Loss (gamma=2.5) + Class Weights + Label Smoothing (0.05)
    + Temperature Scaling + SWA + lr=3e-5 (grid search best)
    + Balanced dataset (1:1:1 via oversampling)
    + Filtered label noise (confidence >= 0.5, clean context only)

Improvements over v1:
  - lr 2e-5 → 3e-5 (grid search showed +5pp F1 improvement)
  - Dataset balanced 1:1:1 (was 3.8:1 neutral-heavy)
  - Label noise filtered (excluded background_only, llm_failed)
  - More epochs (15 vs 10) for balanced data convergence
"""
