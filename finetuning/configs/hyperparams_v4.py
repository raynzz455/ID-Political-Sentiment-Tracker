"""
hyperparams_v4.py
=================
v4 HYPERPARAMS — TUNED for gold standard dataset (~2,200 rows, LLM-verified).

Key UPGRADES over v3:
  1. Oversampling: negative → ~400, positive → ~600 (reduces imbalance)
  2. Focal gamma: 2.5 → 3.0 (stronger focus on minority)
  3. LoRA dropout: 0.15 → 0.20 (more regularization)
  4. Mixup alpha: 0.2 → 0.3, prob: 30% → 40%
  5. Label smoothing: 0.05 → 0.07
  6. Class weights: 1/sqrt(freq) → 1/log(freq+1) (gentler)
  7. Entity-stratified K-fold (GroupKFold by entity)
  8. Epochs: 20 → 18, SWA start: 5 → 4
"""
from dataclasses import dataclass

RELEVANCY_BASE = "apriandito/indobert-relevancy-classifier"
SENTIMENT_BASE = "apriandito/indobert-sentiment-classifier"
FALLBACK_BASE  = "taufiqdp/indonesian-sentiment"

MAX_SEQ_LENGTH = 256
PAIR_FORMAT    = True

@dataclass
class LoRAConfig:
    r: int = 64
    alpha: int = 128
    dropout: float = 0.20
    bias: str = "none"
    task_type = "SEQ_CLS"
    target_modules = ["query", "key", "value", "dense"]
    target_modules_extended = ["query", "key", "value", "dense", "intermediate.dense"]

LORA = LoRAConfig()

LEARNING_RATE = 2.5e-5
WEIGHT_DECAY  = 0.03
ADAM_EPSILON  = 1e-8
ADAM_BETA1    = 0.9
ADAM_BETA2    = 0.999
MAX_GRAD_NORM = 1.0

WARMUP_RATIO  = 0.06
SCHEDULER     = "cosine_with_restarts"
SCHEDULER_NUM_CYCLES = 2

BATCH_SIZE        = 8
GRAD_ACCUM_STEPS  = 8
NUM_EPOCHS        = 18
EARLY_STOP_PATIENCE = 5

FOCAL_GAMMA     = 3.0
LABEL_SMOOTHING = 0.07
CLASS_WEIGHT_FN  = "log"

SWA_ENABLED       = True
SWA_START_EPOCH   = 4
SWA_LR            = 5e-6
SWA_ANNEAL_EPOCHS = 3

ADVERSARIAL_ENABLED = True
ADVERSARIAL_EPSILON = 1e-5
ADVERSARIAL_ALPHA   = 0.5

MIXUP_ENABLED = True
MIXUP_ALPHA   = 0.3
MIXUP_PROB    = 0.4

OVERSAMPLING_ENABLED = True
OVERSAMPLING_TARGETS = {"negative": 400, "positive": 600}

K_FOLD_ENABLED = True
K_FOLD_N       = 5
K_FOLD_STRATIFIED = True
K_FOLD_ENTITY_AWARE = True

VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
TRAIN_SPLIT = 1.0 - VAL_SPLIT - TEST_SPLIT
SEED        = 42

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
RELEVANCY_LABELS = ["not_relevant", "relevant"]

TEMPERATURE    = 1.3
CONFIDENCE_TAU = 0.70

OUT_DIR_RELEVANCY = "./runs/relevancy_v4"
OUT_DIR_SENTIMENT = "./runs/sentiment_v4"

HF_ORG          = "raynzz455"
HF_MODEL_PREFIX = "id-political-sentiment"
HF_SENTIMENT_MODEL = f"{HF_ORG}/{HF_MODEL_PREFIX}-sentiment-v4"
HF_RELEVANCY_MODEL = f"{HF_ORG}/{HF_MODEL_PREFIX}-relevancy-v4"

DETERMINISTIC = True
FP16          = True

METHOD_DESCRIPTION = """
v4: Based on gold standard dataset (~2,200 rows, LLM-verified + re-verified)
  + Oversampling, Focal gamma=3.0, Entity-stratified K-fold
  + Mixup 0.3/40%, Label smoothing 0.07, Log class weights
Expected: macro-F1 0.70 → 0.75+, negative recall 0.45 → 0.60+
"""
