"""
hyperparams_optimized.py
========================
OPTIMIZED hyperparameters for maximum accuracy on Indonesian political sentiment.
Based on empirical testing across 909-row dataset with 5-method comparison.

Target: >=97% kept-set accuracy (at 85% coverage via confidence deferral)
        >=90% macro-F1 (at full coverage)
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
    dropout: float = 0.08
    bias: str = "none"
    task_type = "SEQ_CLS"
    target_modules = ["query", "key", "value", "dense", "classifier"]

LORA = LoRAConfig()

LEARNING_RATE = 1.5e-5
WEIGHT_DECAY  = 0.02
ADAM_EPSILON  = 1e-8
ADAM_BETA1    = 0.9
ADAM_BETA2    = 0.999
MAX_GRAD_NORM = 1.0

WARMUP_RATIO    = 0.08
SCHEDULER       = "cosine_with_restarts"
NUM_RESTARTS    = 2
RESTART_FACTOR   = 1.0

BATCH_SIZE        = 16
GRAD_ACCUM_STEPS  = 4
NUM_EPOCHS        = 15
EARLY_STOP_PATIENCE = 4

FOCAL_GAMMA       = 2.5
LABEL_SMOOTHING   = 0.05

SWA_ENABLED       = True
SWA_START_EPOCH   = 10
SWA_LR            = 5e-6
SWA_ANNEAL_EPOCHS = 3

VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
SEED        = 42

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
RELEVANCY_LABELS = ["not_relevant", "relevant"]

TEMPERATURE     = 1.5
CONFIDENCE_TAU  = 0.80

HF_ORG          = "raynzz455"
HF_MODEL_PREFIX = "id-political-sentiment"
HF_SENTIMENT_MODEL = f"{HF_ORG}/{HF_MODEL_PREFIX}-sentiment-v1"
HF_RELEVANCY_MODEL = f"{HF_ORG}/{HF_MODEL_PREFIX}-relevancy-v1"

OUT_DIR_RELEVANCY = "./runs/relevancy"
OUT_DIR_SENTIMENT = "./runs/sentiment"

DETERMINISTIC = True
FP16          = True
