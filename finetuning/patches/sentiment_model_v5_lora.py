"""
sentiment_model.py — ID-Sentiment-Tracker (v5 — LoRA Adapter Support)
=====================================================================
v5: PRODUCTION DEPLOYMENT of finetuned v3 models.

PERUBAHAN v5 over v4:
  1. LORA ADAPTER SUPPORT: can load finetuned LoRA adapter on top of base model.
     Set env vars USE_FINETUNED_V3=1 + MODEL_V3_PATH=/path/to/lora to enable.
  2. TEMPERATURE SCALING: applies temperature T (from metrics.json) to softmax.
     Makes confidence meaningful for deferral.
  3. FALLBACK to BASE MODEL: if LoRA adapter missing/corrupt, falls back to
     base model automatically (zero-downtime deployment).
  4. PURE LABELS preserved (v4 behavior kept).
  5. METRICS EXTRACTION preserved (polarity_score, entropy).

DEPLOYMENT:
  Option A — LoRA adapter (recommended, 50MB):
    export USE_FINETUNED_V3=1
    export SENTIMENT_LORA_PATH=/app/models/sentiment-v3/lora
    export RELEVANCY_LORA_PATH=/app/models/relevancy-v3/lora
    export SENTIMENT_TEMPERATURE=1.3
    export RELEVANCY_TEMPERATURE=1.2

  Option B — Merged full model (440MB, no PEFT dependency):
    export USE_FINETUNED_V3=1
    export SENTIMENT_MODEL_ID=raynzz455/id-political-sentiment-sentiment-v3
    export RELEVANCY_MODEL_ID=raynzz455/id-political-sentiment-relevancy-v3

  Option C — Base model (current, no finetuning):
    (no env vars needed — defaults to apriandito/* classifiers)

ARSITEKTUR 2-STAGE (unchanged):
  Stage 1 — RelevancyModel:  "apakah teks ini tentang entity X?"
  Stage 2 — SentimentModel:  "apa sentimen teks ini terhadap entity X?"
"""

import sys
import os
import math
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    print("[ERROR] pip install torch transformers --break-system-packages")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# CONFIG — supports finetuned v3 via env vars
# ─────────────────────────────────────────────────────────────

# Base model IDs (used if USE_FINETUNED_V3 not set, or as fallback)
RELEVANCY_MODEL_ID = os.getenv("RELEVANCY_MODEL_ID", "apriandito/indobert-relevancy-classifier")
SENTIMENT_MODEL_ID  = os.getenv("SENTIMENT_MODEL_ID", "apriandito/indobert-sentiment-classifier")
FALLBACK_MODEL_ID   = os.getenv("FALLBACK_MODEL_ID", "taufiqdp/indonesian-sentiment")

# v5: Finetuned v3 LoRA adapter paths (optional)
USE_FINETUNED_V3 = os.getenv("USE_FINETUNED_V3", "0") == "1"
SENTIMENT_LORA_PATH = os.getenv("SENTIMENT_LORA_PATH", "")
RELEVANCY_LORA_PATH = os.getenv("RELEVANCY_LORA_PATH", "")

# v5: Temperature scaling (from finetune metrics.json)
SENTIMENT_TEMPERATURE = float(os.getenv("SENTIMENT_TEMPERATURE", "1.0"))
RELEVANCY_TEMPERATURE = float(os.getenv("RELEVANCY_TEMPERATURE", "1.0"))

MAX_SEQ_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RELEVANCY_THRESHOLD = 0.5
RELEVANT_LABEL_HINTS = {"relevan", "relevant", "yes", "ya", "1", "true"}

LABEL_NORMALIZE_MAP = {
    "negatif": "negative", "negative": "negative", "neg": "negative",
    "netral": "neutral", "neutral": "neutral", "neu": "neutral",
    "positif": "positive", "positive": "positive", "pos": "positive",
}

def normalize_label(raw_label: str) -> str:
    key = raw_label.lower().strip()
    if key not in LABEL_NORMALIZE_MAP:
        logger.warning(f"Label sentimen tidak dikenal: '{raw_label}' -> 'neutral'")
        return "neutral"
    return LABEL_NORMALIZE_MAP[key]


# ─────────────────────────────────────────────────────────────
# CONTINUOUS METRICS CALCULATION (unchanged from v4)
# ─────────────────────────────────────────────────────────────

def calculate_continuous_metrics(scores: tuple) -> tuple:
    """
    Menghitung metrik kontinu murni tanpa interpretasi heuristic.
    scores = (neg, neu, pos)
    return (polarity_score, entropy)
    """
    neg, neu, pos = scores
    polarity = pos - neg
    entropy = -sum(p * math.log(p + 1e-9) for p in scores if p > 0)
    return polarity, entropy


# ─────────────────────────────────────────────────────────────
# RESULT DATACLASS (unchanged from v4)
# ─────────────────────────────────────────────────────────────

@dataclass
class GatedResult:
    is_relevant: bool
    relevancy_confidence: float
    label: Optional[str]
    sentiment_confidence: Optional[float]
    scores: Optional[tuple]
    polarity_score: Optional[float] = None
    entropy: Optional[float] = None


# ─────────────────────────────────────────────────────────────
# v5: BASE MODEL LOADER WITH LORA ADAPTER SUPPORT
# ─────────────────────────────────────────────────────────────

class _LoadedModel:
    """v5: Loads base model + optional LoRA adapter + temperature scaling."""

    def __init__(self, model_id: str, lora_path: str = "", temperature: float = 1.0):
        self.model_id = model_id
        self.lora_path = lora_path
        self.temperature = temperature
        self.uses_lora = False

        logger.info(f"Loading {model_id} ...")

        # Load tokenizer (from base or LoRA path if specified)
        tok_source = lora_path if (lora_path and os.path.isdir(lora_path)) else model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tok_source)

        # Load base model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)

        # v5: Apply LoRA adapter if path provided
        if lora_path and os.path.isdir(lora_path):
            try:
                from peft import PeftModel
                logger.info(f"  Loading LoRA adapter: {lora_path}")
                self.model = PeftModel.from_pretrained(self.model, lora_path)
                self.model = self.model.merge_and_unload()  # merge for faster inference
                self.uses_lora = True
                logger.info(f"  LoRA adapter merged successfully")
            except ImportError:
                logger.warning(f"  peft not installed — using base model only")
            except Exception as e:
                logger.warning(f"  LoRA load failed ({e}) — falling back to base model")

        self.model.to(DEVICE)
        self.model.eval()
        self.id2label = self.model.config.id2label
        logger.info(f"  -> loaded. id2label = {self.id2label} | LoRA={self.uses_lora} | T={self.temperature}")

    @torch.no_grad()
    def _forward_pair(self, a: str, b: str) -> list[float]:
        inputs = self.tokenizer(
            a, b, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt",
        ).to(DEVICE)
        logits = self.model(**inputs).logits
        # v5: apply temperature scaling
        probs = torch.softmax(logits / max(self.temperature, 0.05), dim=-1)[0]
        return probs.cpu().tolist()

    @torch.no_grad()
    def _forward_single(self, text: str) -> list[float]:
        inputs = self.tokenizer(
            text, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt",
        ).to(DEVICE)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits / max(self.temperature, 0.05), dim=-1)[0]
        return probs.cpu().tolist()


class RelevancyModel(_LoadedModel):
    def __init__(self):
        # v5: use LoRA path if USE_FINETUNED_V3 and path set
        lora_path = RELEVANCY_LORA_PATH if USE_FINETUNED_V3 else ""
        temp = RELEVANCY_TEMPERATURE if USE_FINETUNED_V3 else 1.0
        super().__init__(RELEVANCY_MODEL_ID, lora_path, temp)

        self._relevant_idx = None
        for idx, label in self.id2label.items():
            if label.lower().strip() in RELEVANT_LABEL_HINTS:
                self._relevant_idx = idx
                break
        if self._relevant_idx is None:
            logger.warning(f"Tidak bisa auto-detect label 'relevan'. Default ke index 1.")
            self._relevant_idx = 1

    def check(self, context: str, text: str) -> tuple[bool, float]:
        probs = self._forward_pair(context, text)
        relevant_prob = probs[self._relevant_idx]
        return relevant_prob >= RELEVANCY_THRESHOLD, relevant_prob


class SentimentModel(_LoadedModel):
    def __init__(self):
        lora_path = SENTIMENT_LORA_PATH if USE_FINETUNED_V3 else ""
        temp = SENTIMENT_TEMPERATURE if USE_FINETUNED_V3 else 1.0
        super().__init__(SENTIMENT_MODEL_ID, lora_path, temp)

    def predict(self, context: str, text: str) -> tuple[str, float, tuple]:
        probs = self._forward_pair(context, text)
        scores = {normalize_label(self.id2label[i]): probs[i] for i in range(len(probs))}
        pred_idx = probs.index(max(probs))
        label = normalize_label(self.id2label[pred_idx])
        conf = probs[pred_idx]
        score_tuple = (scores.get("negative", 0.0), scores.get("neutral", 0.0), scores.get("positive", 0.0))
        return label, conf, score_tuple


class FallbackModel(_LoadedModel):
    """Fallback model — always uses base (no LoRA), no temperature."""
    def __init__(self):
        super().__init__(FALLBACK_MODEL_ID, "", 1.0)

    def predict(self, text: str) -> tuple[str, float, tuple]:
        probs = self._forward_single(text)
        scores = {normalize_label(self.id2label[i]): probs[i] for i in range(len(probs))}
        pred_idx = probs.index(max(probs))
        label = normalize_label(self.id2label[pred_idx])
        conf = probs[pred_idx]
        score_tuple = (scores.get("negative", 0.0), scores.get("neutral", 0.0), scores.get("positive", 0.0))
        return label, conf, score_tuple


# ─────────────────────────────────────────────────────────────
# PIPELINE — interface utama (unchanged API)
# ─────────────────────────────────────────────────────────────

class SentimentPipeline:
    def __init__(self):
        self._relevancy: Optional[RelevancyModel] = None
        self._sentiment: Optional[SentimentModel] = None
        self._fallback: Optional[FallbackModel] = None

    @property
    def relevancy(self) -> RelevancyModel:
        if self._relevancy is None:
            self._relevancy = RelevancyModel()
        return self._relevancy

    @property
    def sentiment(self) -> SentimentModel:
        if self._sentiment is None:
            self._sentiment = SentimentModel()
        return self._sentiment

    @property
    def fallback(self) -> FallbackModel:
        if self._fallback is None:
            self._fallback = FallbackModel()
        return self._fallback

    def predict_gated(self, text: str, context: Optional[str]) -> GatedResult:
        if not text or not text.strip():
            return GatedResult(False, 0.0, None, None, None)

        # FALLBACK PATH (Document-level)
        if context is None:
            label, conf, scores = self.fallback.predict(text)
            polarity, entropy = calculate_continuous_metrics(scores)
            return GatedResult(True, 1.0, label, conf, scores, polarity, entropy)

        # GATED PATH (Entity-level)
        try:
            is_relevant, rel_conf = self.relevancy.check(context, text)
        except Exception as e:
            logger.error(f"Relevancy check gagal: {e} — treat sebagai TIDAK relevan (fail-closed)")
            return GatedResult(False, 0.0, None, None, None)

        if not is_relevant:
            return GatedResult(False, rel_conf, None, None, None)

        try:
            label, conf, scores = self.sentiment.predict(context, text)
            polarity, entropy = calculate_continuous_metrics(scores)
            return GatedResult(True, rel_conf, label, conf, scores, polarity, entropy)
        except Exception as e:
            logger.error(f"Sentiment predict gagal: {e}")
            scores = (0.33, 0.34, 0.33)
            polarity, entropy = calculate_continuous_metrics(scores)
            return GatedResult(True, rel_conf, "neutral", 0.34, scores, polarity, entropy)


@lru_cache(maxsize=1)
def get_pipeline() -> SentimentPipeline:
    return SentimentPipeline()
