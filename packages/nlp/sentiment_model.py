"""
sentiment_model.py — ID-Sentiment-Tracker
===========================================
v4: Pure ML Output & Derived Continuous Metrics.

PERUBAHAN v4:
  1. PURE LABELS: Mengembalikan label asli model (positive, neutral, negative).
     Tidak ada lagi pemaksaan label buatan (factual/ambiguous) agar selaras
     dengan CHECK CONSTRAINT di database.
  2. METRICS EXTRACTION: Menghitung Polarity Score (pos-neg) dan Entropy
     sebagai feature kontinu murni tanpa threshold heuristic.
  3. FAIL-CLOSED: Relevancy gate tetap fail-closed jika error.

ARSITEKTUR 2-STAGE:
  Stage 1 — RelevancyModel:  "apakah teks ini tentang entity X?"
  Stage 2 — SentimentModel:  "apa sentimen teks ini terhadap entity X?"
"""

import sys
import math
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
# CONFIG
# ─────────────────────────────────────────────────────────────

RELEVANCY_MODEL_ID = "apriandito/indobert-relevancy-classifier"
SENTIMENT_MODEL_ID  = "apriandito/indobert-sentiment-classifier"
FALLBACK_MODEL_ID   = "taufiqdp/indonesian-sentiment"

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
# CONTINUOUS METRICS CALCULATION
# ─────────────────────────────────────────────────────────────

def calculate_continuous_metrics(scores: tuple) -> tuple:
    """
    Menghitung metrik kontinu murni tanpa interpretasi heuristic.
    scores = (neg, neu, pos)
    return (polarity_score, entropy)
    """
    neg, neu, pos = scores
    
    # 1. Continuous Polarity Score (-1.0 to 1.0)
    polarity = pos - neg
    
    # 2. Entropy (Tingkat kebingungan model)
    # Tambahkan epsilon 1e-9 untuk menghindari log(0)
    entropy = -sum(p * math.log(p + 1e-9) for p in scores if p > 0)
    
    return polarity, entropy


# ─────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class GatedResult:
    is_relevant: bool
    relevancy_confidence: float
    label: Optional[str]               # Pure label model: positive, neutral, negative
    sentiment_confidence: Optional[float]
    scores: Optional[tuple]            # (neg, neu, pos)
    polarity_score: Optional[float] = None
    entropy: Optional[float] = None


# ─────────────────────────────────────────────────────────────
# BASE MODEL LOADER
# ─────────────────────────────────────────────────────────────

class _LoadedModel:
    def __init__(self, model_id: str):
        logger.info(f"Loading {model_id} ...")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(DEVICE)
        self.model.eval()
        self.id2label = self.model.config.id2label
        logger.info(f"  -> loaded. id2label = {self.id2label}")

    @torch.no_grad()
    def _forward_pair(self, a: str, b: str) -> list[float]:
        inputs = self.tokenizer(
            a, b, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt",
        ).to(DEVICE)
        logits = self.model(**inputs).logits
        return torch.softmax(logits, dim=-1)[0].cpu().tolist()

    @torch.no_grad()
    def _forward_single(self, text: str) -> list[float]:
        inputs = self.tokenizer(
            text, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt",
        ).to(DEVICE)
        logits = self.model(**inputs).logits
        return torch.softmax(logits, dim=-1)[0].cpu().tolist()


class RelevancyModel(_LoadedModel):
    def __init__(self):
        super().__init__(RELEVANCY_MODEL_ID)
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
        super().__init__(SENTIMENT_MODEL_ID)

    def predict(self, context: str, text: str) -> tuple[str, float, tuple]:
        probs = self._forward_pair(context, text)
        scores = {normalize_label(self.id2label[i]): probs[i] for i in range(len(probs))}
        pred_idx = probs.index(max(probs))
        label = normalize_label(self.id2label[pred_idx])
        conf = probs[pred_idx]
        score_tuple = (scores.get("negative", 0.0), scores.get("neutral", 0.0), scores.get("positive", 0.0))
        return label, conf, score_tuple


class FallbackModel(_LoadedModel):
    def __init__(self):
        super().__init__(FALLBACK_MODEL_ID)

    def predict(self, text: str) -> tuple[str, float, tuple]:
        probs = self._forward_single(text)
        scores = {normalize_label(self.id2label[i]): probs[i] for i in range(len(probs))}
        pred_idx = probs.index(max(probs))
        label = normalize_label(self.id2label[pred_idx])
        conf = probs[pred_idx]
        score_tuple = (scores.get("negative", 0.0), scores.get("neutral", 0.0), scores.get("positive", 0.0))
        return label, conf, score_tuple


# ─────────────────────────────────────────────────────────────
# PIPELINE — interface utama
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
            # FAIL-CLOSED: Jika gate error, anggap tidak relevan agar tidak lolos ke sentimen
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