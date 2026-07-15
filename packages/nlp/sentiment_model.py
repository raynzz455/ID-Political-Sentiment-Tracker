"""
sentiment_model.py — ID-Sentiment-Tracker
===========================================
v2: Tambah RelevancyModel + 2-stage gated pipeline.

KOREKSI dari v1: SentimentPredictor SAJA tidak bisa deteksi entity mismatch
(misal "Prabowo Subianto" vs "Listyo Sigit Prabowo"). Model sentimen hanya
menjawab "apa sentimennya", tidak menjawab "apakah context ini relevan".
Untuk itu perlu model TERPISAH: RelevancyModel.

ARSITEKTUR 2-STAGE (yang benar):
  Stage 1 — RelevancyModel:  "apakah teks ini tentang entity X?"
  Stage 2 — SentimentModel:  "apa sentimen teks ini terhadap entity X?"
  Stage 2 HANYA dipanggil kalau Stage 1 bilang relevan.

MODELS (semua dari riset SocialX + Telkom University + BRIN, April 2026):
  - apriandito/indobert-relevancy-classifier  (F1 0.948, relevansi)
  - apriandito/indobert-sentiment-classifier  (F1 0.856, sentimen)
  - taufiqdp/indonesian-sentiment              (fallback, tanpa context)

Usage:
    from sentiment_model import get_pipeline

    pipeline = get_pipeline()

    result = pipeline.predict_gated(
        text="Kapolri Listyo Sigit Prabowo menyerahkan bansos...",
        context="Prabowo Subianto",
    )
    # result.is_relevant -> False (seharusnya, karena beda orang)
    # result.label -> None (tidak dihitung kalau tidak relevan)

    result2 = pipeline.predict_gated(
        text="Prabowo resmikan program makan siang gratis...",
        context="Prabowo Subianto",
    )
    # result2.is_relevant -> True
    # result2.label -> "positive" / dst
"""

import sys
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

# Threshold relevansi — di bawah ini dianggap TIDAK relevan, skip sentiment.
# Mulai konservatif (0.5 = default decision boundary model). Bisa dituning
# setelah lihat distribusi score relevansi di data nyata.
RELEVANCY_THRESHOLD = 0.5

# Kata kunci untuk auto-detect mana label "relevan" di id2label
# (jaga-jaga kalau urutan/casing label berbeda dari ekspektasi)
RELEVANT_LABEL_HINTS = {"relevan", "relevant", "yes", "ya", "1", "true"}


# ─────────────────────────────────────────────────────────────
# LABEL NORMALIZATION (sentiment)
# ─────────────────────────────────────────────────────────────

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
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class GatedResult:
    is_relevant: bool
    relevancy_confidence: float
    label: Optional[str]               # None kalau is_relevant=False
    sentiment_confidence: Optional[float]
    scores: Optional[tuple]            # (neg, neu, pos), None kalau tidak relevan


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


# ─────────────────────────────────────────────────────────────
# RELEVANCY MODEL
# ─────────────────────────────────────────────────────────────

class RelevancyModel(_LoadedModel):
    def __init__(self):
        super().__init__(RELEVANCY_MODEL_ID)

        # Auto-detect index mana yang berarti "relevan"
        self._relevant_idx = None
        for idx, label in self.id2label.items():
            if label.lower().strip() in RELEVANT_LABEL_HINTS:
                self._relevant_idx = idx
                break

        if self._relevant_idx is None:
            logger.warning(
                f"Tidak bisa auto-detect label 'relevan' dari {self.id2label}. "
                f"Default ke index 1. PERIKSA MANUAL dan koreksi RELEVANT_LABEL_HINTS "
                f"kalau hasil prediksi terbalik."
            )
            self._relevant_idx = 1

    def check(self, context: str, text: str) -> tuple[bool, float]:
        """Return (is_relevant, confidence_relevan)."""
        probs = self._forward_pair(context, text)
        relevant_prob = probs[self._relevant_idx]
        return relevant_prob >= RELEVANCY_THRESHOLD, relevant_prob


# ─────────────────────────────────────────────────────────────
# SENTIMENT MODEL (context-conditioned)
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# FALLBACK MODEL (document-level, tanpa context)
# ─────────────────────────────────────────────────────────────

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
# PIPELINE — interface utama (lazy load semua sub-model)
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
        """
        Pipeline lengkap 2-stage. Kalau context=None, langsung fallback
        document-level (tidak ada yang perlu di-gate).
        """
        if not text or not text.strip():
            return GatedResult(False, 0.0, None, None, None)

        if context is None:
            label, conf, scores = self.fallback.predict(text)
            return GatedResult(True, 1.0, label, conf, scores)

        try:
            is_relevant, rel_conf = self.relevancy.check(context, text)
        except Exception as e:
            logger.error(f"Relevancy check gagal: {e} — treat sebagai relevan (fail-open)")
            is_relevant, rel_conf = True, 0.5

        if not is_relevant:
            return GatedResult(False, rel_conf, None, None, None)

        try:
            label, conf, scores = self.sentiment.predict(context, text)
        except Exception as e:
            logger.error(f"Sentiment predict gagal: {e}")
            return GatedResult(True, rel_conf, "neutral", 0.34, (0.33, 0.34, 0.33))

        return GatedResult(True, rel_conf, label, conf, scores)


@lru_cache(maxsize=1)
def get_pipeline() -> SentimentPipeline:
    return SentimentPipeline()
