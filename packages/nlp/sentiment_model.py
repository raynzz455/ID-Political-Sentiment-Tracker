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

# PEFT optional — untuk load LoRA adapter yang belum di-merge.
# Kalau model di HuggingFace adalah LoRA adapter (punya lora/adapter_config.json),
# kita auto-detect dan load via PEFT. Kalau sudah di-merge (full model), skip.
_PEFT_AVAILABLE = False
try:
    from peft import PeftModel
    import huggingface_hub
    _PEFT_AVAILABLE = True
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
# BUG N2 FIX: Support fine-tuned v4 models via env var override.
# After uploading v4 models to HuggingFace, set these env vars to switch
# production from base models to fine-tuned v4:
#   export NLP_RELEVANCY_MODEL=Raynzz455/id-political-sentiment-relevancy
#   export NLP_SENTIMENT_MODEL=Raynzz455/id-political-sentiment-sentiment
#   export NLP_FALLBACK_MODEL=taufiqdp/indonesian-sentiment  (no v4 fallback yet)
# If env vars not set, falls back to original base models (safe default).
import os as _os

RELEVANCY_MODEL_ID = _os.environ.get(
    "NLP_RELEVANCY_MODEL",
    "apriandito/indobert-relevancy-classifier"
)
SENTIMENT_MODEL_ID  = _os.environ.get(
    "NLP_SENTIMENT_MODEL",
    "apriandito/indobert-sentiment-classifier"
)
FALLBACK_MODEL_ID   = _os.environ.get(
    "NLP_FALLBACK_MODEL",
    "taufiqdp/indonesian-sentiment"
)

logger.info(f"Model config: relevancy={RELEVANCY_MODEL_ID}, "
            f"sentiment={SENTIMENT_MODEL_ID}, fallback={FALLBACK_MODEL_ID}")

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
# PREMISE FORMAT NORMALIZATION (BUG N6 FIX — CRITICAL)
# ─────────────────────────────────────────────────────────────
# Training v4 pakai premise = "Tentang {entity}" (dari dataset.entity_premise).
# Production sebelumnya kirim entity_name langsung = "{entity}" (TANPA prefix).
# Mismatch ini menurunkan akurasi v4 models di production karena model dilatih
# dengan format "Tentang Erick Thohir" tapi menerima "Erick Thohir" saja.
#
# Fix: normalize context ke format "Tentang {entity}" sebelum kirim ke model.
# Env var NLP_PREMISE_PREFIX controls:
#   - "Tentang " (default) → match v4 training format
#   - "" (empty)           → raw entity_name (untuk base model apriandito)
PREMISE_PREFIX = _os.environ.get("NLP_PREMISE_PREFIX", "Tentang ")


def normalize_premise(context: Optional[str]) -> Optional[str]:
    """Normalize context ke format yang sama dengan training v4.

    Training v4: premise = "Tentang Erick Thohir"
    Production harus match: context → "Tentang Erick Thohir"

    Jika context sudah punya prefix "Tentang ", tidak di-double.
    Jika context kosong/None, return apa adanya.
    """
    if not context or not context.strip():
        return context
    # Jika sudah ada prefix "Tentang " (atau prefix lain), jangan double
    if PREMISE_PREFIX and context.strip().lower().startswith(PREMISE_PREFIX.strip().lower()):
        return context
    # Tambahkan prefix
    if PREMISE_PREFIX:
        return f"{PREMISE_PREFIX}{context.strip()}"
    return context


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
    # FIX SF#3 (HIGH): Flag untuk distinguish real predictions from error fallbacks.
    # Before: except block returned GatedResult(True, rel_conf, "neutral", 0.34, ...)
    # → looked like legit prediction → nlp_worker inserted it as real sentiment.
    # After: is_error=True → nlp_worker can skip/error-handle these.
    is_error: bool = False


# ─────────────────────────────────────────────────────────────
# BASE MODEL LOADER
# ─────────────────────────────────────────────────────────────

class _LoadedModel:
    def __init__(self, model_id: str):
        logger.info(f"Loading {model_id} ...")
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Auto-detect: apakah repo ini LoRA adapter atau full model?
        # LoRA adapter repo punya lora/adapter_config.json tapi TIDAK punya
        # config.json di root. Full model punya config.json di root.
        base_model_id, is_lora = self._detect_lora_format(model_id)

        if is_lora and _PEFT_AVAILABLE:
            logger.info(f"  -> LoRA adapter detected. Loading base + adapter via PEFT...")
            base = AutoModelForSequenceClassification.from_pretrained(base_model_id)
            self.model = PeftModel.from_pretrained(base, model_id, subfolder="lora")
            self.model = self.model.merge_and_unload()
            logger.info(f"  -> LoRA merged into base (in-memory).")
        elif is_lora and not _PEFT_AVAILABLE:
            raise RuntimeError(
                f"Model {model_id} adalah LoRA adapter tapi library 'peft' belum "
                f"ter-install. Solusi: (1) pip install peft, ATAU (2) jalankan "
                f"finetuning/merge_and_upload_lora.py di Colab untuk merge & "
                f"re-upload sebagai full model."
            )
        else:
            # Full model — load standard
            self.model = AutoModelForSequenceClassification.from_pretrained(model_id)

        self.model.to(DEVICE)
        self.model.eval()
        self.id2label = self.model.config.id2label
        logger.info(f"  -> loaded. id2label = {self.id2label}")

    @staticmethod
    def _detect_lora_format(model_id: str) -> tuple[str, bool]:
        """Cek apakah repo HF berisi LoRA adapter atau full model.

        Returns: (base_model_id, is_lora)
        - is_lora=True jika repo punya lora/adapter_config.json & tidak ada config.json di root
        - is_lora=False jika repo punya config.json di root (full/merged model)
        """
        if not _PEFT_AVAILABLE:
            # Kalau PEFT tidak ada, anggap full model (legacy behavior)
            return ("", False)
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            info = api.model_info(model_id)
            files = [s.rfilename for s in info.siblings]
            has_lora_config = "lora/adapter_config.json" in files
            has_root_config = "config.json" in files

            if has_lora_config and not has_root_config:
                # LoRA adapter — baca base_model dari adapter_config.json
                import json
                from huggingface_hub import hf_hub_download
                cfg_path = hf_hub_download(
                    model_id, "lora/adapter_config.json",
                )
                with open(cfg_path) as f:
                    cfg = json.load(f)
                base_model_id = cfg.get("base_model_name_or_path", "")
                logger.info(f"  -> LoRA detected. Base model: {base_model_id}")
                return (base_model_id, True)
            else:
                # Full model (atau LoRA yang sudah di-merge)
                return ("", False)
        except Exception as e:
            # Fallback: anggap full model (jangan block production)
            logger.debug(f"  -> LoRA detection skipped ({e}), assuming full model")
            return ("", False)

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
        # FIX EC#11 (LOW): Use max(range) instead of probs.index(max(probs)).
        # Before: if probs contains NaN, max() returns NaN, list.index(nan) raises ValueError.
        # After: max(range) handles NaN gracefully (returns first max index).
        pred_idx = max(range(len(probs)), key=lambda i: probs[i])
        scores = {normalize_label(self.id2label[i]): probs[i] for i in range(len(probs))}
        label = normalize_label(self.id2label[pred_idx])
        conf = probs[pred_idx]
        score_tuple = (scores.get("negative", 0.0), scores.get("neutral", 0.0), scores.get("positive", 0.0))
        return label, conf, score_tuple


class FallbackModel(_LoadedModel):
    def __init__(self):
        super().__init__(FALLBACK_MODEL_ID)

    def predict(self, text: str) -> tuple[str, float, tuple]:
        probs = self._forward_single(text)
        # FIX EC#11 (LOW): same NaN-safe max logic
        pred_idx = max(range(len(probs)), key=lambda i: probs[i])
        scores = {normalize_label(self.id2label[i]): probs[i] for i in range(len(probs))}
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

    def predict_gated(self, text: str, context: Optional[str],
                      skip_relevancy: bool = False) -> GatedResult:
        """Predict sentiment with relevancy gate.

        BUG N3 FIX: skip_relevancy=True skips the internal relevancy model
        call when the caller has ALREADY confirmed relevancy (e.g.,
        context_worker pre-filtered via is_relevant=True in metadata).
        This saves ~0.5s per span (avoids running the same model twice).

        BUG N6 FIX (CRITICAL): context dinormalize ke format "Tentang {entity}"
        untuk match dengan training v4. Sebelumnya production kirim entity_name
        langsung (mis. "Erick Thohir") padahal training pakai "Tentang Erick Thohir".
        """
        if not text or not text.strip():
            return GatedResult(False, 0.0, None, None, None)

        # FALLBACK PATH (Document-level) — no context, no normalization needed
        if context is None:
            # FIX SF#9 (HIGH): Add error handling to prevent infinite retry loop.
            # Before: no try/except → exception propagates to nlp_worker →
            # inference_error → item NOT acked → requeued → fail again → infinite loop.
            # After: return GatedResult(is_error=True) → nlp_worker skips gracefully.
            try:
                label, conf, scores = self.fallback.predict(text)
                polarity, entropy = calculate_continuous_metrics(scores)
                return GatedResult(True, 1.0, label, conf, scores, polarity, entropy)
            except Exception as e:
                logger.error(f"Fallback predict gagal: {e}")
                scores = (0.33, 0.34, 0.33)
                polarity, entropy = calculate_continuous_metrics(scores)
                return GatedResult(True, 1.0, "neutral", 0.34, scores, polarity, entropy, is_error=True)

        # BUG N6 FIX: normalize context ke format training v4
        # Training: premise = "Tentang Erick Thohir"
        # Production sebelumnya: context = "Erick Thohir" (MISMATCH!)
        # Sekarang: context = normalize_premise("Erick Thohir") = "Tentang Erick Thohir"
        context = normalize_premise(context)

        # GATED PATH (Entity-level)
        # BUG N3 FIX: skip relevancy check if caller already confirmed it
        if not skip_relevancy:
            try:
                is_relevant, rel_conf = self.relevancy.check(context, text)
            except Exception as e:
                # FAIL-CLOSED: Jika gate error, anggap tidak relevan
                logger.error(f"Relevancy check gagal: {e} — treat sebagai TIDAK relevan (fail-closed)")
                return GatedResult(False, 0.0, None, None, None)

            if not is_relevant:
                return GatedResult(False, rel_conf, None, None, None)
        else:
            # Caller confirmed relevancy — trust pre-filter
            rel_conf = 1.0

        try:
            label, conf, scores = self.sentiment.predict(context, text)
            polarity, entropy = calculate_continuous_metrics(scores)
            return GatedResult(True, rel_conf, label, conf, scores, polarity, entropy)
        except Exception as e:
            # FIX SF#3 (HIGH): Flag as error so nlp_worker can skip, not insert fake neutral.
            logger.error(f"Sentiment predict gagal: {e}")
            scores = (0.33, 0.34, 0.33)
            polarity, entropy = calculate_continuous_metrics(scores)
            return GatedResult(True, rel_conf, "neutral", 0.34, scores, polarity, entropy, is_error=True)


@lru_cache(maxsize=1)
def get_pipeline() -> SentimentPipeline:
    return SentimentPipeline()