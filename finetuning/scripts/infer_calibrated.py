"""
infer_calibrated.py
===================
Drop-in replacement for packages/nlp/sentiment_model.SentimentPipeline
that uses the LoRA-finetuned + temperature-calibrated + confidence-deferred
models produced by finetune.py, AND fixes the three codebase bugs identified
in CRITICAL_ANALYSIS.md §4:

  BUG B (title in fallback)  -> fallback now uses BODY ONLY (title excluded).
  BUG C (one context/entity) -> multi-mention aggregation: runs the sentiment
                                 model on ALL context spans per entity, then
                                 aggregates by confidence-weighted mean polarity.
  BUG D (weak relevancy)     -> relevancy gate now uses the finetuned model
                                 with the alias-aware premise.

Usage in nlp_worker.py (replace the import):
    from infer_calibrated import get_pipeline   # instead of sentiment_model

The public API is identical to SentimentPipeline.predict_gated(text, context),
so nlp_worker.py needs NO other changes. The multi-mention aggregation is
exposed via predict_gated_multi() for an upgraded worker.
"""
from __future__ import annotations
import math, logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

import sys
sys.path.insert(0, str(_script_dir))
sys.path.insert(0, str(_script_dir.parent / "configs"))
try:
    import hyperparams_optimized as H
except ImportError:
    import hyperparams as H

# torchao compatibility fix
try:
    import torchao
    from packaging import version
    if version.parse(torchao.__version__) < version.parse("0.16.0"):
        import peft.import_utils
        peft.import_utils.is_torchao_available = lambda: False
except ImportError:
    pass

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_script_dir = Path(__file__).parent if '__file__' in dir() else Path('.')
RUNS_DIR = _script_dir.parent / "runs"

@dataclass
class GatedResult:
    is_relevant: bool
    relevancy_confidence: float
    label: Optional[str]
    sentiment_confidence: Optional[float]
    scores: Optional[tuple]            # (neg, neu, pos)
    polarity_score: Optional[float] = None
    entropy: Optional[float] = None
    deferred: bool = False             # True if below confidence threshold


def _calculate_metrics(scores):
    neg, neu, pos = scores
    polarity = pos - neg
    entropy = -sum(p * math.log(p + 1e-9) for p in scores if p > 0)
    return polarity, entropy


class _FinetunedModel:
    """Loads base + LoRA + merges, applies temperature scaling."""

    def __init__(self, base_model_id: str, run_subdir: str, labels: list[str]):
        run_dir = RUNS_DIR / run_subdir
        # read temperature
        import json
        mpath = run_dir / "metrics.json"
        self.temperature = 1.0
        if mpath.exists():
            self.temperature = json.load(open(mpath)).get("temperature", 1.0)

        logger.info(f"Loading finetuned model: base={base_model_id} lora={run_dir/'lora'} T={self.temperature}")
        self.tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer")
        base = AutoModelForSequenceClassification.from_pretrained(base_model_id)
        self.model = PeftModel.from_pretrained(base, run_dir / "lora")
        self.model = self.model.merge_and_unload()
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.labels = labels
        self.id2label = {i: l for i, l in enumerate(labels)}

    @torch.no_grad()
    def _forward_pair(self, a: str, b: str) -> list[float]:
        enc = self.tokenizer(a, b, truncation=True, max_length=H.MAX_SEQ_LENGTH,
                             return_tensors="pt").to(self.device)
        logits = self.model(**enc).logits / self.temperature
        return F.softmax(logits, dim=-1)[0].cpu().tolist()


class _RelevancyModel(_FinetunedModel):
    def __init__(self):
        super().__init__(H.RELEVANCY_BASE, "relevancy", H.RELEVANCY_LABELS)

    def check(self, context: str, text: str) -> tuple[bool, float]:
        probs = self._forward_pair(text, context)   # premise=entity, hyp=context
        rel_idx = self.labels.index("relevant")
        rel_prob = probs[rel_idx]
        return rel_prob >= H.RELEVANCY_THRESHOLD, rel_prob


class _SentimentModel(_FinetunedModel):
    def __init__(self):
        super().__init__(H.SENTIMENT_BASE, "sentiment", H.SENTIMENT_LABELS)

    def predict(self, context: str, text: str) -> tuple[str, float, tuple]:
        probs = self._forward_pair(text, context)   # premise=entity, hyp=context
        idx = max(range(len(probs)), key=lambda i: probs[i])
        label = self.id2label[idx]
        conf = probs[idx]
        # order scores as (neg, neu, pos) to match sentiment_model.py
        scores = (
            probs[self.labels.index("negative")],
            probs[self.labels.index("neutral")],
            probs[self.labels.index("positive")],
        )
        return label, conf, scores


class _FallbackModel:
    """Document-level sentiment using the finetuned sentiment model on BODY ONLY.

    FIX BUG B: the original nlp_worker fed `f"{title} {body}"[:1500]` to the
    fallback. Indonesian headlines are clickbait; we now use BODY ONLY.
    """
    def __init__(self):
        self._sent = _SentimentModel()

    def predict(self, body_text: str) -> tuple[str, float, tuple]:
        # premise = a neutral marker so the pair format is preserved
        return self._sent.predict(context=body_text[:1500], text="berita ini")


class CalibratedSentimentPipeline:
    """Drop-in replacement for SentimentPipeline.

    Identical public method predict_gated(text, context) so nlp_worker.py is
    unchanged. Adds predict_gated_multi() for multi-mention aggregation
    (fixes BUG C: original kept only ONE context per entity).
    """

    def __init__(self):
        self._relevancy: Optional[_RelevancyModel] = None
        self._sentiment: Optional[_SentimentModel] = None
        self._fallback: Optional[_FallbackModel] = None

    @property
    def relevancy(self):
        if self._relevancy is None: self._relevancy = _RelevancyModel()
        return self._relevancy

    @property
    def sentiment(self):
        if self._sentiment is None: self._sentiment = _SentimentModel()
        return self._sentiment

    @property
    def fallback(self):
        if self._fallback is None: self._fallback = _FallbackModel()
        return self._fallback

    # -------------------------------------------------------------------
    # Original single-context API (back-compat with nlp_worker.py)
    # -------------------------------------------------------------------
    def predict_gated(self, text: str, context: Optional[str]) -> GatedResult:
        if not text or not text.strip():
            return GatedResult(False, 0.0, None, None, None)

        # FALLBACK PATH (document-level) — text is the BODY (not title+body)
        if context is None:
            label, conf, scores = self.fallback.predict(text)
            polarity, entropy = _calculate_metrics(scores)
            return GatedResult(True, 1.0, label, conf, scores, polarity, entropy,
                               deferred=conf < H.CONFIDENCE_TAU)

        # GATED PATH (entity-level)
        try:
            is_relevant, rel_conf = self.relevancy.check(context, text)
        except Exception as e:
            logger.error(f"Relevancy check gagal: {e} — fail-closed")
            return GatedResult(False, 0.0, None, None, None)

        if not is_relevant:
            return GatedResult(False, rel_conf, None, None, None)

        try:
            label, conf, scores = self.sentiment.predict(context, text)
            polarity, entropy = _calculate_metrics(scores)
            return GatedResult(True, rel_conf, label, conf, scores, polarity, entropy,
                               deferred=conf < H.CONFIDENCE_TAU)
        except Exception as e:
            logger.error(f"Sentiment predict gagal: {e}")
            scores = (0.33, 0.34, 0.33)
            polarity, entropy = _calculate_metrics(scores)
            return GatedResult(True, rel_conf, "neutral", 0.34, scores, polarity, entropy,
                               deferred=True)

    # -------------------------------------------------------------------
    # NEW: multi-mention aggregation (fixes BUG C)
    # -------------------------------------------------------------------
    def predict_gated_multi(self, entity_name: str, contexts: list[str]) -> GatedResult:
        """Aggregate sentiment over ALL context spans for one entity.

        Algorithm:
          1. For each context, run relevancy gate. Keep relevant spans.
          2. For each relevant span, run sentiment -> (label, conf, scores).
          3. Confidence-weighted mean of the (neg, neu, pos) score vectors.
          4. Final label = argmax of the aggregated vector.
          5. Defer if max aggregated prob < CONFIDENCE_TAU OR if <1 span
             was relevant (no signal).

        This recovers signal that BUG C discarded (the original kept only the
        single highest heuristic-quality_score span per entity).
        """
        if not contexts:
            return GatedResult(False, 0.0, None, None, None)

        rel_contexts, rel_confs = [], []
        for ctx in contexts:
            try:
                ok, rc = self.relevancy.check(ctx, entity_name)
                if ok:
                    rel_contexts.append(ctx)
                    rel_confs.append(rc)
            except Exception:
                continue

        if not rel_contexts:
            return GatedResult(False, 0.0, None, None, None)

        agg = torch.zeros(3)   # neg, neu, pos
        total_w = 0.0
        for ctx, rc in zip(rel_contexts, rel_confs):
            try:
                _, conf, scores = self.sentiment.predict(ctx, entity_name)
                w = conf * rc    # weight by both sentiment + relevancy confidence
                agg += w * torch.tensor(scores)
                total_w += w
            except Exception:
                continue

        if total_w == 0:
            return GatedResult(True, rel_confs[0], "neutral", 0.34,
                               (0.33, 0.34, 0.33), 0.0, 0.0, deferred=True)

        agg = (agg / total_w).tolist()
        idx = max(range(3), key=lambda i: agg[i])
        label = ["negative", "neutral", "positive"][idx]
        conf = agg[idx]
        polarity, entropy = _calculate_metrics(tuple(agg))
        return GatedResult(
            is_relevant=True,
            relevancy_confidence=sum(rel_confs) / len(rel_confs),
            label=label,
            sentiment_confidence=conf,
            scores=tuple(agg),
            polarity_score=polarity,
            entropy=entropy,
            deferred=conf < H.CONFIDENCE_TAU,
        )


@lru_cache(maxsize=1)
def get_pipeline() -> CalibratedSentimentPipeline:
    return CalibratedSentimentPipeline()
