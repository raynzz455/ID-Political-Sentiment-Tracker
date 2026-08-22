"""
sentiment_model_v6.py
=====================
v6 Production inference model — loads v4 LoRA adapter + temperature calibration.

Features: LoRA adapter loading, temperature scaling, confidence-threshold deferral,
batch inference, sentence-pair format (entity + context).
"""
import json, logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "apriandito/indobert-sentiment-classifier"
DEFAULT_MAX_SEQ_LENGTH = 256
DEFAULT_TEMPERATURE = 1.3
DEFAULT_CONFIDENCE_TAU = 0.70
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
LABEL_ID = {l: i for i, l in enumerate(SENTIMENT_LABELS)}
ID_LABEL = {i: l for l, i in LABEL_ID.items()}


class SentimentModelV6:
    """Production sentiment classifier with v4 LoRA adapter + calibration."""

    def __init__(self, adapter_path, base_model=DEFAULT_BASE_MODEL,
                 max_seq_length=DEFAULT_MAX_SEQ_LENGTH, temperature=DEFAULT_TEMPERATURE,
                 confidence_tau=DEFAULT_CONFIDENCE_TAU, device=None, merge_adapter=True):
        self.max_seq_length = max_seq_length
        self.temperature = temperature
        self.confidence_tau = confidence_tau
        self.adapter_path = adapter_path
        if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        base = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=len(SENTIMENT_LABELS))
        self.model = PeftModel.from_pretrained(base, str(adapter_path))
        if merge_adapter and hasattr(self.model, 'merge_and_unload'):
            self.model = self.model.merge_and_unload()
            logger.info("LoRA adapter merged")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model ready. T={temperature}, tau={confidence_tau}")

    def predict(self, entity, context, return_all=False):
        return self.predict_batch([(entity, context)], return_all=return_all)[0]

    def predict_batch(self, items, return_all=False, batch_size=32):
        all_results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            premises = [f"Tentang {item[0]}" for item in batch]  # v4: entity premise
            contexts = [item[1] for item in batch]
            enc = self.tokenizer(premises, contexts, truncation=True, max_length=self.max_seq_length,
                                 padding=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(input_ids=enc["input_ids"].to(self.device),
                                    attention_mask=enc["attention_mask"].to(self.device))
                logits = outputs.logits
                if self.temperature != 1.0: logits = logits / self.temperature
                probs = F.softmax(logits, dim=-1)
            for j, (entity, context) in enumerate(batch):
                prob = probs[j].cpu().tolist()
                pred_id = int(max(range(len(prob)), key=lambda x: prob[x]))
                confidence = float(prob[pred_id])
                result = {"label": ID_LABEL[pred_id], "confidence": confidence,
                          "deferred": confidence < self.confidence_tau, "entity": entity}
                if return_all:
                    result["probabilities"] = {SENTIMENT_LABELS[k]: float(prob[k]) for k in range(len(SENTIMENT_LABELS))}
                all_results.append(result)
        return all_results

    def predict_with_deferral(self, entity, context):
        result = self.predict(entity, context)
        if result["deferred"]: result["label"] = "ABSTAIN"
        return result

    def get_config(self):
        return {"adapter_path": self.adapter_path, "max_seq_length": self.max_seq_length,
                "temperature": self.temperature, "confidence_tau": self.confidence_tau,
                "device": str(self.device), "labels": SENTIMENT_LABELS, "model_version": "v6 (v4 LoRA)"}


def load_model(adapter_path, base_model=DEFAULT_BASE_MODEL, temperature=DEFAULT_TEMPERATURE,
               confidence_tau=DEFAULT_CONFIDENCE_TAU):
    return SentimentModelV6(adapter_path=adapter_path, base_model=base_model,
                           temperature=temperature, confidence_tau=confidence_tau)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--context", required=True)
    args = ap.parse_args()
    model = load_model(args.adapter)
    result = model.predict(args.entity, args.context, return_all=True)
    print(json.dumps(result, indent=2, ensure_ascii=False))
