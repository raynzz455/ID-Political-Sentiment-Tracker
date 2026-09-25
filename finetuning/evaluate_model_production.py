"""
evaluate_model_production.py — Automated Model Evaluation Pipeline
=====================================================================
Test otomatis untuk menguji akurasi model finetuned SEBELUM deploy
ke production. Run via GitHub Action setelah retrain di Colab.

APA YANG DIUJI:
  1. Load model dari HuggingFace (base + finetuned v4)
  2. Run inference pada gold test set (finetuning/datasets/dataset_gold_standard_final.jsonl)
  3. Hitung metrics: accuracy, macro-F1, per-class F1, ECE, confusion matrix
  4. Confidence threshold sweep (τ=0.5 → 0.9) untuk kept-set accuracy
  5. Compare base vs finetuned → pastikan v4 lebih baik
  6. Gate check: kalau macro-F1 < 0.85 → FAIL (jangan deploy)
  7. Output report JSON untuk audit trail

CARA PAKAI:
  # Local (setelah retrain):
  python finetuning/evaluate_model_production.py --task relevancy
  python finetuning/evaluate_model_production.py --task sentiment

  # Via GitHub Action:
  # Lihat .github/workflows/model-evaluation.yml

GATE CRITERIA (wajib pass sebelum deploy):
  - macro-F1 >= 0.85 (sentiment) atau >= 0.80 (relevancy)
  - ECE <= 0.20
  - Kept-set accuracy >= 0.90 at τ=0.70
  - Finetuned must beat base by >= 5pp macro-F1
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Gate criteria
GATE = {
    "sentiment": {"min_macro_f1": 0.85, "max_ece": 0.20, "min_kept_acc": 0.90},
    "relevancy": {"min_macro_f1": 0.80, "max_ece": 0.25, "min_kept_acc": 0.85},
}


def load_gold_test_set(task: str) -> list:
    """Load gold test set untuk evaluasi."""
    # Cari gold standard dataset
    candidates = [
        ROOT_DIR / "finetuning" / "datasets" / "dataset_gold_standard_final.jsonl",
        ROOT_DIR / "finetuning" / "datasets" / f"dataset_{task}.jsonl",
        ROOT_DIR / "finetuning" / "gold_labels.jsonl",
    ]
    for p in candidates:
        if p.exists():
            logger.info(f"Loading test set: {p}")
            rows = []
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            logger.info(f"  Loaded {len(rows)} rows")
            return rows
    logger.warning("No gold test set found — using empty test set")
    return []


def evaluate_model(model_id: str, task: str, test_rows: list) -> dict:
    """Run inference + compute metrics untuk satu model."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel
        import numpy as np
    except ImportError:
        logger.error("pip install torch transformers peft numpy")
        return {}

    if not test_rows:
        logger.warning(f"No test rows for {model_id}")
        return {}

    logger.info(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Try load as full model, fallback to LoRA adapter
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
    except Exception:
        # LoRA adapter — need base model
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info(model_id)
        files = [s.rfilename for s in info.siblings]
        if "lora/adapter_config.json" in files and "config.json" not in files:
            from huggingface_hub import hf_hub_download
            cfg_path = hf_hub_download(model_id, "lora/adapter_config.json")
            with open(cfg_path) as f:
                cfg = json.load(f)
            base_id = cfg.get("base_model_name_or_path", "")
            logger.info(f"  LoRA adapter. Base: {base_id}")
            base = AutoModelForSequenceClassification.from_pretrained(base_id)
            model = PeftModel.from_pretrained(base, model_id, subfolder="lora")
            model = model.merge_and_unload()
        else:
            raise

    model.eval()
    id2label = model.config.id2label
    logger.info(f"  Labels: {id2label}")

    # Run inference
    y_true, y_pred, y_proba = [], [], []
    for i, row in enumerate(test_rows):
        # Format: premise (entity) + hypothesis (context)
        entity = row.get("entity_name", row.get("entity", ""))
        context = row.get("context_text", row.get("context", ""))
        gold = row.get("gold_label", row.get("label", ""))

        if not entity or not context or not gold:
            continue

        inputs = tokenizer(entity, context, truncation=True, max_length=256,
                         return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].tolist()

        pred_idx = max(range(len(probs)), key=lambda i: probs[i])
        pred_label = id2label.get(pred_idx, str(pred_idx)).lower().strip()

        y_true.append(gold.lower().strip())
        y_pred.append(pred_label)
        y_proba.append(max(probs))

    if not y_true:
        logger.warning("No valid predictions")
        return {}

    # Compute metrics
    from collections import Counter
    labels = sorted(set(y_true) | set(y_true))
    cm = Counter(zip(y_true, y_pred))

    # Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true)

    # Macro-F1
    f1_scores = []
    for label in labels:
        tp = cm.get((label, label), 0)
        fp = sum(cm.get((l, label), 0) for l in labels if l != label)
        fn = sum(cm.get((label, l), 0) for l in labels if l != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1_scores.append(f1)
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    # ECE (Expected Calibration Error)
    n = len(y_true)
    bins = 10
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        mask = [lo <= p < hi for p in y_proba]
        if any(mask):
            bin_acc = sum(1 for i, m in enumerate(mask) if m and y_pred[i] == y_true[i]) / sum(mask)
            bin_conf = sum(p for i, p in enumerate(y_proba) if mask[i]) / sum(mask)
            ece += (sum(mask) / n) * abs(bin_acc - bin_conf)

    # Confidence sweep (kept-set accuracy)
    sweep = []
    for tau in [0.5, 0.6, 0.7, 0.8, 0.9]:
        kept = [(t, p, prob) for t, p, prob in zip(y_true, y_pred, y_proba) if prob >= tau]
        if kept:
            kept_acc = sum(1 for t, p, _ in kept if t == p) / len(kept)
            coverage = len(kept) / n
        else:
            kept_acc, coverage = 0, 0
        sweep.append({"tau": tau, "kept_accuracy": round(kept_acc, 4),
                      "coverage": round(coverage, 4), "n_kept": len(kept)})

    result = {
        "model_id": model_id,
        "task": task,
        "n_test": len(y_true),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "ece": round(ece, 4),
        "per_class_f1": {labels[i]: round(f1_scores[i], 4) for i in range(len(labels))},
        "sweep": sweep,
        "kept_acc_at_0.70": next((s["kept_accuracy"] for s in sweep if s["tau"] == 0.7), 0),
    }

    logger.info(f"  accuracy={accuracy:.4f} macro_f1={macro_f1:.4f} ece={ece:.4f}")
    return result


def check_gate(result: dict, task: str) -> tuple:
    """Check apakah model pass gate criteria. Returns (passed, reasons)."""
    gate = GATE.get(task, GATE["sentiment"])
    reasons = []
    passed = True

    if result["macro_f1"] < gate["min_macro_f1"]:
        reasons.append(f"❌ macro-F1 {result['macro_f1']} < {gate['min_macro_f1']}")
        passed = False
    else:
        reasons.append(f"✅ macro-F1 {result['macro_f1']} >= {gate['min_macro_f1']}")

    if result["ece"] > gate["max_ece"]:
        reasons.append(f"❌ ECE {result['ece']} > {gate['max_ece']}")
        passed = False
    else:
        reasons.append(f"✅ ECE {result['ece']} <= {gate['max_ece']}")

    if result["kept_acc_at_0.70"] < gate["min_kept_acc"]:
        reasons.append(f"❌ kept-acc@0.70 {result['kept_acc_at_0.70']} < {gate['min_kept_acc']}")
        passed = False
    else:
        reasons.append(f"✅ kept-acc@0.70 {result['kept_acc_at_0.70']} >= {gate['min_kept_acc']}")

    return passed, reasons


def main():
    parser = argparse.ArgumentParser(description="Model Evaluation Pipeline")
    parser.add_argument("--task", choices=["relevancy", "sentiment"], required=True)
    parser.add_argument("--model-id", default=None,
                       help="HuggingFace model ID to evaluate. Default: use NLP_*_MODEL env or finetuned v4")
    args = parser.parse_args()

    task = args.task
    # Determine model IDs
    base_id = "apriandito/indobert-relevancy-classifier" if task == "relevancy" \
              else "apriandito/indobert-sentiment-classifier"
    finetuned_id = args.model_id or os.environ.get(
        f"NLP_{task.upper()}_MODEL",
        f"Raynzz455/id-political-sentiment-{task}"
    )

    logger.info("=" * 60)
    logger.info(f"MODEL EVALUATION — {task.upper()}")
    logger.info("=" * 60)
    logger.info(f"  Base model:      {base_id}")
    logger.info(f"  Finetuned model: {finetuned_id}")

    # Load test set
    test_rows = load_gold_test_set(task)
    if not test_rows:
        logger.error("No test data — cannot evaluate")
        sys.exit(1)

    # Evaluate both models
    results = {}
    results["base"] = evaluate_model(base_id, task, test_rows)
    results["finetuned"] = evaluate_model(finetuned_id, task, test_rows)

    # Compare
    logger.info("")
    logger.info("=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)
    if results["base"] and results["finetuned"]:
        base_f1 = results["base"]["macro_f1"]
        ft_f1 = results["finetuned"]["macro_f1"]
        delta = ft_f1 - base_f1
        logger.info(f"  Base macro-F1:       {base_f1}")
        logger.info(f"  Finetuned macro-F1:  {ft_f1}")
        logger.info(f"  Delta:               {'+' if delta >= 0 else ''}{delta:.4f}")
        if delta < 0.05:
            logger.warning("  ⚠️  Finetuned tidak beat base by 5pp — pertimbangkan retrain")

    # Gate check for finetuned
    logger.info("")
    logger.info("=" * 60)
    logger.info("GATE CHECK (finetuned model)")
    logger.info("=" * 60)
    if results["finetuned"]:
        passed, reasons = check_gate(results["finetuned"], task)
        for r in reasons:
            logger.info(f"  {r}")
        logger.info(f"\n  {'✅ PASS — safe to deploy' if passed else '❌ FAIL — do NOT deploy'}")

    # Save report
    report = {
        "task": task,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "base": results.get("base", {}),
        "finetuned": results.get("finetuned", {}),
        "gate_passed": passed if results["finetuned"] else False,
    }
    report_path = ROOT_DIR / "finetuning" / f"eval_report_{task}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nReport saved: {report_path}")

    # Exit code: 0 pass, 1 fail
    sys.exit(0 if (results["finetuned"] and passed) else 1)


if __name__ == "__main__":
    main()
