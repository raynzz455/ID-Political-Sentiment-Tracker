"""
evaluate_v4.py
==============
Evaluation + confidence-threshold sweep for v4 fine-tuned models.

EffectiveAccuracy = |{x in K : pred(x)=y(x)}| / |K|     (kept-set accuracy)
Coverage          = |K| / N                               (fraction not deferred)

where K = {x : max_softmax(p(x)) >= tau}.

FIXES applied in this revision:
  - C8: Added row normalization (premise/hypothesis) matching finetune_v4.py,
        so evaluate no longer crashes on missing dataset fields. Also applies
        the sentiment filter (gold_relevancy == "relevant") and exclude_flags.
  - C9: Fixed base model selection — now uses H.SENTIMENT_BASE / H.RELEVANCY_BASE
        instead of the non-existent cfg["base_model"] key.
  - H5: Fixed relevancy stratification — now splits by the correct label field
        (gold_relevancy for relevancy task, label for sentiment task).
  - M4: Removed dead apply_temperature() placeholder.
  - C6: K-fold results reader now compatible with the new flat format
        (mean_accuracy / std_accuracy / fold_results / k).

Run after finetune_v4.py:
    python evaluate_v4.py --task sentiment --run-dir ./runs/sentiment_v4
    python evaluate_v4.py --task relevancy --run-dir ./runs/relevancy_v4
    python evaluate_v4.py --task sentiment --kfold-results ./runs/sentiment_v4/kfold_results.json
"""
from __future__ import annotations
import json, argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent / "configs"))
import hyperparams_v4 as H

# ---------------------------------------------------------------------------
# Task config — mirrors finetune_v4.py so evaluate reproduces the same split
# ---------------------------------------------------------------------------
TASK_CFG = {
    "relevancy": {
        "labels": H.RELEVANCY_LABELS,
        "data": "dataset_gold_standard_final.jsonl",
        "dir": "datasets",
        "base_model": H.RELEVANCY_BASE,
        "label_field": "gold_relevancy",
        "text_field": "text",
        "entity_field": "entity_name",
        "entity_premise_field": "entity_premise",
        "exclude_flags": ["corruption_stitch", "wrong_entity"],
        "filter": None,
        "oversample": False,
    },
    "sentiment": {
        "labels": H.SENTIMENT_LABELS,
        "data": "dataset_gold_standard_final.jsonl",
        "dir": "datasets",
        "base_model": H.SENTIMENT_BASE,
        "label_field": "label",
        "text_field": "text",
        "entity_field": "entity_name",
        "entity_premise_field": "entity_premise",
        "exclude_flags": ["corruption_stitch", "wrong_entity"],
        # FIX C8: sentiment only trains on rows deemed relevant
        "filter": lambda r: r.get("gold_relevancy", "relevant") == "relevant",
        "oversample": False,
    },
}


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def normalize_rows(all_rows, cfg, label2id):
    """FIX C8+C9+H5: Reproduce finetune_v4.py's row normalization so the test
    split used for evaluation matches the split used during training.

    Returns list of dicts with keys: premise, hypothesis, label, confidence, entity.
    """
    label_field = cfg["label_field"]
    text_field = cfg["text_field"]
    entity_field = cfg["entity_field"]
    entity_premise_field = cfg["entity_premise_field"]
    exclude_flags = cfg.get("exclude_flags", [])
    filter_fn = cfg.get("filter")

    rows = []
    excluded = {"bad_flag": 0, "filter": 0, "no_label": 0, "no_text": 0}
    for r in all_rows:
        if r.get("context_flag") in exclude_flags:
            excluded["bad_flag"] += 1
            continue
        label = r.get(label_field) or r.get("gold_label") or r.get("pseudo_label")
        if not label or label not in label2id:
            excluded["no_label"] += 1
            continue
        text = r.get(text_field) or r.get("context_text") or r.get("hypothesis", "")
        if not text or len(text.strip()) < 10:
            excluded["no_text"] += 1
            continue
        if filter_fn and not filter_fn(r):
            excluded["filter"] += 1
            continue
        entity = r.get(entity_field, "")
        premise = r.get(entity_premise_field)
        if not premise:
            premise = f"Tentang {entity}" if entity else ""
        rows.append({
            "premise": premise,
            "hypothesis": text,
            "label": label,
            "confidence": r.get("label_confidence", 0.5),
            "label_source": r.get("label_source", "unknown"),
            "entity": entity,
        })
    print(f"  Normalized: {len(rows)} rows (excluded: {excluded})")
    return rows


def stratified_split(rows, seed=H.SEED):
    """Stratified train/val/test split by r['label']. Mirrors finetune_v4.py."""
    import random
    rng = random.Random(seed)
    by = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    tr, va, te = [], [], []
    for lab, items in by.items():
        items = list(items); rng.shuffle(items)
        n = len(items)
        nt = max(1, int(round(n * H.TEST_SPLIT)))
        nv = max(1, int(round(n * H.VAL_SPLIT)))
        ntr = max(1, n - nt - nv)
        tr.extend(items[:ntr]); va.extend(items[ntr:ntr + nv]); te.extend(items[ntr + nv:])
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te


@torch.no_grad()
def score_all_calibrated(model, tok, rows, labels, T, device=None, max_len=H.MAX_SEQ_LENGTH):
    """Score with temperature scaling on logits."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    label2id = {l: i for i, l in enumerate(labels)}
    probs, golds = [], []
    for r in rows:
        enc = tok(r["premise"], r["hypothesis"], truncation=True,
                  max_length=max_len, return_tensors="pt").to(device)
        out = model(**enc)
        p = F.softmax(out.logits / T, dim=-1)[0].cpu().numpy()
        probs.append(p)
        golds.append(label2id[r["label"]])
    return np.array(probs), np.array(golds)


def confidence_threshold_sweep(probs, golds, taus=None):
    """Return list of (tau, kept_acc, coverage, macro_f1_kept, n_kept, n_deferred)."""
    if taus is None:
        taus = np.arange(0.30, 0.98, 0.03)
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    rows = []
    for tau in taus:
        keep = confs >= tau
        n_kept = int(keep.sum())
        n_def = int((~keep).sum())
        if n_kept == 0:
            continue
        kept_acc = accuracy_score(golds[keep], preds[keep])
        kept_f1 = f1_score(golds[keep], preds[keep], average="macro",
                           labels=list(range(probs.shape[1])), zero_division=0)
        rows.append({
            "tau": round(float(tau), 3),
            "kept_accuracy": round(float(kept_acc), 4),
            "coverage": round(n_kept / len(golds), 4),
            "kept_macro_f1": round(float(kept_f1), 4),
            "n_kept": n_kept,
            "n_deferred": n_def,
        })
    return rows


def evaluate_single_fold(task, run_dir):
    """Evaluate a single-fold trained model on the held-out test split."""
    cfg = TASK_CFG[task]
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    metrics = json.load(open(metrics_path)) if metrics_path.exists() else {}
    T = metrics.get("temperature", 1.0)
    print(f"Task: {task} | run_dir: {run_dir} | temperature: {T}")

    # FIX C9: use base_model from TASK_CFG (was cfg["base_model"] KeyError)
    base = cfg["base_model"]
    tok = AutoTokenizer.from_pretrained(run_dir / "tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained(base)
    model = PeftModel.from_pretrained(model, run_dir / "lora")
    model = model.merge_and_unload()  # merge LoRA for faster inference

    # FIX C8: load + normalize rows the same way finetune_v4.py does
    data_dir = Path(__file__).resolve().parent / cfg.get("dir", "datasets")
    all_rows = load_jsonl(data_dir / cfg["data"])
    label2id = {l: i for i, l in enumerate(cfg["labels"])}
    rows = normalize_rows(all_rows, cfg, label2id)

    # Reproduce the SAME stratified test split as finetune_v4.py
    _, _, test = stratified_split(rows)
    print(f"Test set: {len(test)} rows | balance: {dict(Counter(r['label'] for r in test))}")

    # score (calibrated)
    probs, golds = score_all_calibrated(model, tok, test, cfg["labels"], T)

    # full-coverage metrics
    preds = probs.argmax(axis=1)
    full_acc = accuracy_score(golds, preds)
    full_f1 = f1_score(golds, preds, average="macro",
                       labels=list(range(len(cfg["labels"]))), zero_division=0)
    cm = confusion_matrix(golds, preds, labels=list(range(len(cfg["labels"]))))
    print("\n=== FULL-COVERAGE METRICS (no deferral) ===")
    print(f"  accuracy : {full_acc:.4f}")
    print(f"  macro-F1 : {full_f1:.4f}")
    print(f"  confusion matrix (rows=true, cols=pred):")
    print(f"    labels: {cfg['labels']}")
    print(f"    {cm.tolist()}")
    print(classification_report(golds, preds, target_names=cfg["labels"], zero_division=0))

    # confidence-threshold sweep
    sweep = confidence_threshold_sweep(probs, golds)
    print("\n=== CONFIDENCE-THRESHOLD SWEEP (kept-set accuracy vs coverage) ===")
    print(f"  {'tau':>5} {'kept_acc':>9} {'coverage':>9} {'kept_F1':>8} {'kept':>5} {'defer':>6}")
    for s in sweep:
        flag = "  <-- 97% target" if s["kept_accuracy"] >= 0.97 else ""
        print(f"  {s['tau']:>5} {s['kept_accuracy']:>9.4f} {s['coverage']:>9.4f} "
              f"{s['kept_macro_f1']:>8.4f} {s['n_kept']:>5} {s['n_deferred']:>6}{flag}")

    # find the tau that hits >=97% with max coverage
    hits_97 = [s for s in sweep if s["kept_accuracy"] >= 0.97]
    best = max(hits_97, key=lambda s: s["coverage"]) if hits_97 else None
    if best:
        print(f"\n>> >=97% kept-accuracy ACHIEVED at tau={best['tau']} "
              f"with coverage={best['coverage']:.1%} ({best['n_kept']}/{len(test)} kept, "
              f"{best['n_deferred']} deferred to human/LLM).")
    else:
        max_acc = max(s["kept_accuracy"] for s in sweep)
        print(f"\n>> 97% NOT reached on this test split. Max kept-acc={max_acc:.4f}.")
        print("   Options: (a) label more gold data, (b) raise tau, (c) add the LLM second-pass for deferred.")

    # save
    out = {
        "task": task,
        "temperature": T,
        "full_coverage": {"accuracy": full_acc, "macro_f1": full_f1,
                          "confusion_matrix": cm.tolist(), "labels": cfg["labels"]},
        "sweep": sweep,
        "best_97": best,
    }
    with open(run_dir / "evaluation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {run_dir / 'evaluation.json'}")


def summarize_kfold(kfold_path):
    """FIX C6: Read kfold_results.json in the new flat format and print summary."""
    kfold = json.load(open(kfold_path))
    k = kfold.get("k", "?")
    print(f"\n{'=' * 60}")
    print(f"K-FOLD RESULTS (k={k})")
    print(f"{'=' * 60}")

    # Prefer the flat top-level keys (new format)
    mean_acc = kfold.get("mean_accuracy")
    std_acc = kfold.get("std_accuracy")
    mean_f1 = kfold.get("mean_macro_f1")
    std_f1 = kfold.get("std_macro_f1")

    # Backward-compat: derive from aggregate if flat keys missing
    if mean_acc is None:
        agg = kfold.get("aggregate", {})
        a_acc = agg.get("accuracy", {})
        a_f1 = agg.get("macro_f1", {})
        mean_acc = a_acc.get("mean", 0.0)
        std_acc = a_acc.get("std", 0.0)
        mean_f1 = a_f1.get("mean", 0.0)
        std_f1 = a_f1.get("std", 0.0)

    print(f"Mean Accuracy:  {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Mean Macro-F1:  {mean_f1:.4f} ± {std_f1:.4f}")

    # fold_results is the canonical key; "folds" is a backward-compat alias
    fold_results = kfold.get("fold_results", kfold.get("folds", []))
    print(f"\nPer-fold:")
    for r in fold_results:
        fold_num = r.get("fold", "?")
        print(f"  Fold {fold_num}: acc={r['accuracy']:.4f}, f1={r['macro_f1']:.4f}, "
              f"wf1={r.get('weighted_f1', 0):.4f}, T={r.get('temperature', 0):.3f}")

    # Identify best fold by macro-F1 (used by colab pipeline for upload)
    if fold_results:
        best = max(fold_results, key=lambda x: x.get("macro_f1", 0))
        print(f"\n  Best fold: {best.get('fold', '?')} "
              f"(macro_f1={best['macro_f1']:.4f}, saved_to={best.get('saved_to', 'n/a')})")

    return kfold


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["relevancy", "sentiment"], required=True)
    ap.add_argument("--run-dir", default=None, help="Path to single fold model dir")
    ap.add_argument("--kfold-results", default=None, help="Path to kfold_results.json")
    a = ap.parse_args()

    if a.kfold_results:
        summarize_kfold(a.kfold_results)
    elif a.run_dir:
        evaluate_single_fold(a.task, a.run_dir)
    else:
        ap.error("Either --run-dir or --kfold-results is required")
