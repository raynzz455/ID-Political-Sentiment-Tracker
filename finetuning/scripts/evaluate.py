"""
evaluate.py
===========
Evaluation + the confidence-threshold sweep that produces the ≥97% kept-accuracy
curve promised in CRITICAL_ANALYSIS.md §7.1.

EffectiveAccuracy = |{x in K : pred(x)=y(x)}| / |K|      (kept-set accuracy)
Coverage          = |K| / N                                (fraction not deferred)

where K = {x : max_softmax(p(x)) >= tau}.

Run after finetune.py:
    python evaluate.py --task sentiment --run-dir ./runs/sentiment
    python evaluate.py --task relevancy --run-dir ./runs/relevancy
"""
from __future__ import annotations
import json, argparse, math
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

import sys
from pathlib import Path
_script_dir = Path(__file__).parent if '__file__' in dir() else Path('.')
sys.path.insert(0, str(_script_dir))
sys.path.insert(0, str(_script_dir.parent / 'configs'))
try:
    import hyperparams_optimized as H
except ImportError:
    import hyperparams as H

# torchao compatibility fix (same as finetune.py)
try:
    import torchao
    from packaging import version
    if version.parse(torchao.__version__) < version.parse("0.16.0"):
        import peft.import_utils
        peft.import_utils.is_torchao_available = lambda: False
except ImportError:
    pass

_DATA_FILE = str(_script_dir.parent / "datasets" / "dataset_enhanced.jsonl")

TASK_CFG = {
    "relevancy": {
        "labels": H.RELEVANCY_LABELS,
        "data": _DATA_FILE,
        "base_model": H.RELEVANCY_BASE,
        "label_field": "gold_relevancy",
        "exclude_flags": ["corruption_stitch", "wrong_entity"],
    },
    "sentiment": {
        "labels": H.SENTIMENT_LABELS,
        "data": _DATA_FILE,
        "base_model": H.SENTIMENT_BASE,
        "label_field": "gold_label",
        "filter": lambda r: r.get("gold_relevancy") == "relevant",
        "exclude_flags": ["corruption_stitch", "wrong_entity"],
    },
}

def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def stratified_split(rows, seed=H.SEED):
    import random
    rng = random.Random(seed)
    by = {}
    for r in rows: by.setdefault(r["label"], []).append(r)
    tr, va, te = [], [], []
    for lab, items in by.items():
        items = list(items); rng.shuffle(items)
        n = len(items)
        nt = max(1, int(round(n*H.TEST_SPLIT)))
        nv = max(1, int(round(n*H.VAL_SPLIT)))
        ntr = max(1, n - nt - nv)
        tr.extend(items[:ntr]); va.extend(items[ntr:ntr+nv]); te.extend(items[ntr+nv:])
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te

@torch.no_grad()
def score_all(model, tok, rows, labels, device=None, max_len=H.MAX_SEQ_LENGTH):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    label2id = {l:i for i,l in enumerate(labels)}
    probs, golds = [], []
    for r in rows:
        enc = tok(r["premise"], r["hypothesis"], truncation=True,
                  max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        p = F.softmax(out.logits / 1.0, dim=-1)[0].cpu().numpy()
        probs.append(p)
        golds.append(label2id[r["label"]])
    return np.array(probs), np.array(golds)

def apply_temperature(probs, T):
    # re-softmax with temperature on logits: but we only have probs.
    # Approximate by sharpening: p' = p^(1/T) / sum. Valid for calibration
    # comparison; for exact, re-run score_all with T. We re-run instead.
    return probs  # placeholder; real T applied in score_all via logits/T

@torch.no_grad()
def score_all_calibrated(model, tok, rows, labels, T, device=None, max_len=H.MAX_SEQ_LENGTH):
    """Score with temperature scaling on logits."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    label2id = {l:i for i,l in enumerate(labels)}
    probs, golds = [], []
    for r in rows:
        enc = tok(r["premise"], r["hypothesis"], truncation=True,
                  max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
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

def main(task, run_dir):
    cfg = TASK_CFG[task]
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    metrics = json.load(open(metrics_path)) if metrics_path.exists() else {}
    T = metrics.get("temperature", 1.0)
    print(f"Task: {task} | run_dir: {run_dir} | temperature: {T}")

    # load base + LoRA
    base = cfg["base_model"]
    tok = AutoTokenizer.from_pretrained(run_dir / "tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained(base)
    model = PeftModel.from_pretrained(model, run_dir / "lora")
    model = model.merge_and_unload()  # merge LoRA for faster inference

    # data — reproduce the SAME stratified test split as finetune.py
    all_rows = load_jsonl(cfg["data"])
    # Filter rows for this task
    label_field = cfg.get("label_field", "label")
    exclude_flags = cfg.get("exclude_flags", [])
    filter_fn = cfg.get("filter")
    rows = []
    for r in all_rows:
        if r.get("context_flag") in exclude_flags:
            continue
        if filter_fn and not filter_fn(r):
            continue
        rows.append({"premise": r.get("premise",""), "hypothesis": r.get("hypothesis",""),
                     "label": r.get(label_field, "neutral"), "row_index": r.get("row_index", -1)})
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["relevancy", "sentiment"], required=True)
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    main(a.task, a.run_dir)
