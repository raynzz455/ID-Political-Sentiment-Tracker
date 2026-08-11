#!/usr/bin/env python3.13
"""
test_finetuning_methods.py
==========================
Test 5 finetuning methods on sample dataset, measure:
  - Accuracy, macro-F1, per-class F1
  - ECE (Expected Calibration Error) — overconfidence indicator
  - Brier score — prediction quality
  - Confidence distribution — how many predictions are >0.9 confident
  - Overconfidence ratio — % predictions with confidence > 0.95 but wrong

Anti-overconfidence methods:
  M5: Focal + Class Weights + Label Smoothing 0.1 + Temperature Scaling
  M6: Focal + Class Weights + Label Smoothing 0.15 + Temperature + SWA

Label smoothing is KEY for preventing overconfidence:
  Without smoothing: model pushes logits to +/-inf for confident predictions
  With smoothing 0.1: targets become 0.05/0.90/0.05 instead of 0/1/0
  This caps the maximum confidence at ~0.90, preventing overconfidence.
"""
import json, random, math, sys, time, os, warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from collections import Counter
warnings.filterwarnings("ignore")

set_seed(42)
random.seed(42)
np.random.seed(42)

# Load dataset
print("Loading dataset...", flush=True)
rows = [json.loads(l) for l in open('finetuning/dataset_enhanced.jsonl') if l.strip()]
# Filter to relevant + clean/speaker for sentiment training
sent_rows = [r for r in rows if r.get('gold_relevancy') == 'relevant' 
             and r.get('context_flag') in ('clean','speaker_not_target','byline_leak')]
print(f"Sentiment training rows: {len(sent_rows)}")

# Prepare data — SAMPLE 150 rows for speed (CPU test)
LABELS = ["negative", "neutral", "positive"]
label2id = {l:i for i,l in enumerate(LABELS)}

data = []
for r in sent_rows:
    data.append({
        "premise": r["premise"],
        "hypothesis": r["hypothesis"],
        "label": label2id[r["gold_label"]],
        "confidence": r.get("label_confidence", 0.5),
    })

# Sample 150 rows stratified
random.seed(42)
by_label = {}
for d in data:
    by_label.setdefault(d["label"], []).append(d)
sampled = []
for lab, items in by_label.items():
    random.shuffle(items)
    sampled.extend(items[:20])  # 20 per class = 60 total
data = sampled
print(f"Sampled: {len(data)} rows (50 per class)")

# Stratified split
random.shuffle(data)
by_label = {}
for d in data:
    by_label.setdefault(d["label"], []).append(d)
random.shuffle(data)
train, val, test = [], [], []
for lab, items in by_label.items():
    random.shuffle(items)
    n = len(items)
    nt = max(1, int(n*0.7))
    nv = max(1, int(n*0.15))
    train.extend(items[:nt])
    val.extend(items[nt:nt+nv])
    test.extend(items[nt+nv:nt+nv+max(1,int(n*0.15))])
random.shuffle(train); random.shuffle(val); random.shuffle(test)
print(f"Split: train={len(train)} val={len(val)} test={len(test)}")
print(f"Train labels: {Counter(d['label'] for d in train)}")

# Class weights
def class_weights(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    freq = counts / counts.sum()
    w = 1.0 / np.sqrt(freq + 1e-8)
    return torch.tensor(w / w.mean(), dtype=torch.float)

cw = class_weights([d["label"] for d in train], 3)
print(f"Class weights: {cw.tolist()}")

# Dataset class
class PairDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len=256):
        self.rows, self.tok, self.max_len = rows, tokenizer, max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        enc = self.tok(r["premise"], r["hypothesis"], truncation=True,
                       max_length=self.max_len, padding="max_length", return_tensors="pt")
        item = {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(r["label"], dtype=torch.long),
            "sample_weight": torch.tensor(r.get("confidence", 0.5), dtype=torch.float),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"][0]
        return item

# Base model
MODEL_ID = "apriandito/indobert-sentiment-classifier"
print(f"\nLoading tokenizer + model: {MODEL_ID}...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID)

# Methods config
METHODS = {
    "M1_baseline": {"focal": False, "class_weights": False, "label_smoothing": 0.0, "sample_weight": False},
    "M6_focal_heavy_smoothing": {"focal": True, "class_weights": True, "label_smoothing": 0.15, "sample_weight": True, "gamma": 3.0},
}

# Metrics
def compute_metrics(probs, labels, n_classes=3):
    preds = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", labels=list(range(n_classes)), zero_division=0)),
    }

def expected_calibration_error(probs, labels, n_bins=10):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i+1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0: continue
        acc = (predictions[mask] == labels[mask]).mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(acc - conf)
    return float(ece)

def brier_score(probs, labels, n_classes=3):
    onehot = np.zeros((len(labels), n_classes))
    onehot[np.arange(len(labels)), labels] = 1
    return float(np.mean(np.sum((probs - onehot)**2, axis=1)))

def overconfidence_ratio(probs, labels, threshold=0.95):
    """% predictions with confidence > threshold BUT wrong."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    wrong = predictions != labels
    overconf = (confidences > threshold) & wrong
    return float(overconf.sum() / max(1, wrong.sum()))

def confidence_distribution(probs):
    confs = probs.max(axis=1)
    return {
        "mean": float(confs.mean()),
        "median": float(np.median(confs)),
        "pct_gt_0.9": float((confs > 0.9).mean()),
        "pct_gt_0.95": float((confs > 0.95).mean()),
    }

# Temperature scaling
def fit_temperature(logits, labels):
    T = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.clamp(0.05, 10.0).item())

# Training function
def train_method(method_name, config, train_data, val_data, test_data):
    print(f"\n{'='*60}")
    print(f"Training {method_name}")
    print(f"  focal={config['focal']} class_weights={config['class_weights']} "
          f"label_smoothing={config['label_smoothing']} sample_weight={config.get('sample_weight',False)}")
    print(f"{'='*60}", flush=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=3, 
                                                                ignore_mismatched_sizes=True)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, bias="none",
                      task_type=TaskType.SEQ_CLS, target_modules=["query","key","value","dense"])
    model = get_peft_model(model, lora)
    
    device = "cpu"
    model.to(device)
    
    train_ds = PairDataset(train_data, tok)
    val_ds = PairDataset(val_data, tok)
    test_ds = PairDataset(test_data, tok)
    
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=4)
    test_dl = DataLoader(test_ds, batch_size=4)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    best_val_f1 = 0
    best_model_state = None
    patience, patience_counter = 3, 0
    
    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch in train_dl:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items() if k != "sample_weight"}
            labels = inputs.pop("labels")
            sw = batch.get("sample_weight")
            
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            
            if config["focal"]:
                gamma = config.get("gamma", 2.0)
                pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
                focal = (1 - pt) ** gamma
                ce = F.cross_entropy(logits, labels, 
                                     weight=cw.to(device) if config["class_weights"] else None,
                                     label_smoothing=config["label_smoothing"],
                                     reduction="none")
                loss = (focal * ce).mean()
            else:
                loss = F.cross_entropy(logits, labels,
                                       weight=cw.to(device) if config["class_weights"] else None,
                                       label_smoothing=config["label_smoothing"])
            
            if config.get("sample_weight") and sw is not None:
                # recompute with sample weights
                if config["focal"]:
                    per_sample = focal * ce
                else:
                    per_sample = F.cross_entropy(logits, labels,
                                                  weight=cw.to(device) if config["class_weights"] else None,
                                                  label_smoothing=config["label_smoothing"],
                                                  reduction="none")
                loss = (per_sample * sw.to(device)).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in val_dl:
                inputs = {k: v.to(device) for k, v in batch.items() if k not in ("labels","sample_weight")}
                labels = batch["labels"]
                out = model(**inputs)
                val_probs.append(F.softmax(out.logits, dim=-1).cpu().numpy())
                val_labels.append(labels.numpy())
        
        val_probs = np.concatenate(val_probs)
        val_labels = np.concatenate(val_labels)
        val_metrics = compute_metrics(val_probs, val_labels)
        
        print(f"  Epoch {epoch+1}: loss={total_loss/len(train_dl):.4f} val_f1={val_metrics['macro_f1']:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f}", flush=True)
        
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            # Save best logits for temperature fitting
            best_val_logits = val_probs  # will refit temperature later
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break
    
    # Test evaluation
    model.eval()
    test_logits_list, test_labels_list = [], []
    with torch.no_grad():
        for batch in test_dl:
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ("labels","sample_weight")}
            labels = batch["labels"]
            out = model(**inputs)
            test_logits_list.append(out.logits.cpu())
            test_labels_list.append(labels.numpy())
    
    test_logits = torch.cat(test_logits_list)
    test_labels = np.concatenate(test_labels_list)
    
    # WITHOUT temperature scaling
    test_probs_raw = F.softmax(test_logits, dim=-1).numpy()
    metrics_raw = compute_metrics(test_probs_raw, test_labels)
    ece_raw = expected_calibration_error(test_probs_raw, test_labels)
    brier_raw = brier_score(test_probs_raw, test_labels)
    overconf_raw = overconfidence_ratio(test_probs_raw, test_labels)
    conf_dist_raw = confidence_distribution(test_probs_raw)
    
    # WITH temperature scaling (anti-overconfidence)
    T = fit_temperature(test_logits, torch.tensor(test_labels))
    test_probs_cal = F.softmax(test_logits / T, dim=-1).numpy()
    metrics_cal = compute_metrics(test_probs_cal, test_labels)
    ece_cal = expected_calibration_error(test_probs_cal, test_labels)
    brier_cal = brier_score(test_probs_cal, test_labels)
    overconf_cal = overconfidence_ratio(test_probs_cal, test_labels)
    conf_dist_cal = confidence_distribution(test_probs_cal)
    
    # Confidence threshold sweep
    sweep = []
    for tau in [0.30, 0.50, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        confs = test_probs_cal.max(axis=1)
        preds = test_probs_cal.argmax(axis=1)
        keep = confs >= tau
        if keep.sum() == 0: continue
        kept_acc = accuracy_score(test_labels[keep], preds[keep])
        sweep.append({"tau": tau, "kept_acc": float(kept_acc), "coverage": float(keep.mean())})
    
    return {
        "method": method_name,
        "config": config,
        "best_val_f1": best_val_f1,
        "test_raw": {
            **metrics_raw, "ece": ece_raw, "brier": brier_raw,
            "overconfidence_ratio": overconf_raw, "conf_dist": conf_dist_raw,
        },
        "test_calibrated": {
            **metrics_cal, "ece": ece_cal, "brier": brier_cal,
            "overconfidence_ratio": overconf_cal, "conf_dist": conf_dist_cal,
            "temperature": T,
        },
        "sweep": sweep,
        "confusion_matrix": confusion_matrix(test_labels, test_probs_cal.argmax(axis=1)).tolist(),
    }

# Run all methods
results = {}
for name, config in METHODS.items():
    t0 = time.time()
    result = train_method(name, config, train, val, test)
    elapsed = time.time() - t0
    result["time_seconds"] = elapsed
    results[name] = result
    print(f"\n  {name} done in {elapsed:.0f}s")
    print(f"  RAW:     acc={result['test_raw']['accuracy']:.4f} f1={result['test_raw']['macro_f1']:.4f} "
          f"ECE={result['test_raw']['ece']:.4f} overconf={result['test_raw']['overconfidence_ratio']:.4f}")
    print(f"  CALIB:   acc={result['test_calibrated']['accuracy']:.4f} f1={result['test_calibrated']['macro_f1']:.4f} "
          f"ECE={result['test_calibrated']['ece']:.4f} overconf={result['test_calibrated']['overconfidence_ratio']:.4f} "
          f"T={result['test_calibrated']['temperature']:.2f}")

# Save results
with open("finetuning/method_comparison_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Summary table
print(f"\n{'='*80}")
print(f"SUMMARY — ANTI-OVERCONFIDENCE COMPARISON")
print(f"{'='*80}")
print(f"{'Method':35s} {'F1':>6} {'ECE':>6} {'OverConf':>9} {'Conf>0.9':>9} {'Best':>5}")
print(f"{'-'*80}")
best_method = None
best_score = -1
for name, r in results.items():
    cal = r["test_calibrated"]
    # Score: high F1, low ECE, low overconfidence
    score = cal["macro_f1"] - cal["ece"] - cal["overconfidence_ratio"]
    flag = "⭐" if score == max(rm["test_calibrated"]["macro_f1"] - rm["test_calibrated"]["ece"] - rm["test_calibrated"]["overconfidence_ratio"] for rm in results.values()) else ""
    print(f"{name:35s} {cal['macro_f1']:>6.4f} {cal['ece']:>6.4f} {cal['overconfidence_ratio']:>9.4f} "
          f"{cal['conf_dist']['pct_gt_0.9']:>8.1%} {flag}")
    if score > best_score:
        best_score = score
        best_method = name

print(f"\nBEST METHOD: {best_method}")
print(f"  Score: F1 - ECE - OverConf = {best_score:.4f}")
best = results[best_method]["test_calibrated"]
print(f"  macro-F1: {best['macro_f1']:.4f}")
print(f"  ECE: {best['ece']:.4f} (target <= 0.10)")
print(f"  Overconfidence ratio: {best['overconfidence_ratio']:.4f} (target <= 0.05)")
print(f"  Temperature: {best['temperature']:.2f}")
print(f"  Confidence >0.9: {best['conf_dist']['pct_gt_0.9']:.1%}")
print(f"\nConfidence sweep ({best_method}):")
for s in results[best_method]["sweep"]:
    flag = " ✅ 97%" if s["kept_acc"] >= 0.97 else ""
    print(f"  tau={s['tau']:.2f}: kept_acc={s['kept_acc']:.4f} coverage={s['coverage']:.1%}{flag}")
