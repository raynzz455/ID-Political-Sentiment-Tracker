"""
finetune.py
===========
LoRA finetuning for the ID-Political-Sentiment-Tracker base models.

Trains BOTH heads of the 2-stage pipeline (matches packages/nlp/sentiment_model.py):

    python finetune.py --task relevancy   # -> apriandito/indobert-relevancy-classifier + LoRA
    python finetune.py --task sentiment   # -> apriandito/indobert-sentiment-classifier + LoRA

Key design choices (justified in CRITICAL_ANALYSIS.md §7 and hyperparams.py):

  * Sentence-pair NLI format  tokenizer(premise=entity_name+alias, hypothesis=context)
    — identical to _forward_pair() in sentiment_model.py. Zero train/infer skew.
  * LoRA (r=16, alpha=32) on Q/K/V/dense — <1% trainable params, stable on 909 rows.
  * Focal loss (gamma=2) + class-balanced weights (1/sqrt(freq)) — kills the
    68% neutral majority bias and forces the model to learn pos/neg.
  * Stratified 70/15/15 split — preserves the rare negative class in val/test.
  * Early stopping on val macro-F1 (patience=3) — prevents neutral collapse.
  * Temperature scaling on val — turns softmax into a real probability so the
    confidence-threshold deferral in evaluate.py is meaningful.
  * FP16 + gradient clipping.

Run:
    pip install -r requirements_finetune.txt
    python finetune.py --task relevancy
    python finetune.py --task sentiment
"""
from __future__ import annotations
import os, json, random, argparse, math
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

import sys
from pathlib import Path
_script_dir = Path(__file__).parent if '__file__' in dir() else Path('.')
sys.path.insert(0, str(_script_dir))
sys.path.insert(0, str(_script_dir.parent / 'configs'))

_DATA_FILE = str(_script_dir.parent / 'datasets' / 'dataset_enhanced.jsonl')

try:
    import hyperparams_optimized as H
    print('[INFO] Using OPTIMIZED hyperparams (M5 anti-overconfidence)')
except ImportError:
    import hyperparams as H
    print('[INFO] Using base hyperparams')

# ---------------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------------
set_seed(H.SEED)
random.seed(H.SEED)
np.random.seed(H.SEED)

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
TASK_CFG = {
    "relevancy": {
        # Uses the ENHANCED dataset (dataset_enhanced.jsonl) which has:
        #   - gold_relevancy field (relevant | not_relevant)
        #   - label_confidence for sample weighting
        #   - context_flag to exclude corruption_stitch / wrong_entity
        "data_file": _DATA_FILE,
        "label_field": "gold_relevancy",
        "base_model": H.RELEVANCY_BASE,
        "labels": H.RELEVANCY_LABELS,
        "out_dir": H.OUT_DIR_RELEVANCY,
        "exclude_flags": ["corruption_stitch", "wrong_entity"],
    },
    "sentiment": {
        # Same enhanced dataset, filtered to gold_relevancy == "relevant"
        "data_file": _DATA_FILE,
        "label_field": "gold_label",
        "filter": lambda r: r.get("gold_relevancy") == "relevant",
        "base_model": H.SENTIMENT_BASE,
        "labels": H.SENTIMENT_LABELS,
        "out_dir": H.OUT_DIR_SENTIMENT,
        "exclude_flags": ["corruption_stitch", "wrong_entity"],
    },
}

class PairDataset(Dataset):
    """Sentence-pair dataset: (premise, hypothesis) -> label.

    premise   = entity_name (+ alias hint)
    hypothesis = cleaned context_text

    Also carries per-sample `confidence` (from label_source) so the trainer
    can down-weight unverified pseudo-labels.
    """
    def __init__(self, rows, tokenizer, label2id, max_len=H.MAX_SEQ_LENGTH):
        self.rows = rows
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        enc = self.tokenizer(
            r["premise"], r["hypothesis"],
            truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": torch.tensor(self.label2id[r["label"]], dtype=torch.long),
            "sample_weight": torch.tensor(r.get("confidence", 0.5), dtype=torch.float),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"][0]
        return item

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def stratified_split(rows, label_key, train_p, val_p, seed=H.SEED):
    """Stratified split preserving class proportions in val/test."""
    rng = random.Random(seed)
    by_label = {}
    for r in rows:
        by_label.setdefault(r[label_key], []).append(r)
    train, val, test = [], [], []
    for lab, items in by_label.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        test_split = getattr(H, "TEST_SPLIT", 0.15)
        n_test = max(1, int(round(n * test_split)))
        val_split = getattr(H, "VAL_SPLIT", 0.15)
        n_val  = max(1, int(round(n * val_split)))
        n_train = n - n_test - n_val
        # guarantee at least 1 in train when class is tiny
        if n_train < 1:
            n_train, n_val, n_test = max(1, n-2), max(0, min(1, n-1)), max(0, min(1, n-1))
            n_train = n - n_val - n_test
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train+n_val])
        test.extend(items[n_train+n_val:n_train+n_val+n_test])
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test

# ---------------------------------------------------------------------------
# 2. Class-balanced focal loss
# ---------------------------------------------------------------------------
def class_weights_from_freq(labels, num_classes):
    """1/sqrt(freq) reweighting (Cui et al. 2019, 'Class-Balanced Loss')."""
    counts = np.bincount([labels[i] for i in range(len(labels))], minlength=num_classes).astype(float)
    freq = counts / counts.sum()
    w = 1.0 / np.sqrt(freq + 1e-8)
    w = w / w.mean()   # normalise so mean weight = 1
    return torch.tensor(w, dtype=torch.float)

class FocalLossTrainer(Trainer):
    """Trainer with focal loss + class-balanced weights.

    L = - alpha_y * (1 - p_y)^gamma * log(p_y)
    """
    def __init__(self, *args, class_weights=None, focal_gamma=H.FOCAL_GAMMA, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights          # tensor (C,)
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        sample_weights = inputs.pop("sample_weight", None)   # (B,) per-sample confidence
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        # focal modulation
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
        focal = (1.0 - pt) ** self.focal_gamma
        ce = F.cross_entropy(logits, labels, weight=self.class_weights.to(logits.device), reduction="none")
        per_sample = focal * ce
        # scale by per-sample confidence (down-weight unverified pseudo-labels)
        if sample_weights is not None:
            per_sample = per_sample * sample_weights.to(logits.device)
        loss = per_sample.mean()
        return (loss, outputs) if return_outputs else loss

# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main(task: str):
    cfg = TASK_CFG[task]
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\nFINETUNE TASK: {task}\nbase: {cfg['base_model']}\nout:  {out_dir}\n{'='*70}\n")

    # 3.1 data — load ENHANCED dataset and prepare rows for this task
    all_rows = load_jsonl(cfg["data_file"])
    print(f"Loaded {len(all_rows)} rows from {cfg['data_file']}")
    label2id = {l: i for i, l in enumerate(cfg["labels"])}
    id2label = {i: l for l, i in label2id.items()}
    label_field = cfg.get("label_field", "label")
    exclude_flags = cfg.get("exclude_flags", [])
    filter_fn = cfg.get("filter")

    # Filter + normalize: map enhanced rows to {premise, hypothesis, label, confidence}
    rows = []
    excluded = {"bad_flag": 0, "filter": 0}
    for r in all_rows:
        if r.get("context_flag") in exclude_flags:
            excluded["bad_flag"] += 1
            continue
        if filter_fn and not filter_fn(r):
            excluded["filter"] += 1
            continue
        rows.append({
            "premise": r["premise"],
            "hypothesis": r["hypothesis"],
            "label": r[label_field],
            "confidence": r.get("label_confidence", 0.5),
            "label_source": r.get("label_source", "unknown"),
            "row_index": r.get("row_index", -1),
        })
    print(f"After filter: {len(rows)} rows (excluded: {excluded})")

    train_split = getattr(H, "TRAIN_SPLIT", 0.70)
    val_split = getattr(H, "VAL_SPLIT", 0.15)
    train_rows, val_rows, test_rows = stratified_split(rows, "label", train_split, val_split)
    print(f"Split: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    # class balance report
    from collections import Counter
    print("Train class balance:", dict(Counter(r["label"] for r in train_rows)))
    print("Val   class balance:", dict(Counter(r["label"] for r in val_rows)))
    print("Test  class balance:", dict(Counter(r["label"] for r in test_rows)))
    print("Train label_source:", dict(Counter(r["label_source"] for r in train_rows)))

    # 3.2 tokenizer + model
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["base_model"], num_labels=len(cfg["labels"]),
        id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # 3.3 LoRA
    lora_cfg = LoraConfig(
        r=H.LORA.r, lora_alpha=H.LORA.alpha, lora_dropout=H.LORA.dropout,
        bias=H.LORA.bias, task_type=TaskType.SEQ_CLS,
        target_modules=H.LORA.target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 3.4 class weights
    train_label_ids = [label2id[r["label"]] for r in train_rows]
    cw = class_weights_from_freq(train_label_ids, len(cfg["labels"]))
    print(f"Class weights: {dict(zip(cfg['labels'], cw.tolist()))}")

    # 3.5 datasets
    train_ds = PairDataset(train_rows, tok, label2id)
    val_ds   = PairDataset(val_rows,   tok, label2id)
    test_ds  = PairDataset(test_rows,  tok, label2id)

    # 3.6 training args
    steps_per_epoch = max(1, len(train_ds) // (H.BATCH_SIZE * H.GRAD_ACCUM_STEPS))
    warmup_steps = int(H.WARMUP_RATIO * steps_per_epoch * H.NUM_EPOCHS)

    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=H.NUM_EPOCHS,
        per_device_train_batch_size=H.BATCH_SIZE,
        per_device_eval_batch_size=H.BATCH_SIZE * 2,
        gradient_accumulation_steps=H.GRAD_ACCUM_STEPS,
        learning_rate=H.LEARNING_RATE,
        weight_decay=H.WEIGHT_DECAY,
        adam_beta1=H.ADAM_BETA1, adam_beta2=H.ADAM_BETA2, adam_epsilon=H.ADAM_EPSILON,
        max_grad_norm=H.MAX_GRAD_NORM,
        warmup_steps=warmup_steps,
        lr_scheduler_type=H.SCHEDULER,
        fp16=H.FP16,
        # eval_strategy works in transformers >=4.41, evaluation_strategy for older versions
        **({"eval_strategy": "epoch"} if "eval_strategy" in TrainingArguments.__init__.__code__.co_varnames else {"evaluation_strategy": "epoch"}),
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=H.SEED,
        report_to="none",
        logging_steps=max(1, steps_per_epoch // 4),
    )

    # 3.7 metrics
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
            "weighted_f1": f1_score(labels, preds, average="weighted"),
        }

    trainer = FocalLossTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        # processing_class (transformers v5) vs tokenizer (v4) — check Trainer not TrainingArguments
        **({"processing_class": tok} if "processing_class" in Trainer.__init__.__code__.co_varnames else {"tokenizer": tok}),
        compute_metrics=compute_metrics,
        class_weights=cw,
        focal_gamma=H.FOCAL_GAMMA,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=H.EARLY_STOP_PATIENCE)],
    )

    # 3.8 train
    trainer.train()

    # 3.9 test eval
    print("\n=== TEST EVAL ===")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print(json.dumps(test_metrics, indent=2))

    # 3.10 confusion matrix on test
    preds_out = trainer.predict(test_ds)
    preds = np.argmax(preds_out.predictions, axis=-1)
    cm = confusion_matrix(preds_out.label_ids, preds, labels=list(range(len(cfg["labels"]))))
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(f"  labels: {cfg['labels']}")
    print(cm)

    # 3.11 temperature scaling on val (calibration)
    temperature = calibrate_temperature(model, val_ds, tok)
    print(f"\nCalibrated temperature: {temperature:.3f}")

    # 3.12 save
    model.save_pretrained(out_dir / "lora")
    tok.save_pretrained(out_dir / "tokenizer")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "task": task,
            "test_metrics": test_metrics,
            "confusion_matrix": cm.tolist(),
            "labels": cfg["labels"],
            "temperature": temperature,
            "hyperparams": {k: asdict(v) if hasattr(v, '__dataclass_fields__') else v
                            for k, v in vars(H).items() if k.isupper()},
            "train_size": len(train_rows),
            "val_size": len(val_rows),
            "test_size": len(test_rows),
            "class_weights": dict(zip(cfg["labels"], cw.tolist())),
        }, f, indent=2)
    print(f"\nSaved LoRA adapter + metrics -> {out_dir}")

# ---------------------------------------------------------------------------
# 4. Temperature scaling (Guo et al. 2017)
# ---------------------------------------------------------------------------
@torch.no_grad()
def calibrate_temperature(model, val_ds, tokenizer, device=None):
    """Find T that minimises NLL on val set. Makes softmax a real probability."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    logits_all, labels_all = [], []
    for i in range(len(val_ds)):
        item = val_ds[i]
        labels_all.append(int(item["labels"].item()))
        # Prepare inputs (remove non-model keys)
        model_inputs = {}
        for k, v in item.items():
            if k in ("labels", "sample_weight"):
                continue
            model_inputs[k] = v.unsqueeze(0).to(device)
        out = model(**model_inputs)
        logits_all.append(out.logits.squeeze(0).cpu())
    logits = torch.stack(logits_all)
    labels = torch.tensor(labels_all)

    T = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.clamp(0.05, 10.0).item())

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["relevancy", "sentiment"], required=True)
    args = ap.parse_args()
    main(args.task)
