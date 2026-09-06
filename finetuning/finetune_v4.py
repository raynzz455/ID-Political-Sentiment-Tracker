"""
finetune_v4.py
==============
v4 ENHANCED finetuning script — for gold standard dataset (LLM-verified + re-verified).

UPGRADES over v3:
  1. Oversampling: negative→400, positive→600 (reduces imbalance)
  2. Focal gamma: 2.5→3.0, Label smoothing: 0.05→0.07
  3. LoRA dropout: 0.15→0.20, Mixup alpha: 0.2→0.3, prob: 40%
  4. Class weights: 1/sqrt(freq)→1/log(freq+1) (gentler)
  5. Entity-stratified K-fold (GroupKFold by entity)
  6. Entity as explicit premise: "Tentang {entity}"
  7. Epochs: 20→18, SWA start: 5→4

UPGRADES over finetune.py (v1):
  1. K-Fold Cross-Validation (5-fold stratified) — robust evaluation
  2. Adversarial Training (PGD on embeddings) — fights input perturbations
  3. Mixup Augmentation — interpolates sentence pairs to fight overfitting
  4. LoRA r=64 (upgraded from 16/32) — more capacity
  5. Effective batch 64 (batch=8 x grad_accum=8)
  6. 18 epochs + SWA from epoch 4
  7. Per-sample confidence weighting (kept from v1)

Scientific basis:
  - K-fold CV: Kohavi (1995) — 5-fold has optimal bias/variance tradeoff
  - Adversarial: Miyato et al. (2017) — Virtual Adversarial Training
  - Mixup: Zhang et al. (2018) — mixup: Beyond Empirical Risk Minimization
  - SWA: Izmailov et al. (2018) — Averaging Weights Leads to Wider Optima
  - Focal Loss: Lin et al. (2017) — down-weights easy examples
  - Label Smoothing: Szegedy et al. (2016) — prevents overconfidence
  - Temperature: Guo et al. (2017) — calibrates softmax
  - LoRA: Hu et al. (2021) — parameter-efficient fine-tuning

Usage:
  python finetune_v4.py --task sentiment --kfold 5
  python finetune_v4.py --task relevancy --kfold 5
  python finetune_v4.py --task sentiment            # single split mode
"""
import os
import json
import random
import argparse
import numpy as np
import logging
from pathlib import Path
from dataclasses import asdict
from collections import Counter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_h)

# v3.1: Set CUDA memory allocator config BEFORE torch import
# Helps prevent OOM by using expandable segments (less fragmentation)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
)
# v3.2: SWA (Stochastic Weight Averaging) — Izmailov et al. 2018
# Averages model weights from SWA_START_EPOCH onwards for flatter optimum
import copy

# Fix: Uninstall incompatible torchao before importing peft
# PEFT 0.16+ requires torchao >= 0.16.0, but Colab has 0.10.0
import subprocess
try:
    import torchao
    torchao_version = getattr(torchao, '__version__', '0.0.0')
    if tuple(int(x) for x in torchao_version.split('.')[:2]) < (0, 16):
        subprocess.run(['pip', 'uninstall', '-y', 'torchao'], capture_output=True)
        print(f"[FIX] Uninstalled incompatible torchao {torchao_version}")
except ImportError:
    pass

from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GroupKFold

# Import hyperparams
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "configs"))
import hyperparams_v4 as H

# ---------------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(H.SEED)
random.seed(H.SEED)
np.random.seed(H.SEED)

# ---------------------------------------------------------------------------
# 0.1 Path helpers — resolve dataset/output dirs relative to THIS script so
# the script works no matter what cwd it is launched from.
# (BUG#2 fix: OUT_DIR_* in hyperparams are relative strings like
#  './runs/sentiment_v4'. Without resolution, running from project root vs
#  finetuning/ would scatter outputs to different places.)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_out_dir(cfg):
    """Return absolute path to the output directory, creating it if needed.

    Resolves cfg['out_dir'] against the script directory when it is a
    relative path, so outputs always land next to the script regardless of cwd.
    """
    d = cfg["out_dir"]
    p = Path(d)
    if not p.is_absolute():
        # strip leading ./ so './runs/x' -> 'runs/x' before joining
        p = _SCRIPT_DIR / p.as_posix().lstrip("./").lstrip("/")
    p.mkdir(parents=True, exist_ok=True)
    return p

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
TASK_CFG = {
    "relevancy": {
        "data_file": "dataset_gold_standard_final.jsonl",
        "label_field": "gold_relevancy",
        "text_field": "text",
        "entity_field": "entity_name",
        "entity_premise_field": "entity_premise",
        "base_model": H.RELEVANCY_BASE,
        "labels": H.RELEVANCY_LABELS,
        "out_dir": H.OUT_DIR_RELEVANCY,
        "exclude_flags": ["corruption_stitch", "wrong_entity"],
        "oversample": False,
    },
    "sentiment": {
        "data_file": "dataset_gold_standard_final.jsonl",
        "label_field": "label",
        "text_field": "text",
        "entity_field": "entity_name",
        "entity_premise_field": "entity_premise",
        "base_model": H.SENTIMENT_BASE,
        "labels": H.SENTIMENT_LABELS,
        "out_dir": H.OUT_DIR_SENTIMENT,
        "exclude_flags": ["corruption_stitch", "wrong_entity"],
        "oversample": True,
        "filter": lambda r: r.get("gold_relevancy", "relevant") == "relevant",
    },
}


def oversample_minority(rows, targets, seed=H.SEED):
    """Oversample minority classes to target counts via random duplication."""
    import random as _rng
    rng = _rng.Random(seed)
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    result = []
    for label, items in by_label.items():
        target = targets.get(label, len(items))
        pool = list(items); rng.shuffle(pool)
        result.extend(pool)
        needed = target - len(pool)
        for _ in range(max(0, needed)):
            result.append(rng.choice(pool))
    rng.shuffle(result)
    return result


class PairDataset(Dataset):
    """Sentence-pair dataset with per-sample confidence weighting."""

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
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "token_type_ids": enc.get("token_type_ids", torch.zeros_like(enc["input_ids"][0]))[0]
                if self.tokenizer.model_max_length and "token_type_ids" in enc
                else torch.zeros(self.max_len, dtype=torch.long),
            "labels": torch.tensor(self.label2id[r["label"]], dtype=torch.long),
            "sample_weight": torch.tensor(r.get("confidence", 0.5), dtype=torch.float),
        }


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def stratified_split(rows, label_key="label", seed=H.SEED):
    """Stratified train/val/test split by label_key.

    Uses H.VAL_SPLIT and H.TEST_SPLIT for proportions.
    """
    rng = random.Random(seed)
    by_label = {}
    for r in rows:
        by_label.setdefault(r[label_key], []).append(r)
    train, val, test = [], [], []
    for lab, items in by_label.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(round(n * H.TEST_SPLIT)))
        n_val = max(1, int(round(n * H.VAL_SPLIT)))
        n_train = max(1, n - n_test - n_val)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:n_train + n_val + n_test])
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


# ---------------------------------------------------------------------------
# 2. Class-balanced focal loss with mixup support
# ---------------------------------------------------------------------------
def class_weights_from_freq(labels, num_classes, method="log"):
    """Class reweighting. method='sqrt' (Cui 2019) or 'log' (v4, gentler)."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    freq = counts / counts.sum()
    if method == "sqrt":
        w = 1.0 / np.sqrt(freq + 1e-8)
    else:
        w = 1.0 / np.log(freq + 1.0 + 1e-8)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float)


class FocalLossTrainerV4(Trainer):
    """v4 Trainer: Focal Loss + log class weights + per-sample confidence +
    adversarial training + mixup."""

    def __init__(self, *args, class_weights=None, focal_gamma=H.FOCAL_GAMMA,
                 label_smoothing=H.LABEL_SMOOTHING,
                 adversarial=H.ADVERSARIAL_ENABLED,
                 mixup=H.MIXUP_ENABLED, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        # Move class_weights to GPU once (avoid repeated .to(device) every forward)
        if class_weights is not None and torch.cuda.is_available():
            class_weights = class_weights.to("cuda")
        self.class_weights = class_weights
        # v3.1: Auto-disable adversarial on low-memory GPUs (< 12GB) to prevent OOM
        if adversarial and torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_mem_gb < 12:
                logger.warning(f"GPU memory {gpu_mem_gb:.1f}GB < 12GB — disabling adversarial training (OOM prevention)")
                adversarial = False
        self.adversarial = adversarial
        self.mixup = mixup

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        sample_weights = inputs.pop("sample_weight", None)

        # Get embeddings (for mixup + adversarial)
        # NOTE: HuggingFace Trainer passes `model` param to compute_loss.
        # We use `model.training` to check if we're in train mode.
        is_training = model.training
        # v3.1: NEVER use output_hidden_states=True (saves ~2.8GB GPU memory)
        # Adversarial training uses model.get_input_embeddings() directly,
        # NOT output_hidden_states — so we don't need it at all.
        outputs = model(**inputs)
        logits = outputs.logits

        cw = self.class_weights
        if cw is not None and cw.device != logits.device:
            cw = cw.to(logits.device)

        # Mixup on logits (simpler than embedding-level)
        if self.mixup and is_training and random.random() < H.MIXUP_PROB:
            lam = np.random.beta(H.MIXUP_ALPHA, H.MIXUP_ALPHA)
            batch_size = logits.size(0)
            index = torch.randperm(batch_size, device=logits.device)
            mixed_logits = lam * logits + (1 - lam) * logits[index]
            mixed_labels = labels[index]

            # Soft labels for mixup
            onehot_orig = F.one_hot(labels, num_classes=logits.size(-1)).float()
            onehot_mixed = F.one_hot(mixed_labels, num_classes=logits.size(-1)).float()
            soft_labels = lam * onehot_orig + (1 - lam) * onehot_mixed
            soft_labels = (1 - self.label_smoothing) * soft_labels + \
                          self.label_smoothing / logits.size(-1)

            log_probs = F.log_softmax(mixed_logits, dim=-1)
            # FIX C5: compute per-sample CE from mixed logits + soft labels
            ce = -(soft_labels * log_probs).sum(dim=-1)
            # FIX C5: focal pt must come from MIXED probs (not original probs)
            mixed_probs = F.softmax(mixed_logits, dim=-1)
            pt = (soft_labels * mixed_probs).sum(dim=-1).clamp(min=1e-8)
            focal = (1.0 - pt) ** self.focal_gamma
            # FIX C5: apply class weights in mixup branch too (was missing)
            if cw is not None:
                # weight each sample by its original-label class weight
                sample_cw = cw[labels]
                per_sample = focal * ce * sample_cw
            else:
                per_sample = focal * ce
        else:
            # Standard focal loss with label smoothing
            probs = F.softmax(logits, dim=-1)
            pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
            focal = (1.0 - pt) ** self.focal_gamma
            ce = F.cross_entropy(logits, labels, weight=cw,
                                  label_smoothing=self.label_smoothing, reduction="none")
            per_sample = focal * ce

        # Per-sample confidence weighting
        if sample_weights is not None:
            per_sample = per_sample * sample_weights.to(logits.device)

        loss = per_sample.mean()

        # Adversarial training (PGD on embeddings)
        if self.adversarial and is_training:
            try:
                adv_loss = self._adversarial_loss(model, inputs, labels, sample_weights)
                loss = loss + H.ADVERSARIAL_ALPHA * adv_loss
            except AttributeError:
                # _adversarial_loss not available — skip adversarial
                pass

        return (loss, outputs) if return_outputs else loss

    def _adversarial_loss(self, model, inputs, labels, sample_weights):
        """PGD adversarial perturbation on input_ids embeddings (Miyato et al. 2017)."""
        try:
            embed_layer = model.get_input_embeddings()
            input_ids = inputs["input_ids"]
            embeds = embed_layer(input_ids)
            embeds = embeds.detach().requires_grad_(True)

            cw = self.class_weights
            if cw is not None and cw.device != embeds.device:
                cw = cw.to(embeds.device)

            with torch.enable_grad():
                inputs_adv = {k: v for k, v in inputs.items() if k != "input_ids"}
                inputs_adv["inputs_embeds"] = embeds
                outputs_adv = model(**inputs_adv)
                logits_adv = outputs_adv.logits
                probs_adv = F.softmax(logits_adv, dim=-1)
                pt_adv = probs_adv.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
                focal_adv = (1.0 - pt_adv) ** self.focal_gamma
                ce_adv = F.cross_entropy(logits_adv, labels,
                                          weight=cw,
                                          label_smoothing=self.label_smoothing, reduction="none")
                loss_adv = (focal_adv * ce_adv).mean()

            grad = torch.autograd.grad(loss_adv, embeds)[0]
            perturb = H.ADVERSARIAL_EPSILON * grad.sign()
            embeds_perturbed = embeds + perturb

            inputs_pert = {k: v for k, v in inputs.items() if k != "input_ids"}
            inputs_pert["inputs_embeds"] = embeds_perturbed.detach()
            outputs_pert = model(**inputs_pert)
            logits_pert = outputs_pert.logits
            probs_pert = F.softmax(logits_pert, dim=-1)
            pt_pert = probs_pert.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
            focal_pert = (1.0 - pt_pert) ** self.focal_gamma
            ce_pert = F.cross_entropy(logits_pert, labels,
                                       weight=cw,
                                       label_smoothing=self.label_smoothing, reduction="none")
            per_sample_pert = focal_pert * ce_pert
            if sample_weights is not None:
                per_sample_pert = per_sample_pert * sample_weights.to(logits_pert.device)
            return per_sample_pert.mean()
        except Exception as e:
            return torch.tensor(0.0, device=labels.device)

# ---------------------------------------------------------------------------
# v3.2: SWA (Stochastic Weight Averaging) Callback
# ---------------------------------------------------------------------------
class SWACallback(TrainerCallback):
    """Averages LoRA weights from SWA_START_EPOCH for flatter optimum.
    
    Paper: Izmailov et al. (2018) "Averaging Weights Leads to Wider Optima"
    Expected: +1-2% F1, better generalization on small datasets.
    """
    def __init__(self, start_epoch=5, anneal_epochs=3):
        self.start_epoch = start_epoch
        self.anneal_epochs = anneal_epochs
        self.swa_weights = None
        self.swa_count = 0
        self.n_averaged = 0

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        current_epoch = int(state.epoch)
        if current_epoch < self.start_epoch:
            return

        # Get LoRA parameters (only adapter weights, not frozen base)
        lora_params = {n: p.detach().clone() for n, p in model.named_parameters() if 'lora_' in n.lower()}

        if self.swa_weights is None:
            # First SWA epoch — initialize
            self.swa_weights = lora_params
            self.n_averaged = 1
        else:
            # Running average: swa = (swa * n + new) / (n + 1)
            for n in lora_params:
                self.swa_weights[n] = (
                    self.swa_weights[n] * self.n_averaged + lora_params[n]
                ) / (self.n_averaged + 1)
            self.n_averaged += 1

        logger.info(f"  [SWA] Epoch {current_epoch}: averaged {self.n_averaged} weight snapshots")

    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Apply SWA weights to model at end of training."""
        if self.swa_weights is None or self.n_averaged < 2:
            return

        # Replace LoRA weights with SWA-averaged version
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.swa_weights:
                    p.copy_(self.swa_weights[n])

        logger.info(f"  [SWA] Applied {self.n_averaged}-epoch weight average to model")


# ---------------------------------------------------------------------------
# 3. Temperature scaling (Guo et al. 2017)
# ---------------------------------------------------------------------------
def calibrate_temperature(model, val_ds, tokenizer, device=None):
    """Fit a scalar temperature T on the validation set via LBFGS.

    FIX C4: Previously decorated with @torch.no_grad(), which disabled the
    autograd graph for the ENTIRE function — including the LBFGS closure that
    calls loss.backward(). That raised RuntimeError("element 0 of tensors does
    not require grad"). Now we only wrap the INFERENCE loop in no_grad, and run
    the optimization outside that context so gradients flow for T.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    # --- Phase 1: collect logits under no_grad (pure inference) ---
    logits_all, labels_all = [], []
    with torch.no_grad():
        for i in range(len(val_ds)):
            item = {k: v.unsqueeze(0).to(device) for k, v in val_ds[i].items()}
            labels_all.append(int(item.pop("labels").item()))
            out = model(**item)
            logits_all.append(out.logits.squeeze(0).cpu())
    logits = torch.stack(logits_all)
    labels = torch.tensor(labels_all)

    # --- Phase 2: fit temperature T WITH grad enabled (LBFGS needs autograd) ---
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
# 4. K-Fold Cross-Validation
# ---------------------------------------------------------------------------
def run_kfold(task, all_rows, label2id, id2label, k=H.K_FOLD_N):
    """Run entity-aware K-fold CV. Uses GroupKFold by entity to prevent leakage."""
    print(f"\n{'='*70}")
    print(f"K-FOLD CV (entity-aware, k={k})")
    print(f"{'='*70}")

    labels_array = np.array([label2id[r["label"]] for r in all_rows])
    groups = [r.get("entity", r.get("entity_name", "unknown")) for r in all_rows]
    n_entities = len(set(groups))
    print(f"  Rows: {len(all_rows)}, Unique entities: {n_entities}")

    # Use GroupKFold if enough entities, else StratifiedKFold
    if n_entities >= k * 3:
        gkf = GroupKFold(n_splits=k)
        splits = gkf.split(np.zeros(len(all_rows)), labels_array, groups)
    else:
        print(f"  Warning: too few entities for GroupKFold, using StratifiedKFold")
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=H.SEED)
        splits = skf.split(np.zeros(len(all_rows)), labels_array)

    cfg = TASK_CFG[task]
    fold_results = []
    import gc
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Fold {fold+1}/{k} ---")
        train_rows = [all_rows[i] for i in train_idx]
        val_rows = [all_rows[i] for i in val_idx]

        # v4: Oversample training set only
        if H.OVERSAMPLING_ENABLED and cfg.get("oversample"):
            print(f"  Oversampling train: {len(train_rows)} -> ", end="")
            train_rows = oversample_minority(train_rows, H.OVERSAMPLING_TARGETS, seed=H.SEED + fold)
            print(f"{len(train_rows)}")

        print(f"  train: {len(train_rows)} | val: {len(val_rows)}")
        print(f"  train balance: {dict(Counter(r['label'] for r in train_rows))}")
        print(f"  val   balance: {dict(Counter(r['label'] for r in val_rows))}")

        metrics = train_single_fold(task, train_rows, val_rows, label2id, id2label,
                                     out_suffix=f"_fold{fold+1}")
        # FIX C6: tag each fold result with its fold index for downstream consumers
        metrics["fold"] = fold + 1
        fold_results.append(metrics)
        print(f"  metrics: {metrics}")

        # FIX H2: Clear GPU memory AFTER EACH fold (was outside the loop before)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate
    print(f"\n{'='*70}")
    print(f"K-FOLD RESULTS (k={k})")
    print(f"{'='*70}")
    avg_metrics = {}
    for key in fold_results[0]:
        if isinstance(fold_results[0][key], (int, float)):
            values = [r[key] for r in fold_results]
            avg = np.mean(values)
            std = np.std(values)
            avg_metrics[key] = {"mean": avg, "std": std, "values": values}
            print(f"  {key:20s}: {avg:.4f} ± {std:.4f}")

    # FIX C6: Return a format compatible with evaluate_v4.py and colab pipeline.
    # Consumers expect flat keys: mean_accuracy, std_accuracy, mean_macro_f1,
    # std_macro_f1, plus fold_results list (each with "fold" key) and "k".
    def _flat(metric_key):
        a = avg_metrics.get(metric_key, {"mean": 0.0, "std": 0.0})
        return a["mean"], a["std"]

    mean_acc, std_acc = _flat("accuracy")
    mean_f1, std_f1 = _flat("macro_f1")
    mean_wf1, std_wf1 = _flat("weighted_f1")

    return {
        "k": k,
        "task": task,
        "fold_results": fold_results,
        "folds": fold_results,  # backward-compat alias
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_macro_f1": mean_f1,
        "std_macro_f1": std_f1,
        "mean_weighted_f1": mean_wf1,
        "std_weighted_f1": std_wf1,
        "aggregate": avg_metrics,  # backward-compat: detailed per-key breakdown
    }


# ---------------------------------------------------------------------------
# 5. Single fold training
# ---------------------------------------------------------------------------
def train_single_fold(task, train_rows, val_rows, label2id, id2label,
                       out_suffix=""):
    cfg = TASK_CFG[task]
    out_dir = resolve_out_dir(cfg)  # FIX BUG#2: absolute via script dir

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["base_model"], num_labels=len(cfg["labels"]),
        id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    lora_cfg = LoraConfig(
        r=H.LORA.r, lora_alpha=H.LORA.alpha, lora_dropout=H.LORA.dropout,
        bias=H.LORA.bias, task_type=TaskType.SEQ_CLS,
        target_modules=H.LORA.target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    if not out_suffix:
        model.print_trainable_parameters()

    # Class weights
    train_label_ids = [label2id[r["label"]] for r in train_rows]
    cw = class_weights_from_freq(train_label_ids, len(cfg["labels"]))

    # Datasets
    train_ds = PairDataset(train_rows, tok, label2id)
    val_ds = PairDataset(val_rows, tok, label2id)

    steps_per_epoch = max(1, len(train_ds) // (H.BATCH_SIZE * H.GRAD_ACCUM_STEPS))
    warmup_steps = int(H.WARMUP_RATIO * steps_per_epoch * H.NUM_EPOCHS)

    # Build TrainingArguments dict (compatible with transformers 4.40+)
    train_args_dict = dict(
        output_dir=str(out_dir),
        num_train_epochs=H.NUM_EPOCHS,
        per_device_train_batch_size=H.BATCH_SIZE,
        per_device_eval_batch_size=H.BATCH_SIZE,  # v3.1: reduced from *2 (OOM fix)
        # v3.1: OOM prevention settings
        dataloader_pin_memory=False,  # saves pinned memory
        gradient_checkpointing=False,
        gradient_accumulation_steps=H.GRAD_ACCUM_STEPS,
        learning_rate=H.LEARNING_RATE,
        weight_decay=H.WEIGHT_DECAY,
        adam_beta1=H.ADAM_BETA1, adam_beta2=H.ADAM_BETA2, adam_epsilon=H.ADAM_EPSILON,
        max_grad_norm=H.MAX_GRAD_NORM,
        warmup_steps=warmup_steps,
        lr_scheduler_type=H.SCHEDULER,
        # v3.2: pass num_cycles for cosine_with_restarts
        lr_scheduler_kwargs={"num_cycles": H.SCHEDULER_NUM_CYCLES} if H.SCHEDULER == "cosine_with_restarts" else None,
        fp16=H.FP16,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",  # v3.2: full key for transformers 5.x
        greater_is_better=True,
        seed=H.SEED,
        report_to="none",
        logging_steps=max(1, steps_per_epoch // 4),
    )
    # eval_strategy: renamed in 4.46+ (try new name, fallback to old)
    try:
        train_args_dict["eval_strategy"] = "epoch"
        targs = TrainingArguments(**train_args_dict)
    except TypeError:
        # Old transformers (< 4.46) uses evaluation_strategy
        train_args_dict["evaluation_strategy"] = "epoch"
        targs = TrainingArguments(**train_args_dict)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
            "weighted_f1": f1_score(labels, preds, average="weighted"),
        }

    trainer = FocalLossTrainerV4(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        compute_metrics=compute_metrics,
        class_weights=cw,
        focal_gamma=H.FOCAL_GAMMA,
        adversarial=H.ADVERSARIAL_ENABLED,
        mixup=H.MIXUP_ENABLED,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=H.EARLY_STOP_PATIENCE),
        ] + ([SWACallback(start_epoch=H.SWA_START_EPOCH, anneal_epochs=H.SWA_ANNEAL_EPOCHS)]
             if H.SWA_ENABLED else []),
    )

    trainer.train()
    # v3.1: Clear cache before eval to prevent OOM
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    val_metrics = trainer.evaluate(val_ds, metric_key_prefix="val")

    # Temperature calibration
    T = calibrate_temperature(model, val_ds, tok)

    # FIX C7: Save model for BOTH single-split and K-fold modes.
    # Previously K-fold skipped save entirely (if not out_suffix), leaving
    # colab_complete_pipeline_v4.py unable to find per-fold dirs to upload.
    # Now: single-split → out_dir/lora; K-fold → out_dir/fold_{N}/lora
    if out_suffix:
        fold_dir = out_dir / f"fold_{out_suffix.replace('_fold', '')}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_dir = fold_dir
    else:
        save_dir = out_dir

    model.save_pretrained(save_dir / "lora")
    tok.save_pretrained(save_dir / "tokenizer")
    metrics_payload = {
        "task": task,
        "fold": int(out_suffix.replace("_fold", "")) if out_suffix else None,
        "val_metrics": val_metrics,
        "temperature": T,
        "hyperparams": {k: asdict(v) if hasattr(v, '__dataclass_fields__') else v
                        for k, v in vars(H).items() if k.isupper()},
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "class_weights": dict(zip(cfg['labels'], cw.tolist())),
    }
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"\nSaved LoRA adapter + metrics -> {save_dir}")

    return {
        "accuracy": float(val_metrics.get("val_accuracy", 0)),
        "macro_f1": float(val_metrics.get("val_macro_f1", 0)),
        "weighted_f1": float(val_metrics.get("val_weighted_f1", 0)),
        "temperature": T,
        "saved_to": str(save_dir),
    }


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main(task: str, kfold: int = 0, dataset: str = None):
    cfg = TASK_CFG[task]
    print(f"\n{'='*70}")
    print(f"FINETUNE v4 TASK: {task}")
    print(f"base: {cfg['base_model']}")
    print(f"out:  {cfg['out_dir']}")
    print(f"{'='*70}\n")

    # Load dataset
    data_path = Path(__file__).resolve().parent / "datasets" / (dataset or cfg["data_file"])
    all_rows = load_jsonl(str(data_path))
    print(f"Loaded {len(all_rows)} rows from {cfg['data_file']}")

    label2id = {l: i for i, l in enumerate(cfg["labels"])}
    id2label = {i: l for l, i in label2id.items()}
    label_field = cfg.get("label_field", "label")
    exclude_flags = cfg.get("exclude_flags", [])
    filter_fn = cfg.get("filter")

    # v4: Normalize rows — handle gold standard field names
    text_field = cfg.get("text_field", "text")
    entity_field = cfg.get("entity_field", "entity_name")
    entity_premise_field = cfg.get("entity_premise_field", "entity_premise")
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
        # FIX H4: premise fallback was `r.get(field) or f"Tentang {entity}" or entity`
        # but the f-string is always truthy, so `or entity` was dead code.
        # Now: prefer stored entity_premise, else synthesize from entity.
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
    print(f"After filter: {len(rows)} rows (excluded: {excluded})")

    if kfold > 1:
        # K-fold mode
        results = run_kfold(task, rows, label2id, id2label, k=kfold)
        out_dir = resolve_out_dir(cfg)  # FIX BUG#2
        with open(out_dir / "kfold_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nK-fold results saved -> {out_dir / 'kfold_results.json'}")
    else:
        # Single train/val/test split
        train_rows, val_rows, test_rows = stratified_split(rows, "label")
        # v4: Oversample training set
        if H.OVERSAMPLING_ENABLED and cfg.get("oversample"):
            print(f"Oversampling train: {len(train_rows)} -> ", end="")
            train_rows = oversample_minority(train_rows, H.OVERSAMPLING_TARGETS, seed=H.SEED)
            print(f"{len(train_rows)}")
        print(f"Split: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
        print(f"Train class balance: {dict(Counter(r['label'] for r in train_rows))}")
        print(f"Val   class balance: {dict(Counter(r['label'] for r in val_rows))}")
        print(f"Test  class balance: {dict(Counter(r['label'] for r in test_rows))}")

        metrics = train_single_fold(task, train_rows, val_rows, label2id, id2label)
        print(f"\nFinal metrics: {metrics}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["relevancy", "sentiment"], required=True)
    ap.add_argument("--dataset", default=None, help="Override dataset file")
    ap.add_argument("--kfold", type=int, default=0,
                    help="K-fold CV (0=disabled, 5=recommended)")
    args = ap.parse_args()
    main(args.task, kfold=args.kfold, dataset=args.dataset)
