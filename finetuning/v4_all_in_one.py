"""
v4_all_in_one.py
================
Self-contained v4 fine-tuning + evaluation script for Colab.

Combines: hyperparams + finetune + evaluate in ONE file, so you can upload
just this file + the dataset to Colab and run the full pipeline.

FIXES applied (see finetune_v4.py / evaluate_v4.py changelog for details):
  - C1: from __future__ import annotations now at file top (was mid-file → SyntaxError)
  - C2: hyperparams are module-level constants (no more H = None AttributeError)
  - C3: single main() + single __main__ block with subcommand dispatch
  - C4: calibrate_temperature no longer wrapped in @torch.no_grad() (LBFGS needs grad)
  - C5: mixup focal weight now uses MIXED probs + class weights applied in mixup branch
  - C6: K-fold JSON output uses flat keys (mean_accuracy, fold_results, k, ...)
  - C7: K-fold mode now saves per-fold models (fold_N/lora) for upload
  - C8: evaluate normalizes rows (premise/hypothesis) matching finetune
  - C9: base model selected from H.SENTIMENT_BASE / H.RELEVANCY_BASE
  - H1: stratified_split params cleaned up
  - H2: GPU memory cleared between folds (not just after all folds)
  - H5: relevancy task stratifies by gold_relevancy (not sentiment label)

Usage in Colab:
  1. Upload this file (v4_all_in_one.py) to /content/
  2. Upload dataset_gold_standard_final.jsonl to /content/finetuning/datasets/
  3. !pip install -q transformers peft scikit-learn accelerate sentencepiece
  4. !python v4_all_in_one.py finetune --task sentiment --kfold 5
  5. !python v4_all_in_one.py evaluate --task sentiment --run-dir ./runs/sentiment_v4
  6. !python v4_all_in_one.py kfold-summary --kfold-results ./runs/sentiment_v4/kfold_results.json
"""
from __future__ import annotations

# ============================================================================
# SECTION 1: HYPERPARAMETERS (from hyperparams_v4.py)
# ============================================================================
import os
import json
import random
import argparse
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter
from types import SimpleNamespace

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_h)

# v3.1: Set CUDA memory allocator config BEFORE torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

RELEVANCY_BASE = "apriandito/indobert-relevancy-classifier"
SENTIMENT_BASE = "apriandito/indobert-sentiment-classifier"
FALLBACK_BASE = "taufiqdp/indonesian-sentiment"

MAX_SEQ_LENGTH = 256
PAIR_FORMAT = True


@dataclass
class LoRAConfig:
    r: int = 64
    alpha: int = 128
    dropout: float = 0.20
    bias: str = "none"
    task_type: str = "SEQ_CLS"
    target_modules: list = None
    target_modules_extended: list = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["query", "key", "value", "dense"]
        if self.target_modules_extended is None:
            self.target_modules_extended = ["query", "key", "value", "dense", "intermediate.dense"]


# Build a namespace object H so the rest of the code can use H.SEED, H.LORA, etc.
_LORA = LoRAConfig()
H = SimpleNamespace(
    RELEVANCY_BASE=RELEVANCY_BASE,
    SENTIMENT_BASE=SENTIMENT_BASE,
    FALLBACK_BASE=FALLBACK_BASE,
    MAX_SEQ_LENGTH=MAX_SEQ_LENGTH,
    PAIR_FORMAT=PAIR_FORMAT,
    LORA=_LORA,
    LEARNING_RATE=2.5e-5,
    WEIGHT_DECAY=0.03,
    ADAM_EPSILON=1e-8,
    ADAM_BETA1=0.9,
    ADAM_BETA2=0.999,
    MAX_GRAD_NORM=1.0,
    WARMUP_RATIO=0.06,
    SCHEDULER="cosine_with_restarts",
    SCHEDULER_NUM_CYCLES=2,
    BATCH_SIZE=8,
    GRAD_ACCUM_STEPS=8,
    NUM_EPOCHS=18,
    EARLY_STOP_PATIENCE=5,
    FOCAL_GAMMA=3.0,
    LABEL_SMOOTHING=0.07,
    CLASS_WEIGHT_FN="log",
    SWA_ENABLED=True,
    SWA_START_EPOCH=4,
    SWA_LR=5e-6,
    SWA_ANNEAL_EPOCHS=3,
    ADVERSARIAL_ENABLED=True,
    ADVERSARIAL_EPSILON=1e-5,
    ADVERSARIAL_ALPHA=0.5,
    MIXUP_ENABLED=True,
    MIXUP_ALPHA=0.3,
    MIXUP_PROB=0.4,
    OVERSAMPLING_ENABLED=True,
    OVERSAMPLING_TARGETS={"negative": 400, "positive": 600},
    K_FOLD_ENABLED=True,
    K_FOLD_N=5,
    K_FOLD_STRATIFIED=True,
    K_FOLD_ENTITY_AWARE=True,
    VAL_SPLIT=0.15,
    TEST_SPLIT=0.15,
    TRAIN_SPLIT=0.70,
    SEED=42,
    SENTIMENT_LABELS=["negative", "neutral", "positive"],
    RELEVANCY_LABELS=["not_relevant", "relevant"],
    TEMPERATURE=1.3,
    CONFIDENCE_TAU=0.70,
    OUT_DIR_RELEVANCY="./runs/relevancy_v4",
    OUT_DIR_SENTIMENT="./runs/sentiment_v4",
    HF_ORG="raynzz455",
    HF_MODEL_PREFIX="id-political-sentiment",
    HF_SENTIMENT_MODEL="raynzz455/id-political-sentiment-sentiment-v4",
    HF_RELEVANCY_MODEL="raynzz455/id-political-sentiment-relevancy-v4",
    DETERMINISTIC=True,
    FP16=True,
)

# ============================================================================
# SECTION 2: TORCH + TRANSFORMERS IMPORTS
# ============================================================================
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback,
)
import copy

# Fix: Uninstall incompatible torchao before importing peft
import subprocess
try:
    import torchao
    _tv = getattr(torchao, "__version__", "0.0.0")
    if tuple(int(x) for x in _tv.split(".")[:2]) < (0, 16):
        subprocess.run(["pip", "uninstall", "-y", "torchao"], capture_output=True)
        print(f"[FIX] Uninstalled incompatible torchao {_tv}")
except ImportError:
    pass

from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, GroupKFold

# Reproducibility
torch.manual_seed(H.SEED)
random.seed(H.SEED)
np.random.seed(H.SEED)

# ---------------------------------------------------------------------------
# Path helpers — resolve dataset/output dirs relative to THIS script so the
# script works no matter what cwd it is launched from.
# (BUG#1+BUG#2 fix: previously data_dir and out_dir were relative to cwd, so
#  launching from project root vs finetuning/ vs /content/ produced different
#  paths and the script could not find the dataset / scattered outputs.)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_data_dir(cfg):
    """Return absolute path to the dataset directory.

    Priority:
      1. --data-dir CLI flag (handled in cmd_finetune via cfg override)
      2. cfg['data_dir'] resolved against script dir
    """
    d = cfg.get("data_dir", "datasets")
    p = Path(d)
    if not p.is_absolute():
        p = _SCRIPT_DIR / p
    return p


def resolve_out_dir(cfg):
    """Return absolute path to the output directory, creating it if needed.

    Resolves cfg['out_dir'] (which may be a relative string like
    './runs/sentiment_v4') against the script directory so outputs always
    land next to the script regardless of cwd.
    """
    d = cfg["out_dir"]
    p = Path(d)
    if not p.is_absolute():
        # strip leading ./ so './runs/x' -> 'runs/x' before joining
        p = _SCRIPT_DIR / p.as_posix().lstrip("./").lstrip("/")
    p.mkdir(parents=True, exist_ok=True)
    return p


# ============================================================================
# SECTION 2.5: Adaptive GPU VRAM scaling (v4.1)
# ============================================================================
# Maximize GPU utilization instead of leaving 30-50% VRAM idle.
# Auto-scales batch_size + max_seq_length + grad_accum based on detected VRAM.
_VRAM_TIERS = [
    (0,   4,  256, False, 16),  # < 8 GB:  CPU fallback / tiny GPU
    (8,   8,  256, False, 8),   # 8-12 GB: T4 free tier
    (12, 16,  256, True,  4),   # 12-16 GB: T4 full / P100
    (16, 24,  320, True,  4),   # 16-24 GB: V100 / A10
    (24, 32,  384, True,  2),   # > 24 GB: A100 / A6000
]


def auto_scale_gpu_config(base_batch=H.BATCH_SIZE,
                           base_seq=H.MAX_SEQ_LENGTH,
                           base_adversarial=H.ADVERSARIAL_ENABLED,
                           base_grad_accum=H.GRAD_ACCUM_STEPS,
                           verbose=True):
    """Return (batch_size, max_seq_length, adversarial, grad_accum) tuned to GPU.

    Keeps effective batch size (batch * accum) consistent with base config
    so gradient statistics stay stable across GPU sizes.
    """
    if not torch.cuda.is_available():
        if verbose:
            logger.info("[GPU] No CUDA — using base config (CPU mode)")
        return base_batch, base_seq, False, base_grad_accum

    if os.environ.get("DISABLE_AUTO_SCALE") == "1":
        if verbose:
            logger.info(f"[GPU] Auto-scale DISABLED — using base config "
                        f"(batch={base_batch}, seq={base_seq}, accum={base_grad_accum})")
        return base_batch, base_seq, base_adversarial, base_grad_accum

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    name = props.name

    batch, seq, adv, accum = base_batch, base_seq, False, base_grad_accum
    for threshold, b, s, a, ga in _VRAM_TIERS:
        if vram_gb >= threshold:
            batch, seq, adv, accum = b, s, a, ga

    # Preserve effective batch size
    target_effective = base_batch * base_grad_accum
    new_accum = max(1, target_effective // batch)

    if not base_adversarial:
        adv = False

    if verbose:
        logger.info(
            f"[GPU] {name} ({vram_gb:.1f} GB VRAM) → "
            f"batch={batch}, seq={seq}, accum={new_accum} "
            f"(effective batch={batch*new_accum}), adversarial={adv}"
        )
    return batch, seq, adv, new_accum


# ============================================================================
# SECTION 3: TASK CONFIG + DATASET
# ============================================================================
TASK_CFG = {
    "relevancy": {
        "data_file": "dataset_gold_standard_final.jsonl",
        "data_dir": "datasets",
        "label_field": "gold_relevancy",
        "text_field": "text",
        "entity_field": "entity_name",
        "entity_premise_field": "entity_premise",
        "base_model": H.RELEVANCY_BASE,
        "labels": H.RELEVANCY_LABELS,
        "out_dir": H.OUT_DIR_RELEVANCY,
        "exclude_flags": ["corruption_stitch", "wrong_entity"],
        "oversample": False,
        "filter": None,
    },
    "sentiment": {
        "data_file": "dataset_gold_standard_final.jsonl",
        "data_dir": "datasets",
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


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def normalize_rows(all_rows, cfg, label2id):
    """Reproduce finetune_v4.py row normalization so premise/hypothesis exist."""
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


def oversample_minority(rows, targets, seed=H.SEED):
    """Oversample minority classes to target counts via random duplication."""
    rng = random.Random(seed)
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    result = []
    for label, items in by_label.items():
        target = targets.get(label, len(items))
        pool = list(items)
        rng.shuffle(pool)
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
        tti = enc.get("token_type_ids")
        if tti is not None:
            tti = tti[0]
        else:
            tti = torch.zeros(self.max_len, dtype=torch.long)
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "token_type_ids": tti,
            "labels": torch.tensor(self.label2id[r["label"]], dtype=torch.long),
            "sample_weight": torch.tensor(r.get("confidence", 0.5), dtype=torch.float),
        }


def stratified_split(rows, label_key="label", seed=H.SEED):
    """Stratified train/val/test split by label_key. Uses H.VAL_SPLIT/TEST_SPLIT."""
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
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# ============================================================================
# SECTION 4: FOCAL LOSS TRAINER + ADVERSARIAL + MIXUP
# ============================================================================
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
        # Move class_weights to GPU once
        if class_weights is not None and torch.cuda.is_available():
            class_weights = class_weights.to("cuda")
        self.class_weights = class_weights
        if adversarial and torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if gpu_mem_gb < 12:
                logger.warning(f"GPU memory {gpu_mem_gb:.1f}GB < 12GB — disabling adversarial (OOM)")
                adversarial = False
        self.adversarial = adversarial
        self.mixup = mixup

    def _cw(self, device):
        cw = self.class_weights
        if cw is not None and cw.device != device:
            cw = cw.to(device)
        return cw

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        sample_weights = inputs.pop("sample_weight", None)
        is_training = model.training
        outputs = model(**inputs)
        logits = outputs.logits
        cw = self._cw(logits.device)

        if self.mixup and is_training and random.random() < H.MIXUP_PROB:
            lam = np.random.beta(H.MIXUP_ALPHA, H.MIXUP_ALPHA)
            batch_size = logits.size(0)
            index = torch.randperm(batch_size, device=logits.device)
            mixed_logits = lam * logits + (1 - lam) * logits[index]
            mixed_labels = labels[index]

            onehot_orig = F.one_hot(labels, num_classes=logits.size(-1)).float()
            onehot_mixed = F.one_hot(mixed_labels, num_classes=logits.size(-1)).float()
            soft_labels = lam * onehot_orig + (1 - lam) * onehot_mixed
            soft_labels = (1 - self.label_smoothing) * soft_labels + \
                          self.label_smoothing / logits.size(-1)

            log_probs = F.log_softmax(mixed_logits, dim=-1)
            ce = -(soft_labels * log_probs).sum(dim=-1)
            # FIX C5: focal pt from MIXED probs, not original probs
            mixed_probs = F.softmax(mixed_logits, dim=-1)
            pt = (soft_labels * mixed_probs).sum(dim=-1).clamp(min=1e-8)
            focal = (1.0 - pt) ** self.focal_gamma
            # FIX C5: apply class weights in mixup branch too
            if cw is not None:
                sample_cw = cw[labels]
                per_sample = focal * ce * sample_cw
            else:
                per_sample = focal * ce
        else:
            probs = F.softmax(logits, dim=-1)
            pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
            focal = (1.0 - pt) ** self.focal_gamma
            ce = F.cross_entropy(logits, labels, weight=cw,
                                  label_smoothing=self.label_smoothing, reduction="none")
            per_sample = focal * ce

        if sample_weights is not None:
            per_sample = per_sample * sample_weights.to(logits.device)
        loss = per_sample.mean()

        if self.adversarial and is_training:
            try:
                adv_loss = self._adversarial_loss(model, inputs, labels, sample_weights)
                loss = loss + H.ADVERSARIAL_ALPHA * adv_loss
            except AttributeError:
                pass

        return (loss, outputs) if return_outputs else loss

    def _adversarial_loss(self, model, inputs, labels, sample_weights):
        """PGD adversarial perturbation on input_ids embeddings (Miyato et al. 2017)."""
        try:
            embed_layer = model.get_input_embeddings()
            input_ids = inputs["input_ids"]
            embeds = embed_layer(input_ids)
            embeds = embeds.detach().requires_grad_(True)
            cw = self._cw(embeds.device)

            with torch.enable_grad():
                inputs_adv = {k: v for k, v in inputs.items() if k != "input_ids"}
                inputs_adv["inputs_embeds"] = embeds
                outputs_adv = model(**inputs_adv)
                logits_adv = outputs_adv.logits
                probs_adv = F.softmax(logits_adv, dim=-1)
                pt_adv = probs_adv.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
                focal_adv = (1.0 - pt_adv) ** self.focal_gamma
                ce_adv = F.cross_entropy(logits_adv, labels, weight=cw,
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
            ce_pert = F.cross_entropy(logits_pert, labels, weight=cw,
                                       label_smoothing=self.label_smoothing, reduction="none")
            per_sample_pert = focal_pert * ce_pert
            if sample_weights is not None:
                per_sample_pert = per_sample_pert * sample_weights.to(logits_pert.device)
            return per_sample_pert.mean()
        except Exception:
            return torch.tensor(0.0, device=labels.device)


# ============================================================================
# SECTION 5: SWA CALLBACK
# ============================================================================
class SWACallback(TrainerCallback):
    """Averages LoRA weights from SWA_START_EPOCH for flatter optimum."""

    def __init__(self, start_epoch=5, anneal_epochs=3):
        self.start_epoch = start_epoch
        self.anneal_epochs = anneal_epochs
        self.swa_weights = None
        self.n_averaged = 0

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        current_epoch = int(state.epoch)
        if current_epoch < self.start_epoch:
            return
        lora_params = {n: p.detach().clone() for n, p in model.named_parameters()
                       if "lora_" in n.lower()}
        if self.swa_weights is None:
            self.swa_weights = lora_params
            self.n_averaged = 1
        else:
            for n in lora_params:
                self.swa_weights[n] = (
                    self.swa_weights[n] * self.n_averaged + lora_params[n]
                ) / (self.n_averaged + 1)
            self.n_averaged += 1
        logger.info(f"  [SWA] Epoch {current_epoch}: averaged {self.n_averaged} snapshots")

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if self.swa_weights is None or self.n_averaged < 2:
            return
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.swa_weights:
                    p.copy_(self.swa_weights[n])
        logger.info(f"  [SWA] Applied {self.n_averaged}-epoch weight average")


# ============================================================================
# SECTION 6: TEMPERATURE CALIBRATION
# ============================================================================
def calibrate_temperature(model, val_ds, tokenizer, device=None):
    """Fit scalar temperature T on val set via LBFGS (Guo et al. 2017).

    FIX C4: no longer decorated with @torch.no_grad() — that disabled the
    autograd graph for the LBFGS closure (loss.backward crashed). Now only
    the inference loop is wrapped in no_grad.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    logits_all, labels_all = [], []
    with torch.no_grad():
        for i in range(len(val_ds)):
            item = {k: v.unsqueeze(0).to(device) for k, v in val_ds[i].items()}
            labels_all.append(int(item.pop("labels").item()))
            out = model(**item)
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


# ============================================================================
# SECTION 7: K-FOLD + SINGLE FOLD TRAINING
# ============================================================================
def run_kfold(task, all_rows, label2id, id2label, k=H.K_FOLD_N):
    """Run entity-aware K-fold CV."""
    print(f"\n{'=' * 70}\nK-FOLD CV (entity-aware, k={k})\n{'=' * 70}")
    labels_array = np.array([label2id[r["label"]] for r in all_rows])
    groups = [r.get("entity", r.get("entity_name", "unknown")) for r in all_rows]
    n_entities = len(set(groups))
    print(f"  Rows: {len(all_rows)}, Unique entities: {n_entities}")

    if n_entities >= k * 3:
        gkf = GroupKFold(n_splits=k)
        splits = gkf.split(np.zeros(len(all_rows)), labels_array, groups)
    else:
        print("  Warning: too few entities for GroupKFold, using StratifiedKFold")
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=H.SEED)
        splits = skf.split(np.zeros(len(all_rows)), labels_array)

    cfg = TASK_CFG[task]
    fold_results = []
    import gc
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Fold {fold + 1}/{k} ---")
        train_rows = [all_rows[i] for i in train_idx]
        val_rows = [all_rows[i] for i in val_idx]
        if H.OVERSAMPLING_ENABLED and cfg.get("oversample"):
            print(f"  Oversampling train: {len(train_rows)} -> ", end="")
            train_rows = oversample_minority(train_rows, H.OVERSAMPLING_TARGETS, seed=H.SEED + fold)
            print(f"{len(train_rows)}")
        print(f"  train: {len(train_rows)} | val: {len(val_rows)}")
        metrics = train_single_fold(task, train_rows, val_rows, label2id, id2label,
                                     out_suffix=f"_fold{fold + 1}")
        metrics["fold"] = fold + 1
        fold_results.append(metrics)
        print(f"  metrics: {metrics}")
        # FIX H2: clear between folds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'=' * 70}\nK-FOLD RESULTS (k={k})\n{'=' * 70}")
    avg_metrics = {}
    for key in fold_results[0]:
        if isinstance(fold_results[0][key], (int, float)):
            values = [r[key] for r in fold_results]
            avg_metrics[key] = {"mean": float(np.mean(values)),
                                "std": float(np.std(values)), "values": values}
            print(f"  {key:20s}: {avg_metrics[key]['mean']:.4f} ± {avg_metrics[key]['std']:.4f}")

    def _flat(mk):
        a = avg_metrics.get(mk, {"mean": 0.0, "std": 0.0})
        return a["mean"], a["std"]

    ma, sa = _flat("accuracy")
    mf, sf = _flat("macro_f1")
    mw, sw = _flat("weighted_f1")
    return {
        "k": k, "task": task,
        "fold_results": fold_results,
        "folds": fold_results,
        "mean_accuracy": ma, "std_accuracy": sa,
        "mean_macro_f1": mf, "std_macro_f1": sf,
        "mean_weighted_f1": mw, "std_weighted_f1": sw,
        "aggregate": avg_metrics,
    }


def train_single_fold(task, train_rows, val_rows, label2id, id2label, out_suffix=""):
    cfg = TASK_CFG[task]
    out_dir = resolve_out_dir(cfg)  # FIX BUG#2: absolute path via script dir

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

    train_label_ids = [label2id[r["label"]] for r in train_rows]
    cw = class_weights_from_freq(train_label_ids, len(cfg["labels"]))

    train_ds = PairDataset(train_rows, tok, label2id)
    val_ds = PairDataset(val_rows, tok, label2id)

    # v4.1: Adaptive GPU VRAM scaling — maximize batch size for available VRAM
    auto_batch, auto_seq, auto_adv, auto_accum = auto_scale_gpu_config()
    effective_batch = auto_batch * auto_accum
    steps_per_epoch = max(1, len(train_ds) // effective_batch)
    warmup_steps = int(H.WARMUP_RATIO * steps_per_epoch * H.NUM_EPOCHS)

    train_args_dict = dict(
        output_dir=str(out_dir),
        num_train_epochs=H.NUM_EPOCHS,
        per_device_train_batch_size=auto_batch,       # v4.1: adaptive
        per_device_eval_batch_size=auto_batch * 2,     # v4.1: eval 2x (no backward)
        dataloader_pin_memory=torch.cuda.is_available(),
        gradient_checkpointing=False,
        gradient_accumulation_steps=auto_accum,        # v4.1: adaptive
        learning_rate=H.LEARNING_RATE,
        weight_decay=H.WEIGHT_DECAY,
        adam_beta1=H.ADAM_BETA1, adam_beta2=H.ADAM_BETA2, adam_epsilon=H.ADAM_EPSILON,
        max_grad_norm=H.MAX_GRAD_NORM,
        warmup_steps=warmup_steps,
        lr_scheduler_type=H.SCHEDULER,
        lr_scheduler_kwargs={"num_cycles": H.SCHEDULER_NUM_CYCLES}
            if H.SCHEDULER == "cosine_with_restarts" else None,
        fp16=H.FP16 and torch.cuda.is_available(),    # v4.1: only on CUDA
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        seed=H.SEED,
        report_to="none",
        logging_steps=max(1, steps_per_epoch // 4),
    )
    try:
        train_args_dict["eval_strategy"] = "epoch"
        targs = TrainingArguments(**train_args_dict)
    except TypeError:
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
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tok, compute_metrics=compute_metrics,
        class_weights=cw, focal_gamma=H.FOCAL_GAMMA,
        adversarial=auto_adv,  # v4.1: auto-scaled
        mixup=H.MIXUP_ENABLED,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=H.EARLY_STOP_PATIENCE),
        ] + ([SWACallback(start_epoch=H.SWA_START_EPOCH, anneal_epochs=H.SWA_ANNEAL_EPOCHS)]
             if H.SWA_ENABLED else []),
    )

    trainer.train()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    val_metrics = trainer.evaluate(val_ds, metric_key_prefix="val")
    T = calibrate_temperature(model, val_ds, tok)

    # FIX C7: save for both single-split and K-fold
    if out_suffix:
        fold_dir = out_dir / f"fold_{out_suffix.replace('_fold', '')}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_dir = fold_dir
    else:
        save_dir = out_dir
    model.save_pretrained(save_dir / "lora")
    tok.save_pretrained(save_dir / "tokenizer")
    payload = {
        "task": task,
        "fold": int(out_suffix.replace("_fold", "")) if out_suffix else None,
        "val_metrics": val_metrics,
        "temperature": T,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "class_weights": dict(zip(cfg["labels"], cw.tolist())),
    }
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved LoRA adapter + metrics -> {save_dir}")

    return {
        "accuracy": float(val_metrics.get("val_accuracy", 0)),
        "macro_f1": float(val_metrics.get("val_macro_f1", 0)),
        "weighted_f1": float(val_metrics.get("val_weighted_f1", 0)),
        "temperature": T,
        "saved_to": str(save_dir),
    }


# ============================================================================
# SECTION 8: EVALUATION
# ============================================================================
@torch.no_grad()
def score_all_calibrated(model, tok, rows, labels, T, device=None, max_len=H.MAX_SEQ_LENGTH):
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
    if taus is None:
        taus = np.arange(0.30, 0.98, 0.03)
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    out = []
    for tau in taus:
        keep = confs >= tau
        n_kept = int(keep.sum())
        if n_kept == 0:
            continue
        kept_acc = accuracy_score(golds[keep], preds[keep])
        kept_f1 = f1_score(golds[keep], preds[keep], average="macro",
                           labels=list(range(probs.shape[1])), zero_division=0)
        out.append({
            "tau": round(float(tau), 3),
            "kept_accuracy": round(float(kept_acc), 4),
            "coverage": round(n_kept / len(golds), 4),
            "kept_macro_f1": round(float(kept_f1), 4),
            "n_kept": n_kept,
            "n_deferred": int((~keep).sum()),
        })
    return out


def cmd_finetune(args):
    task = args.task
    cfg = TASK_CFG[task]
    # Apply CLI overrides for data-dir / out-dir before anything else
    if getattr(args, "data_dir", None):
        cfg["data_dir"] = args.data_dir
    if getattr(args, "out_dir", None):
        cfg["out_dir"] = args.out_dir
    # v4.1: Apply GPU overrides (mutates H namespace so auto_scale picks them up)
    if getattr(args, "batch_size", None) is not None:
        H.BATCH_SIZE = args.batch_size
    if getattr(args, "max_seq_length", None) is not None:
        H.MAX_SEQ_LENGTH = args.max_seq_length
    if getattr(args, "grad_accum", None) is not None:
        H.GRAD_ACCUM_STEPS = args.grad_accum
    if getattr(args, "no_adversarial", False):
        H.ADVERSARIAL_ENABLED = False
    if getattr(args, "no_auto_scale", False):
        os.environ["DISABLE_AUTO_SCALE"] = "1"
    print(f"\n{'=' * 70}\nFINETUNE v4 TASK: {task}\nbase: {cfg['base_model']}\n"
          f"out:  {resolve_out_dir(cfg)}\n{'=' * 70}\n")

    data_path = resolve_data_dir(cfg) / (args.dataset or cfg["data_file"])
    # FIX BUG#1: absolute path via __file__, not cwd
    if not data_path.exists():
        print(f"ERROR: dataset not found at {data_path}")
        print(f"  Place dataset_gold_standard_final.jsonl in: {resolve_data_dir(cfg)}")
        sys.exit(1)
    all_rows = load_jsonl(str(data_path))
    print(f"Loaded {len(all_rows)} rows from {data_path}")

    label2id = {l: i for i, l in enumerate(cfg["labels"])}
    id2label = {i: l for l, i in label2id.items()}
    rows = normalize_rows(all_rows, cfg, label2id)

    if args.kfold > 1:
        results = run_kfold(task, rows, label2id, id2label, k=args.kfold)
        out_dir = resolve_out_dir(cfg)  # FIX BUG#2
        with open(out_dir / "kfold_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nK-fold results saved -> {out_dir / 'kfold_results.json'}")
    else:
        train_rows, val_rows, test_rows = stratified_split(rows, "label")
        if H.OVERSAMPLING_ENABLED and cfg.get("oversample"):
            print(f"Oversampling train: {len(train_rows)} -> ", end="")
            train_rows = oversample_minority(train_rows, H.OVERSAMPLING_TARGETS, seed=H.SEED)
            print(f"{len(train_rows)}")
        print(f"Split: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
        metrics = train_single_fold(task, train_rows, val_rows, label2id, id2label)
        print(f"\nFinal metrics: {metrics}")


def cmd_evaluate(args):
    task = args.task
    cfg = TASK_CFG[task]
    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.json"
    metrics = json.load(open(metrics_path)) if metrics_path.exists() else {}
    T = metrics.get("temperature", 1.0)
    print(f"Task: {task} | run_dir: {run_dir} | temperature: {T}")

    base = cfg["base_model"]
    tok = AutoTokenizer.from_pretrained(run_dir / "tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained(base)
    model = PeftModel.from_pretrained(model, run_dir / "lora")
    model = model.merge_and_unload()

    all_rows = load_jsonl(str(resolve_data_dir(cfg) / cfg["data_file"]))
    label2id = {l: i for i, l in enumerate(cfg["labels"])}
    rows = normalize_rows(all_rows, cfg, label2id)
    _, _, test = stratified_split(rows)
    print(f"Test set: {len(test)} rows | balance: {dict(Counter(r['label'] for r in test))}")

    probs, golds = score_all_calibrated(model, tok, test, cfg["labels"], T)
    preds = probs.argmax(axis=1)
    full_acc = accuracy_score(golds, preds)
    full_f1 = f1_score(golds, preds, average="macro",
                       labels=list(range(len(cfg["labels"]))), zero_division=0)
    cm = confusion_matrix(golds, preds, labels=list(range(len(cfg["labels"]))))
    print("\n=== FULL-COVERAGE METRICS (no deferral) ===")
    print(f"  accuracy : {full_acc:.4f}")
    print(f"  macro-F1 : {full_f1:.4f}")
    print(f"  confusion matrix (labels={cfg['labels']}): {cm.tolist()}")
    print(classification_report(golds, preds, target_names=cfg["labels"], zero_division=0))

    sweep = confidence_threshold_sweep(probs, golds)
    print("\n=== CONFIDENCE-THRESHOLD SWEEP ===")
    print(f"  {'tau':>5} {'kept_acc':>9} {'coverage':>9} {'kept_F1':>8} {'kept':>5} {'defer':>6}")
    for s in sweep:
        flag = "  <-- 97% target" if s["kept_accuracy"] >= 0.97 else ""
        print(f"  {s['tau']:>5} {s['kept_accuracy']:>9.4f} {s['coverage']:>9.4f} "
              f"{s['kept_macro_f1']:>8.4f} {s['n_kept']:>5} {s['n_deferred']:>6}{flag}")
    hits_97 = [s for s in sweep if s["kept_accuracy"] >= 0.97]
    best = max(hits_97, key=lambda s: s["coverage"]) if hits_97 else None
    if best:
        print(f"\n>> >=97% kept-accuracy at tau={best['tau']} "
              f"coverage={best['coverage']:.1%} ({best['n_kept']}/{len(test)} kept)")
    else:
        print(f"\n>> 97% NOT reached. Max kept-acc={max(s['kept_accuracy'] for s in sweep):.4f}")

    out = {
        "task": task, "temperature": T,
        "full_coverage": {"accuracy": full_acc, "macro_f1": full_f1,
                          "confusion_matrix": cm.tolist(), "labels": cfg["labels"]},
        "sweep": sweep, "best_97": best,
    }
    with open(run_dir / "evaluation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {run_dir / 'evaluation.json'}")


def cmd_kfold_summary(args):
    kfold = json.load(open(args.kfold_results))
    k = kfold.get("k", "?")
    print(f"\n{'=' * 60}\nK-FOLD RESULTS (k={k})\n{'=' * 60}")
    ma = kfold.get("mean_accuracy")
    sa = kfold.get("std_accuracy")
    mf = kfold.get("mean_macro_f1")
    sf = kfold.get("std_macro_f1")
    if ma is None:
        agg = kfold.get("aggregate", {})
        a_acc = agg.get("accuracy", {})
        a_f1 = agg.get("macro_f1", {})
        ma = a_acc.get("mean", 0.0); sa = a_acc.get("std", 0.0)
        mf = a_f1.get("mean", 0.0); sf = a_f1.get("std", 0.0)
    print(f"Mean Accuracy:  {ma:.4f} ± {sa:.4f}")
    print(f"Mean Macro-F1:  {mf:.4f} ± {sf:.4f}")
    fr = kfold.get("fold_results", kfold.get("folds", []))
    print("\nPer-fold:")
    for r in fr:
        print(f"  Fold {r.get('fold', '?')}: acc={r['accuracy']:.4f}, "
              f"f1={r['macro_f1']:.4f}, T={r.get('temperature', 0):.3f}")
    if fr:
        best = max(fr, key=lambda x: x.get("macro_f1", 0))
        print(f"\n  Best fold: {best.get('fold', '?')} (f1={best['macro_f1']:.4f}, "
              f"saved_to={best.get('saved_to', 'n/a')})")


# ============================================================================
# SECTION 9: SINGLE MAIN ENTRY POINT
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="v4 all-in-one: finetune + evaluate")
    sub = ap.add_subparsers(dest="command", required=True)

    p_ft = sub.add_parser("finetune", help="Run fine-tuning")
    p_ft.add_argument("--task", choices=["relevancy", "sentiment"], required=True)
    p_ft.add_argument("--dataset", default=None, help="Override dataset filename")
    p_ft.add_argument("--data-dir", default=None,
                     help="Override dataset directory (absolute or relative to script)")
    p_ft.add_argument("--kfold", type=int, default=0, help="0=disabled, 5=recommended")
    p_ft.add_argument("--out-dir", default=None,
                     help="Override output directory (absolute or relative to script)")
    # v4.1: GPU VRAM auto-scale overrides
    p_ft.add_argument("--batch-size", type=int, default=None,
                     help="Override auto batch-size (default: auto-scale to VRAM)")
    p_ft.add_argument("--max-seq-length", type=int, default=None,
                     help="Override max sequence length (default: 256, or 320/384 on big GPUs)")
    p_ft.add_argument("--grad-accum", type=int, default=None,
                     help="Override gradient accumulation steps (default: auto)")
    p_ft.add_argument("--no-adversarial", action="store_true",
                     help="Force disable adversarial training (auto-disabled < 12GB VRAM)")
    p_ft.add_argument("--no-auto-scale", action="store_true",
                     help="Disable VRAM auto-scaling, use hyperparams defaults")

    p_ev = sub.add_parser("evaluate", help="Evaluate a single-fold model")
    p_ev.add_argument("--task", choices=["relevancy", "sentiment"], required=True)
    p_ev.add_argument("--run-dir", required=True)

    p_kf = sub.add_parser("kfold-summary", help="Summarize kfold_results.json")
    p_kf.add_argument("--kfold-results", required=True)

    args = ap.parse_args()
    if args.command == "finetune":
        cmd_finetune(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "kfold-summary":
        cmd_kfold_summary(args)


if __name__ == "__main__":
    main()
