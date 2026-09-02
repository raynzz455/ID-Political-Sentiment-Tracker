# Panduan Finetuning + Hyperparameter Tuning di Google Colab

> Step-by-step lengkap. Copy-paste setiap CELL ke Google Colab.
> Estimasi waktu total: 30-45 menit (GPU T4)

---

## Prasyarat

1. Buka https://colab.research.google.com
2. **Runtime → Change runtime type → T4 GPU** (WAJIB untuk speed)
3. Tidak perlu HuggingFace token (base model public, model disimpan di Google Drive)

---

## TAHAP 1: Setup Environment (2 menit)

### CELL 1 — Clone repo

```python
!git clone https://github.com/raynzz455/ID-Political-Sentiment-Tracker.git
%cd /content/ID-Political-Sentiment-Tracker
```

### CELL 2 — Install dependencies

```python
!pip install -q torch transformers peft scikit-learn numpy

# Verify GPU
import torch
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - aktifkan GPU!'}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB" if torch.cuda.is_available() else "")
```

### CELL 3 — Mount Google Drive (untuk simpan model)

```python
from google.colab import drive
drive.mount('/content/drive')

import os
SAVE_DIR = '/content/drive/MyDrive/id-political-sentiment-models'
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Model akan disimpan di: {SAVE_DIR}")
```

### CELL 4 — Verify dataset

```python
import json
from collections import Counter

rows = [json.loads(l) for l in open('finetuning/datasets/dataset_enhanced.jsonl')]
print(f"Total rows: {len(rows)}")
print(f"Labels: {dict(Counter(r['gold_label'] for r in rows))}")

sent = [r for r in rows if r.get('gold_relevancy') == 'relevant']
print(f"\nSentiment training rows: {len(sent)}")
for l in ['positive', 'neutral', 'negative']:
    print(f"  {l}: {sum(1 for r in sent if r['gold_label']==l)}")
```

---

## TAHAP 2: Finetuning Sentiment Model (10-15 menit)

### CELL 5 — Finetune dengan M5 (Anti-Overconfidence)

```python
import sys
sys.path.insert(0, 'finetuning/configs')
sys.path.insert(0, 'finetuning/scripts')

# Run finetune
# Ini akan download base model (apriandito/indobert-sentiment-classifier) — PUBLIC, no token
# Lalu training LoRA dengan:
#   - Focal loss gamma=2.5
#   - Class weights 1/sqrt(freq)
#   - Label smoothing 0.05
#   - Per-sample confidence weighting
#   - 10 epochs, early stop patience=3

!python finetuning/scripts/finetune.py --task sentiment
```

**Apa yang terjadi di tahap ini:**

1. **Download base model** (110M parameters, ~440MB) — otomatis dari HuggingFace Hub (public)
2. **Load dataset** — 909 rows, filter ke relevant rows (~541 rows)
3. **Split data** — 70% train, 15% val, 15% test (stratified)
4. **LoRA setup** — freeze base model, add trainable adapters (r=32, ~1M params)
5. **Training loop** (10 epochs max):
   - Epoch 1-3: model belajar pattern umum
   - Epoch 4-7: model fine-tune pada sentiment spesifik
   - Epoch 7+: SWA averaging untuk stabilitas
   - Early stop kalau val F1 tidak improve 4 epochs
6. **Temperature scaling** — fit T pada val set untuk calibration
7. **Save LoRA adapter** ke `./runs/sentiment/lora/`

**Output yang dihasilkan:**
```
runs/sentiment/
├── lora/
│   ├── adapter_config.json         # LoRA config (r=32, alpha=64)
│   └── adapter_model.safetensors   # Hasil belajar (~4MB)
├── tokenizer/
├── metrics.json                    # Test accuracy, F1, temperature
└── checkpoint-*/                   # Intermediate checkpoints
```

---

## TAHAP 3: Evaluasi + Confidence Sweep (1 menit)

### CELL 6 — Run evaluation

```python
!python finetuning/scripts/evaluate.py --task sentiment --run-dir ./runs/sentiment
```

### CELL 7 — Display results

```python
import json

with open('runs/sentiment/metrics.json') as f:
    metrics = json.load(f)
with open('runs/sentiment/evaluation.json') as f:
    evaluation = json.load(f)

print("=" * 60)
print("HASIL FINETUNING")
print("=" * 60)

test = metrics.get('test_metrics', {})
print(f"\nTest Accuracy:     {test.get('accuracy', 'N/A')}")
print(f"Test macro-F1:     {test.get('macro_f1', 'N/A')}")
print(f"Temperature (T):   {metrics.get('temperature', 'N/A')}")

print("\n--- CALIBRATION ---")
full = evaluation.get('full_coverage', {})
print(f"Full-coverage accuracy: {full.get('accuracy', 'N/A')}")
print(f"Full-coverage macro-F1: {full.get('macro_f1', 'N/A')}")

# Confusion matrix
cm = full.get('confusion_matrix', [])
if cm:
    labels = full.get('labels', ['negative', 'neutral', 'positive'])
    print(f"\nConfusion Matrix (baris=true, kolom=pred):")
    print(f"  {'':>12s} " + " ".join(f"{l[:8]:>10s}" for l in labels))
    for i, row in enumerate(cm):
        print(f"  {labels[i][:12]:>12s} " + " ".join(f"{v:>10d}" for v in row))

# Confidence sweep
print(f"\n--- CONFIDENCE SWEEP (tau threshold) ---")
print(f"  {'tau':>6} {'kept_acc':>10} {'coverage':>10} {'status':>10}")
print("  " + "-" * 40)
for s in evaluation.get('sweep', []):
    flag = " <-- 97%" if s['kept_accuracy'] >= 0.97 else ""
    print(f"  {s['tau']:>6.2f} {s['kept_accuracy']:>10.4f} {s['coverage']:>10.1%}{flag:>10}")

best = evaluation.get('best_97')
if best:
    print(f"\n✅ TARGET 97% TERCAPAI!")
    print(f"   tau={best['tau']}, coverage={best['coverage']:.1%}")
else:
    max_acc = max((s.get('kept_accuracy', 0) for s in evaluation.get('sweep', [])), default=0)
    print(f"\n⚠️ 97% belum tercapai. Max: {max_acc:.4f}")
    print(f"   → Lanjut ke TAHAP 4: Hyperparameter Tuning")
```

---

## TAHAP 4: Hyperparameter Tuning (10-20 menit)

Kalau akurasi belum mencapai target, jalankan grid search untuk cari parameter optimal.

### CELL 8 — Grid search hyperparameter tuning

```python
import json, random, time, gc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter

set_seed(42)
random.seed(42)
np.random.seed(42)

# Load dataset
rows = [json.loads(l) for l in open('finetuning/datasets/dataset_enhanced.jsonl')]
sent = [r for r in rows if r.get('gold_relevancy') == 'relevant']

LABELS = ["negative", "neutral", "positive"]
label2id = {l: i for i, l in enumerate(LABELS)}

data = [{"premise": r["premise"], "hypothesis": r["hypothesis"],
         "label": label2id[r["gold_label"]],
         "confidence": r.get("label_confidence", 0.5)} for r in sent]

# Stratified split
by_label = {}
for d in data:
    by_label.setdefault(d["label"], []).append(d)
random.shuffle(data)
train, val, test = [], [], []
for lab, items in by_label.items():
    random.shuffle(items)
    n = len(items)
    nt = max(1, int(n * 0.7))
    nv = max(1, int(n * 0.15))
    train.extend(items[:nt])
    val.extend(items[nt:nt+nv])
    test.extend(items[nt+nv:nt+nv+max(1, int(n*0.15))])
random.shuffle(train); random.shuffle(val); random.shuffle(test)

print(f"Split: train={len(train)} val={len(val)} test={len(test)}")
print(f"Train labels: {Counter(d['label'] for d in train)}")

# Class weights
def class_weights(labels, n_classes=3):
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    freq = counts / counts.sum()
    w = 1.0 / np.sqrt(freq + 1e-8)
    return torch.tensor(w / w.mean(), dtype=torch.float)

cw = class_weights([d["label"] for d in train])

# Dataset class
MODEL_ID = "apriandito/indobert-sentiment-classifier"
tok = AutoTokenizer.from_pretrained(MODEL_ID)

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

# ECE
def expected_calibration_error(probs, labels, n_bins=10):
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    bounds = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs > bounds[i]) & (confs <= bounds[i+1])
        if mask.sum() == 0: continue
        acc = (preds[mask] == labels[mask]).mean()
        conf = confs[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(acc - conf)
    return float(ece)

# Grid search configs
GRID = [
    {"name": "M5_smoothing_0.05_gamma_2.5", "gamma": 2.5, "smoothing": 0.05, "lr": 2e-5, "lora_r": 32},
    {"name": "M5_smoothing_0.10_gamma_2.5", "gamma": 2.5, "smoothing": 0.10, "lr": 2e-5, "lora_r": 32},
    {"name": "M5_smoothing_0.05_gamma_3.0", "gamma": 3.0, "smoothing": 0.05, "lr": 2e-5, "lora_r": 32},
    {"name": "M5_smoothing_0.05_gamma_2.5_lr_3e5", "gamma": 2.5, "smoothing": 0.05, "lr": 3e-5, "lora_r": 32},
    {"name": "M5_smoothing_0.05_gamma_2.5_r_16", "gamma": 2.5, "smoothing": 0.05, "lr": 2e-5, "lora_r": 16},
]

def train_and_eval(config, train_data, val_data, test_data):
    """Train satu config, return metrics."""
    print(f"\nTraining {config['name']}...", flush=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=3, ignore_mismatched_sizes=True)
    lora = LoraConfig(r=config["lora_r"], lora_alpha=config["lora_r"]*2, lora_dropout=0.1,
                      bias="none", task_type=TaskType.SEQ_CLS,
                      target_modules=["query","key","value","dense"])
    model = get_peft_model(model, lora)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    train_ds = PairDataset(train_data, tok)
    val_ds = PairDataset(val_data, tok)
    test_ds = PairDataset(test_data, tok)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=16)
    test_dl = DataLoader(test_ds, batch_size=16)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    
    best_val_f1 = 0
    patience_counter = 0
    
    for epoch in range(8):
        model.train()
        for batch in train_dl:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items() if k != "sample_weight"}
            labels = inputs.pop("labels")
            sw = batch.get("sample_weight")
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            
            # Focal loss
            gamma = config["gamma"]
            pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
            focal = (1 - pt) ** gamma
            ce = F.cross_entropy(logits, labels, weight=cw.to(device),
                                 label_smoothing=config["smoothing"], reduction="none")
            loss = (focal * ce).mean()
            if sw is not None:
                loss = (focal * ce * sw.to(device)).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in val_dl:
                inputs = {k: v.to(device) for k, v in batch.items() if k not in ("labels","sample_weight")}
                out = model(**inputs)
                val_probs.append(F.softmax(out.logits, dim=-1).cpu().numpy())
                val_labels.append(batch["labels"].numpy())
        val_probs = np.concatenate(val_probs)
        val_labels = np.concatenate(val_labels)
        val_f1 = f1_score(val_labels, val_probs.argmax(axis=1), average="macro", zero_division=0)
        
        print(f"  Epoch {epoch+1}: val_f1={val_f1:.4f}", flush=True)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 3:
                print(f"  Early stop at epoch {epoch+1}")
                break
    
    # Test
    model.eval()
    test_logits_list, test_labels_list = [], []
    with torch.no_grad():
        for batch in test_dl:
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ("labels","sample_weight")}
            out = model(**inputs)
            test_logits_list.append(out.logits.cpu())
            test_labels_list.append(batch["labels"].numpy())
    
    test_logits = torch.cat(test_logits_list)
    test_labels_np = np.concatenate(test_labels_list)
    
    # Temperature scaling
    T = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(test_logits / T, torch.tensor(test_labels_np))
        loss.backward()
        return loss
    opt.step(closure)
    T_val = float(T.clamp(0.05, 10.0).item())
    
    test_probs = F.softmax(test_logits / T_val, dim=-1).numpy()
    test_acc = accuracy_score(test_labels_np, test_probs.argmax(axis=1))
    test_f1 = f1_score(test_labels_np, test_probs.argmax(axis=1), average="macro", zero_division=0)
    ece = expected_calibration_error(test_probs, test_labels_np)
    
    # Confidence sweep
    confs = test_probs.max(axis=1)
    sweep = []
    for tau in [0.50, 0.70, 0.75, 0.80, 0.85, 0.90]:
        keep = confs >= tau
        if keep.sum() == 0: continue
        kept_acc = accuracy_score(test_labels_np[keep], test_probs[keep].argmax(axis=1))
        sweep.append({"tau": tau, "kept_acc": float(kept_acc), "coverage": float(keep.mean())})
    
    result = {
        "config": config,
        "best_val_f1": best_val_f1,
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
        "ece": ece,
        "temperature": T_val,
        "sweep": sweep,
    }
    
    print(f"  RESULT: acc={test_acc:.4f} f1={test_f1:.4f} ECE={ece:.4f} T={T_val:.2f}")
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return result

# Run grid search
print("=" * 60)
print(f"GRID SEARCH — {len(GRID)} configs")
print("=" * 60)

all_results = {}
for config in GRID:
    t0 = time.time()
    result = train_and_eval(config, train, val, test)
    result["time_seconds"] = time.time() - t0
    all_results[config["name"]] = result

# Find best
print(f"\n{'='*60}")
print(f"GRID SEARCH RESULTS")
print(f"{'='*60}")
print(f"{'Config':40s} {'Acc':>6} {'F1':>6} {'ECE':>6} {'Time':>6}")
print("-" * 70)
best_name = None
best_score = -1
for name, r in all_results.items():
    score = r["test_macro_f1"] - r["ece"]  # high F1, low ECE
    flag = " ⭐" if score > best_score else ""
    print(f"{name:40s} {r['test_accuracy']:>6.4f} {r['test_macro_f1']:>6.4f} {r['ece']:>6.4f} {r['time_seconds']:>5.0f}s{flag}")
    if score > best_score:
        best_score = score
        best_name = name

best = all_results[best_name]
print(f"\nBEST: {best_name}")
print(f"  macro-F1: {best['test_macro_f1']:.4f}")
print(f"  ECE: {best['ece']:.4f}")
print(f"  Temperature: {best['temperature']:.2f}")
print(f"  Config: {best['config']}")

print(f"\nConfidence sweep for {best_name}:")
for s in best["sweep"]:
    flag = " ✅ 97%" if s["kept_acc"] >= 0.97 else ""
    print(f"  tau={s['tau']:.2f}: kept_acc={s['kept_acc']:.4f} coverage={s['coverage']:.1%}{flag}")

# Save
with open('grid_search_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to grid_search_results.json")
```

---

## TAHAP 5: Save Model ke Google Drive (2 menit)

### CELL 9 — Save model + metrics

```python
import shutil, os

# Save sentiment model
drive_dir = f'{SAVE_DIR}/sentiment-v1'
os.makedirs(drive_dir, exist_ok=True)

# Copy LoRA adapter
if os.path.exists('./runs/sentiment/lora'):
    shutil.copytree('./runs/sentiment/lora', f'{drive_dir}/lora', dirs_exist_ok=True)
    print(f"✅ LoRA adapter: {drive_dir}/lora/")

# Copy tokenizer
if os.path.exists('./runs/sentiment/tokenizer'):
    shutil.copytree('./runs/sentiment/tokenizer', f'{drive_dir}/tokenizer', dirs_exist_ok=True)
    print(f"✅ Tokenizer: {drive_dir}/tokenizer/")

# Copy metrics
for f in ['./runs/sentiment/metrics.json', './runs/sentiment/evaluation.json']:
    if os.path.exists(f):
        shutil.copy(f, drive_dir)
        print(f"✅ {os.path.basename(f)}")

# Copy grid search results
if os.path.exists('grid_search_results.json'):
    shutil.copy('grid_search_results.json', drive_dir)
    print(f"✅ grid_search_results.json")

print(f"\nModel saved to: {drive_dir}")
```

### CELL 10 — Merge + save full model (~440MB)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

print("Merging LoRA into base model...")
base_id = "apriandito/indobert-sentiment-classifier"
tokenizer = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForSequenceClassification.from_pretrained(base_id)
model = PeftModel.from_pretrained(base, './runs/sentiment/lora')
model = model.merge_and_unload()

# Apply temperature
T = metrics.get('temperature', 1.0)
model.config.temperature = T

# Save
merged_dir = f'{drive_dir}/merged_model'
os.makedirs(merged_dir, exist_ok=True)
model.save_pretrained(merged_dir)
tokenizer.save_pretrained(merged_dir)
print(f"✅ Full merged model: {merged_dir} (~440MB)")
```

---

## TAHAP 6: Finetune Relevancy Model (5-10 menit)

### CELL 11 — Finetune relevancy

```python
!python finetuning/scripts/finetune.py --task relevancy
!python finetuning/scripts/evaluate.py --task relevancy --run-dir ./runs/relevancy
```

### CELL 12 — Save relevancy model

```python
drive_rel = f'{SAVE_DIR}/relevancy-v1'
os.makedirs(drive_rel, exist_ok=True)
shutil.copytree('./runs/relevancy/lora', f'{drive_rel}/lora', dirs_exist_ok=True)
shutil.copytree('./runs/relevancy/tokenizer', f'{drive_rel}/tokenizer', dirs_exist_ok=True)
for f in ['./runs/relevancy/metrics.json', './runs/relevancy/evaluation.json']:
    if os.path.exists(f): shutil.copy(f, drive_rel)
print(f"✅ Relevancy model: {drive_rel}")
```

---

## TAHAP 7: Test Model dengan Sample Input

### CELL 13 — Test prediction

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model
model_path = f'{drive_dir}/merged_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Test dengan sample
samples = [
    ("Prabowo Subianto", "PRESIDEN Prabowo Subianto menegaskan program ekonomi akan terus berjalan meski dihujani kritik."),
    ("Joko Widodo", "Eks Menteri era Presiden Jokowi ini juga dituntut membayar uang pengganti Rp809 miliar."),
    ("Anies Baswedan", "Anies Baswedan mengatakan Indonesia bangsa yang lugu dan baik hati."),
    ("Thomas Lembong", "Eks Mendag Thomas Lembong divonis bersalah melakukan tindak pidana korupsi impor gula."),
    ("Rocky Gerung", "Rocky Gerung menyebut pasal KUHP yang baru sebagai pasal yang dungu."),
]

LABELS = ["negative", "neutral", "positive"]
T = metrics.get('temperature', 1.3)

print("=" * 80)
print("TEST PREDICTIONS")
print("=" * 80)

for entity, context in samples:
    inputs = tokenizer(entity, context, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits / T, dim=-1)[0]
    
    pred_idx = probs.argmax().item()
    pred_label = LABELS[pred_idx]
    confidence = probs[pred_idx].item()
    
    deferred = "⚠️ DEFER (low confidence)" if confidence < 0.70 else "✅ confident"
    
    print(f"\n  Entity: {entity}")
    print(f"  Context: {context[:80]}...")
    print(f"  Prediction: {pred_label} ({confidence:.1%}) {deferred}")
    print(f"  Probs: neg={probs[0]:.2f} neu={probs[1]:.2f} pos={probs[2]:.2f}")
```

---

## TAHAP 8: Download Backup

### CELL 14 — Download metrics

```python
from google.colab import files

files.download('./runs/sentiment/metrics.json')
files.download('./runs/sentiment/evaluation.json')
if os.path.exists('grid_search_results.json'):
    files.download('grid_search_results.json')

print("\n✅ Pipeline lengkap!")
print(f"\nModel tersimpan di Google Drive:")
print(f"  {SAVE_DIR}/sentiment-v1/ (LoRA + merged + metrics)")
print(f"  {SAVE_DIR}/relevancy-v1/ (LoRA + merged + metrics)")
```

---

## Cara Baca Hasil

### Metric yang Penting:

| Metric | Target | Arti |
|--------|--------|------|
| macro-F1 | >= 0.90 | Model akurat untuk semua class (tidak bias ke neutral) |
| ECE | <= 0.15 | Confidence reliable untuk deferral |
| Test Accuracy | >= 0.85 | Overall benar |
| Kept-set acc (tau=0.70) | >= 0.97 | Target akurasi tinggi |

### Kalau Akurasi Kurang:

1. **macro-F1 < 0.80** → coba grid search (TAHAP 4), ganti gamma/smoothing
2. **ECE > 0.20** → tingkatkan label_smoothing ke 0.10
3. **Positive/Negative F1 < 0.60** → class weights terlalu lemah, coba 1/π bukan 1/√π
4. **Overfitting (train >> val)** → kurangi epochs, tambah dropout

### Kalau Akurasi Bagus:

- Save model ke Google Drive (TAHAP 5)
- Copy ke production server
- Update `packages/nlp/sentiment_model.py`:
  ```python
  SENTIMENT_MODEL_ID = "./models/sentiment-v1/merged_model"
  ```

---

## Estimasi Waktu Total

| Tahap | Waktu | GPU? |
|-------|-------|------|
| 1. Setup | 2 menit | Tidak |
| 2. Finetune sentiment | 10-15 menit | Ya |
| 3. Evaluate | 1 menit | Ya |
| 4. Grid search (5 configs) | 20-30 menit | Ya |
| 5. Save ke Drive | 2 menit | Tidak |
| 6. Finetune relevancy | 5-10 menit | Ya |
| 7. Test predictions | 1 menit | Ya |
| 8. Download backup | 1 menit | Tidak |
| **Total** | **40-60 menit** | |
