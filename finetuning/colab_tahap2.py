"""
GOOGLE COLAB — TAHAP 2: FINETUNE v2 + EVALUATE + SAVE
======================================================
Copy-paste TIAP CELL ke Google Colab (GPU runtime).
Estimasi waktu: 15-20 menit.

=== CELL 1: Clone + Setup ===
"""
# Cell 1
!git clone https://github.com/raynzz455/ID-Political-Sentiment-Tracker.git
%cd /content/ID-Political-Sentiment-Tracker

# Install deps (clean reinstall to fix torchao/torchvision conflict)
!pip uninstall -y torch torchvision torchaudio torchao torchtext 2>/dev/null
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
!pip install -q torchao transformers peft scikit-learn numpy

import torch, transformers, peft
print(f"torch: {torch.__version__}, transformers: {transformers.__version__}, peft: {peft.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# === CELL 2: Mount Google Drive ===
# Cell 2
from google.colab import drive
drive.mount('/content/drive')

import os
SAVE_DIR = '/content/drive/MyDrive/id-political-sentiment-models'
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Save dir: {SAVE_DIR}")

# === CELL 3: Build dataset v2 (balanced 1:1:1) ===
# Cell 3
!python finetuning/scripts/build_dataset_v2.py

# Verify
import json
from collections import Counter
rows = [json.loads(l) for l in open('finetuning/datasets/dataset_v2.jsonl')]
print(f"\nDataset v2: {len(rows)} rows, balanced: {dict(Counter(r['gold_label'] for r in rows))}")

# === CELL 4: Edit hyperparams to use v2 config (lr=3e-5 from grid search) ===
# Cell 4
# Overwrite hyperparams_optimized.py with v2 params (best from grid search)
!cp finetuning/configs/hyperparams_v2.py finetuning/configs/hyperparams_optimized.py
print("Using v2 hyperparams: lr=3e-5, gamma=2.5, smoothing=0.05, epochs=15")

# === CELL 5: Finetune v2 (GPU ~10-15 min) ===
# Cell 5 — MAIN TRAINING
!python finetuning/scripts/finetune.py --task sentiment

# === CELL 6: Evaluate v2 ===
# Cell 6
!python finetuning/scripts/evaluate.py --task sentiment --run-dir ./runs/sentiment

# === CELL 7: Display results ===
# Cell 7
import json

with open('runs/sentiment/metrics.json') as f:
    metrics = json.load(f)
with open('runs/sentiment/evaluation.json') as f:
    evaluation = json.load(f)

print("=" * 60)
print("HASIL FINETUNING v2")
print("=" * 60)

test = metrics.get('test_metrics', {})
print(f"\nTest Accuracy:     {test.get('accuracy', 'N/A')}")
print(f"Test macro-F1:     {test.get('macro_f1', 'N/A')}")
print(f"Temperature (T):   {metrics.get('temperature', 'N/A')}")

full = evaluation.get('full_coverage', {})
print(f"\nFull-coverage accuracy: {full.get('accuracy', 'N/A')}")
print(f"Full-coverage macro-F1: {full.get('macro_f1', 'N/A')}")

cm = full.get('confusion_matrix', [])
if cm:
    labels = full.get('labels', ['negative', 'neutral', 'positive'])
    print(f"\nConfusion Matrix:")
    print(f"  {'':>12s} " + " ".join(f"{l[:8]:>10s}" for l in labels))
    for i, row in enumerate(cm):
        print(f"  {labels[i][:12]:>12s} " + " ".join(f"{v:>10d}" for v in row))

# Per-class F1
from sklearn.metrics import f1_score
if cm:
    import numpy as np
    cm_np = np.array(cm)
    for i, label in enumerate(labels):
        tp = cm_np[i][i]
        fp = cm_np[:, i].sum() - tp
        fn = cm_np[i, :].sum() - tp
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        print(f"\n  {label}: P={precision:.2f} R={recall:.2f} F1={f1:.2f}")

print(f"\nConfidence Sweep:")
print(f"  {'tau':>6} {'kept_acc':>10} {'coverage':>10}")
for s in evaluation.get('sweep', []):
    flag = " <-- 97%" if s['kept_accuracy'] >= 0.97 else ""
    print(f"  {s['tau']:>6.2f} {s['kept_accuracy']:>10.4f} {s['coverage']:>10.1%}{flag}")

best = evaluation.get('best_97')
if best:
    print(f"\n✅ 97% TARGET: tau={best['tau']} coverage={best['coverage']:.1%}")
else:
    print(f"\n⚠️ 97% belum tercapai. Lihat docs/DATASET_METHOD_TUNING_GUIDE.md untuk tuning.")

# Compare v1 vs v2
print(f"\n{'='*60}")
print(f"COMPARISON v1 vs v2")
print(f"{'='*60}")
print(f"  v1: Accuracy=67.1%, F1=59.6% (lr=2e-5, imbalanced 3.8:1)")
print(f"  v2: Accuracy={test.get('accuracy', '?')}, F1={test.get('macro_f1', '?')} (lr=3e-5, balanced 1:1:1)")

# === CELL 8: Save v2 model to Google Drive ===
# Cell 8
import shutil

drive_v2 = f'{SAVE_DIR}/sentiment-v2'
os.makedirs(drive_v2, exist_ok=True)

if os.path.exists('./runs/sentiment/lora'):
    shutil.copytree('./runs/sentiment/lora', f'{drive_v2}/lora', dirs_exist_ok=True)
    print(f"✅ LoRA: {drive_v2}/lora")

if os.path.exists('./runs/sentiment/tokenizer'):
    shutil.copytree('./runs/sentiment/tokenizer', f'{drive_v2}/tokenizer', dirs_exist_ok=True)
    print(f"✅ Tokenizer: {drive_v2}/tokenizer")

for f_name in ['./runs/sentiment/metrics.json', './runs/sentiment/evaluation.json']:
    if os.path.exists(f_name):
        shutil.copy(f_name, drive_v2)
        print(f"✅ {os.path.basename(f_name)}")

# === CELL 9: Merge + save full model ===
# Cell 9
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

print("Merging LoRA...")
base_id = "apriandito/indobert-sentiment-classifier"
tok = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForSequenceClassification.from_pretrained(base_id)
model = PeftModel.from_pretrained(base, './runs/sentiment/lora')
model = model.merge_and_unload()

T = metrics.get('temperature', 3.837)
model.config.temperature = T

merged_dir = f'{drive_v2}/merged_model'
os.makedirs(merged_dir, exist_ok=True)
model.save_pretrained(merged_dir)
tok.save_pretrained(merged_dir)
print(f"✅ Full model: {merged_dir} (~440MB)")

# === CELL 10: Test predictions ===
# Cell 10
import torch

model_path = merged_dir
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

samples = [
    ("Prabowo Subianto", "PRESIDEN Prabowo Subianto menegaskan program ekonomi akan terus berjalan meski dihujani kritik."),
    ("Joko Widodo", "Eks Menteri era Presiden Jokowi ini juga dituntut membayar uang pengganti Rp809 miliar."),
    ("Thomas Lembong", "Eks Mendag Thomas Lembong divonis bersalah melakukan tindak pidana korupsi impor gula."),
    ("Anies Baswedan", "Anies Baswedan mengatakan Indonesia bangsa yang lugu dan baik hati."),
    ("Rocky Gerung", "Rocky Gerung menyebut pasal KUHP yang baru sebagai pasal yang dungu."),
]

LABELS = ["negative", "neutral", "positive"]
T = metrics.get('temperature', 3.837)

print("=" * 80)
print("TEST PREDICTIONS (v2 model)")
print("=" * 80)

for entity, context in samples:
    inputs = tokenizer(entity, context, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits / T, dim=-1)[0]
    pred = LABELS[probs.argmax()]
    conf = probs.max().item()
    status = "✅ confident" if conf >= 0.70 else "⚠️ DEFER"
    print(f"\n  {entity}: {pred} ({conf:.1%}) {status}")
    print(f"  probs: neg={probs[0]:.2f} neu={probs[1]:.2f} pos={probs[2]:.2f}")

# === CELL 11: Download metrics ===
# Cell 11
from google.colab import files
files.download('./runs/sentiment/metrics.json')
files.download('./runs/sentiment/evaluation.json')

print("\n✅ Tahap 2 selesai!")
print(f"Model di Google Drive: {SAVE_DIR}/sentiment-v2/")
