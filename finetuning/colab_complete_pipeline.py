"""
GOOGLE COLAB — COMPLETE FINETUNING PIPELINE (NO HUGGINGFACE REQUIRED)
=====================================================================
Copy-paste TIAP CELL ke Google Colab.
Set runtime ke GPU: Runtime -> Change runtime type -> T4 GPU.

=== CELL 1: Clone repo ===
"""
# Cell 1
!git clone https://github.com/raynzz455/ID-Political-Sentiment-Tracker.git
%cd /content/ID-Political-Sentiment-Tracker

# === CELL 2: Install dependencies (ALL LATEST VERSIONS) ===
# Cell 2
# Fix: torchao 0.10.0 (Colab default) incompatible dengan peft terbaru.
# Solution: upgrade torchao to latest (>=0.16.0) instead of uninstalling.
!pip install -q --upgrade torch torchao transformers peft scikit-learn numpy

# Verify
import torch, transformers, peft
print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"peft: {peft.__version__}")
try:
    import torchao
    print(f"torchao: {torchao.__version__}")
except:
    print("torchao: not installed (OK)")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - aktifkan GPU!'}")

# === CELL 3: Mount Google Drive ===
# Cell 3
from google.colab import drive
drive.mount('/content/drive')

import os
SAVE_DIR = '/content/drive/MyDrive/id-political-sentiment-models'
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Model akan disimpan di: {SAVE_DIR}")

# === CELL 4: Verify dataset ===
# Cell 4
import json
from collections import Counter

rows = [json.loads(l) for l in open('finetuning/datasets/dataset_enhanced.jsonl')]
print(f"Dataset: {len(rows)} rows")
print(f"Labels: {dict(Counter(r['gold_label'] for r in rows))}")

sent = [r for r in rows if r.get('gold_relevancy') == 'relevant']
print(f"\nSentiment training rows: {len(sent)}")
for l in ['positive', 'neutral', 'negative']:
    print(f"  {l}: {sum(1 for r in sent if r['gold_label']==l)}")

# === CELL 5: Run finetune (GPU ~10 min) ===
# Cell 5
!python finetuning/scripts/finetune.py --task sentiment

# === CELL 6: Run evaluation ===
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
print("HASIL FINETUNING")
print("=" * 60)

test = metrics.get('test_metrics', {})
print(f"Test Accuracy:     {test.get('accuracy', 'N/A')}")
print(f"Test macro-F1:     {test.get('macro_f1', 'N/A')}")
print(f"Temperature (T):   {metrics.get('temperature', 'N/A')}")

full = evaluation.get('full_coverage', {})
print(f"\nFull-coverage accuracy: {full.get('accuracy', 'N/A')}")
print(f"Full-coverage macro-F1: {full.get('macro_f1', 'N/A')}")

cm = full.get('confusion_matrix', [])
if cm:
    labels = full.get('labels', ['negative', 'neutral', 'positive'])
    print(f"\nConfusion Matrix (baris=true, kolom=pred):")
    print(f"  {'':>12s} " + " ".join(f"{l[:8]:>10s}" for l in labels))
    for i, row in enumerate(cm):
        print(f"  {labels[i][:12]:>12s} " + " ".join(f"{v:>10d}" for v in row))

print(f"\nConfidence Threshold Sweep:")
print(f"  {'tau':>6} {'kept_acc':>10} {'coverage':>10}")
for s in evaluation.get('sweep', []):
    flag = " <-- 97%" if s['kept_accuracy'] >= 0.97 else ""
    print(f"  {s['tau']:>6.2f} {s['kept_accuracy']:>10.4f} {s['coverage']:>10.1%}{flag}")

best = evaluation.get('best_97')
if best:
    print(f"\n97% TARGET ACHIEVED at tau={best['tau']} (coverage={best['coverage']:.1%})")
else:
    print(f"\n97% not reached. See docs/FINETUNING_SCIENCE.md for tuning guide.")

# === CELL 8: SAVE MODEL TO GOOGLE DRIVE ===
# Cell 8
import shutil

drive_dir = f'{SAVE_DIR}/sentiment-v1'
os.makedirs(drive_dir, exist_ok=True)

if os.path.exists('./runs/sentiment/lora'):
    shutil.copytree('./runs/sentiment/lora', f'{drive_dir}/lora', dirs_exist_ok=True)
    print(f"LoRA adapter saved: {drive_dir}/lora")

if os.path.exists('./runs/sentiment/tokenizer'):
    shutil.copytree('./runs/sentiment/tokenizer', f'{drive_dir}/tokenizer', dirs_exist_ok=True)
    print(f"Tokenizer saved: {drive_dir}/tokenizer")

for f_name in ['./runs/sentiment/metrics.json', './runs/sentiment/evaluation.json']:
    if os.path.exists(f_name):
        shutil.copy(f_name, drive_dir)
        print(f"{os.path.basename(f_name)} saved")

# === CELL 9: MERGE + SAVE FULL MODEL (~440MB) ===
# Cell 9
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

print("Merging LoRA into base model...")
base_id = "apriandito/indobert-sentiment-classifier"
tok = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForSequenceClassification.from_pretrained(base_id)
model = PeftModel.from_pretrained(base, './runs/sentiment/lora')
model = model.merge_and_unload()

T = metrics.get('temperature', 1.0)
model.config.temperature = T

merged_dir = f'{drive_dir}/merged_model'
os.makedirs(merged_dir, exist_ok=True)
model.save_pretrained(merged_dir)
tok.save_pretrained(merged_dir)
print(f"Full model saved: {merged_dir}")

# === CELL 10: Also finetune + save relevancy model ===
# Cell 10
!python finetuning/scripts/finetune.py --task relevancy
!python finetuning/scripts/evaluate.py --task relevancy --run-dir ./runs/relevancy

drive_rel = f'{SAVE_DIR}/relevancy-v1'
os.makedirs(drive_rel, exist_ok=True)
shutil.copytree('./runs/relevancy/lora', f'{drive_rel}/lora', dirs_exist_ok=True)
shutil.copytree('./runs/relevancy/tokenizer', f'{drive_rel}/tokenizer', dirs_exist_ok=True)
for f_name in ['./runs/relevancy/metrics.json', './runs/relevancy/evaluation.json']:
    if os.path.exists(f_name):
        shutil.copy(f_name, drive_rel)
print(f"Relevancy model saved: {drive_rel}")

# === CELL 11: Download metrics ===
# Cell 11
from google.colab import files
files.download('./runs/sentiment/metrics.json')
files.download('./runs/sentiment/evaluation.json')

print("\n✅ Pipeline lengkap!")
print(f"Model di Google Drive: {SAVE_DIR}")
