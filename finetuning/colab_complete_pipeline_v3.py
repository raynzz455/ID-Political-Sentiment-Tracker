"""
GOOGLE COLAB — V3 FINETUNING PIPELINE (K-Fold CV + Adversarial + Mixup)
=========================================================================
Copy-paste TIAP CELL ke Google Colab.
Set runtime ke GPU: Runtime -> Change runtime type -> T4 GPU.

v3 improvements:
  - Uses finetune_v3.py (K-fold CV + adversarial + mixup)
  - Uses dataset_v9.jsonl (1378 rows, 94.4% LLM-verified)
  - LoRA r=64 (upgraded from 16/32)
  - 20 epochs + SWA from epoch 5
  - Effective batch 64

=== CELL 1: Clone repo ===
"""
# Cell 1
!git clone https://github.com/raynzz455/ID-Political-Sentiment-Tracker.git
%cd /content/ID-Political-Sentiment-Tracker
!git pull  # ensure latest

# === CELL 2: Install dependencies (NUCLEAR OPTION) ===
# Cell 2
!pip uninstall -y torch torchvision torchaudio torchao torchtext 2>/dev/null
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
!pip install -q torchao transformers peft scikit-learn numpy

import torch, transformers, peft
print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"peft: {peft.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - aktifkan GPU!'}")

# === CELL 3: Mount Google Drive ===
# Cell 3
from google.colab import drive
drive.mount('/content/drive')

import os
SAVE_DIR = '/content/drive/MyDrive/id-political-sentiment-models-v3'
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Model v3 akan disimpan di: {SAVE_DIR}")

# === CELL 4: Verify dataset v9 ===
# Cell 4
import json
from collections import Counter

rows = [json.loads(l) for l in open('finetuning/datasets/dataset_v9.jsonl')]
print(f"Dataset v9: {len(rows)} rows")
print(f"Labels: {dict(Counter(r['gold_label'] for r in rows))}")

sent = [r for r in rows if r.get('gold_relevancy') == 'relevant']
print(f"\nSentiment training rows: {len(sent)}")
for l in ['positive', 'neutral', 'negative']:
    cnt = sum(1 for r in sent if r['gold_label']==l)
    print(f"  {l}: {cnt} ({100*cnt/len(sent):.1f}%)")

conf_buckets = Counter()
for r in sent:
    c = r.get('label_confidence', 0)
    if c >= 0.85: conf_buckets['>=0.85 (LLM/gold)'] += 1
    elif c >= 0.7: conf_buckets['0.70-0.84 (trusted)'] += 1
    elif c >= 0.55: conf_buckets['0.55-0.69 (low)'] += 1
    else: conf_buckets['<0.55 (unverified)'] += 1
print(f"\nConfidence distribution:")
for k, v in conf_buckets.most_common():
    print(f"  {k:30s}: {v:4d} ({100*v/len(sent):.1f}%)")

# === CELL 5: Run finetune v3 — K-FOLD CV mode (~40 min) ===
# Cell 5
print("=" * 60)
print("RUNNING FINETUNE v3 — K-FOLD CV (5-fold)")
print("Expected: ~40 minutes on T4 GPU")
print("Features: LoRA r=64 + Focal + Adversarial + Mixup + SWA")
print("=" * 60)

!python finetuning/finetune_v3.py --task sentiment --kfold 5

# === CELL 6: Run standard train/val/test split (optional) ===
# Cell 6 (OPTIONAL — uncomment to run)
# !python finetuning/finetune_v3.py --task sentiment

# === CELL 7: Display K-Fold results ===
# Cell 7
import json

kfold_path = 'runs/sentiment_v3/kfold_results.json'
try:
    with open(kfold_path) as f:
        results = json.load(f)

    print("=" * 60)
    print("K-FOLD CROSS-VALIDATION RESULTS (v3)")
    print("=" * 60)

    agg = results.get('aggregate', {})
    for key, stats in agg.items():
        if isinstance(stats, dict) and 'mean' in stats:
            print(f"  {key:20s}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    print(f"\nPer-fold details:")
    for i, fold in enumerate(results.get('folds', [])):
        print(f"  Fold {i+1}: acc={fold.get('accuracy',0):.4f} f1={fold.get('macro_f1',0):.4f} T={fold.get('temperature',0):.3f}")
except FileNotFoundError:
    print(f"K-fold results not found at {kfold_path}")
    print("Did you run CELL 5? Or run standard mode (CELL 6) instead.")

# === CELL 8: SAVE MODEL TO GOOGLE DRIVE ===
# Cell 8
import shutil

drive_dir = f'{SAVE_DIR}/sentiment-v3'
os.makedirs(drive_dir, exist_ok=True)

if os.path.exists(kfold_path):
    shutil.copy(kfold_path, drive_dir)
    print(f"K-fold results saved: {drive_dir}/kfold_results.json")

lora_path = 'runs/sentiment_v3/lora'
if os.path.exists(lora_path):
    shutil.copytree(lora_path, f'{drive_dir}/lora', dirs_exist_ok=True)
    print(f"LoRA adapter saved: {drive_dir}/lora")

tokenizer_path = 'runs/sentiment_v3/tokenizer'
if os.path.exists(tokenizer_path):
    shutil.copytree(tokenizer_path, f'{drive_dir}/tokenizer', dirs_exist_ok=True)
    print(f"Tokenizer saved: {drive_dir}/tokenizer")

metrics_path = 'runs/sentiment_v3/metrics.json'
if os.path.exists(metrics_path):
    shutil.copy(metrics_path, drive_dir)
    print(f"Metrics saved: {drive_dir}/metrics.json")

# === CELL 9: MERGE LoRA + SAVE FULL MODEL (~440MB) ===
# Cell 9
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

if os.path.exists(lora_path):
    print("Merging LoRA into base model...")
    base_id = "apriandito/indobert-sentiment-classifier"
    tok = AutoTokenizer.from_pretrained(base_id)
    base = AutoModelForSequenceClassification.from_pretrained(base_id)
    model = PeftModel.from_pretrained(base, lora_path)
    model = model.merge_and_unload()

    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
        T = metrics.get('temperature', 1.0)
    except:
        T = 1.3
    model.config.temperature = T
    print(f"Temperature applied: T={T}")

    merged_dir = f'{drive_dir}/merged_model'
    os.makedirs(merged_dir, exist_ok=True)
    model.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)
    print(f"Full merged model saved: {merged_dir}")
    print(f"  Size: ~440MB")
    print(f"  Ready for HuggingFace upload or production deployment")
else:
    print("LoRA adapter not found. Run standard mode (CELL 6) to get single-fold save.")

# === CELL 10: Upload to HuggingFace Hub (optional) ===
# Cell 10 (OPTIONAL — requires HF_TOKEN)
# from huggingface_hub import login, create_repo, upload_folder
# HF_TOKEN = "hf_xxx"
# login(HF_TOKEN)
# REPO_NAME = "raynzz455/id-political-sentiment-sentiment-v3"
# create_repo(REPO_NAME, exist_ok=True)
# upload_folder(folder_path=merged_dir, repo_id=REPO_NAME)
# print(f"Model uploaded: https://huggingface.co/{REPO_NAME}")

# === CELL 11: Also finetune + save relevancy model ===
# Cell 11
print("\n" + "=" * 60)
print("FINETUNE RELEVANCY MODEL (v3)")
print("=" * 60)

!python finetuning/finetune_v3.py --task relevancy --kfold 5

drive_rel = f'{SAVE_DIR}/relevancy-v3'
os.makedirs(drive_rel, exist_ok=True)

kfold_rel_path = 'runs/relevancy_v3/kfold_results.json'
if os.path.exists(kfold_rel_path):
    shutil.copy(kfold_rel_path, drive_rel)
    print(f"K-fold results saved: {drive_rel}/kfold_results.json")

lora_rel_path = 'runs/relevancy_v3/lora'
if os.path.exists(lora_rel_path):
    shutil.copytree(lora_rel_path, f'{drive_rel}/lora', dirs_exist_ok=True)
    print(f"LoRA adapter saved: {drive_rel}/lora")

# === CELL 12: Summary ===
# Cell 12
print("\n" + "=" * 60)
print("V3 PIPELINE COMPLETE")
print("=" * 60)
print(f"\nModels saved to Google Drive: {SAVE_DIR}")
print(f"\nv3 improvements over v1:")
print(f"  - LoRA r=64 (was 16/32) — more capacity")
print(f"  - K-Fold CV (5-fold) — robust evaluation")
print(f"  - Adversarial training (PGD) — fights input perturbations")
print(f"  - Mixup augmentation — fights overfitting")
print(f"  - SWA from epoch 5 — flatter optimum")
print(f"  - 20 epochs (was 10/15) — more training")
print(f"  - Effective batch 64 (was 32) — stable gradients")
print(f"\nExpected: macro-F1 0.64 (v1) -> 0.70+ (v3)")
