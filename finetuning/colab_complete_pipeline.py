"""
GOOGLE COLAB — COMPLETE FINETUNING PIPELINE (NO HUGGINGFACE REQUIRED)
=====================================================================
Copy-paste TIAP CELL ke Google Colab.
Set runtime ke GPU: Runtime -> Change runtime type -> T4 GPU.

Model akan disimpan LOKAL di Google Colab + Google Drive.
TIDAK PERLU HuggingFace token.

=== CELL 1: Clone repo ===
"""
# Cell 1
!git clone https://github.com/raynzz455/ID-Political-Sentiment-Tracker.git
%cd /content/ID-Political-Sentiment-Tracker
!git log --oneline -3

# === CELL 2: Install dependencies (NO HF token needed) ===
# Cell 2
!pip install -q torch transformers peft scikit-learn numpy

import torch
print(f"PyTorch: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# === CELL 3: Mount Google Drive ===
# Cell 3
from google.colab import drive
drive.mount('/content/drive')

import os
MODEL_SAVE_DIR = '/content/drive/MyDrive/id-political-sentiment-models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
print(f"Model akan disimpan di: {MODEL_SAVE_DIR}")

# === CELL 4: Verify dataset ===
# Cell 4
import json
from collections import Counter

rows = [json.loads(l) for l in open('finetuning/datasets/dataset_enhanced.jsonl')]
print(f"Dataset: {len(rows)} rows")
print(f"Labels: {dict(Counter(r['gold_label'] for r in rows))}")

sent_rows = [r for r in rows if r.get('gold_relevancy') == 'relevant']
print(f"Sentiment training rows: {len(sent_rows)}")
print(f"  positive: {sum(1 for r in sent_rows if r['gold_label']=='positive')}")
print(f"  neutral: {sum(1 for r in sent_rows if r['gold_label']=='neutral')}")
print(f"  negative: {sum(1 for r in sent_rows if r['gold_label']=='negative')}")

# === CELL 5: Run finetune (GPU ~10 min) ===
# Cell 5 — MAIN TRAINING
# Base model download otomatis dari HF Hub (PUBLIC, no token needed)
import sys
sys.path.insert(0, 'finetuning/configs')
sys.path.insert(0, 'finetuning/scripts')

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
print("FINETUNING RESULTS")
print("=" * 60)

test_m = metrics.get('test_metrics', {})
print(f"Test Accuracy:  {test_m.get('accuracy', 'N/A')}")
print(f"Test macro-F1:  {test_m.get('macro_f1', 'N/A')}")
print(f"Temperature:    {metrics.get('temperature', 'N/A')}")

full = evaluation.get('full_coverage', {})
print(f"\nFull-coverage accuracy: {full.get('accuracy', 'N/A')}")
print(f"Full-coverage macro-F1: {full.get('macro_f1', 'N/A')}")

cm = full.get('confusion_matrix', [])
if cm:
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    labels = full.get('labels', ['neg','neu','pos'])
    print(f"  {'':>10s} " + " ".join(f"{l:>10s}" for l in labels))
    for i, row in enumerate(cm):
        print(f"  {labels[i]:>10s} " + " ".join(f"{v:>10d}" for v in row))

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
# Cell 8 — simpan model ke Google Drive (NO HuggingFace!)
import shutil

drive_dir = f'{MODEL_SAVE_DIR}/sentiment-v1'
os.makedirs(drive_dir, exist_ok=True)

# Copy LoRA adapter (~4MB)
if os.path.exists('./runs/sentiment/lora'):
    shutil.copytree('./runs/sentiment/lora', f'{drive_dir}/lora', dirs_exist_ok=True)
    print(f"LoRA adapter saved: {drive_dir}/lora")

# Copy tokenizer
if os.path.exists('./runs/sentiment/tokenizer'):
    shutil.copytree('./runs/sentiment/tokenizer', f'{drive_dir}/tokenizer', dirs_exist_ok=True)
    print(f"Tokenizer saved: {drive_dir}/tokenizer")

# Copy metrics
for f in ['./runs/sentiment/metrics.json', './runs/sentiment/evaluation.json']:
    if os.path.exists(f):
        shutil.copy(f, drive_dir)
        print(f"{os.path.basename(f)} saved")

# === CELL 9: MERGE + SAVE FULL MODEL (~440MB) ===
# Cell 9 — merge LoRA ke base, simpan full model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

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

drive_rel = f'{MODEL_SAVE_DIR}/relevancy-v1'
os.makedirs(drive_rel, exist_ok=True)
shutil.copytree('./runs/relevancy/lora', f'{drive_rel}/lora', dirs_exist_ok=True)
shutil.copytree('./runs/relevancy/tokenizer', f'{drive_rel}/tokenizer', dirs_exist_ok=True)
for f in ['./runs/relevancy/metrics.json', './runs/relevancy/evaluation.json']:
    if os.path.exists(f): shutil.copy(f, drive_rel)
print(f"Relevancy model saved: {drive_rel}")

# === CELL 11: Download metrics (optional backup) ===
# Cell 11
from google.colab import files
files.download('./runs/sentiment/metrics.json')
files.download('./runs/sentiment/evaluation.json')

# === CELL 12: How to use model in production ===
# Cell 12
print("""
PIPELINE COMPLETE — Model saved to Google Drive!

Location: /content/drive/MyDrive/id-political-sentiment-models/
  sentiment-v1/
    lora/              <- LoRA adapter (~4MB)
    tokenizer/         <- Tokenizer config
    merged_model/      <- Full merged model (~440MB)
    metrics.json       <- Performance metrics
    evaluation.json    <- Confidence sweep results
  relevancy-v1/
    (same structure)

HOW TO USE IN PRODUCTION (without HuggingFace):

1. Copy model folder from Google Drive to your server:

   # Option A: Manual copy
   # Download from Google Drive, upload to server

   # Option B: Git LFS (if repo has large file support)
   git lfs install
   git lfs track "*.safetensors"
   git add .gitattributes
   cp -r /path/to/models ./models/
   git add models/
   git commit -m "add finetuned models"

2. Load model in Python:

   from transformers import AutoTokenizer, AutoModelForSequenceClassification
   import torch

   # Load from local folder
   model_path = "./models/sentiment-v1/merged_model"
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   model = AutoModelForSequenceClassification.from_pretrained(model_path)

   # Predict
   entity = "Prabowo Subianto"
   context = "Presiden Prabowo menegaskan program ekonomi akan berjalan."
   inputs = tokenizer(entity, context, truncation=True, max_length=256, return_tensors="pt")

   with torch.no_grad():
       T = 1.3  # from metrics.json
       probs = torch.softmax(model(**inputs).logits / T, dim=-1)

   labels = ["negative", "neutral", "positive"]
   pred = labels[probs.argmax()]
   conf = probs.max().item()
   print(f"Sentiment: {pred} (confidence: {conf:.1%})")

   # Defer if low confidence
   if conf < 0.70:
       print("Low confidence - defer to human review")

3. Update packages/nlp/sentiment_model.py:

   SENTIMENT_MODEL_ID = "./models/sentiment-v1/merged_model"
   RELEVANCY_MODEL_ID = "./models/relevancy-v1/merged_model"
""")
