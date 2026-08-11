"""
GOOGLE COLAB — COMPLETE FINETUNING PIPELINE
============================================
Copy-paste TIAP CELL di bawah ke Google Colab cell terpisah.
Set runtime ke GPU: Runtime → Change runtime type → T4 GPU.

=== CELL 1: Clone repo ===
"""
# Cell 1 — copy to Colab
!git clone https://github.com/raynzz455/ID-Political-Sentiment-Tracker.git
%cd /content/ID-Political-Sentiment-Tracker
!git log --oneline -3

# === CELL 2: Install dependencies ===
# Cell 2
!pip install -q torch transformers peft scikit-learn numpy huggingface_hub

# Verify GPU
import torch
print(f"PyTorch: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# === CELL 3: Set HuggingFace token ===
# Cell 3 — replace with YOUR token from https://huggingface.co/settings/tokens
import os
os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN_HERE"  # <-- GANTI dengan token Anda

# Login (for model download + upload)
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

# === CELL 4: Verify dataset ===
# Cell 4
import json
from collections import Counter

rows = [json.loads(l) for l in open('finetuning/datasets/dataset_enhanced.jsonl')]
print(f"Dataset: {len(rows)} rows")
print(f"Labels: {dict(Counter(r['gold_label'] for r in rows))}")
print(f"Sources: {dict(Counter(r['label_source'] for r in rows))}")

# Filter to relevant rows for sentiment training
sent_rows = [r for r in rows if r.get('gold_relevancy') == 'relevant']
print(f"\nSentiment training rows (relevant only): {len(sent_rows)}")
print(f"  positive: {sum(1 for r in sent_rows if r['gold_label']=='positive')}")
print(f"  neutral: {sum(1 for r in sent_rows if r['gold_label']=='neutral')}")
print(f"  negative: {sum(1 for r in sent_rows if r['gold_label']=='negative')}")

# === CELL 5: Run finetune (GPU ~10 min) ===
# Cell 5 — this is the MAIN training cell
import sys
sys.path.insert(0, 'finetuning/configs')
sys.path.insert(0, 'finetuning/scripts')

# Run finetune — will take ~10 minutes on T4 GPU
!python finetuning/scripts/finetune.py --task sentiment

# === CELL 6: Run evaluation + confidence sweep ===
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
print("FINETUNING RESULTS — M5 (Anti-Overconfidence)")
print("=" * 60)

test_m = metrics.get('test_metrics', {})
print(f"\nTest Accuracy:  {test_m.get('accuracy', 'N/A'):.4f}")
print(f"Test macro-F1:  {test_m.get('macro_f1', 'N/A'):.4f}")
print(f"Temperature:    {metrics.get('temperature', 'N/A')}")
print(f"Class weights:  {metrics.get('class_weights', 'N/A')}")

print("\n" + "=" * 60)
print("CALIBRATION METRICS")
print("=" * 60)
full = evaluation.get('full_coverage', {})
print(f"Full-coverage accuracy: {full.get('accuracy', 'N/A'):.4f}")
print(f"Full-coverage macro-F1: {full.get('macro_f1', 'N/A'):.4f}")

# Show confusion matrix if available
cm = full.get('confusion_matrix', [])
if cm:
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"  Labels: {full.get('labels', ['neg','neu','pos'])}")
    for row in cm:
        print(f"  {row}")

print("\n" + "=" * 60)
print("CONFIDENCE THRESHOLD SWEEP")
print("=" * 60)
print(f"  {'tau':>6} {'kept_acc':>10} {'coverage':>10} {'deferred':>10}")
print("-" * 40)

for s in evaluation.get('sweep', []):
    flag = " <-- 97% TARGET" if s['kept_accuracy'] >= 0.97 else ""
    print(f"  {s['tau']:>6.2f} {s['kept_accuracy']:>10.4f} {s['coverage']:>10.1%} {1-s['coverage']:>10.1%}{flag}")

best = evaluation.get('best_97')
if best:
    print(f"\n✅ 97% TARGET ACHIEVED!")
    print(f"   tau={best['tau']}, kept_accuracy={best['kept_accuracy']:.4f}, coverage={best['coverage']:.1%}")
else:
    max_acc = max(s['kept_accuracy'] for s in evaluation.get('sweep', [{}]))
    print(f"\n⚠️ 97% target NOT reached. Max kept-acc: {max_acc:.4f}")
    print(f"   Try: increase label_smoothing, or increase tau, or add more data")

# === CELL 8: Upload to HuggingFace ===
# Cell 8
!python finetuning/scripts/upload_huggingface.py --task sentiment --hf-token $HF_TOKEN

# === CELL 9: Also upload relevancy model ===
# Cell 9
!python finetuning/scripts/finetune.py --task relevancy
!python finetuning/scripts/evaluate.py --task relevancy --run-dir ./runs/relevancy
!python finetuning/scripts/upload_huggingface.py --task relevancy --hf-token $HF_TOKEN

# === CELL 10: Download results for backup ===
# Cell 10
from google.colab import files

# Download key files
files.download('runs/sentiment/metrics.json')
files.download('runs/sentiment/evaluation.json')

print("\n✅ Pipeline complete!")
print(f"   Sentiment model: https://huggingface.co/raynzz455/id-political-sentiment-sentiment-v1")
print(f"   Relevancy model: https://huggingface.co/raynzz455/id-political-sentiment-relevancy-v1")
print(f"\n   Next: Update packages/nlp/sentiment_model.py to use finetuned models.")

# === CELL 11: Update production code (optional) ===
# Cell 11 — after verifying model works, update production
print("""
To update production code, edit packages/nlp/sentiment_model.py:

  RELEVANCY_MODEL_ID = "raynzz455/id-political-sentiment-relevancy-v1"
  SENTIMENT_MODEL_ID  = "raynzz455/id-political-sentiment-sentiment-v1"
  FALLBACK_MODEL_ID   = "taufiqdp/indonesian-sentiment"  # keep as fallback

Then deploy to GitHub Actions. The finetuned model will be downloaded
automatically on first run (cached for subsequent runs).
""")
