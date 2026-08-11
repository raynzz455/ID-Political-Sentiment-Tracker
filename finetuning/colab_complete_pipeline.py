# ============================================================
# GOOGLE COLAB — COMPLETE FINETUNING PIPELINE
# Copy-paste each cell to Colab (with GPU enabled)
# ============================================================

# === CELL 1: Clone repo + setup ===
!git clone https://github.com/raynzz455/ID-Political-Sentiment-Tracker.git
%cd /content/ID-Political-Sentiment-Tracker
!git checkout feature/finetuning-optimized

# === CELL 2: Install dependencies ===
!pip install torch transformers peft scikit-learn numpy huggingface_hub --quiet

# === CELL 3: Set HuggingFace token (for model download + upload) ===
import os
os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN_HERE"  # <-- REPLACE with your HF token

# Login to HuggingFace
!huggingface-cli login --token $HF_TOKEN

# === CELL 4: Verify dataset ===
import json
rows = [json.loads(l) for l in open('finetuning/datasets/dataset_enhanced.jsonl')]
print(f"Dataset: {len(rows)} rows")
from collections import Counter
print(f"Labels: {dict(Counter(r['gold_label'] for r in rows))}")
print(f"Sources: {dict(Counter(r['label_source'] for r in rows))}")

# === CELL 5: Run finetune (sentiment task, GPU ~10 min) ===
import sys
sys.path.insert(0, 'finetuning/configs')
sys.path.insert(0, 'finetuning/scripts')

# Run finetune
!python finetuning/scripts/finetune.py --task sentiment

# === CELL 6: Run evaluation + confidence sweep ===
!python finetuning/scripts/evaluate.py --task sentiment --run-dir ./runs/sentiment

# === CELL 7: Display results ===
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
print()

full = evaluation.get('full_coverage', {})
print(f"Full-coverage accuracy: {full.get('accuracy', 'N/A')}")
print(f"Full-coverage macro-F1: {full.get('macro_f1', 'N/A')}")
print()

print("Confidence Threshold Sweep:")
print(f"  {'tau':>6} {'kept_acc':>10} {'coverage':>10}")
for s in evaluation.get('sweep', []):
    flag = " <-- 97% TARGET" if s['kept_accuracy'] >= 0.97 else ""
    print(f"  {s['tau']:>6.2f} {s['kept_accuracy']:>10.4f} {s['coverage']:>10.1%}{flag}")

best = evaluation.get('best_97')
if best:
    print(f"\n✅ 97% TARGET ACHIEVED at tau={best['tau']} (coverage={best['coverage']:.1%})")
else:
    print(f"\n❌ 97% target NOT reached. Max kept-acc: {max(s['kept_accuracy'] for s in evaluation.get('sweep',[])):.4f}")

# === CELL 8: Upload to HuggingFace ===
!python finetuning/scripts/upload_huggingface.py --task sentiment --hf-token $HF_TOKEN
!python finetuning/scripts/upload_huggingface.py --task relevancy --hf-token $HF_TOKEN

# === CELL 9: Download results for backup ===
from google.colab import files
files.download('runs/sentiment/metrics.json')
files.download('runs/sentiment/evaluation.json')
files.download('runs/sentiment/lora/adapter_config.json')

print("\n✅ Complete! Model uploaded to HuggingFace.")
print(f"   URL: https://huggingface.co/raynzz455/id-political-sentiment-sentiment-v1")
