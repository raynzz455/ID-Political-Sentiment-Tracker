# Tahap 2: Finetuning v2 + LLM Hybrid Pipeline

> Target: naikkan akurasi dari 67% ke 85%+ (model) / 90%+ (hybrid)

## 1. Apa yang Dikerjakan di Tahap 2

### A. Dataset v2 — Balanced + Filtered
| Metric | v1 | v2 |
|--------|-----|-----|
| Total rows | 909 | 777 |
| Class balance | 3.8:1 (neutral-heavy) | 1:1:1 (balanced) |
| Min confidence | 0.30 | 0.50 |
| Context quality | All flags | Clean + speaker only |
| Label noise | ~10% | ~5% (filtered) |

### B. Best Params dari Grid Search
| Param | v1 | v2 | Source |
|-------|-----|-----|--------|
| Learning rate | 2e-5 | 3e-5 | Grid search winner |
| Epochs | 10 | 15 | More training on balanced data |
| Others | Same | Same | gamma=2.5, smoothing=0.05, r=32 |

### C. LLM Hybrid Pipeline
Input -> Model v2 predict -> confidence >= 0.70? -> YES: use model -> NO: LLM second-pass -> Combine 90%+

## 2. Cara Run Tahap 2 di Google Colab

### Cell 1: Clone + Install
```python
!git clone https://github.com/raynzz455/ID-Political-Sentiment-Tracker.git
%cd /content/ID-Political-Sentiment-Tracker
!pip uninstall -y torch torchvision torchaudio torchao torchtext 2>/dev/null
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
!pip install -q torchao transformers peft scikit-learn numpy
```

### Cell 2: Build dataset v2
```python
!python finetuning/scripts/build_dataset_v2.py
```

### Cell 3: Finetune v2
```python
import sys
sys.path.insert(0, 'finetuning/configs')
sys.path.insert(0, 'finetuning/scripts')
import hyperparams_v2 as H
print(f"lr={H.LEARNING_RATE} gamma={H.FOCAL_GAMMA} smoothing={H.LABEL_SMOOTHING}")
!python finetuning/scripts/finetune.py --task sentiment
```

### Cell 4: Evaluate v2
```python
!python finetuning/scripts/evaluate.py --task sentiment --run-dir ./runs/sentiment_v2
```

### Cell 5: LLM Hybrid Pipeline (optional)
```python
!python finetuning/scripts/tahap2_llm_hybrid_pipeline.py --model-path ./runs/sentiment --dataset finetuning/datasets/dataset_enhanced.jsonl --output hybrid_predictions.jsonl --limit 50
```

### Cell 6: Save to Google Drive
```python
import shutil, os
SAVE_DIR = '/content/drive/MyDrive/id-political-sentiment-models'
drive_v2 = f'{SAVE_DIR}/sentiment-v2'
os.makedirs(drive_v2, exist_ok=True)
shutil.copytree('./runs/sentiment_v2/lora', f'{drive_v2}/lora', dirs_exist_ok=True)
shutil.copytree('./runs/sentiment_v2/tokenizer', f'{drive_v2}/tokenizer', dirs_exist_ok=True)
for f in ['./runs/sentiment_v2/metrics.json', './runs/sentiment_v2/evaluation.json']:
    if os.path.exists(f): shutil.copy(f, drive_v2)
print(f"v2 model saved: {drive_v2}")
```

## 3. Expected Results

| Metric | v1 | v2 (projected) | v2 + LLM hybrid |
|--------|-----|----------------|-----------------|
| Accuracy | 67% | 75-80% | 90%+ |
| Macro-F1 | 60% | 70-75% | 85%+ |
| ECE | 0.13 | 0.13 | 0.10 |

## 4. Files
| File | Purpose |
|------|---------|
| finetuning/datasets/dataset_v2.jsonl | Balanced dataset (777 rows, 1:1:1) |
| finetuning/scripts/build_dataset_v2.py | Script rebuild v2 |
| finetuning/configs/hyperparams_v2.py | Best params (lr=3e-5) |
| finetuning/scripts/tahap2_llm_hybrid_pipeline.py | LLM hybrid inference |
