# Panduan Script: Penguji Worker + Finetuning NLP
# =================================================

## 1. SCRIPT PENGUJI WORKER (Krusial untuk NLP)
==============================================

Mengapa krusial? Jika entity resolution salah → context extraction salah
→ model belajar dari data yang salah → prediksi tidak akurat.

### Lokasi:
  finetuning/tests/test_entity_resolution.py      ← uji entity worker
  finetuning/tests/test_context_extraction.py     ← uji context worker

### Cara menjalankan (di sandbox atau Colab):

```bash
# Test entity resolution (default 20 rows, semua expert)
python3 finetuning/tests/test_entity_resolution.py

# Test dengan 50 rows
python3 finetuning/tests/test_entity_resolution.py --n 50

# Test expert spesifik
python3 finetuning/tests/test_entity_resolution.py --expert stanza
python3 finetuning/tests/test_entity_resolution.py --expert rapidfuzz

# Test context extraction
python3 finetuning/tests/test_context_extraction.py
python3 finetuning/tests/test_context_extraction.py --n 50 --expert stanza
```

### Metrics yang diukur:

**Entity Resolution:**
- Accuracy: expected entity ditemukan (target: >95%)
- Precision: found entities yang benar (target: >80%)
- Recall: semua expected ditemukan (target: >95%)
- F1 Score: balanced metric (target: >85%)

**Context Extraction:**
- Extraction rate: berapa row berhasil di-extract (target: >95%)
- High quality: entity + good length + clean boundary (target: >90%)
- Entity coverage: entity ada di context (target: >95%)
- Avg length: optimal 200-400 chars (untuk MAX_SEQ=256)

### Output:
  - Console: summary metrics + comparison table
  - JSON report: finetuning/tests/entity_resolution_report.json
  - JSON report: finetuning/tests/context_extraction_report.json

---

## 2. SCRIPT FINETUNING + HYPERPARAMETER (Model Belajar Sendiri)
==================================================================

Base model: IndoBERT (apriandito/indobert-sentiment-classifier)
Method: LoRA (Low-Rank Adaptation) — parameter-efficient fine-tuning
        Hanya melatih adapter layers (r=64), bukan full model
        → model "belajar sendiri" dari dataset gold standard

### Lokasi script:

```
finetuning/
├── finetune_v4.py                    ← SCRIPT TRAINING UTAMA
├── evaluate_v4.py                    ← SCRIPT EVALUATION
├── configs/
│   └── hyperparams_v4.py             ← HYPERPARAMETERS
├── colab_complete_pipeline_v4.py     ← PIPELINE COLOMB (7 langkah)
├── patches/
│   └── sentiment_model_v6.py         ← PRODUCTION INFERENCE
└── datasets/
    └── dataset_gold_standard_final.jsonl  ← DATA TRAINING (2,238 rows)
```

### Hyperparameters (hyperparams_v4.py):

```python
# Base model ( IndoBERT — pre-trained untuk Bahasa Indonesia)
SENTIMENT_BASE = "apriandito/indobert-sentiment-classifier"

# LoRA — parameter-efficient fine-tuning
LORA.r = 64           # rank (kapasitas adapter)
LORA.alpha = 128      # scaling
LORA.dropout = 0.20   # regularization

# Training
LEARNING_RATE = 2.5e-5
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 8  # effective batch = 64
NUM_EPOCHS = 18

# Loss — untuk handle imbalance
FOCAL_GAMMA = 3.0           # focus pada hard examples
LABEL_SMOOTHING = 0.07      # anti-overconfidence
CLASS_WEIGHT_FN = "log"     # 1/log(freq+1) — gentler reweighting

# Augmentation
MIXUP_ALPHA = 0.3            # interpolation
MIXUP_PROB = 0.4             # 40% batches
ADVERSARIAL_ENABLED = True   # PGD perturbation

# Imbalance handling
OVERSAMPLING_TARGETS = {"negative": 400, "positive": 600}

# Cross-validation
K_FOLD_N = 5
K_FOLD_ENTITY_AWARE = True   # GroupKFold by entity

# Stochastic Weight Averaging
SWA_START_EPOCH = 4
```

### Cara menjalankan di Colab:

```python
# 1. Clone repo
!git clone https://github.com/raynzz455/ID-Political-Sentiment-Tracker.git
%cd ID-Political-Sentiment-Tracker/finetuning

# 2. Install dependencies
!pip install -q transformers peft scikit-learn accelerate stanza rapidfuzz

# 3. Download Stanza Indonesian model
!python -c "import stanza; stanza.download('id')"

# 4. Fine-tune sentiment (K-fold, ~30 min on T4 GPU)
!python finetune_v4.py --task sentiment --kfold 5

# 5. Fine-tune relevancy (optional)
!python finetune_v4.py --task relevancy --kfold 5

# 6. Evaluate
!python evaluate_v4.py --task sentiment --kfold-results runs/sentiment_v4/kfold_results.json
```

### Atau gunakan pipeline otomatis:

```python
# Jalankan semua: install → clone → finetune → evaluate → upload
!python colab_complete_pipeline_v4.py --steps all

# Atau sentiment only
!python colab_complete_pipeline_v4.py --steps sentiment-only
```

### Apa yang model pelajari?

Setelah fine-tuning, model IndoBERT + LoRA adapter akan:

1. **Memahami sentimen politik Indonesia**
   - Pre-trained IndoBERT hanya tahu bahasa umum
   - LoRA adapter belajar: "Prabowo + dicopot = negative"
   - Bukan: "Prabowo + pelantikan = positive"

2. **Context-aware sentiment**
   - Input: sentence-pair (entity_premise + context)
   - Model belajar: sentimen TERHADAP entity, bukan YANG DIKATAKAN entity
   - "Prabowo mengkritik X" → neutral untuk Prabowo (pembicara)

3. **Handle class imbalance**
   - Oversampling: negative 131 → 400, positive 325 → 600
   - Focal loss: focus pada hard examples (minority class)
   - Class weights: 1/log(freq+1) — gentler reweighting

4. **Generalization**
   - Entity-aware K-fold: tidak ada entity leakage antar fold
   - Adversarial training: robust terhadap input perturbation
   - SWA: flatter optimum, better generalization

### Expected results:

| Metric | v3 (sebelumnya) | v4 (target) |
|--------|-----------------|-------------|
| macro-F1 | 0.70 | **0.75+** |
| Negative recall | 0.45 | **0.60+** |
| ECE (calibration) | 0.10 | **0.08** |

### Setelah fine-tuning:

Model LoRA adapter disimpan di:
```
runs/sentiment_v4/fold_1/final_model/
runs/sentiment_v4/fold_2/final_model/
...
runs/sentiment_v4/kfold_results.json   ← metrics
```

Upload ke HuggingFace:
```python
!huggingface-cli login --token YOUR_HF_TOKEN
!huggingface-cli upload raynzz455/id-political-sentiment-v4 runs/sentiment_v4/fold_1/final_model
```

Production inference (sentiment_model_v6.py):
```python
from sentiment_model_v6 import load_model
model = load_model("path/to/adapter")
result = model.predict("Prabowo Subianto", "Prabowo divonis korupsi hari ini.")
# → {"label": "negative", "confidence": 0.92}
```

---

## 3. URUTAN EKSEKUSI LENGKAP
==============================

```bash
# Step 1: Test worker precision (sebelum training, pastikan data bagus)
python3 finetuning/tests/test_entity_resolution.py --n 20
python3 finetuning/tests/test_context_extraction.py --n 20

# Step 2: Fine-tune (di Colab dengan GPU)
!python finetune_v4.py --task sentiment --kfold 5

# Step 3: Evaluate
!python evaluate_v4.py --task sentiment --kfold-results runs/sentiment_v4/kfold_results.json

# Step 4: Upload model ke HuggingFace
!python colab_complete_pipeline_v4.py --steps upload

# Step 5: Deploy production
# Gunakan sentiment_model_v6.py dengan adapter yang sudah dilatih
```
