# Dataset + Metode Finetuning + Manual Tuning Guide

## 1. DATASET — Dari Mana dan Bagaimana Dibentuk

### Sumber Data
Dataset berasal dari pipeline production — 909 artikel berita politik Indonesia yang di-scrape dari media (Tempo, CNN Indonesia, Detik, dll) lalu di-extract.

### Proses Labeling
```
909 rows raw (pseudo_label dari model v12)
    ↓
27 rows → human gold labels (critical review manual)
375 rows → LLM second-pass labels (z-ai chat, strict prompt)
144 rows → LLM-verified heuristic labels
263 rows → heuristic upgrade (cue-based rules)
    ↓
909 rows all labeled — DATASET v1 (dataset_enhanced.jsonl)
```

### Dataset v2 (untuk finetune v2)
```
909 rows v1
    ↓ filter: exclude background_only (296), llm_failed (181), corruption (1), wrong_entity (7)
376 rows clean
    ↓ oversample: duplicate negative (+209) dan positive (+192) sampai match neutral (259)
777 rows balanced 1:1:1 — DATASET v2 (dataset_v2.jsonl)
```

## 2. METODE FINETUNING

### 2.1 LoRA (Low-Rank Adaptation)
Freeze base model (110M params), hanya train adapter matrix kecil (~14M params, 4%).
Mencegah overfit pada data kecil (597 rows).

### 2.2 Focal Loss (gamma=2.5)
Down-weight "easy" examples. Fokus ke hard examples (minority class).
Mengatasi class imbalance 3.8:1.

### 2.3 Class-Balanced Weights (1/sqrt(freq))
Weight class berbanding terbalik dengan frekuensi.
neutral (mayoritas) diturunkan, negative/positive (minoritas) dinaikkan.

### 2.4 Label Smoothing (epsilon=0.05)
Soft target [0.025, 0.95, 0.025] instead of hard [0, 1, 0].
Caps max confidence di ~0.90. Mencegah overconfidence.

### 2.5 Temperature Scaling (T=3.837)
Bagi logits dengan T sebelum softmax. Confidence lebih honest.
T di-fit pada validation set via LBFGS optimizer.

### 2.6 SWA (Stochastic Weight Averaging)
Average weights dari epoch 10-15. Flatter optimum, generalisasi lebih baik.

### 2.7 Per-Sample Confidence Weighting
Setiap sample di-weight berdasarkan kualitas label:
- gold_human (conf=1.0): full gradient
- llm_second_pass (conf=0.85): 85% gradient
- heuristic (conf=0.5): 50% gradient

### 2.8 Early Stopping (patience=4)
Stop kalau val macro-F1 tidak improve 4 epoch.

### 2.9 Stratified Split (70/15/15)
Class proportion dipertahankan di train/val/test.

## 3. MANUAL TUNING — Parameter yang Bisa Anda Atur

### 3.1 Parameter yang Bisa di-Tuning

| Parameter | Current | Range | Efek |
|-----------|---------|-------|------|
| learning_rate | 3e-5 | 1e-5 ~ 5e-5 | Tinggi=unstable, Rendah=underfit |
| num_epochs | 15 | 5 ~ 30 | Sedikit=underfit, Banyak=overfit |
| batch_size | 16 | 8 ~ 32 | Besar=stabil butuh memory |
| focal_gamma | 2.5 | 1.0 ~ 4.0 | Tinggi=fokus hard examples |
| label_smoothing | 0.05 | 0.0 ~ 0.15 | Tinggi=kurangi overconfidence |
| lora_r | 32 | 8 ~ 64 | Tinggi=kapasitas+, overfit+ |
| lora_dropout | 0.1 | 0.0 ~ 0.3 | Tinggi=regularisasi+ |
| weight_decay | 0.01 | 0.001 ~ 0.1 | Tinggi=regularisasi+ |
| confidence_tau | 0.70 | 0.50 ~ 0.90 | Tinggi=lebih banyak DEFER |
| temperature | 3.837 | 1.0 ~ 5.0 | Tinggi=confidence lebih soft |
| early_stop_patience | 4 | 2 ~ 10 | Tinggi=training lebih lama |

### 3.2 Cara Tuning di Colab

Edit file `finetuning/configs/hyperparams_v2.py`:
```python
# Contoh: coba lr lebih tinggi
LEARNING_RATE = 5e-5

# Contoh: coba gamma lebih agresif
FOCAL_GAMMA = 3.0

# Contoh: kurangi overconfidence lebih
LABEL_SMOOTHING = 0.10

# Contoh: lebih banyak DEFER
CONFIDENCE_TAU = 0.80
```

Lalu run finetune lagi:
```python
!python finetuning/scripts/finetune.py --task sentiment
```

### 3.3 Workflow Tuning

```
Step 1: Run v2 dengan default params → catat F1 dan ECE
Step 2: Kalau F1 < 0.75 → coba lr=5e-5 atau gamma=3.0
Step 3: Kalau ECE > 0.15 → tingkatkan smoothing ke 0.10
Step 4: Kalau minority F1 < 0.60 → oversampling lebih aggressive
Step 5: Kalau overfit → kurangi epochs atau tambah dropout
Step 6: Kalau masih kurang → LLM hybrid pipeline (DEFER ke LLM)
```

### 3.4 Parameter yang TIDAK Perlu di-Tuning

| Parameter | Value | Kenapa |
|-----------|-------|--------|
| MAX_SEQ_LENGTH | 256 | Match production |
| SEED | 42 | Reproducibility |
| optimizer | AdamW | Standard untuk BERT |
| scheduler | cosine | Sudah optimal |
| target_modules | Q/K/V/dense | Sudah optimal |
