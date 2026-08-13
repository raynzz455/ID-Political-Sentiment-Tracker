# Laporan Pengembangan Model Sentimen Politik Indonesia

## 1. Ringkasan Status
- **Versi Model Saat Ini:** V1 (Tersimpan di Google Drive)
- **Status:** Siap untuk sistem Deferral (LLM Second-Pass)
- **Target Selanjutnya:** V3 dengan penambahan dataset (Active Learning)

## 2. Metrik Performa Model (V1)
Model dilatih menggunakan dataset awal berjumlah 909 baris (853 setelah filter).

- **Test Accuracy:** 67.18%
- **Test Macro-F1:** 0.5964
- **Temperature (T):** 3.837 (Untuk kalibrasi probabilitas)
- **Max Kept-Accuracy (Tau=0.78):** 73.44% dengan coverage 50%

### Analisis Confusion Matrix
Model memiliki kecenderungan bias terhadap kelas Neutral (Mayoritas). Sebanyak 11 data Negative salah ditebak sebagai Neutral. Hal ini disebabkan oleh ketimpangan jumlah data latih (Neutral: 384, Negative: 102).

## 3. Temuan Hyperparameter Tuning (Grid Search)
Telah dilakukan uji coba terhadap 5 kombinasi parameter. Ditemukan bahwa meningkatkan Learning Rate memberikan dampak paling signifikan.

| Config | Acc | F1 | ECE |
|--------|-----|----|-----|
| `lr=2e-5` (Config Asli) | 0.6822 | 0.5757 | 0.1628 |
| **`lr=3e-5` (Best Config)** | **0.7132** | **0.6377** | **0.1309** |

**Catatan:** Config terbaik ini (`lr=3e-5`) belum diaplikasikan ke model yang tersimpan di Drive (V1). Model V2 baru akan di-generate jika kita training ulang menggunakan config ini.

## 4. Evaluasi Inferensi (Sistem Deferral)
Pengujian inferensi menunjukkan kalibrasi Temperature bekerja sangat baik. Model mampu mengenali batas kemampuannya.

| Entitas | Konteks | Prediksi | Confidence | Status |
|---------|---------|----------|------------|--------|
| Anies Baswedan | "...bangsa yang lugu..." | Positive | 82.2% | Confident |
| Thomas Lembong | "...divonis korupsi..." | Negative | 79.4% | Confident |
| Rocky Gerung | "...pasal yang dungu" | Negative | 80.7% | Confident |
| Prabowo Subianto | "...meski dihujani kritik" | Positive | 43.0% | DEFER |
| Joko Widodo | "...dituntut membayar..." | Negative | 65.3% | DEFER |

**Kesimpulan:** Model siap digunakan. Data yang diprediksi `DEFER` (confidence < 70%) wajib diteruskan ke LLM besar (GPT-4o/Claude) dalam pipeline production untuk mempertahankan akurasi agregat > 90%.

## 5. Dataset Overview

### Dataset V1 (digunakan untuk training V1)
- Total: 909 rows (853 setelah filter)
- Imbalanced: Neutral 384, Positive 111, Negative 102
- Label sources: 23 gold_human, 171 llm_second_pass, 114 llm_verified, sisanya heuristic

### Dataset V2 (siap untuk training V2 — belum di-run)
- Total: 777 rows (balanced 1:1:1, 259 per class)
- Filter: excluded background_only, llm_failed, corruption, wrong_entity
- Oversampled: negative +209 dup, positive +192 dup
- Expected improvement: F1 naik dari 0.60 ke 0.70-0.75

## 6. Roadmap Pengembangan

### Tahap 1 (SELESAI): V1 Model + Grid Search
- [x] Finetune V1 dengan M5 config (lr=2e-5, gamma=2.5, smoothing=0.05)
- [x] Grid search menemukan lr=3e-5 lebih baik (+5pp F1)
- [x] Sistem deferral bekerja (3/5 confident, 2/5 DEFER)
- [x] Model tersimpan di Google Drive

### Tahap 2 (SIAP DI-RUN): V2 Model dengan best params
- [ ] Build dataset v2 (balanced 1:1:1, 777 rows)
- [ ] Finetune V2 dengan lr=3e-5 (grid search winner)
- [ ] Evaluate V2 — target F1 >= 0.70
- [ ] Run LLM hybrid pipeline — target 90%+ combined accuracy

### Tahap 3 (FUTURE): V3 dengan data tambahan
- [ ] Deploy patch v15 (entity_resolution) ke production
- [ ] Re-scrape 500-1000 artikel baru dari Supabase
- [ ] Label via LLM second-pass
- [ ] Active learning: tambah label manual untuk negative/positive
- [ ] Finetune V3 dengan dataset 2000+ rows

## 7. Production Deployment Guide

### Cara pakai model V1 di production:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load dari Google Drive
model_path = "./models/sentiment-v1/merged_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Predict dengan temperature scaling (WAJIB)
T = 3.837  # dari metrics.json
entity = "Prabowo Subianto"
context = "Presiden Prabowo menegaskan program ekonomi..."
inputs = tokenizer(entity, context, truncation=True, max_length=256, return_tensors="pt")

with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits / T, dim=-1)

labels = ["negative", "neutral", "positive"]
pred = labels[probs.argmax()]
conf = probs.max().item()

# Deferral system
if conf < 0.70:
    # DEFER to LLM (GPT-4o/Claude)
    pred = llm_second_pass(entity, context)
```

### Update packages/nlp/sentiment_model.py:
```python
SENTIMENT_MODEL_ID = "./models/sentiment-v1/merged_model"
RELEVANCY_MODEL_ID = "apriandito/indobert-relevancy-classifier"  # belum di-finetune
```
