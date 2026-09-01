# Laporan Finetuning Model Sentimen Politik Indonesia (v1)

## 1. Ringkasan Eksekutif
Proyek ini berfokus pada fine-tuning model bahasa (IndoBERT) untuk mendeteksi sentimen (Positif, Netral, Negatif) pada teks berita/pernyataan politik Indonesia. Model dilatih menggunakan teknik LoRA (Low-Rank Adaptation) dengan Focal Loss dan Kalibrasi Temperature untuk mengatasi ketidakseimbangan kelas dan *overconfidence*. Model dievaluasi menggunakan metrik Macro-F1 dan Confidence Threshold (Deferral System).

## 2. Detail Data & Preprocessing
- **Sumber Data:** `dataset_enhanced.jsonl` (Hasil scraping & labeling heuristik/LLM)
- **Total Data Mentah:** 909 baris
- **Data Setelah Filter (Relevan):** 853 baris (eksklusi data dengan flag `corruption_stitch` dan `wrong_entity`)
- **Pembagian Data (Stratified Split):**
  - Train: 597 baris (70%)
  - Validation: 128 baris (15%)
  - Test: 128 baris (15%)
- **Distribusi Kelas (Imbalanced):**
  - Netral: 384 (Mayoritas)
  - Positif: 111
  - Negatif: 102 (Minoritas)

## 3. Arsitektur & Strategi Training
- **Base Model:** `apriandito/indobert-sentiment-classifier` (Public HuggingFace)
- **Metode Training:** LoRA (Parameter-Efficient Fine-Tuning)
  - Trainable params: 14,224,387 (4.07% dari total 349 Juta params)
  - LoRA Config: `r=32`, `alpha=64`, `target_modules=["query","key","value","dense"]`
- **Loss Function:** Focal Loss (`gamma=2.5`) dengan Class-Balanced Weights (`1/sqrt(freq)`)
- **Kalibrasi:** Temperature Scaling (`T=3.837`) di-fit pada Validation Set.
- **Hardware:** Google Colab (Tesla T4 GPU, 15GB VRAM)

## 4. Hasil Evaluasi (Model Awal - CELL 5)
- **Test Accuracy:** 0.6718 (67.1%)
- **Test Macro-F1:** 0.5964 (59.6%)
- **Confusion Matrix:**
  | True \ Pred | Negative | Neutral | Positive |
  |-------------|----------|---------|----------|
  | **Negative**| 9        | 11      | 2        |
  | **Neutral** | 8        | 60      | 14       |
  | **Positive**| 3        | 4       | 17       |

**Analisis:** Model cukup kesulitan membedakan kelas *Negative* dan *Neutral* (11 data negatif ditebak netral). Hal ini wajar mengingat jumlah data training kelas *Negative* sangat sedikit (102 baris).

## 5. Hasil Hyperparameter Tuning (Grid Search)
Dilakukan Grid Search pada 5 kombinasi parameter berbeda.

| Config | Acc | F1 | ECE | Time |
|--------|-----|----|-----|------|
| `smoothing_0.05_gamma_2.5` | 0.6822 | 0.5757 | 0.1628 | 361s |
| `smoothing_0.10_gamma_2.5` | 0.6977 | 0.5946 | 0.1832 | 534s |
| `smoothing_0.05_gamma_3.0` | 0.6899 | 0.5826 | 0.1590 | 620s |
| **`smoothing_0.05_gamma_2.5_lr_3e5` ⭐** | **0.7132** | **0.6377** | **0.1309** | 707s |
| `smoothing_0.05_gamma_2.5_r_16` | 0.6744 | 0.6182 | 0.1900 | 527s |

**Best Config:** Menaikkan Learning Rate dari `2e-5` ke `3e-5` terbukti paling optimal, menaikkan F1 menjadi 0.6377 dengan ECE terbaik (0.1309).

## 6. Uji Inferensi & Sistem Deferral
Pengujian inferensi dilakukan pada 5 sampel kalimat politik nyata dengan Threshold Deferral (`tau = 0.70`).

| Entitas | Konteks | Prediksi | Confidence | Status |
|---------|---------|----------|------------|--------|
| Prabowo | "...meski dihujani kritik" | Positive | 43.0% | ⚠️ DEFER |
| Jokowi | "...dituntut membayar uang..." | Negative | 65.3% | ⚠️ DEFER |
| Anies | "...bangsa yang lugu dan baik hati" | Positive | 82.2% | ✅ Confident |
| T. Lembong | "...divonis bersalah... korupsi" | Negative | 79.4% | ✅ Confident |
| R. Gerung | "...pasal yang dungu" | Negative | 80.7% | ✅ Confident |

**Analisis Kalibrasi:** Sistem kalibrasi suhu (Temperature=3.837) bekerja sangat baik. Model menolak menebak (DEFER) pada kalimat yang ambigu (konteks campuran, seperti pada kasus Prabowo & Jokowi), dan memberikan confidence tinggi pada kalimat dengan kata sifat eksplisit.

## 7. Limitasi & Rekomendasi Pengembangan
1. **Keterbatasan Data:** Akurasi mentok di ~67% karena jumlah data latih (597 baris) terlalu kecil untuk variasi bahasa politik yang kompleks.
2. **Target Akurasi 90%:** Tidak dapat dicapai dengan fine-tuning LoRA saja pada dataset ini. Disarankan menjalankan *Pipeline LLM Second-Pass*: data dengan status `DEFER` (confidence < 70%) diteruskan ke LLM besar (GPT-4o/Claude) untuk mendapat akurasi gabungan >90%.
3. **Active Learning:** Menambah data *gold_human* secara manual khusus untuk kelas Negatif & Positif untuk melatih versi model v2 di masa depan.

## 8. Status Deployment
- ✅ Model disimpan di Google Drive: `/content/drive/MyDrive/id-political-sentiment-models/sentiment-v1/`
- ✅ Format: Merged Model (Full ~440MB) & LoRA Adapter (~4MB).
- 🔲 **Next Step:** Copy folder model ke server production dan arahkan path `SENTIMENT_MODEL_ID` ke folder `merged_model`.
- ⚠️ **Penting Saat Inferensi:** Output logits model WAJIB dibagi dengan Temperature (`T = 3.837`) sebelum di-passing ke fungsi Softmax agar probabilitas terkalibrasi dengan benar.

---

## Appendix: File Structure

```
Google Drive/
└── id-political-sentiment-models/
    ├── sentiment-v1/
    │   ├── lora/
    │   │   ├── adapter_config.json
    │   │   └── adapter_model.safetensors (~4MB)
    │   ├── tokenizer/
    │   │   ├── tokenizer_config.json
    │   │   ├── tokenizer.json
    │   │   └── special_tokens_map.json
    │   ├── merged_model/
    │   │   ├── config.json
    │   │   ├── model.safetensors (~440MB)
    │   │   ├── tokenizer files
    │   │   └── README.md (model card)
    │   ├── metrics.json
    │   └── evaluation.json
    └── relevancy-v1/
        └── (same structure)
```

## Appendix: Cara Pakai Model di Production

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load dari Google Drive / server lokal
model_path = "./models/sentiment-v1/merged_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Predict dengan temperature scaling
entity = "Prabowo Subianto"
context = "Presiden Prabowo menegaskan program ekonomi akan berjalan."
inputs = tokenizer(entity, context, truncation=True, max_length=256, return_tensors="pt")

with torch.no_grad():
    T = 3.837  # WAJIB: dari metrics.json
    probs = torch.softmax(model(**inputs).logits / T, dim=-1)

labels = ["negative", "neutral", "positive"]
pred = labels[probs.argmax()]
conf = probs.max().item()

print(f"Sentiment: {pred} (confidence: {conf:.1%})")

# Defer jika low confidence
if conf < 0.70:
    print("⚠️ Low confidence — defer to LLM/human review")
```
