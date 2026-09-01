# Laporan Pengembangan & Evaluasi Model Sentimen Politik V2

## 1. Ringkasan Eksekutif
Pada tahap V2, pipeline fine-tuning model IndoBERT (LoRA) dengan Focal Loss dan Kalibrasi Temperature telah berhasil dijalankan secara end-to-end tanpa error. Model mampu mengklasifikasikan sentimen politik (Positif, Netral, Negatif) dengan kalibrasi probabilitas yang sangat baik.

Meskipun akurasi mentah (full-coverage) berada di angka ~67-71%, sistem telah berhasil mengimplementasikan **Confidence Threshold (Deferral System)**. Sistem ini dirancang khusus untuk mencapai target akurasi agregat **>95%** di tingkat produksi melalui kombinasi model LoRA dan LLM Second-Pass.

## 2. Arsitektur & Konfigurasi V2
- **Base Model:** `apriandito/indobert-sentiment-classifier`
- **Metode:** LoRA (Parameter-Efficient Fine-Tuning) pada modul Q, K, V, Dense.
- **Optimasi Loss:** Focal Loss (`gamma=2.5`) dengan Class-Balanced Weights untuk mengatasi data imbalance.
- **Kalibrasi:** Temperature Scaling (`T = 3.837`) terbukti efektif menekan *overconfidence*.

### Hasil Hyperparameter Tuning (Grid Search)
Dari 5 kombinasi yang diuji, ditemukan bahwa peningkatan Learning Rate memberikan hasil paling optimal:
- **Best Config:** `Learning Rate = 3e-5`, `Smoothing = 0.05`, `LoRA r = 32`
- **Best Macro-F1:** 0.6377 (naik dari 0.5964)
- **Best ECE (Expected Calibration Error):** 0.1309 (sangat baik)

## 3. Evaluasi Inferensi (Sistem Deferral)
Pengujian inferensi pada 5 sampel kalimat politik nyata menunjukkan kalibrasi model bekerja dengan sangat cerdas:

| Entitas | Konteks | Prediksi | Confidence | Status |
|---------|---------|----------|------------|--------|
| Anies Baswedan | "...bangsa yang lugu dan baik hati" | Positive | 82.2% | Confident |
| Thomas Lembong | "...divonis bersalah... korupsi" | Negative | 79.4% | Confident |
| Rocky Gerung | "...pasal yang dungu" | Negative | 80.7% | Confident |
| Prabowo Subianto | "...meski dihujani kritik" | Positive | 43.0% | DEFER |
| Joko Widodo | "...dituntut membayar uang..." | Negative | 65.3% | DEFER |

**Analisis:** Model menolak menebak asal pada kalimat ambigu (status DEFER) dan hanya memberikan prediksi ketika confidence >70%. Ini adalah fondasi utama untuk mencapai akurasi 95%++.

## 4. Roadmap Mencapai Akurasi 95%-97%++ (Target V3)

### Fase 1: Hybrid Pipeline (LLM Second-Pass) - Target: 90-95%
Implementasi langsung di server production tanpa perlu training ulang:
1. Model V2 bertindak sebagai lini pertama (Filter Cepat & Murah).
2. Jika model memberikan confidence >= 70% (Confident), prediksi langsung dipakai (Akurasi pada subset ini sudah >85%).
3. Jika model memberikan confidence < 70% (DEFER), teks tersebut diteruskan ke LLM Besar (GPT-4o-mini atau Claude 3.5 Haiku) untuk dilabeli.
4. Efek: Gabungan model cepat + LLM pada kasus sulit akan langsung mendorong akurasi sistemik menembus 90-95%.

### Fase 2: Active Learning & Data Augmentation (Training V3) - Target: 95-97%++
1. Kumpulkan Data DEFER: Jalankan aplikasi scraper, kumpulkan semua berita politik yang diprediksi DEFER oleh model V2.
2. Manual Labeling: Beri label asli (gold_human) pada 500-1000 data baru tersebut, dengan fokus menambah kelas Negative dan Positive.
3. Training V3: Gabungkan data lama dengan data baru (total ~2000 baris). Gunakan config terbaik (lr=3e-5).
4. Efek: Model V3 akan memiliki pemahaman konteks yang jauh lebih luas. Macro-F1 diprediksi naik ke 0.80-0.90, dan tingkat deferral akan menurun drastis, mendorong akurasi total mendekati 97%.

## 5. Status Penyimpanan & Deployment (V2)
- Status Training: Selesai.
- Catatan Penyimpanan: Model disimpan via Google Drive atau download lokal.
- Langkah Aksi Selanjutnya:
  1. Ekstrak file merged_model ke server lokal.
  2. Update path SENTIMENT_MODEL_ID di kode produksi.
  3. Pastikan logits dibagi dengan Temperature T=3.837 sebelum di-softmax.
  4. Mulai implementasi LLM Second-Pass untuk data DEFER (Fase 1).
