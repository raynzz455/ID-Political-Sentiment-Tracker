# Roadmap Peningkatan Kualitas Model — V2 → V3 → V4

## Status Saat Ini (V2)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Accuracy | 67-71% | 85%+ | ❌ |
| Macro-F1 | 0.60-0.64 | 0.80+ | ❌ |
| ECE | 0.13 | 0.15 | ✅ |
| Deferral rate | 30-40% | <15% | ❌ |
| Unique rows | 376 | 2000+ | ❌ |

### Masalah Utama
1. **Negative class: hanya 50 unique rows** — terlalu sedikit untuk belajar
2. **Positive class: hanya 67 unique rows** — sama, terlalu sedikit
3. **64% label masih heuristic** — label noise tinggi
4. **Context masih v17** — patch v18.1 belum deploy

---

## Tahap 3A: LLM Hybrid Pipeline (1 jam — PALING CEPAT)

**Target: 90%+ akurasi gabungan**

Script sudah ada: `finetuning/scripts/tahap2_llm_hybrid_pipeline.py`

### Cara Kerja:
```
Input artikel → Model V2 predict
                   ↓
           confidence >= 0.70?
              ↓              ↓
           YES            NO (DEFER)
              ↓              ↓
         Pakai model     LLM second-pass
         (cepat, 0.1s)   (akurat, 3s)
              ↓              ↓
              Combine → 90%+ accuracy
```

### Yang Perlu Dilakukan:
1. Copy model V2 dari Google Drive ke server production
2. Update `packages/nlp/sentiment_model.py`:
   ```python
   SENTIMENT_MODEL_ID = "./models/sentiment-v2/merged_model"
   ```
3. Implementasi LLM second-pass di `nlp_worker.py` untuk DEFER cases
4. Test: run 50 artikel, cek akurasi gabungan

### Estimasi Hasil:
- 70% artikel: model handle (cepat, gratis)
- 30% artikel: LLM handle (akurat, ada cost)
- Combined: 90%+ accuracy

---

## Tahap 3B: Tambah Data dari Production (2-3 jam)

**Target: 1500+ unique rows**

Patch v15 sudah deploy, pipeline sedang jalan. Setelah pipeline selesai:

### Yang Perlu Dilakukan:
1. **Tunggu pipeline process 500+ artikel baru** (sudah jalan via GitHub Actions)
2. **Export data dari Supabase:**
   ```sql
   SELECT rc.raw_text_id, rc.entity_id, rc.context_text, 
          pe.canonical_name, pe.aliases, pe.era, pe.party_affiliation,
          rt.title, rt.text, rt.source_url
   FROM entity_contexts rc
   JOIN political_entities pe ON pe.id = rc.entity_id
   JOIN raw_texts rt ON rt.id = rc.raw_text_id
   WHERE rc.context_version = 'v17_threaded_gpu'
   AND rt.entity_resolved_at IS NOT NULL
   LIMIT 1000;
   ```
3. **Label via LLM second-pass** (pakai `llm_verify_all.py`)
4. **Rebuild dataset v3** (pakai `build_enhanced_dataset.py`)
5. **Finetune V3** di Colab (15 menit)

### Estimasi Hasil:
- 376 unique → 1500+ unique rows
- F1: 0.64 → 0.72-0.78
- Deferral: 35% → 25%

---

## Tahap 3C: Active Learning (1-2 minggu)

**Target: 2000+ gold_human labels, F1 0.80+**

### Yang Perlu Dilakukan:
1. **Kumpulkan DEFER cases dari production** — artikel yang model V2 tidak confident
2. **Label manual** 500-1000 rows, fokus pada:
   - Negative class (butuh 500+ unique)
   - Positive class (butuh 500+ unique)
3. **Format labeling:**
   ```json
   {"row_index": N, "entity_name": "...", "gold_label": "negative", 
    "reasoning": "Entity dikritik karena...", "label_source": "gold_human"}
   ```
4. **Retrain V3** dengan 2000+ rows

### Estimasi Hasil:
- F1: 0.72 → 0.80-0.85
- Deferral: 25% → 15%
- Accuracy standalone: 80-85%

---

## Tahap 3D: Architecture Upgrade (R&D, opsional)

**Target: F1 0.85+ standalone**

### Opsi:
1. **IndoBERT-large** (jika tersedia di HuggingFace)
   - 335M params (vs 110M BERT-base)
   - Lebih banyak kapasitas untuk pattern kompleks
   - Butuh GPU dengan 8GB+ VRAM

2. **Cross-encoder** (bukan sentence-pair)
   - Format: `[CLS] entity [SEP] context [SEP]`
   - Lebih powerful untuk klasifikasi pairwise
   - Tapi butuh data lebih banyak

3. **Ensemble** (relevancy + sentiment + fallback)
   - Vote dari 3 model
   - Lebih robust untuk edge cases
   - Tapi 3x inference time

---

## Prioritas Rekomendasi

### Sekarang (minggu ini):
```
1. Run Tahap 3A: LLM Hybrid Pipeline → 90%+ accuracy
   - Script sudah ada
   - Tinggal implementasi di production
   - Hasil langsung terlihat
```

### Minggu depan:
```
2. Run Tahap 3B: Export data baru dari Supabase
   - Pipeline sudah jalan (v15 deploy)
   - Tunggu 500+ artikel processed
   - Label via LLM
   - Retrain V3 → F1 0.72+
```

### Bulan depan:
```
3. Run Tahap 3C: Active Learning
   - Kumpulkan DEFER cases dari production
   - Label manual 500 rows
   - Retrain V3 → F1 0.80+
```

### Q3 2026 (opsional):
```
4. Run Tahap 3D: Architecture upgrade
   - Coba IndoBERT-large
   - Atau cross-encoder
   - Target F1 0.85+ standalone
```

---

## Decision Matrix

| Tahap | Effort | Target Accuracy | Cost | Dependency |
|-------|--------|---------------|------|------------|
| 3A (LLM Hybrid) | 1 jam | 90%+ | LLM API cost | Model V2 ready |
| 3B (New Data) | 2-3 jam | 72-78% | Gratis | Pipeline v15 jalan |
| 3C (Active Learning) | 1-2 minggu | 80-85% | Manual labor | Tahap 3A jalan |
| 3D (Architecture) | 1 bulan | 85%+ | GPU compute | Tahap 3C selesai |

## Recommended Path

```
V2 (done) → 3A (LLM hybrid) → 3B (new data) → 3C (active learning) → 3D (architecture)
   67%         90%+               72-78%          80-85%                85%+
```

**Mulai dari 3A dulu** — paling cepat, hasil langsung 90%+. 
Kalau sudah jalan di production, baru kerjakan 3B dan 3C bertahap.
