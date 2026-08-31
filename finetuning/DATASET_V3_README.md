# Dataset v3 — LLM-Verified Finetuning Dataset

> Generated: 2026-08-16
> Source: `dataset_v2.jsonl` (777 rows) + LLM verification of 149 low-confidence unique base rows
> Final: `datasets/dataset_v3.jsonl` (777 rows, 748 sentiment-relevant, **77.9% LLM-verified >= 0.85 confidence**)

---

## 1. Apa yang ditingkatkan dari v2 → v3

| Metric                                   | v2          | v3 (final)        | Improvement                       |
|------------------------------------------|-------------|-------------------|-----------------------------------|
| High-confidence (>=0.85) rows            | 36.4%       | **77.9%**         | +41.5 pp (2x lipat)               |
| Low-confidence (<0.55) rows              | 26.8%       | **1.9%**          | -24.9 pp (14x lebih sedikit)      |
| LLM-verified label source                 | 5.4%        | **45.7%**         | +40.3 pp                          |
| Rows excluded as `not_relevant`           | 0           | **29**            | LLM deteksi background mentions   |
| Label flips (heuristic salah → LLM benar)| -           | **186**           | Speaker-vs-target confusion fixed |
| Label confirms (LLM setuju dgn heuristic)| -           | **114**           | Heuristics validated              |

**Label distribution (sentiment-relevant, 748 rows):**
- positive: 227 (30.3%)
- neutral:  383 (51.2%)
- negative: 138 (18.4%)

Distribusi ini **lebih realistis** daripada v2 yang artificial 33/33/33 (hasil oversampling paksa). Neutral tetap dominan (wajar untuk berita politik), tapi minor class (negative 138) sudah cukup untuk training LoRA.

---

## 2. File yang dihasilkan

| File | Deskripsi |
|------|-----------|
| `scripts/verify_dataset_v2.mjs` | Script verifikasi LLM (z-ai-web-dev-sdk, batch 5, resume, rate-limit handling) |
| `scripts/build_dataset_v3.py`   | Script merge: v2 + LLM-verified → dataset_v3.jsonl |
| `llm_verified_v3.jsonl`         | 135 label LLM-verified (1 baris per base row_index) |
| `verify_v3_report.json`         | Statistik verifikasi (flip rate, distribusi label) |
| `datasets/dataset_v3.jsonl`     | **Dataset final untuk finetuning** |
| `dataset_v3_report.json`        | Statistik final dataset v3 |

---

## 3. Cara menjalankan verifikasi (dari awal)

```bash
cd /home/z/my-project

# STEP 1: Verifikasi LLM baris berconfidence rendah dari dataset_v2
# - Opsi --limit N: hanya verifikasi N baris (untuk test)
# - Opsi --batch N: ukuran batch (default 5, turunkan ke 3 jika sering 429)
# - Opsi --delay MS: jeda antar batch ms (default 2500, naikkan ke 4000 jika 429)
# - Resume otomatis: baris yang sudah diverify di llm_verified_v3.jsonl di-skip
node finetuning/scripts/verify_dataset_v2.mjs --batch 5 --delay 2500

# Jika ada baris yang gagal (429 rate-limit), ulangi sampai semua selesai:
node finetuning/scripts/verify_dataset_v2.mjs --batch 1 --delay 4000

# STEP 2: Build dataset_v3 dari hasil verifikasi
python3 finetuning/scripts/build_dataset_v3.py

# STEP 3: Cek laporan final
cat finetuning/dataset_v3_report.json
cat finetuning/verify_v3_report.json
```

---

## 4. Cara menggunakan dataset_v3 untuk finetuning

`finetune.py` **sudah di-update** untuk default pakai `dataset_v3.jsonl`:

```bash
cd /home/z/my-project/finetuning

# Finetune sentiment model (di GPU, ~25 menit di Colab T4)
python finetune.py --task sentiment

# Finetune relevancy model
python finetune.py --task relevancy

# Evaluasi + confidence threshold sweep (untuk deferral 97% kept-accuracy)
python evaluate.py --task sentiment
```

**Yang berbeda dari training v1:**
- 77.9% baris punya confidence >= 0.85 → sample weighting FocalLossTrainer akan memberi bobot lebih tinggi ke baris terverifikasi
- 29 baris `not_relevant` di-exclude otomatis dari training sentiment (filter `gold_relevancy == "relevant"`)
- 14 baris `llm_verify_pending` (confidence 0.45) akan di-down-weight → tidak merusak training

---

## 5. Quality highlights — contoh perbaikan label

### Case 1: Speaker-vs-target confusion (Prabowo)
- **v2:** `negative` (heuristic, conf=0.45) — salah karena heuristic ketemu kata "disalahgunakan"
- **v3:** `neutral` (LLM, conf=0.85) — BENAR, Prabowo adalah pembicara yang menyatakan sesuatu
- **Reasoning:** "Prabowo sebagai pembicara, bukan target sentimen."

### Case 2: Background mention (Rizal Ramli)
- **v2:** relevant, ikut training sentiment
- **v3:** `not_relevant` — LLM deteksi Rizal Ramli hanya disebut di latar ("menjenguk SBY")
- **Reasoning:** "Entitas hanya disebut sebagai bagian dari konteks latar."
- Baris ini di-exclude dari training sentiment → mengurangi noise.

### Case 3: Polarity flip (Fadli Zon)
- **v2:** `neutral` (heuristic speaker_upgraded)
- **v3:** `positive` (LLM) — BENAR, Fadli dipuji karena inisiatif revitalisasi Keraton
- **Reasoning:** "Fadli dipuji karena inisiatif revitalisasi budaya."

---

## 6. Status & sisa pekerjaan

| Status | Detail |
|--------|--------|
| ✅ Verifikasi LLM | 135 / 149 unique base rows (90.6%) |
| ⚠️ Pending | 14 baris gagal karena rate-limit 429. Label heuristic dipertahankan, confidence diturunkan ke 0.45. Bisa diulang nanti saat quota reset. |
| ✅ Dataset v3 siap | 777 rows, 748 sentiment-relevant, 77.9% high-confidence |
| ⏳ Finetuning | Belum dijalankan (butuh GPU). Script siap: `python finetune.py --task sentiment` |
| ⏳ Eval + calibration | Belum dijalankan. Script siap: `python evaluate.py` |

**Next steps:**
1. Jalankan `python finetune.py --task sentiment` di Colab GPU (T4 cukup, ~25 menit)
2. Jalankan `python evaluate.py` untuk dapat confidence threshold optimal (target >=97% kept-accuracy)
3. Jalankan `tahap2_llm_hybrid_pipeline.py` untuk hybrid model+LLM (target 90%+)
4. Upload model ke HuggingFace via `upload_huggingface.py`
