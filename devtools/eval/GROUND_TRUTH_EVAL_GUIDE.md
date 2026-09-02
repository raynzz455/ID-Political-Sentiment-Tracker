# Ground Truth Evaluation — Panduan Lengkap

Urutan kerja: bersihkan kontaminasi → perbaiki logic NULL → kumpulkan data
bersih → evaluasi statistik. JANGAN lompat ke evaluasi sebelum 2 langkah
pertama selesai, atau angkanya tidak bisa dipercaya.

## Langkah 0 — Diagnosa kontaminasi dummy vs real (WAJIB paling dulu)

Jalankan di Supabase SQL Editor:

```sql
SELECT confidence, COUNT(*) AS jumlah
FROM sentiment_scores
GROUP BY confidence
ORDER BY jumlah DESC
LIMIT 20;
```

Dummy model lama hardcode confidence persis `0.65` (untuk label positive/
negative dominant) atau `0.60` (neutral dominant) — lihat kode lama di
`predict_sentiment()`. Kalau ada cluster besar di angka itu, itu sisa dummy.

```sql
TRUNCATE TABLE sentiment_scores;
```

Aman dijalankan — `raw_texts` (960+ pending) tidak tersentuh, tinggal
diproses ulang dengan pipeline yang benar.

## Langkah 1 — Patch `cli_test.py` (logic NULL + model_version)

Lihat detail lengkap di `INTEGRATION_GUIDE.md` (sudah ada sebelumnya) untuk
wiring relevancy gate. Tambahan untuk sesi ini:

```python
predictor = get_pipeline()

# A. SELALU hitung document-level untuk national index
#    (entity_id=NULL, ini BUKAN sampah -- untuk mv_national_monthly_summary)
fallback_result = predictor.predict_gated(text=text, context=None)
sb.rpc("insert_sentiment_score", {
    "p_raw_text_id": raw_id,
    "p_entity_id": None,
    "p_label": fallback_result.label,
    "p_neg": float(fallback_result.scores[0]),
    "p_neu": float(fallback_result.scores[1]),
    "p_pos": float(fallback_result.scores[2]),
    "p_confidence": float(fallback_result.sentiment_confidence),
    "p_model_version": "indobert-fallback-v1",   # <-- TAG EKSPLISIT
}).execute()

# B. Untuk SETIAP entity yang match via alias regex, cek relevancy
for e in matched:
    result = predictor.predict_gated(text=text, context=e["canonical_name"])

    if not result.is_relevant:
        print(f"       -> SKIP {e['canonical_name']}: relevancy={result.relevancy_confidence:.3f}")
        continue  # TIDAK insert apa pun untuk entity ini

    sb.rpc("insert_sentiment_score", {
        "p_raw_text_id": raw_id,
        "p_entity_id": e["id"],
        "p_label": result.label,
        "p_neg": float(result.scores[0]),
        "p_neu": float(result.scores[1]),
        "p_pos": float(result.scores[2]),
        "p_confidence": float(result.sentiment_confidence),
        "p_model_version": "indobert-ctx-relevancy-gated-v1",   # <-- TAG BEDA
    }).execute()
```

Catatan performa: ini berarti setiap artikel sekarang minimal 1 inference
call (fallback) + 2 call per entity kandidat (relevancy + sentiment).
Lebih berat, tapi sesuai prioritas yang sudah disepakati (akurasi dulu).

## Langkah 2 — Jalankan batch untuk kumpulkan data bersih

```powershell
python cli_test.py batch 200
```

Ulangi beberapa kali (queue 960+ pending) sampai punya cukup data untuk
sampling ground truth — target minimal 300-500 sentiment_scores baru
dengan `model_version` yang sudah ter-tag jelas.

## Langkah 3 — Export sample untuk dilabeli manual

```powershell
pip install scikit-learn pandas --break-system-packages

# Stage 2 (sentiment) -- dari sentiment_scores yang sudah bersih
python export_sentiment_ground_truth.py --n 150 --model-version indobert-ctx-relevancy-gated-v1

# Stage 1 (relevancy) -- scan ulang raw_texts, termasuk yang DITOLAK gate
python export_relevancy_review.py --n 300 --max-candidates 150
```

## Langkah 4 — Labeli manual (yang paling penting di seluruh proses ini)

Buka kedua file CSV di Excel/Google Sheets.

**Untuk `relevancy_ground_truth_TEMPLATE.csv`:** baca `text_preview`,
tentukan apakah teks itu BENAR tentang `entity_candidate` yang disebut
(bukan orang lain dengan nama mirip). Isi `gold_relevant`: `yes` / `no`.
**Prioritaskan baris dengan `gate_decision=not_relevant`** — itu yang
paling penting diverifikasi, soalnya itu klaim utama yang perlu dibuktikan.

**Untuk `sentiment_ground_truth_TEMPLATE.csv`:** baca teks, tentukan
sentimen SEBENARNYA terhadap entity yang disebut. Isi `gold_label`:
`negative` / `neutral` / `positive`.

Target realistis: 150-200 baris per file untuk metrik yang cukup stabil
secara statistik (bandingkan dengan n=13 di laporan Gemini yang terlalu
kecil untuk disimpulkan apa pun).

## Langkah 5 — Hitung metrik

```powershell
python eval_metrics.py \
  --relevancy relevancy_ground_truth_TEMPLATE.csv \
  --sentiment sentiment_ground_truth_TEMPLATE.csv
```

Output: precision/recall/F1 per kelas, confusion matrix, DAN calibration
check (apakah confidence tinggi benar-benar berkorelasi akurasi tinggi --
ini yang menjawab kritik "ilusi high confidence" dengan data nyata, bukan
spekulasi).

## Cara baca hasil

```
F1 tinggi di kelas relevant, F1 rendah di not_relevant
  -> gate cenderung "permisif", banyak false positive lolos
  -> pertimbangkan naikkan RELEVANCY_THRESHOLD di sentiment_model.py

Akurasi sentiment < 70%
  -> model context-conditioned mungkin perlu fine-tuning tambahan,
     bukan dipakai langsung apa adanya

Calibration check: confidence >=0.95 tapi akurasi aktual <80%
  -> model OVERCONFIDENT, jangan percaya confidence mentah untuk
     keputusan downstream (misal threshold filter di dashboard nanti)
```

## Setelah evaluasi selesai

Baru kembali ke agenda yang sempat ditunda: ekspansi entitas dan
pengumpulan data historis. Urutan ini disengaja — percuma menambah lebih
banyak data/entitas kalau model yang memprosesnya belum tervalidasi
akurasinya dengan benar.
