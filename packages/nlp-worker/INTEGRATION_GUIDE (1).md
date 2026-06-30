# Upgrade Model Sentimen — 2-Stage Pipeline (v2)

## Perubahan dari v1

v1 cuma pakai SentimentModel sendirian dan salah desain test — confidence
sentimen dikira bisa mendeteksi entity mismatch (Kapolri vs Presiden Prabowo).
Itu keliru: model sentimen menjawab "apa sentimennya", bukan "apakah context
ini relevan". Dua task berbeda, dua model berbeda.

v2 memisahkan jadi 2 stage yang benar:

```
Stage 1 (RelevancyModel) -> "apakah teks ini tentang entity X?"
         |
   TIDAK relevan -> skip, jangan hitung sentimen sama sekali
   RELEVAN       -> lanjut ke Stage 2
         |
Stage 2 (SentimentModel) -> "apa sentimen teks ini terhadap entity X?"
```

Ini sekaligus menyelesaikan masalah false-positive entity matching yang sudah
beberapa kali muncul di project ini (alias "Prabowo" match Kapolri Listyo
Sigit Prabowo, alias "RK"/"Emil" match artikel kriminal tidak terkait Ridwan
Kamil) — masalah itu sekarang ditangani di level model, bukan cuma di level
regex/alias curation.

## Langkah wajib

### 1. Install dependency (sama seperti v1, tidak ada tambahan)

```powershell
pip install torch transformers --break-system-packages
```

### 2. Jalankan test relevancy gate

```powershell
cd packages/nlp-worker
python test_sentiment_model.py
```

Perhatikan baris `HASIL RELEVANCY GATE: X/Y sesuai ekspektasi` di akhir.
Test case #3 (Kapolri) dan #6 (mayat Bandara vs Ridwan Kamil) adalah test
yang SEBENARNYA relevan untuk masalah false-positive — bukan test case #3
di v1 yang salah desain.

**Kalau hasil gate tidak sesuai ekspektasi (model bilang relevan padahal
seharusnya tidak, atau sebaliknya):** kemungkinan `id2label` model terbalik
dari asumsi. Cek log `[INFO] -> loaded. id2label = {...}` saat model
pertama kali load — sesuaikan `RELEVANT_LABEL_HINTS` di `sentiment_model.py`
kalau labelnya tidak match daftar hint yang sudah disediakan.

**Jangan lanjut ke step 3 sebelum semua/mayoritas kasus gate sesuai ekspektasi.**

### 3. Patch `cli_test.py`

Tambahkan import:
```python
from sentiment_model import get_pipeline
```

Di `cmd_sample` dan `cmd_batch`, ganti logic lama (predict sekali per artikel,
lalu apply ke semua matched entity) dengan loop per-entity yang di-gate:

```python
pipeline = get_pipeline()

# ... di dalam loop `for e in matched:` ...
result = pipeline.predict_gated(text=text, context=e["canonical_name"])

if not result.is_relevant:
    print(f"       -> SKIP {e['canonical_name']}: tidak relevan "
          f"(confidence={result.relevancy_confidence:.3f})")
    continue   # JANGAN insert_sentiment_score untuk entity ini

# Hanya insert kalau relevan
sb.rpc("insert_sentiment_score", {
    "p_raw_text_id": raw_id,
    "p_entity_id": e["id"],
    "p_label": result.label,
    "p_neg": float(result.scores[0]),
    "p_neu": float(result.scores[1]),
    "p_pos": float(result.scores[2]),
    "p_confidence": float(result.sentiment_confidence),
}).execute()
print(f"       -> inserted score for {e['canonical_name']} "
      f"(relevancy={result.relevancy_confidence:.3f})")
```

Untuk artikel tanpa entity match (skip+ack), tidak berubah dari sebelumnya.

### 4. Konsekuensi performa (sesuai keputusan: akurasi dulu)

Setiap entity yang match sekarang butuh 2 inference call (relevancy +
sentiment), bukan 1. Untuk artikel dengan 2-3 entity match, itu 4-6 call.
Lebih lambat dari v1, jauh lebih lambat dari dummy. Ini trade-off yang
sudah disepakati secara eksplisit (akurasi diutamakan, optimasi performa
menyusul lewat ONNX quantization setelah akurasi tervalidasi).

### 5. Dampak ke entity match rate

Ekspektasi realistis: relevancy gate akan MENGURANGI jumlah sentiment_scores
yang ter-insert dibanding sebelumnya (karena sekarang false-positive di-skip),
tapi MENINGKATKAN kualitas data yang tersisa (semua yang ter-insert benar-benar
tentang entity tersebut). Jangan kaget kalau angka total sentiment_scores
turun setelah upgrade ini — itu tanda gate bekerja, bukan regresi.

## Yang belum dilakukan (tetap sesuai rencana sebelumnya)

- ONNX quantization (2 model x 335M = lebih berat dari sebelumnya, makin
  relevan untuk dioptimasi nanti, tapi tetap belakangan)
- True batch tensor inference
- RELEVANCY_THRESHOLD masih default 0.5 -- mungkin perlu tuning setelah
  lihat distribusi confidence di data nyata (kalau banyak kasus borderline
  0.45-0.55, pertimbangkan naikkan threshold ke 0.6-0.7 untuk lebih konservatif)
