# Upgrade Model Sentimen — Panduan Integrasi

Ganti dummy `predict_sentiment()` dengan model real yang context-aware.

## Mengapa model ini, bukan taufiqdp/indonesian-sentiment langsung

Riset terbaru (SocialX + Telkom University + BRIN, April 2026) menguji 3 model
sentimen Indonesia paling banyak diunduh — semua fine-tuned di dataset review
(SmSA). Hasilnya: akurasi turun ke 59-63% dan F1 pada kelas positif di bawah
0.211 saat dievaluasi di luar domain aslinya.

Project ini mengolah **artikel berita**, bukan review — domain berbeda dari
SmSA. Risiko domain-shift ini nyata.

Model pengganti yang dipilih (`apriandito/indobert-sentiment-classifier`)
didesain context-conditioned: menerima nama entity SEKALIGUS teks artikel,
menghasilkan sentimen yang spesifik untuk entity tersebut — bukan sentimen
dokumen generik yang diterapkan rata ke semua entity yang match.

## Langkah wajib sebelum integrasi

### 1. Install dependency

```powershell
pip install torch transformers --break-system-packages
```

### 2. Jalankan verifikasi manual DULU

```powershell
cd packages/nlp-worker
python test_sentiment_model.py
```

Perhatikan khusus test case #3 (Kapolri "Listyo Sigit Prabowo" dengan context
"Prabowo Subianto") — ini kasus false positive yang sudah ditemukan di regex
matching sebelumnya. Kalau model context-aware bekerja baik, confidence-nya
harus terasa berbeda dibanding test case #1 yang benar-benar tentang Presiden
Prabowo.

**Jangan lanjut ke step 3 kalau hasil test case ini tidak masuk akal.**

### 3. Patch `cli_test.py`

File `sentiment_model.py` TIDAK menggantikan `cli_test.py` — ia berdiri sendiri
sebagai module yang dipanggil. Ubah bagian ini di `cli_test.py`:

**Tambahkan import di bagian atas:**
```python
from sentiment_model import get_predictor
```

**Cari baris di `cmd_sample` dan `cmd_batch` yang memanggil:**
```python
label, conf, scores = predict_sentiment(combined)
```

**Ganti jadi (perhatikan: sekarang butuh context per-entity, dipanggil di
DALAM loop matched entities, bukan sekali per artikel):**
```python
predictor = get_predictor()

# ... di dalam loop `for e in matched:` ...
label, conf, scores = predictor.predict(
    text=text,                    # body artikel asli, BUKAN combined title+text
    context=e["canonical_name"],  # nama entity yang match
)
```

**Perubahan penting:** sebelumnya satu sentimen dihitung sekali per artikel
lalu di-apply ke semua matched entity. Sekarang, karena model context-aware,
sentimen dihitung ULANG per entity (entity berbeda di artikel yang sama bisa
punya sentimen berbeda). Ini lebih lambat (N kali inference untuk N entity
yang match dalam satu artikel) tapi itu justru tujuan upgrade ini — akurasi
dulu, optimasi performa nanti.

**Untuk kasus tanpa entity match (skip+ack di `cmd_batch`):** kalau nanti
ingin tetap menyimpan sentimen dokumen-level untuk artikel non-politik
(opsional, bukan wajib), panggil tanpa context:
```python
label, conf, scores = predictor.predict(text=text, context=None)
```

### 4. Jangan hapus dummy predict_sentiment() dulu

Biarkan fungsi dummy lama tetap ada di file (cukup tidak dipanggil). Kalau
model baru ternyata bermasalah di tengah jalan, kamu bisa rollback cepat
tanpa kehilangan kode lama.

## Yang BELUM dilakukan (sengaja, sesuai keputusan)

- **ONNX quantization** — model ini 335M parameter, 3x lebih besar dari
  IndoBERT-base biasa. Belum dioptimasi karena fokus saat ini akurasi dulu.
  Begitu akurasi confirmed bagus dan mau pindah ke production worker
  (HF Spaces), baru optimize ke ONNX INT8.
- **True batch tensor inference** — `predict_batch()` di `sentiment_model.py`
  saat ini masih loop satu-satu di level Python, bukan true batching di level
  tensor (padding + attention mask). Cukup untuk testing CLI sekarang.
- **Wiring otomatis ke `insert_sentiment_score()` RPC** — biarkan kamu yang
  uji manual hasil prediksinya dulu sebelum confirm tulis ke production DB.

## Bonus yang belum diaktifkan: model relevancy

`apriandito/indobert-relevancy-classifier` (F1 0.948 untuk "apakah teks ini
tentang topik X") berpotensi menggantikan regex alias matching yang sudah
beberapa kali bermasalah dengan false positive (kasus "Prabowo" vs "Listyo
Sigit Prabowo"). Belum diimplementasi di paket ini — simpan sebagai langkah
upgrade berikutnya setelah model sentimen ini stabil.

## Catatan kejujuran ilmiah

Model primary berasal dari satu paper riset (download masih ~336x di
HuggingFace, belum widely-validated seperti model SmSA yang sudah dipakai
puluhan ribu kali). Arah temuannya (model umum gagal out-of-domain) sangat
masuk akal secara ML, tapi tetap perlu divalidasi dengan mata sendiri lewat
`test_sentiment_model.py` sebelum dipercaya penuh untuk production.
