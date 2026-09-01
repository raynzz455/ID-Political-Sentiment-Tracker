Sudah baca repo-mu lengkap. Ini yang perlu kamu jelaskan ke dosen — terstruktur dari yang paling penting ke detail teknis.

---

## 1. Mulai dari masalah, bukan teknologi

Dosen paling tertarik pada **relevansi dan urgensi**. Buka dengan ini:

> "Tidak ada sistem yang bisa menjawab pertanyaan: bagaimana opini publik terhadap tokoh politik A berubah dari tahun 2022 ke 2024? Media mainstream hanya memberikan snapshot hari ini. Project ini membangun infrastruktur yang bisa menjawab pertanyaan itu secara kuantitatif, berbasis data, dan real-time."

Lalu kaitkan ke konteks Indonesia:
- Pemilu 2024 menghasilkan dinamika politik yang sangat cepat berubah
- Tidak ada platform publik yang tracking sentimen per-tokoh secara longitudinal
- Peneliti dan jurnalis tidak punya tool yang terjangkau untuk ini

---

## 2. Jelaskan constraint sebagai keputusan desain, bukan keterbatasan

Ini yang membedakan project riset dari project mainan. Kamu punya tiga constraint nyata yang memaksa keputusan arsitektur yang menarik:

**Constraint 1 — Free tier only**
Jelaskan ini bukan karena tidak mampu bayar, tapi karena constraint ini membuktikan efisiensi desain. Kalau sistem bisa jalan di free tier, artinya setiap komponen dipilih dengan sangat deliberate.

**Constraint 2 — UU PDP (Perlindungan Data Pribadi)**
Ini yang paling impressive untuk dosen hukum/etika:
> "Sistem dirancang sejak awal untuk tidak menyimpan identitas penulis apapun. Yang disimpan adalah teks artikel dan skor sentimen — bukan siapa yang menulis. Ini diimplementasikan di level database via Row Level Security, bukan hanya di aplikasi."

**Constraint 3 — Skalabilitas tanpa infrastruktur berbayar**
Partitioned table di PostgreSQL untuk time-series, pgmq untuk queue tanpa Redis, GitHub Actions untuk scheduling tanpa dedicated server.

---

## 3. Jelaskan pipeline sebagai narasi, bukan diagram

Diagram boleh ditunjukkan, tapi jelaskan dalam kalimat:

> "Data masuk dari 24 sumber berita secara otomatis setiap 30 menit via Supabase Edge Function yang berjalan di Deno runtime. Setiap artikel di-hash SHA-256 sebelum masuk — kalau artikel yang sama muncul di dua sumber berbeda, sistem hanya simpan satu. Artikel yang lolos dedup masuk ke antrian pgmq, lalu diambil oleh NLP worker yang menjalankan model IndoBERT untuk mengklasifikasi sentimen dan mengidentifikasi tokoh yang disebut. Hasilnya disimpan di tabel terpartisi per bulan — ini penting untuk query time-series yang efisien."

Tekankan tiga keputusan teknis yang non-obvious:
- **Mengapa partitioned table?** Query "sentimen Prabowo sepanjang 2024" hanya scan 12 partisi, bukan seluruh tabel
- **Mengapa pgmq bukan Redis?** Satu dependency lebih sedikit, queue persisten di database yang sama, tidak ada data loss kalau worker crash
- **Mengapa ONNX bukan model langsung dari HuggingFace?** Quantisasi INT8 membuat model 4x lebih kecil dan 3x lebih cepat di CPU tanpa GPU

---

## 4. Model NLP — ini yang paling perlu dijelaskan detail

Dosen pasti fokus di sini karena ini core academic value-nya.

**Pilihan model:**
> "Dipilih IndoBERT karena merupakan BERT pre-trained khusus Bahasa Indonesia oleh peneliti ITTB. Di domain politik Indonesia, IndoBERT mencapai akurasi 78-84% baseline. Untuk mencapai target >90%, model di-fine-tune dengan data berlabel dari domain politik spesifik."

**Kenapa bukan model multilingual umum seperti mBERT atau XLM-R?**
> "Model multilingual membagi kapasitas parameter untuk 100+ bahasa. IndoBERT mengalokasikan seluruh kapasitasnya untuk Bahasa Indonesia, termasuk memahami slang gaul, code-switching Indonesia-Inggris, dan struktur kalimat bahasa daerah yang sering muncul di media sosial."

**Pipeline NLP yang perlu dijelaskan:**
1. Input teks artikel/komentar
2. Tokenisasi dengan tokenizer IndoBERT
3. Forward pass → tiga output sekaligus: skor sentimen, NER tokoh, embedding 768 dimensi
4. Skor disimpan per artikel per tokoh → memungkinkan query "sentimen terhadap tokoh X khusus"
5. Embedding disimpan di pgvector untuk pencarian semantik

**Kenapa embedding disimpan padahal tidak langsung dipakai?**
> "Forward pass sudah terjadi — zero additional cost. Embedding ini akan dipakai untuk tiga hal: mendeteksi artikel duplikat secara semantik meski teksnya berbeda, clustering topik tanpa labeling manual, dan mendeteksi kampanye bot yang post konten mirip dalam waktu singkat."

---

## 5. Keputusan data yang perlu dijelaskan

Ini yang sering dilewatkan mahasiswa tapi dosen sangat appreciate:

**Mengapa RSS bukan scraping langsung?**
> "RSS adalah protokol yang memang dirancang untuk dibaca mesin. Scraping langsung ke HTML site adalah grey area legal dan teknis sangat fragile — satu perubahan template web langsung break semua scraper. RSS memberikan data terstruktur yang stabil."

**Mengapa Google News RSS sebagai proxy?**
> "IP datacenter Supabase diblock oleh Cloudflare yang dipakai Detik dan Kompas. Solusinya: fetch lewat Google News yang sudah index arsip mereka. Google adalah trusted source, tidak pernah diblock siapapun."

**Mengapa perlu data historis?**
> "Sentiment tracker yang hanya punya data hari ini tidak bisa menjawab pertanyaan paling menarik: apakah popularitas tokoh X meningkat setelah kebijakan Y? Apakah ada korelasi antara peristiwa politik dan perubahan sentimen publik? Untuk itu dibutuhkan data minimal 2-3 tahun ke belakang, yang diambil dari GDELT — database global yang sudah meng-index berita Indonesia sejak 2015."

---

## 6. Output yang bisa dijelaskan ke dosen

Tanpa masuk ke implementasi frontend:

| Output | Nilai Akademik |
|---|---|
| **Trend sentimen per tokoh per bulan** | Bisa menjawab hipotesis "apakah sentimen publik terpengaruh oleh peristiwa politik tertentu" |
| **Head-to-head comparison** | Bisa menjawab "apakah kandidat A lebih populer dari B di bulan elektoral" |
| **National mood index per tahun** | Agregasi iklim politik Indonesia — bisa dijadikan variabel dalam penelitian sosial |
| **Semantic search** | "Temukan semua artikel yang membahas kebijakan subsidi bahkan jika kata 'subsidi' tidak muncul" |
| **Anomaly detection** | Deteksi otomatis ketika sentimen tokoh berubah drastis dalam 1 jam — indikator peristiwa penting |
| **Public read API** | Peneliti lain bisa query data tanpa akses langsung ke database |

---

## 7. Yang membuat project ini berbeda dari TA/skripsi biasa

Sampaikan ini dengan percaya diri:

> "Kebanyakan penelitian NLP menggunakan dataset yang sudah ada, melatih model, lalu melaporkan akurasi. Project ini membangun seluruh infrastruktur dari nol — dari pengambilan data, antrian pemrosesan, inferensi model, penyimpanan terstruktur, hingga penyajian. Ini adalah production-grade system, bukan eksperimen Jupyter notebook."

Tiga hal yang jarang ada di project mahasiswa:
- **Deduplication di level kriptografi** — SHA-256 hash sebelum masuk database
- **Compliance by design** — UU PDP diimplementasikan di database layer, bukan aplikasi layer
- **Partitioning strategy** — time-series query yang scalable tanpa perlu ganti database

---

## Urutan presentasi yang disarankan

```
1. Masalah yang diselesaikan (2 menit)
2. Constraint yang dipilih dan mengapa (2 menit)  
3. Arsitektur pipeline — narasi, bukan diagram dulu (3 menit)
4. Model NLP — IndoBERT, mengapa, bagaimana (5 menit)
5. Keputusan data — sumber, historis, GDELT (3 menit)
6. Output yang bisa dilihat (2 menit + demo kalau ada)
7. Apa yang belum selesai dan roadmap (1 menit)
```

Yang terakhir itu penting — dosen sangat respect mahasiswa yang bisa jujur tentang limitasi dan punya rencana jelas untuk mengatasinya, dibanding yang overclaim.