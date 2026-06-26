# ID Political Sentiment Tracker

> Basically ini kayak "termometer digital" buat ngukur suhu publik terhadap tokoh politik Indonesia — positif, netral, atau negatif — otomatis, real-time, dan gratis total.

**Status:** 🚧 Work in progress — dibangun layer by layer  
**Budget:** Rp 0 (literally free tier semua)  
**Vibe:** Iseng tapi serius

---

## 🤔 Ini apaan sih sebenarnya?

Pernah nggak ngerasa bingung: "sebenernya orang-orang suka nggak sih sama si X?" — tapi daripada baca ratusan artikel satu-satu, gimana kalau ada yang lakuin itu otomatis terus hasilnya ditampilkan dalam bentuk grafik yang gampang dibaca?

Nah, **itu persis yang project ini lakuin.**

Setiap 30 menit, sistem ini otomatis:
1. 📰 Ngumpulin ribuan berita dari media Indonesia
2. 🧠 Baca dan analisis sentimen tiap artikel pakai AI
3. 📊 Tampilin hasilnya di dashboard publik yang bisa diakses siapa aja

Output-nya simpel: **"Minggu ini, 68% berita tentang si X bernada negatif."**

---

## 🎯 Kenapa dibuat?

Real talk — informasi politik di Indonesia itu banyak banget tapi berserakan. Lu harus buka Detik, Kompas, CNN Indonesia, Republika satu-satu. Belum lagi lu nggak tau mana yang objektif mana yang udah di-framing.

Project ini nggak nyari mana yang bener atau salah — dia cuma ngukur: **dari semua yang ditulis media, tone-nya ke tokoh X lagi ke arah mana?**

Berguna buat:
- Mahasiswa yang lagi riset opini publik
- Yang penasaran tapi males baca berita seharian
- Siapapun yang mau lihat tren sentimen politik tanpa perlu jadi analis

---

## 🏗️ Cara Kerjanya — Dijelasin Kayak Ngobrol Sama Temen

Bayangin sistem ini kayak **dapur restoran**, dari bahan mentah sampai makanan nyampe ke meja tamu:

```
Berita dari internet
      ↓
  Dikumpulin          ← L1 + L2
      ↓
  Diantrekan          ← L3
      ↓
  Dianalisis AI       ← L4
      ↓
  Disimpan rapi       ← L5
      ↓
  Ditampilin ke lu    ← L6
```

---

## ⚙️ Tech Stack — Layer by Layer

### 🌐 Layer 1 — Sumber Data
**Apa yang terjadi:** Ini titik awal — dari mana beritanya diambil.

Pakai **RSS Feed** dari berbagai media Indonesia. RSS itu basically format khusus yang memang dibuat supaya mesin bisa baca berita otomatis — jadi ini bukan scraping ilegal, ini fitur yang memang disediain media buat agregator.

**Sumber yang dipake:**
- CNN Indonesia, Tempo, Republika (direct RSS)
- Detik, Kompas, Tribun, Antara (lewat Google News RSS — karena kalo langsung diblock)

**Analogi:** Ini kayak lu subscribe newsletter dari banyak media sekaligus, terus isinya masuk ke satu kotak.

---

### ⚡ Layer 2 — Ingestion Worker (Deno / TypeScript)
**Apa yang terjadi:** Yang bertugas pergi "belanja" berita tadi dan masukin ke sistem.

Dibangun pakai **Deno** (bahasa TypeScript yang jalan di server), di-deploy ke **Supabase Edge Functions** yang gratis.

**Teknisnya:** Tiap 30 menit, GitHub Actions ngirim sinyal "hei, kerja!" ke Edge Function ini. Terus dia langsung:
- Fetch semua RSS feed secara bersamaan (bukan satu-satu, jadi cepet)
- Bersihin kontennya dari HTML, iklan, dan sampah digital lainnya
- Cek duplikat pakai SHA-256 hash (biar artikel yang sama nggak masuk dua kali)
- Masukin ke database

**Analogi:** Ini kayak kurir yang tiap 30 menit keliling ngumpulin koran dari berbagai kios, terus anter ke gudang — dan dia hafal mana yang udah pernah diantar jadi nggak perlu dobel.

**Kenapa Deno, bukan Python?**  
Edge Function punya batas CPU 150ms. Kerjaan Layer 2 ini mostly nunggu internet (network I/O) — bukan ngitung berat-berat — jadi Deno/TypeScript yang ringan udah lebih dari cukup. Python disimpen buat Layer 4 yang emang butuh komputasi berat.

---

### 📬 Layer 3 — Antrian (pgmq)
**Apa yang terjadi:** Artikel yang baru masuk dimasukin ke antrian biar diproses tertib.

Pakai **pgmq** — ini ekstensi PostgreSQL yang bikin sistem antrian langsung di dalam database. Nggak perlu server Redis atau Kafka yang ribet dan bayar.

**Analogi:** Bayangin antrian kasir Indomaret. Artikel-artikel yang baru dateng ngantri dulu. Nanti si kasir (NLP worker) ambil satu-satu, proses, terus lanjut ke artikel berikutnya. Kalau gagal diproses, otomatis masuk antrian lagi — nggak ilang.

**Kenapa ini penting?**  
Tanpa antrian, kalau tiba-tiba ada 500 artikel masuk sekaligus, sistem bisa crash. Dengan antrian, semuanya diproses pelan-pelan tapi pasti.

---

### 🧠 Layer 4 — NLP Worker (Python + IndoBERT)
**Apa yang terjadi:** Ini bagian paling "otak"-nya — AI yang baca artikel dan mutusin sentimen-nya.

Ini layer paling keren sekaligus paling berat secara komputasi.

**Tech yang dipake:**
- **Python** — bahasa standar untuk AI/ML
- **IndoBERT** — model AI yang dilatih khusus pakai teks Bahasa Indonesia (bukan English AI yang dipaksa baca Indo)
- **ONNX Runtime** — versi IndoBERT yang udah "dipress" jadi lebih ringan, bisa jalan di CPU biasa tanpa GPU
- **Hugging Face Spaces** — hosting gratis buat AI worker ini

**Yang dikerjain per artikel:**
1. Baca teks artikel
2. Deteksi nama tokoh yang disebut (Prabowo? Sri Mulyani? Anies?)
3. Analisis tone-nya: positif / netral / negatif
4. Kasih confidence score (seberapa yakin AI-nya)
5. Buat "sidik jari" artikel (embedding 768 angka) buat keperluan pencarian semantik nanti

**Analogi:** Ini kayak ada editor politik senior yang udah baca jutaan artikel Indo, terus kerjanya nonstop baca artikel baru dan nulis laporan singkat: "Artikel ini ngomongin Prabowo, tone-nya negatif, confidence 87%."

**Kenapa IndoBERT, bukan ChatGPT/GPT-4?**  
Dua alasan: pertama, gratis total. Kedua, IndoBERT emang dilatih khusus Bahasa Indonesia termasuk slang dan konteks politik lokal — lebih akurat untuk use case ini dibanding model general yang dilatih dominan pakai teks Inggris.

---

### 🗄️ Layer 5 — Database (Supabase / PostgreSQL)
**Apa yang terjadi:** Gudang utama tempat semua data disimpan rapi dan terorganisir.

Ini tulang punggung seluruh sistem. Pakai **Supabase** yang di baliknya pakai **PostgreSQL** — database yang udah terbukti battle-tested selama puluhan tahun.

**Yang bikin special di project ini:**

**Partitioning bulanan** — Data artikel dibagi per bulan secara otomatis. Kalau mau query data bulan Januari, database langsung tau cukup buka "laci Januari" — nggak perlu scan jutaan baris dari semua waktu.

**pgvector** — Extension yang bikin PostgreSQL bisa nyimpen dan nyari "sidik jari" artikel (embedding dari Layer 4). Ini yang enable fitur pencarian semantik: "cari semua artikel yang konteksnya mirip kebijakan ekonomi" — tanpa harus exact keyword match.

**Materialized View** — Versi "summary yang udah pre-kalkulasi". Daripada dashboard harus ngitung ulang dari jutaan baris tiap ada user buka, ada tabel ringkasan yang di-refresh tiap jam. Query ke dashboard jadi instan.

**pg_cron** — Penjadwal tugas yang jalan di dalam database. Tiap jam auto-refresh summary, tiap hari auto-bersihin data lama.

**Analogi:** Ini kayak perpustakaan yang punya:
- Rak per bulan (partitioning)
- Sistem pencarian yang ngerti sinonim dan konteks, bukan cuma keyword (pgvector)
- Papan rangkuman yang diupdate tiap hari (materialized view)
- Penjaga yang otomatis beberes jadwal (pg_cron)

---

### 🖥️ Layer 6 — Frontend Dashboard (Next.js + Vercel)
**Apa yang terjadi:** Tampilan yang user liat — tempat semua data tadi divisualisasikan.

Dibangun pakai **Next.js** (framework React yang dibuat sama tim Vercel), di-hosting gratis di **Vercel**.

**Yang bisa diliat user:**
- 📈 Grafik tren sentimen per tokoh (mingguan / bulanan)
- ⚖️ Head-to-head comparison: sentimen Prabowo vs Anies bulan ini
- 🔴 Live feed artikel terbaru yang masuk + label sentimen-nya (update otomatis tanpa refresh)
- 🔍 Pencarian semantik: "cari artikel yang bahas kebijakan BBM" — nemuin meski kata-katanya beda
- 🚨 Alert kalau sentimen tokoh tertentu anjlok atau naik drastis dalam waktu singkat

**Analogi:** Ini bagian yang kelihatan sama tamu restoran. Semua masakan dari dapur (Layer 1-5) ditata rapi di piring yang enak dilihat.

**Kenapa Next.js, bukan Streamlit?**  
Streamlit bagus buat prototipe riset yang tampilannya nggak terlalu penting. Tapi buat dashboard publik yang mau live update tanpa refresh dan tampilannya bisa dikontrol penuh — Next.js jauh lebih proper. Dan karena Layer 5 (Supabase) udah punya PostgREST + Realtime built-in, Next.js bisa langsung ngobrol ke database tanpa perlu API server tambahan.

---

## 💰 Breakdown Biaya (Rp 0)

| Layer | Komponen | Platform | Biaya |
|---|---|---|---|
| L1 | RSS Sources | — (public) | Gratis |
| L2 | Edge Function | Supabase | Gratis |
| L2 | Scheduler | GitHub Actions | Gratis (public repo) |
| L3 | Queue (pgmq) | Supabase | Gratis |
| L4 | NLP Worker | Hugging Face Spaces | Gratis |
| L5 | Database | Supabase | Gratis (500MB) |
| L6 | Dashboard | Vercel | Gratis |
| **Total** | | | **Rp 0** |

---

## 🛡️ Soal Privasi (UU PDP)

Project ini patuh UU Perlindungan Data Pribadi Indonesia:

- ❌ **Tidak nyimpen username / profil penulis** — yang dianalisis cuma isi artikelnya, bukan siapa yang nulis
- ❌ **Teks artikel mentah tidak ditampilkan ke publik** — cuma headline + link + skor sentimen yang bisa diakses umum
- ✅ **Data tokoh politik** masuk kategori data publik — analisis sentimen terhadap figur publik dalam kapasitas jabatan mereka adalah legal dan lazim dilakukan lembaga riset

---

## 📁 Struktur Repo

```
ID-Political-Sentiment-Tracker/
├── db/              # Skema database (SQL) — L5
├── ingestion/       # RSS fetcher (Deno/TS) — L2
├── nlp-worker/      # AI sentiment analyzer (Python) — L4
├── frontend/        # Dashboard publik (Next.js) — L6
└── docs/            # Diagram arsitektur + dokumentasi teknis
```

---

## 🚀 Mau Nyoba?

Cek [`docs/architecture.md`](docs/architecture.md) buat penjelasan teknis lengkap, atau langsung ke folder masing-masing layer — tiap folder punya `README.md` sendiri dengan panduan setup-nya.

---

*Dibuat dengan serius meski awalnya iseng. Sisanya jangan tangkap saya ya tukang bakso.*
