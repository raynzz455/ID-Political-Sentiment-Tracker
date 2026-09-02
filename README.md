# ID Political Sentiment Tracker

> Basically ini kayak "termometer digital" buat ngukur suhu publik terhadap tokoh politik Indonesia — positif, netral, atau negatif — otomatis, real-time, dan gratis total.

**Status:** 🚧 Work in progress — dibangun layer by layer  
**Budget:** Rp 0 (literally free tier semua)  
**Vibe:** Iseng tapi serius

---

## 🤔 Ini apaan sih sebenarnya?

Pernah nggak ngerasa bingung: "sebenernya orang-orang suka nggak sih sama si X?" — tapi daripada baca ratusan artikel satu-satu, gimana kalau ada yang lakuin itu otomatis terus hasilnya ditampilkan dalam bentuk grafik yang gampang dibaca?

Nah, **itu persis yang project ini lakuin.**

Sistem ini otomatis:
1. 📰 Ngumpulin ribuan berita dari media Indonesia (RSS, Google News, DDG)
2. 🧹 Bersihin "sampah" digital (halaman foto, iklan, salah redirect) secara cerdas
3. 🧠 Baca dan analisis sentimen tiap artikel pakai AI (IndoBERT) dengan deteksi tokoh spesifik
4. 📊 Tampilin hasilnya di dashboard publik yang bisa diakses siapa aja

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

Bayangin sistem ini kayak **dapur restoran**, dari bahan mentah sampai makanan nyampe ke meja tamu. Tapi dapur ini punya stasiun kerja yang sangat spesifik biar rasa masakannya (data) konsisten dan gak ada sampah yang lolos:

```text
Berita dari internet
      ↓
  Dikumpulin & Dinormalisasi    ← L1 + L2 (Ingestion & Enrichment)
      ↓
  Quality Control & Cleaning    ← L2.5 + L3 (Validation & Preprocessing)
      ↓
  Identifikasi Tokoh & Konteks  ← L3.2 + L3.5 (Entity Resolution & Context)
      ↓
  Diantrekan                    ← L3.7 (Readiness & Final Gatekeeper)
      ↓
  Dianalisis AI                 ← L4 (NLP Worker)
      ↓
  Disimpan rapi                 ← L5 (Database)
      ↓
  Ditampilin ke lu              ← L6 (Dashboard)
```

---

## ⚙️ Tech Stack — Layer by Layer

### 🌐 Layer 1-2 — Ingestion & Enrichment (Python & Deno)
**Apa yang terjadi:** Yang bertugas pergi "belanja" berita tadi dan masukin ke sistem. 

Dibangun memakai **Deno/TypeScript** (untuk RSS cepat) dan **Python** (untuk Enrichment berat). Di-deploy ke **Supabase Edge Functions** dan **GitHub Actions** yang gratis.

**Teknisnya:** 
- Fetch semua RSS feed secara bersamaan.
- **Expert Gate:** Menggunakan Trafilatura (`favor_precision=True`) dan ekstraksi JSON-LD prioritas.
- **Anti-Sampah:** Menolak teks >20.000 karakter (section leakage/halaman list) dan *Title Mismatch* (salah redirect ke homepage).
- **Deduplication:** Cek duplikat judul sebelum HTTP *fetch* agar tidak buang bandwidth mengunduh HTML yang sama dua kali.

**Analogi:** Ini kayak kurir yang tiap 30 menit keliling ngumpulin koran, terus di-stasiun pertama korannya dibuang iklan-iklan sampahnya. Kalau ada koran yang judulnya udah pernah dibawa, dia langsung skip.

---

### ⚖️ Layer 2.5 & 3 — Validation & Preprocessing
**Apa yang terjadi:** Quality Control menilai teks (0-100). Teks yang lolos dibersihkan (normalisasi unicode, hapus URL, dst) dan dihitung *hash*-nya untuk mencegah duplikat konten lintas bulanan.

**Analogi:** Kayak *Quality Control* di pabrik. Bahan mentah yang busuk (teks terlalu pendek/campur aduk) langsung dibuang. Yang bagus dicuci dan dipotong rapi.

---

### 🎯 Layer 3.2 & 3.5 — Entity Resolution & Context Extraction
**Apa yang terjadi:** Mendeteksi tokoh dalam teks (Prabowo, Gibran, dll). Mengambil kalimat di sekitar tokoh (*context span*) agar AI tidak bingung menganalisis artikel utuh yang mungkin membahas banyak tokoh sekaligus.

**Analogi:** Sistem ini tau kalo di artikel ada 3 tokoh, dia pisah-pisah. "Oke, kalimat ini tentang Prabowo, kalimat itu tentang Anies." Dia ngga mencampur aduk.

---

### 📬 Layer 3.7 — Readiness & Antrian (pgmq)
**Apa yang terjadi:** *Final Gatekeeper*. Mengecek kelengkapan artikel. Jika lolos, dimasukkan ke antrian PGMQ di dalam PostgreSQL. Nggak perlu server Redis atau Kafka yang ribet.

**Analogi:** Bayangin antrian kasir Indomaret. Artikel-artikel yang udah bersih ngantri dulu. Nanti si kasir (NLP worker) ambil satu-satu, proses, terus lanjut ke artikel berikutnya.

---

### 🧠 Layer 4 — NLP Worker (Python + IndoBERT)
**Apa yang terjadi:** Ini bagian paling "otak"-nya — AI yang baca artikel dan mutusin sentimen-nya. Memakai model **IndoBERT** dari HuggingFace yang dilatih khusus Bahasa Indonesia.

AI memproses teks menggunakan **2-Stage Pipeline**:
1. **Relevancy Gate:** "Apakah konteks ini benar-benar membahas tokoh X?" (Mencegah *false positive* seperti "Listyo Sigit Prabowo" vs "Prabowo Subianto").
2. **Sentiment Classifier:** Menilai positif/netral/negatif. Murni output ML tanpa *heuristic mapping* yang merusak data.

**Analogi:** Ini kayak ada editor politik senior yang udah baca jutaan artikel Indo. Dia ngga asal baca. Dia pastiin dulu "ini ngomongin siapa?", baru kasih nilai "berita ini lagi negatif".

---

### 🗄️ Layer 5 — Database (Supabase / PostgreSQL)
**Apa yang terjadi:** Gudang utama tempat semua data disimpan rapi dan terorganisir.

**Yang bikin special di project ini:**
- **Global Dedup & Auto-Partition:** Tabel dibagi per bulan otomatis agar hemat storage dan querynya cepat.
- **Row Level Security (RLS):** Teks mentah berita diblokir untuk publik (patuh UU PDP). Hanya *headline*, *link*, dan *skor sentimen* yang dilempar ke *frontend* melalui tabel cache `entity_highlights`.
- **Materialized View:** Merangkum data sentimen 90 hari terakhir agar *dashboard* load-nya instan tanpa ngitung ulang jutaan baris.

**Analogi:** Ini kayak perpustakaan yang punya rak per bulan (partitioning), sistem keamanan ketat soal siapa yang boleh baca arsip asli (RLS), dan papan rangkuman yang diupdate tiap hari (Materialized View).

---

### 🖥️ Layer 6 — Frontend Dashboard (Next.js + Vercel)
**Apa yang terjadi:** Tampilan yang user liat — tempat semua data tadi divisualisasikan. Dibangun pakai **Next.js**, di-hosting gratis di **Vercel**.

**Yang bisa diliat user:**
- 📈 Grafik tren sentimen per tokoh (mingguan / bulanan)
- ⚖️ Head-to-head comparison: sentimen Prabowo vs Anies bulan ini
- 🔴 Live feed artikel terbaru yang masuk + label sentimen-nya
- 🚨 Alert kalau sentimen tokoh tertentu anjlok atau naik drastis

**Analogi:** Ini bagian yang kelihatan sama tamu restoran. Semua masakan dari dapur (Layer 1-5) ditata rapi di piring yang enak dilihat.

---

## 💰 Breakdown Biaya (Rp 0)

| Layer | Komponen | Platform | Biaya |
|---|---|---|---|
| L1 | RSS Sources | — (public) | Gratis |
| L2 | Edge Function & Scheduler | Supabase & GitHub Actions | Gratis |
| L3 | Queue (pgmq) | Supabase | Gratis |
| L4 | NLP Worker | GitHub Actions / HF Spaces | Gratis |
| L5 | Database | Supabase | Gratis (500MB) |
| L6 | Dashboard | Vercel | Gratis |
| **Total** | | | **Rp 0** |

---

## 🛡️ Soal Privasi (UU PDP)

Project ini patuh UU Perlindungan Data Pribadi Indonesia:

- ❌ **Tidak nyimpen username / profil penulis** — yang dianalisis cuma isi artikelnya, bukan siapa yang nulis
- ❌ **Teks artikel mentah tidak ditampilkan ke publik** — dikunci via RLS. Cuma headline + link + skor sentimen yang bisa diakses umum
- ✅ **Data tokoh politik** masuk kategori data publik — analisis sentimen terhadap figur publik dalam kapasitas jabatan mereka adalah legal dan lazim dilakukan lembaga riset

---

## 📁 Struktur Repo (Monorepo)

```text
ID-Political-Sentiment-Tracker/
├── apps/
│   └── pipeline/
│       └── orchestrator.py       # Entry point untuk jalanin semua worker
├── packages/
│   ├── shared/                   # Modul bersama (constants, db_client, logger)
│   ├── ingestion/                # Fetcher RSS/GNews/DDG
│   ├── enrichment/               # Trafilatura extraction & Resolver
│   ├── validation/               # Quality control & Preprocessing
│   ├── entity/                   # Entity Resolution (NER & Alias)
│   ├── context/                  # Context Extraction & NLP Readiness
│   ├── nlp/                      # AI sentiment analyzer (IndoBERT)
│   └── db/                       # Skema SQL final & Seeds (Single Source of Truth)
├── devtools/                     # Script testing, evaluasi, & local recovery (Playwright)
├── infra/
│   └── supabase/
│       └── functions/            # Edge Functions (Deno/TS)
└── .github/
    └── workflows/                # GitHub Actions scheduler (Event-driven Sequential)
```

---

## 🚀 Mau Nyoba?

Sistem ini dijalankan menggunakan satu file `main.py` sebagai orchestrator.

```bash
# Jalankan Layer 2 hingga 3.7 (Prep Pipeline)
python main.py prep --limit 100

# Jalankan Layer 4 (NLP Inference)
python main.py nlp --target 500

# Jalankan Prep + NLP berurutan
python main.py all

# Cek status database
python main.py status
```

Cek dokumentasi teknis di folder `docs/` atau buka `README.md` di masing-masing folder `packages/` untuk detail implementasi spesifik tiap layer.

---

*Dibuat dengan serius meski awalnya iseng. Sisanya jangan tangkap saya ya tukang bakso.*
```