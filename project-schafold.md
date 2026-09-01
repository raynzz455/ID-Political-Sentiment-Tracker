ID-Political-Sentiment-Tracker/
├── apps/
│   └── pipeline/
│       ├── __init__.py
│       └── orchestrator.py       <-- (BARU) Entry point untuk jalanin semua worker
├── packages/
│   ├── shared/                   <-- (BARU) Modul bersama lintas layer
│   │   ├── __init__.py
│   │   ├── constants.py          <-- (Pindahan dari pipeline_constants.py)
│   │   ├── db_client.py          <-- (BARU) Inisialisasi Supabase client
│   │   └── logger.py             <-- (Pindahan dari pipeline_logger.py)
│   │
│   ├── ingestion/                <-- (BARU)
│   │   ├── __init__.py
│   │   ├── gnews_fetcher.py
│   │   └── ddg_fetcher.py
│   │
│   ├── enrichment/               <-- (BARU)
│   │   ├── __init__.py
│   │   ├── enricher_worker.py
│   │   └── universal_resolver.py
│   │
│   ├── validation/               <-- (BARU)
│   │   ├── __init__.py
│   │   ├── validation_worker.py
│   │   └── preprocessing_worker.py
│   │
│   ├── entity/                   <-- (BARU)
│   │   ├── __init__.py
│   │   └── entity_resolution_worker.py
│   │
│   ├── context/                  <-- (BARU)
│   │   ├── __init__.py
│   │   ├── context_worker.py
│   │   └── nlp_readiness_worker.py
│   │
│   └── nlp/                      <-- (BARU)
│       ├── __init__.py
│       ├── nlp_worker.py         <-- (Pindahan dari drain_queue.py)
│       ├── sentiment_model.py
│       └── cli_test.py
│
├── devtools/                     <-- (BARU) Untuk script testing/evaluasi
│   ├── eval/
│   │   ├── eval_metrics.py
│   │   ├── export_sentiment_ground_truth.py
│   │   └── export_relevancy_review.py
│   └── sql_tools/
│       └── check_db_stats.py
│
├── infra/
│   └── supabase/
│       └── functions/
│           └── rss-ingestion/
│               └── index.ts
├── docs/
└── .github/

                  ┌──────────────────────────────┐
                  │        RSS INGESTION         │
                  │ (RSS, Media, GNews Feed)     │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                     raw_texts (status=PENDING)
                                 │
                                 ▼
══════════════════════════════════════════════════════════════
                 LAYER 2 — ENRICHMENT
══════════════════════════════════════════════════════════════

Enricher Worker
    │
    ├── Fetch halaman asli
    ├── Trafilatura Extraction
    ├── Metadata
    └── Update

status = ENRICHED
content_type =
    FULLTEXT
    atau
    SNIPPET (GNews)

                                 │
                                 ▼
══════════════════════════════════════════════════════════════
              LAYER 2.5 — QUALITY VALIDATION
══════════════════════════════════════════════════════════════

Validation Worker

Input:
status = ENRICHED

Melakukan:

• Quality Score
• Noise Detection
• Bahasa
• Panjang artikel
• Stopword
• Title Matching

Output:

VALIDATED
atau

FAILED

(Tidak melakukan routing lagi.)

                                 │
                                 ▼
══════════════════════════════════════════════════════════════
             LAYER 3 — PREPROCESSING
══════════════════════════════════════════════════════════════

Preprocessing Worker

Input

status = VALIDATED

Melakukan

• Cleaning
• Unicode normalize
• Remove URL
• Remove emoji
• Remove HTML
• Sentence normalize
• dst

Output

preprocessed_text
preprocessed_at
preprocessing_version

status tetap VALIDATED

                                 │
                                 ▼
══════════════════════════════════════════════════════════════
            LAYER 3.2 — ENTITY RESOLUTION
══════════════════════════════════════════════════════════════

Entity Resolution Worker

Input

status = VALIDATED

Menghasilkan

article_entity_map

entity_mentions

main entity

confidence

resolver source

status tetap VALIDATED

                                 │
                                 ▼
══════════════════════════════════════════════════════════════
              LAYER 3.5 — CONTEXT
══════════════════════════════════════════════════════════════

Context Worker

Input

artikel VALIDATED
+
entity

Menghasilkan

entity_contexts

context span

window sentence

normalized context

status tetap VALIDATED

                                 │
                                 ▼
══════════════════════════════════════════════════════════════
            LAYER 3.7 — NLP READINESS
══════════════════════════════════════════════════════════════

Readiness Worker

Mengecek apakah artikel sudah memiliki

✓ preprocessed text

✓ entity

✓ context

Jika lengkap

↓

masuk queue NLP (pgmq)

status = NLP_READY

                                 │
                                 ▼
══════════════════════════════════════════════════════════════
             LAYER 4 — NLP PIPELINE
══════════════════════════════════════════════════════════════

Drain Queue

↓

Ambil artikel dari pgmq

↓

Fallback Sentiment

↓

Cari alias entity

↓

Relevancy Classifier

↓

Jika relevan

↓

Context Sentiment Classifier

↓

Insert sentiment_scores

↓

ACK Queue

↓

status = PROCESSED


Tentu, mari kita pajang kembali **Peta Arsitektur Lengkap (Final Pipeline)** dari hulu ke hilir. 

Sistem ini dirancang dengan prinsip *Single Responsibility* (Satu pekerjaan untuk satu worker), sehingga jika ada bug (seperti kasus Hamzah Haz/Gus Dur tadi), kita tahu persis di layer mana masalahnya terjadi.

---

### 🌐 LAYER 1 — INGESTION (Pengumpul Berita)
*   **Modul:** `packages/ingestion/` (gnews_fetcher, ddg_fetcher) & Supabase Edge Function (RSS).
*   **Tugas:** Menjaring artikel mentah dari internet (RSS Feed media, Google News, DuckDuckGo).
*   **Input:** URL RSS / Query Pencarian.
*   **Output:** Baris baru di tabel `raw_texts`. Pada tahap ini, teks artikel biasanya kosong atau hanya berisi *snippet* singkat dari RSS.
*   **Status DB:** `pending`

### ⚡ LAYER 2 — ENRICHMENT (Pengambil & Pembersih HTML)
*   **Modul:** `enricher_worker.py` & `universal_resolver.py`
*   **Tugas:** Mengambil URL dari tabel `raw_texts` (status `pending`), melakukan HTTP *fetch* ke portal berita, mengunduh HTML, dan mengekstrak teks utuh (Fulltext) memakai Trafilatura/JSON-LD. Di sini juga ada **Deduplication Gate** (membuang artikel dengan judul sama).
*   **Input:** Artikel `pending`.
*   **Output:** Teks artikel utuh bersih (tanpa iklan/sidebar).
*   **Status DB:** `enriched` (jika berhasil) atau `failed` (jika URL mati/sampah).

### ⚖️ LAYER 2.5 — VALIDATION (Quality Control)
*   **Modul:** `validation_worker.py`
*   **Tugas:** Menilai kualitas teks (0-100). Mengecek panjang teks, apakah berbahasa Indonesia, dan apakah judul RSS cocok dengan isi teksnya (mencegah *salah redirect*).
*   **Input:** Artikel `enriched`.
*   **Output:** Menentukan apakah teks ini layak diproses oleh AI atau dibuang.
*   **Status DB:** `validated` (jika lolos) atau `failed` (jika sampah/terlalu pendek).

### 🧹 LAYER 3 — PREPROCESSING (Pencuci Teks)
*   **Modul:** `preprocessing_worker.py`
*   **Tugas:** Membersihkan teks yang sudah tervalidasi. Menghapus URL, email, normalisasi Unicode (tanda baca aneh), dan menghitung *hash* konten untuk membuang duplikat isi yang lolos di Layer 2.
*   **Input:** Artikel `validated`.
*   **Output:** Teks rapi siap baca mesin. Mengisi kolom `preprocessed_at`.
*   **Status DB:** Tetap `validated` (hanya update timestamp).

### 🎯 LAYER 3.2 — ENTITY RESOLUTION (Pendeteksi Tokoh)
*   **Modul:** `entity_resolution_worker.py`
*   **Tugas:** Membaca teks artikel, mencari nama tokoh politik (Prabowo, Jokowi, dll) menggunakan *Regex Matcher* dan *Fuzzy Matching* berdasarkan tabel `political_entities`.
*   **Input:** Artikel `validated` yang sudah di-preprocess.
*   **Output:** Baris baru di tabel `entity_mentions` (mencatat offset/kalimat di mana tokoh disebut).
*   *(Catatan: Di layer inilah bug Hamzah Haz = Gus Dur terjadi, karena Regex salah mencocokkan alias).*

### ✂️ LAYER 3.5 — CONTEXT EXTRACTION (Pemotong Kalimat)
*   **Modul:** `context_worker.py`
*   **Tugas:** Mengambil offset tokoh dari `entity_mentions`, lalu memotong teks artikel di sekitar offset tersebut menjadi potongan kalimat pendek (Konteks). Memilih potongan kalimat terbaik menggunakan *Attribution Hunting* (mencari kata "ujar", "tegas") atau *Dense Embedding*.
*   **Input:** Data dari `entity_mentions`.
*   **Output:** Baris baru di tabel `entity_contexts` berisi `context_text` (kalimat pendek yang relevan dengan tokoh).
*   **Status DB:** Tetap `validated` (hanya update `context_extracted_at`).

### 📬 LAYER 3.7 — NLP READINESS (Gatekeeper Antrian)
*   **Modul:** `nlp_readiness_worker.py`
*   **Tugas:** Mengecek apakah artikel sudah punya konteks tokoh. Jika ada, artikel dimasukkan ke antrian fisik PGMQ (`pgmq.q_nlp_processing_queue`) agar siap dimakan oleh AI.
*   **Input:** Artikel `validated` yang sudah punya konteks.
*   **Output:** ID artikel masuk ke pipa PGMQ.
*   **Status DB:** `queued` (mengisi `nlp_ready_at`).

### 🧠 LAYER 4 — NLP WORKER (Mesin AI Sentimen)
*   **Modul:** `nlp_worker.py` & `sentiment_model.py`
*   **Tugas:** Mengambil ID dari antrian PGMQ, mengambil `context_text` dari DB, lalu menjalankan 2-Stage Pipeline AI:
    1. *Relevancy Gate:* Apakah konteks ini benar-benar relevan dengan tokoh X?
    2. *Sentiment Model:* Jika relevan, tebak sentimennya (Positif/Negatif/Netral).
*   **Input:** Antrian PGMQ.
*   **Output:** Baris baru di tabel `sentiment_scores` (berisi label, confidence, skor).
*   **Status DB:** `processed` (Pipeline selesai).

---

### 🛠️ LAYER 5.5 — FINE-TUNING (Offline Model Training) *Sedang dalam proses*
*   **Modul:** `export_finetune_dataset.py` & Script Training (PyTorch).
*   **Tugas:** Mengambil sampel data dari `entity_contexts` dan `sentiment_scores`, mengeluarkannya ke CSV, dikoreksi manual oleh manusia di Excel, lalu dipakai untuk melatih ulang otak IndoBERT agar lebih pintar politik.

### 🖥️ LAYER 6 — DASHBOARD (Frontend)
*   **Modul:** Next.js (Vercel) & Supabase Materialized View.
*   **Tugas:** Membaca tabel `sentiment_scores` dan `entity_highlights`, lalu menampilkan grafik "Termometer Digital" ke publik.

---

### Penjelasan Bug Hamzah Haz - Gus Dur
Dari peta di atas, jelas bahwa bug terjadi di **Layer 3.2 (Entity Resolution)**. Regex Matcher salah menangkap kata umum (kemungkinan "Wapres" atau "Wakil Rakyat") dan mengirasnya ke profile Abdurahman Wahid. Akibatnya, di Layer 3.5, teks tentang Hamzah Haz ikut dipotong dan dimasukkan ke konteks Gus Dur. Lalu di Layer 4, AI membaca teks Hamzah Haz dan menyimpannya di tabel sentimen milik Gus Dur.

Untuk memperbaikinya, kita bisa pasang *Sanity Check* di awal Layer 4 (NLP Worker) atau akhir Layer 3.2 (Entity Resolver).