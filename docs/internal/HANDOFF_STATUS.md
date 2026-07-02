# HANDOFF STATUS — ID-Sentiment-Tracker

---

## 📋 SESI 8 REPORT — 2026-06-30

### Yang dikerjakan

**A. Kritik terhadap laporan GLM sesi 7 (Decision 7 & 8)**
- Ditemukan inkonsistensi: Decision 7 menyatakan "data quality > model quality"
  tapi prioritas aktualnya menaruh ekspansi entitas DI ATAS fix parser bug
  (Issue B, 5 feed mati) — bertentangan dengan filosofi sendiri
- Gating metric "≥5000 raw_texts" salah sasaran — masalahnya per-entity
  (skewed: Prabowo 36 vs Muhaimin 2), bukan agregat. Reaching 5000 total
  tidak menjamin coverage per-tokoh yang dibutuhkan untuk breakdown tahunan
- Keputusan 3 (evaluasi distribusi sebelum ONNX) dibatalkan tanpa basis
  konsisten — dummy run sebelumnya (56% netral) sebenarnya di BAWAH
  threshold 80% yang menurut rule asli berarti "lanjut ke ONNX", bukan
  "tunda terus". Tapi juga dicatat: dummy heuristic bukan instrumen valid
  untuk ukur distribusi real sejak awal
- **Verified via search:** GitHub Actions scheduled workflow delay (observed
  ~3-4 jam vs konfigurasi 30 menit) adalah documented behavior resmi GitHub
  ("best effort", no SLA) — bukan bug project. Mitigasi: offset cron minute
  dari `:00`/`:30`, atau terima cadence lambat dan hitung ulang timeline
- **Verified via search:** "ModernBERT" tidak native multilingual/Indonesia.
  Turunannya (mmBERT, Sept 2025) belum proven untuk sentimen Indonesia.
  Literatur akademik konsisten: IndoBERT monolingual masih unggul vs model
  multilingual generik untuk domain ini (F1 0.9353 vs mBERT lebih rendah)

**B. Upgrade model sentimen — dari dummy ke real, context-aware**
- Riset: ditemukan studi SocialX+Telkom University+BRIN (April 2026) yang
  membuktikan model SmSA-based umum (kemungkinan termasuk
  `taufiqdp/indonesian-sentiment`) collapse ke 59-63% akurasi & F1 kelas
  positif <0.211 saat dievaluasi out-of-domain (review vs berita politik)
- Model pengganti dipilih: `apriandito/indobert-sentiment-classifier`
  (context-conditioned, F1 0.856) + `apriandito/indobert-relevancy-classifier`
  (F1 0.948) — keduanya dari paper yang sama, didesain dipakai berpasangan
- `sentiment_model.py` v1 dibuat (single context-conditioned model) →
  ditemukan FLAW desain test: sentiment confidence TIDAK BISA deteksi
  entity mismatch (kasus Kapolri "Listyo Sigit Prabowo" vs Presiden Prabowo
  tetap dapat confidence 0.906, karena itu task relevansi bukan task sentimen)
- **Dikoreksi ke v2: 2-stage gated pipeline**
  `Stage 1 (RelevancyModel) -> Stage 2 (SentimentModel, hanya jika relevan)`
  Ini sekaligus jadi solusi permanen untuk masalah false-positive alias
  matching yang berulang kali muncul sejak sesi awal (Prabowo/Listyo Sigit,
  RK/Ridwan Kamil vs kriminal)
- `test_sentiment_model.py` v2 ditulis ulang — test relevancy gate eksplisit
  (bukan sentiment confidence), termasuk 2 kasus false-positive historis

**C. Kritik terhadap laporan evaluasi Gemini (CLI testing sesi ini)**
- Klaim "confidence 0.985 rata-rata" tidak jelas dari stage mana (relevancy
  atau sentiment) — dua pertanyaan berbeda, tidak boleh dicampur
- Rekomendasi "hapus logic insert NULL" **DITOLAK** — kalau diikuti akan
  menghapus kemampuan `mv_national_monthly_summary` /
  `mv_national_yearly_summary` (fitur national mood index yang sudah
  disepakati eksplisit beberapa sesi lalu). NULL bukan sampah, itu data
  untuk agregat nasional — yang salah cuma konsistensi KAPAN insert-nya
- **Temuan kritis yang Gemini lewatkan:** 88 baris `sentiment_scores` dari
  dummy model lama (sesi 7) kemungkinan BELUM dibersihkan sebelum testing
  model real — total sekarang 111 baris bisa jadi campuran dummy+real
  tanpa cara membedakan. Ini mengkontaminasi SEMUA statistik di laporan
  Gemini (confidence avg, distribusi)
- n=13 untuk klaim "distribusi sentimen dinamis" — terlalu kecil secara
  statistik untuk disimpulkan apa pun
- "Siap jadi daemon production" — conflate stabilitas rekayasa (terbukti)
  dengan validitas model/data (BELUM terbukti, belum ada ground truth)

**D. Toolkit ground truth evaluation dibangun**
- SQL diagnostic kontaminasi (deteksi via confidence persis 0.65/0.60 —
  hardcoded value dummy lama)
- Redesain logic NULL: SELALU hitung fallback document-level untuk national
  index (`model_version='indobert-fallback-v1'`), entity-specific HANYA
  kalau lolos relevancy gate (`model_version='indobert-ctx-relevancy-gated-v1'`)
- `export_sentiment_ground_truth.py` — stratified sample per label
- `export_relevancy_review.py` — re-scan raw_texts, capture SEMUA kandidat
  (lolos maupun ditolak gate) untuk validasi manusia
- `eval_metrics.py` — precision/recall/F1/confusion matrix TERPISAH per
  stage + calibration check (uji empiris klaim "overconfident")
- `GROUND_TRUTH_EVAL_GUIDE.md` — urutan kerja lengkap end-to-end

### Belum dikerjakan (lanjutkan sesi berikutnya)
- ⏳ Jalankan migration 007 di Supabase SQL Editor
- ⏳ Jalankan seed 03 + 04 di Supabase SQL Editor
- ⏳ Run `auto_discover.py --source all` (Wikipedia + title scan)
- ⏳ Fix alias Prabowo (false positive "Listyo Sigit") — SQL 1 baris
- ⏳ Run SQL diagnostic kontaminasi dummy data (confidence = 0.65/0.60)
- ⏳ Jalankan ground truth evaluation (`export_sentiment_ground_truth.py`)
- ⏳ Jalankan relevancy review (`export_relevancy_review.py`)
- ⏳ Historical data backfill (archive scraper per tokoh per tahun)

### File baru di sesi ini
```
packages/nlp-worker/sentiment_model.py              ← 2-stage gated pipeline (v2)
packages/nlp-worker/test_sentiment_model.py         ← test relevancy gate v2
packages/nlp-worker/export_sentiment_ground_truth.py ← stratified sample export
packages/nlp-worker/export_relevancy_review.py       ← kandidat relevancy review
packages/nlp-worker/eval_metrics.py                 ← precision/recall/F1 evaluator
packages/nlp-worker/GROUND_TRUTH_EVAL_GUIDE.md      ← panduan evaluasi end-to-end
```

### Model NLP yang dipilih (sesi 8)
| Stage | Model | F1 | Fungsi |
|-------|-------|-----|--------|
| 1 (Relevancy) | `apriandito/indobert-relevancy-classifier` | 0.948 | Filter artikel relevan vs non-relevan |
| 2 (Sentiment) | `apriandito/indobert-sentiment-classifier` | 0.856 | Predict positive/negative/neutral |

**Catatan:** Model sudah ditest (test_sentiment_model.py v2) dan bisa handle false-positive
alias (Prabowo/Listyo Sigit). Tapi **belum production-ready** — perlu ground truth evaluation dulu.

### model_version tagging (sesi 8)
| Tag | Kapan dipakai | Arti |
|-----|--------------|------|
| `indobert-fallback-v1` | Semua artikel (document-level) | Sentimen tanpa entity, untuk national mood index |
| `indobert-ctx-relevancy-gated-v1` | Hanya artikel yang lolos relevancy gate | Sentimen entity-specific, akurat |

### Temuan kritis: kontaminasi data dummy
- 88 baris sentiment_scores dari dummy model (sesi 3-7) **mungkin masih ada di DB**
- Total sekarang 111 baris = kemungkinan campuran dummy + real (tidak bisa dibedakan)
- **Solusi:** Jalankan SQL diagnostic sebelum test lebih lanjut:
  ```sql
  SELECT COUNT(*) FROM sentiment_scores
  WHERE confidence IN (0.65, 0.60)  -- nilai hardcoded dummy
  ```
- Jika ditemukan → hapus sebelum lanjut evaluasi model real

---

## 📐 Architectural Notes (AI Recommendation — 2026-06-30)

> **Status:** Informational (bukan prioritas implementasi saat ini)
>
> Catatan ini merupakan hasil evaluasi arsitektur. Saat ini **tetap mengikuti fokus utama project**, yaitu:
>
> 1. Ekspansi entity
> 2. Historical data collection
> 3. Perbaikan parser
>
> Optimasi NLP dilakukan setelah data dianggap cukup.

### 1. Bottleneck project bukan latency model

Evaluasi pipeline menunjukkan bahwa waktu inferensi model **bukan bottleneck utama**.

Pipeline saat ini:

```
RSS
    ↓
Download HTML
    ↓
Trafilatura Extract
    ↓
Entity Matching
    ↓
Sentiment Model
    ↓
Insert Database
```

Estimasi waktu:

| Tahapan          | Estimasi    |
| ---------------- | ----------- |
| Download article | 300-1000 ms |
| HTML extraction  | 100-400 ms  |
| Entity matching  | <10 ms      |
| NLP inference    | 100-300 ms  |
| Database insert  | 50-150 ms   |

Mayoritas waktu habis pada network dan preprocessing, bukan inference.

**Kesimpulan:**
Jangan mengorbankan akurasi hanya demi mengurangi latency model.

---

### 2. Prioritaskan throughput dibanding latency

Project menggunakan asynchronous queue (`pgmq`) sehingga dashboard tidak melakukan inferensi secara realtime.

Yang lebih penting adalah:

```
berapa artikel selesai diproses per jam
```

daripada

```
berapa cepat satu artikel selesai diproses
```

Untuk workload ±1000 artikel/hari, latency 200 ms maupun 500 ms hampir tidak berpengaruh terhadap operasional pipeline.

---

### 3. Batch inference lebih penting daripada model lebih cepat

Jika production worker dibuat nanti, lebih disarankan:

```
dequeue 16-32 artikel

↓

predict sekaligus

↓

insert batch
```

daripada

```
dequeue 1

↓

predict

↓

ulang
```

Batch inference meningkatkan throughput CPU secara signifikan dibanding optimasi latency individual.

---

### 4. Hugging Face Free diperkirakan masih mencukupi

Estimasi workload:

```
1000 artikel / hari

≈42 artikel / jam

≈1 artikel setiap ±1.4 menit
```

Dengan inference ±0.8 detik/artikel, worker masih memiliki kapasitas yang cukup.

Namun perlu diperhatikan:

* Hugging Face Free memiliki cold start
* Space dapat sleep
* Tidak ada SLA
* Resource tidak dijamin

Untuk skripsi dan proof-of-concept masih sangat layak digunakan.

Jika nanti project berkembang menjadi production service, worker sebaiknya dipindahkan ke VPS/container yang selalu aktif tanpa mengubah arsitektur pipeline.

---

### 5. Evaluasi model setelah data cukup

Saat ini **belum disarankan mengganti dummy model**.

Urutan yang direkomendasikan tetap:

```
Perbanyak entity

↓

Perbanyak historical data

↓

Perbaiki parser

↓

Minimal ±5000 raw_texts

↓

Evaluasi model NLP

↓

Production worker
```

Pemilihan model dilakukan setelah tersedia dataset representatif.

Kandidat yang layak dievaluasi:

* ModernBERT (arsitektur modern, CPU friendly)
* IndoBERT (baseline Indonesia)
* XLM-RoBERTa
* MiniLM Multilingual

Pemilihan akhir didasarkan pada trade-off:

* akurasi
* ukuran model
* throughput CPU
* kemudahan ONNX

bukan semata-mata latency inferensi.

---

### 6. Potensi peningkatan akurasi terbesar

Peningkatan akurasi terbesar kemungkinan **bukan berasal dari mengganti model**, melainkan dari peningkatan kualitas pipeline:

* full article extraction
* entity detection yang lebih baik
* alias management
* historical corpus lebih besar
* evaluasi Targeted Sentiment Analysis (ABSA) setelah pipeline stabil

Dengan kata lain:

```
Data Quality
>
Model Quality
>
Latency Optimization
```

untuk kebutuhan project ini.


> **Tgl update:** 2026-06-30 (sesi 7, GLM/ZCode)
> **Project Ref:** `bawvxti***` (lihat Supabase Dashboard)
> **Repo:** `raynzz455/ID-Political-Sentiment-Tracker.git`
> **Status:** ✅ Pipeline otomatis jalan (GitHub Actions). Fokus: EKSPANSI ENTITAS + HISTORICAL DATA.
> Dummy model di-Tunda — NLP real (IndoBERT ONNX) setelah data cukup.

---

## 🎯 FOKUS UTAMA SAAT INI (Keputusan User, 2026-06-30)

### Keputusan strategi: Kumpulkan Data Dulu, Proses NLP Nanti

**Alasan:**
- Dummy model hanya menghasilkan distribusi palsu (confidence semua 0.6-0.65)
- 88 sentiment_scores dari dummy = sampah data — akan dihapus saat IndoBERT ONNX gantikan
- Lebih efisien kumpulkan data sebanyak mungkin dulu, baru proses dengan model real
- Historical data (orde lama-2025) adalah requirement user utama — belum ada satupun

**Prioritas (dari tertinggi ke terendah):**
```
1. 🔴 EKSPANSI ENTITAS — 50+ tokoh (seed 03), auto-discovery, alias fix
2. 🔴 HISTORICAL DATA — backfill arsip berita 2019-2025 per tokoh
3. 🟡 PARSING FIX — perbaiki 5 feed yang 0 item, body kosong
4. 🟢 INDOBERT ONNX — GANTI DUMMY (setelah data cukup, target >5000 raw_texts)
5. 🟢 PRODUCTION NLP WORKER — auto-dequeue (setelah ONNX ready)
6. ⚪ FRONTEND — Next.js dashboard (paling akhir)
```

### Yang TIDAK diprioritaskan saat ini:
- ❌ IndoBERT ONNX — **ditunda** sampai data cukup
- ❌ Production NLP worker — **ditunda** sampai ONNX ready
- ❌ Frontend — **ditunda** sampai ada data sentimen real
- ❌ YouTube comments (Lapis 3) — **belum diputuskan** (bertentangan dengan Keputusan 1)

---

## 📋 SESI 7 REPORT — 2026-06-30 (GLM/ZCode)

### Yang dikerjakan
- ✅ **Full codebase review** — baca & analisis seluruh kode
- ✅ **Analisis keamanan secrets** — verdict: **AMAN**, nilai asli tidak ada di file
- ✅ **Analisis ZCode error** — 1305 (overloaded) + context window exceeded, bukan bug project
- ✅ **Pertama kali lihat distribusi data real** via `cli_test.py batch 100`:
  - 17% entity match, 83% skip (masuk akal, hanya 18 tokoh aktif)
  - Distribusi dummy: positive 16%, neutral 56%, negative 28%
  - Confidence semua 0.5-0.7 (hardcoded dummy, tidak bermakna)
- ✅ **Verifikasi GitHub Actions otomatis** — run #20-#26 semua sukses, 19-67 detik/run
- ✅ **Data pipeline status** — 1000 raw_texts, 88 sentiment_scores (dummy)

### Data pipeline saat ini
```
raw_texts:          1.000 (981 pending, 19 queued) — terus bertambah otomatis
sentiment_scores:      88 (dummy model, distribusi tidak bermakna)
Entity coverage:      22% (hanya 4 dari 18 tokoh muncul)
Top entities:          Prabowo 36, Gibran 16, Anies 4, Muhaimin 2, ?(NULL) 30
GitHub Actions:        ✅ berjalan otomatis, ~3-4 jam sekali
```

### Temuan dari code review (bug yang masih ada)
| # | Bug | Lokasi | Severity | Prioritas |
|---|-----|--------|----------|-----------|
| B1 | Text kosong lolos guard `text.length < 20` | `index.ts:118` | KRITIS | 🟡 Fix saat parsing fix |
| B2 | `cmd_sample` insert NULL, `cmd_batch` skip — inkonsisten | `cli_test.py` | Low | ⚪ Tidak urgent |
| B3 | `cmd_batch` statistik mencampur item yang tidak masuk DB | `cli_test.py` | Low | ⚪ Tidak urgent |
| B4 | `cmd_single` tidak pakai title+text combined | `cli_test.py` | Low | ⚪ Tidak urgent |
| B5 | `--no-insert` flag dead code | `cli_test.py` | Low | ⚪ Tidak urgent |
| B6 | `last_run_at` tidak update untuk 0-item feed | `index.ts:283` | Medium | 🟡 Fix saat parsing fix |

> **Catatan:** Bug B2-B5 terkait dummy model/CLI testing — tidak perlu fix sekarang
> karena dummy model akan diganti IndoBERT ONNX nanti. Fokus ke B1 dan B6.

---

## 📋 SESI 6 REPORT — 2026-06-29

### Yang dikerjakan
- ✅ **Migration 007** dibuat: ekspansi schema `political_entities`
  - `entity_type` CHECK constraint diperluas: tambah `commentator`, `influencer`,
    `academic`, `journalist`, `former_minister`, `former_official`, `party_official`,
    `governor`, `mayor`
  - Kolom baru: `era[]`, `birth_year`, `active_since_year`, `last_relevant_year`,
    `mention_count_7d`, `mention_count_30d`, `last_mentioned_at`,
    `auto_discovered`, `discovery_source`, `discovery_confidence`,
    `wikipedia_id_url`, `wikipedia_en_url`
  - Tabel baru: `entity_candidates` (staging auto-discovery)
  - Function baru: `auto_promote_candidates()`, `refresh_entity_hotness()`
  - View baru: `hotline_tokoh` (siapa yang sedang ramai, realtime)
  - pg_cron: `refresh-entity-hotness` tiap malam jam 02:00 UTC

- ✅ **Seed 03** dibuat: 50+ entitas komprehensif
  - Presiden & Wapres semua era yang masih relevan (Gus Dur, SBY, Jokowi, dll)
  - Full kabinet Prabowo aktif
  - Ketua & tokoh partai semua partai parlemen
  - Gubernur strategis (Bobby, Dedi Mulyadi, Pramono, dll)
  - Pengamat politik: Rocky Gerung, Refly Harun, Ferry Irwandi, dll
  - Jurnalis/presenter: Najwa Shihab, Karni Ilyas
  - Ekonom/komentator: Rizal Ramli, Faisal Basri (alm), Chatib Basri
  - Mantan pejabat yang masih hot: Mahfud MD, Tom Lembong, Wiranto, dll

- ✅ **Seed 04** dibuat: Google News RSS untuk semua entitas baru (47 configs)

- ✅ **auto_discover.py** dibuat: sistem auto-discovery 3 sumber
  - Source 1: Wikipedia API (id.wikipedia.org) — daftar politisi dari 8 kategori
  - Source 2: Title scan — nama yang sering muncul di `raw_texts` tapi belum di DB
  - Source 3: Google News validation — validasi relevansi politik
  - Auto-promote: confidence >= 0.8 + mention >= 3 + gnews >= 2

### Belum dikerjakan (lanjutkan sesi berikutnya)
- ⏳ Jalankan migration 007 di Supabase SQL Editor
- ⏳ Jalankan seed 03 + 04 di Supabase SQL Editor
- ⏳ Run auto_discover.py pertama kali (Wikipedia + title scan)
- ⏳ Fix alias Prabowo (false positive Listyo Sigit) — SQL 1 baris
- ⏳ GitHub Actions 3 secrets — cron otomatis tiap 30 menit
- ⏳ IndoBERT ONNX (setelah data cukup)

### Prioritas sesi berikutnya (urutan)
```
1. Jalankan 007 → 03 → 04 di SQL Editor (urutan wajib)
2. Verifikasi: SELECT COUNT(*) FROM political_entities; → harus ~50+
3. python entity_discovery/auto_discover.py --source all
4. Fix alias Prabowo + setup GitHub Actions
5. Monitor data masuk (total_inserted dari curl)
6. Baru lanjut IndoBERT ONNX setelah >500 artikel masuk
```

### File baru di sesi ini
```
packages/db/migrations/007_entity_expansion_schema.sql  ← WAJIB run dulu
packages/db/seeds/03_entities_comprehensive.sql          ← 50+ tokoh
packages/db/seeds/04_scraping_configs_expanded.sql       ← 47 RSS configs baru
packages/nlp-worker/entity_discovery/auto_discover.py    ← auto-discovery script
packages/nlp-worker/entity_discovery/requirements.txt
packages/nlp-worker/entity_discovery/README.md
```

### Konteks untuk GLM / AI lain
- Sistem auto-discovery menggunakan `entity_candidates` sebagai staging.
  JANGAN langsung insert ke `political_entities` tanpa validasi.
- `hotline_tokoh` VIEW sudah bisa di-query untuk tahu siapa yang sedang ramai.
- Migration 007 WAJIB dijalankan sebelum seed 03/04 — ada kolom baru yang dipakai.
- `auto_promote_candidates()` RPC aman dipanggil berulang kali — idempotent.

---
> **NOTE:** Section "Belum dikerjakan" dan "Prioritas" versi sesi 5 sudah diarsipkan.
> Lihat versi terbaru di SESI 6 REPORT di atas.
---
Dokumen ini adalah **single source of truth** untuk sinkronisasi antar asisten AI
(GLM/ZCode ↔ Claude). Setiap perubahan production DB atau code WAJIB update dokumen ini.

---

> **Tgl update:** 2026-06-26 (sesi 4)
> **Project Ref:** `bawvxti***` (lihat Supabase Dashboard)
> **Repo:** `raynzz455/ID-Political-Sentiment-Tracker.git`
> **Status:** ✅ Arsitektur 3-lapis didesain. Lapis 2 (2-stage scraping) terimplementasi.
> Repo di-restructure ke monorepo.
> ⚠️ Status ini sudah usang — lihat FOKUS UTAMA di atas untuk state terkini.

Dokumen ini adalah **single source of truth** untuk sinkronisasi antar asisten AI
(GLM/ZCode ↔ Claude). Setiap perubahan production DB atau code WAJIB update dokumen ini.

---

## 🏗️ ARSITEKTUR 3-LAPIS (BARU — sesi 4)

### Latar belakang
Sesi 3 menemukan keterbatasan fundamental: Google News RSS hanya mengirim
`<title>` + `<description>` (HTML link fragment), TIDAK mengirim body artikel.
Body kosong → sentiment tidak akurat.

**Validasi kritis:** gnews encoded link (`/rss/articles/CBMi...`) menggunakan
**server-side HTTP redirect (301/302)**, BUKAN JavaScript redirect.
Diverifikasi manual: link → load Google News bentar → redirect ke domain asli
(detik.com, beritanasional.com, dll). Artinya `fetch()`/`requests` bisa follow.

### Lapis 1 — Discovery (Edge Function, sudah ada)
- RSS gnews + RSS native → `raw_texts` (title + source_url, body boleh kosong)
- Enqueue ke pgmq dengan entity_id
- **Tidak ada perubahan besar** — edge function tetap ringan

### Lapis 2 — Enrich + Predict (NLP Worker, BARU terimplementasi)
Saat body RSS kosong/pendek (<80 chars) & ada `source_url`:
1. `requests.get(source_url, allow_redirects=True)` → follow gnews redirect → HTML asli
2. `trafilatura.extract(html)` → main content bersih 300-500 kata (buang menu/ad/footer)
3. Combined = title + full body → predict sentiment

**Lokasi:** `packages/nlp-worker/cli_test.py`
- `fetch_full_body(url)` — fetch + trafilatura extract
- `enrich_if_needed(item, min_len=80)` — fallback logic: RSS body → fetch → title
- **Dependency baru:** `requests`, `trafilatura` (lihat requirements.txt)

**Perubahan RPC:** `dequeue_nlp_batch` sekarang return `source_url` kolom
(migration `005_dequeue_add_source_url.sql` — WAJIB run sebelum test Lapis 2).

### Lapis 3 — Public Opinion (YouTube Data API, belum dimulai)
- Beda segment: media tone (Lapis 1-2) vs public opinion (Lapis 3)
- Komentar video tokoh → sentiment opini publik
- Free tier: 10.000 quota/day — cukup untuk 18 tokoh
- Status: **BELUM DIMULAI**

---

## 🚨 STATUS TERKINI — Sesi 3 (2026-06-25)

### Progress sesi 3
1. ✅ Queue orphan dibersihkan + data re-ingested (curl berhasil)
2. ✅ `dequeue_nlp_batch` RPC error **RESOLVED** — setelah queue dibersihkan, dequeue jalan normal
3. ✅ CLI tool `batch` command jalan — 100 item berhasil dequeue + ack (queue drain bersih)
4. 🐛 **BUG BARU ditemukan:** `entity_id = NULL` untuk 100% item — text kosong di queue

### Bug #7: RSS text body kosong → entity matching 0%
**Gejala:** `python cli_test.py batch 100` → 0 entity matched, semua skip+ack
**Root cause:** RSS feed (khususnya CNN, Detik, dll) cuma kirim **title**, body/text kosong.
Title berisi nama tokoh ("Jokowi Mania", "Kapolri", "Roy Suryo") tapi `match_entities()`
hanya scan kolom `text`, bukan `title`.

**Evidence dari `inspect`:**
```
[1] source: cnnindonesia_nasional
    title:  Jokowi Mania Pertanyakan Keputusan Kejaksaan Tak Tahan Roy Suryo-Tifa
    text:   (KOSONG)
```

**Fix (sudah diaplikasi ke `cli_test.py`):**
- `match_entities()` sekarang gabungkan `title + text` untuk scanning aliases
- `predict_sentiment()` di `cmd_sample` dan `cmd_batch` juga pakai `title + text` combined
- Signature: `match_entities(text, title, entities)` — `title` default ""

**Catatan:** `cmd_sample` masih insert dengan `entity_id=NULL` saat no match.
`cmd_batch` sudah diubah ke **skip + ack** (jangan insert non-politik).

### Konsekuensi data hilang (sesi 2)
- Sesi 1: Pipeline berhasil — curl return `total_inserted: 50, enqueued: 200`
- Sesi 2: `raw_texts TOTAL 0` — semua data hilang
- Penyebab diduga: `schema.sql` di-run ulang (DROP TABLE CASCADE)
- **PEMBELAJARAN:** JANGAN run `schema.sql` ulang. Gunakan `migration_*.sql` saja.

---

## 🎯 SCHEMA READINESS — VERIFIED (sesi 1)

| Komponen | Status | Evidence |
|---|---|---|
| Tabel + partisi | ✅ | Dump: 6 partisi 2026-06/07/08 (raw + sentiment) |
| RLS policies | ✅ 12 policies | UU PDP compliant, raw_texts/sentiment blocked anon |
| RPC functions | ✅ 13 functions | get_entity_ranking overload sudah di-drop, tinggal 1 versi |
| MV `mv_dashboard_summary` | ✅ ADA | Frontend bisa query agregat |
| Trigger partition key | ✅ FIXED (DROPPED) | RPC isi `ingested_month` eksplisit, bukan trigger |
| Bucket `politik` | ✅ PUBLIC | foto tokoh, anon read, service_role write |

---

## 🔧 BUG KRITIKAL YANG SUDAH DIOBATI — JANGAN DIROLLBACK

### Bug #1: `ingested_month = NULL` → insert gagal ke partitioned table
**Root cause** (sudah dikonfirmasi production):
PostgreSQL melakukan **partition routing SEBELUM BEFORE INSERT trigger** fire. Saat
`ingested_month` NULL saat INSERT, PG langsung throw `23514 "no partition found"`.
Trigger tidak pernah sempat mengisinya, meskipun trigger function-nya benar.

**Fix yang diterapkan (oleh Claude, disetujui ZCode):**
1. DROP trigger `set_raw_texts_month` + `set_sentiment_scores_month` + function
   `trg_set_partition_month()` — **trigger di partitioned table unreliable, JANGAN recreate.**
2. RPC `batch_insert_raw_texts` mengisi `ingested_at` + `ingested_month` secara eksplisit.
3. RPC `insert_sentiment_score` mengisi `scored_at` + `scored_month` eksplisit.
4. `GRANT EXECUTE` ke `service_role` untuk kedua function.

**File referensi:** `packages/db/migrations/001_fix_partition_key.sql`

### Bug #2: Materialized View tidak bisa punya RLS
**Root cause:** PostgreSQL tidak mendukung `ALTER TABLE ... ENABLE ROW LEVEL SECURITY` pada
materialized view (error 42809).
**Fix:** `GRANT SELECT ON mv_dashboard_summary TO anon, authenticated;` (bukan RLS policy).

### Bug #3: `CREATE OR REPLACE` tidak bisa ubah PROCEDURE → FUNCTION
**Root cause:** `batch_insert_raw_texts` di v1 dibuat sebagai PROCEDURE. PostgreSQL tidak
mengizinkan `CREATE OR REPLACE FUNCTION` mengubah routine kind.
**Fix:** `DROP ROUTINE IF EXISTS batch_insert_raw_texts(...)` sebelum CREATE FUNCTION.

---

## 📋 KEPUTUSAN ARSITEKTURAL YANG SUDAH DIAMBIL

### Keputusan 1: Free-tier only (dompet kosong)
- ✅ Supabase Free (500MB DB, 1GB storage)
- ✅ Hugging Face Spaces Free (NLP worker, CPU)
- ✅ Vercel Free (frontend)
- ✅ GitHub Actions free (public repo = unlimited minutes)
- ❌ NO Twitter/X API, NO YouTube Data API, NO Play Store scraper (semua berbayar)
- ❌ NO pgvector/embedding (boros CPU free tier)

### Keputusan 2: RSS-only untuk sumber data (Layer 1)
- Sumber aktif: Detik, CNN, Tempo, Republika, Antara, Kompas, Liputan6, JPNN, Tribun
- Google News RSS per-tokoh (14 feed) — saat ini 503 (rate-limited), belum di-fix
- Tidak ada sumber opini publik (Disqus, Twitter) — PENDING EVALUASI

### Keputusan 3: Framing project REFRAMED
Diskusi dengan Claude mengungkap bahwa RSS = tone media, bukan sentimen publik.
**Keputusan:**
- JANGAN reframe sekarang — lihat distribusi label real dulu (CLI NLP)
- Kalau distribusi >80% netral → pertimbangkan tambah sumber opini (YouTube comments)
- Kalau distribusi reasonable → news tone cukup representative, lanjut
- Target akurasi: 85-90% confidence di domain news (achievable)
- Target akurasi di domain social media: 75-80% (realistic)

### Keputusan 4: Historical data via archive scraper (BUKAN RSS)
Kebutuhan user: sentiment per-tokoh per-tahun (kumulatif per tahun, bukan total).
RSS tidak bisa kasih data 2019-2025. Solusi yang disepakati:
- **Arsip berita media** (scrape arsip Detik/Kompas/Tempo per-tanggal) — sumber utama
- ❌ Wayback Machine — terlalu noisy
- ❌ Kaggle dataset — hanya untuk validasi model, bukan production data
- Implementasi archive scraper: **BELUM DIMULAI**

### Keputusan 5: Hotline tokoh (dynamic priority scraping)
- Banyak tokoh di DB (18+), tapi scraping fokus ke tokoh yang sedang "hot"
- Deteksi: volume artikel N hari terakhir → top-N tokoh mendapat deep scrape
- Implementasi: **BELUM DIMULAI** (butuh tabel `hotlist_tokoh` + trend detector)

### Keputusan 6: CLI NLP tool untuk testing SEBELUM production worker
- Build CLI tool dulu (terminal, lihat distribusi real)
- Kalau distribusi OK → ganti ke ONNX model
- Kalau distribusi buruk → perbaiki data/sumber sebelum invest model
- CLI sudah dibuat: `nlp-worker/cli_test.py`
- Status CLI: **ERROR** (RPC dequeue gagal, perlu debug)

---

## ⚠️ MASALAH YANG MASIH ADA (Known Issues)

### Issue A — Google News RSS return 503 (semua gnews_* feed)
14+7 feed di-fetch concurrent → rate-limited. Non-blocking, general RSS cukup.

### Issue B — 5 feed general return 0 item (parser mismatch)
detik/antara/kompas/liputan6/jpnn: parser XML regex tidak match. Parser perlu update.

### Issue C — Tribunnews HTTP 403 Forbidden
Publisher blokir User-Agent non-browser. Skip dulu.

### Issue D — `last_run_at` tidak ter-update
`scraping_configs.last_run_at` hanya update kalau feed return ≥1 item. Design flaw.

### Issue E — `dequeue_nlp_batch` RPC error ~~(BARU)~~ **FIXED (sesi 3)**
Error `invalid input syntax for type json` — disebabkan queue orphan (LEFT JOIN NULL).
Fix: bersihkan queue, re-ingest data. RPC jalan normal setelah itu.

### Issue F — RSS body kosong, entity matching 0% **(BARU, sesi 3)**
Beberapa RSS feed (CNN, dll) hanya kirim title, body/text kosong.
`match_entities()` hanya scan `text` → 0 match.
**Fix:** Sudah diaplikasi — gabungkan `title + text` untuk scanning.
**Masih perlu verifikasi:** run `batch 50` setelah fix untuk cek match rate.
**Root cause di Edge Function:** mungkin perlu improve `fetchAndParse` untuk extract full article body.

---

> **NOTE:** Section "APA YANG MASIH HARUS DILAKUKAN" versi lama (berbasis layer L1-L6) sudah
> diarsipkan. Semua tugas sudah dikonsolidasikan ke:
> - **FOKUS UTAMA SAAT INI** (bagian atas file) — 4 fase
> - **Urutan Eksekusi FINAL** (di bawah Keputusan Arsitektural)
> - **SESI 8 REPORT** — Belum dikerjakan

---

## 📋 KEPUTUSAN ARSITEKTURAL

| # | Keputusan | Status | Note |
|---|---|---|---|
| 1 | Free-tier only (Supabase + HF Spaces + Vercel + GitHub Actions) | ✅ Final | |
| 2 | RSS-only Lapis 1 (23 feed aktif → target 70+ setelah seed 04) | ✅ Active | |
| 3 | ~~Evaluasi distribusi SEBELUM commit ke ONNX~~ | 🔄 **DIUBAH (sesi 7)** | Fokus kumpulkan data dulu, ONNX nanti |
| 4 | Historical: archive scraper / GDELT, bukan Kaggle/Wayback | ✅ Final | **PRIORITY #2 sekarang** |
| 5 | Hotline tokoh: dynamic priority scraping | ✅ Desain done | Implementasi di seed 03 |
| 6 | CLI dulu sebelum production worker | 🔄 **DIUBAH (sesi 7)** | CLI untuk debug saja, production worker nanti |
| 7 | Dummy model diabaikan, fokus ekspansi entitas + historical data | ✅ Final (sesi 7) | ONNX setelah data cukup |
| 8 | GitHub Actions sudah berjalan otomatis | ✅ Aktif (sesi 7) | Run #20-#26 sukses |
| 9 | **BARU:** Model NLP = 2-stage gated pipeline (relevancy → sentiment) | ✅ Final (sesi 8) | `indobert-relevancy-classifier` + `indobert-sentiment-classifier` |
| 10 | **BARU:** NULL entity score penting untuk national mood index | ✅ Final (sesi 8) | JANGAN hapus logic insert NULL — pakai `model_version` tag untuk bedakan |
| 11 | **BARU:** IndoBERT monolingual > ModernBERT multilingual untuk Indonesia | ✅ Final (sesi 8) | F1 0.9353 IndoBERT vs lower untuk mBERT/multilingual generik |
| 12 | **BARU:** Data Quality > Model Quality > Latency Optimization | ✅ Final (sesi 8) | Bottleneck = network, bukan inference |
| 13 | **BARU:** Ground truth evaluation WAJIB sebelum production | ✅ Final (sesi 8) | Toolkit sudah dibuat (export + eval_metrics) |
| 14 | **BARU:** Clean dummy data (confidence 0.65/0.60) sebelum eval model real | ✅ Final (sesi 8) | SQL diagnostic + delete contaminated rows |

---
## 🗺️ Urutan Eksekusi (FINAL — konsolidasi sesi 1-8)

```
SEKARANG (FASE 1: EKSPANSI DATA)
  → E1: Migration 007 di SQL Editor (WAJIB sebelum seed 03)
  → E2: Seed 03 (50+ tokoh) di SQL Editor
  → E3: Seed 04 (47 RSS configs) di SQL Editor
  → E4: Fix alias Prabowo (SQL 1 baris)
  → E5: Verifikasi entity count → harus 50+
  → Diagnostic: hapus sentiment_scores dari dummy (confidence 0.65/0.60)
      ↓
  → E6: Run auto_discover.py --source all
  → H1: Desain + implementasi archive scraper (historical 2019-2025)
  → H2: Backfill data per tokoh per tahun
  → H3: RPC get_yearly_sentiment
      ↓
PARALEL (bisa sambil historical berjalan)
  → 2c: Fix 5 feed parser mismatch
  → 2b: Fix gnews concurrency (503)
  → 2d: Fix last_run_at
      ↓
FASE 2: NLP MODEL (setelah data cukup)
  → Ground truth evaluation (export + eval_metrics)
  → IndoBERT ONNX production (2-stage gated)
  → Production NLP worker (auto-dequeue, batch inference)
      ↓
FASE 3: FRONTEND (paling akhir)
  → 6a-6d: Next.js dashboard
```

> **Catatan:** Prioritas Fase 1 adalah keputusan user (sesi 7).
> Sesi 8 menambahkan: clean dummy data, ground truth evaluation, 2-stage pipeline.
> Urutan ini mengkonsolidasi semua rekomendasi dari sesi 1-8.

---

## 🔐 Secrets & Credentials

| Secret | Lokasi | Status |
|---|---|---|
| `CRON_SECRET` | Supabase Dashboard → Edge Functions → Secrets | ✅ Set |
| `SUPABASE_SERVICE_ROLE_KEY` | Auto-injected ke Edge Function | ✅ |
| `SUPABASE_URL` | Auto-injected | ✅ |
| `SUPABASE_EDGE_FUNCTION_URL` | GitHub Actions Secrets | ✅ **Sudah set (run #20-#26 sukses)** |
| `SUPABASE_ANON_KEY` | GitHub Actions Secrets | ✅ **Sudah set** |
| `CRON_SECRET` | GitHub Actions Secrets | ✅ **Sudah set** |

---

## 📁 Struktur repo (current — konsolidasi sesi 1-8)

```
ID-Political-Sentiment-Tracker/
├── apps/
│   └── web/                               ← Next.js (belum dibangun)
├── packages/
│   ├── db/
│   │   ├── schema.sql                     ← ⚠️ JANGAN RUN ULANG
│   │   ├── migrations/
│   │   │   ├── 001_fix_partition_key.sql  ✅ applied
│   │   │   ├── 002_pgmq_queue.sql         ✅ applied
│   │   │   ├── 003_allow_null_entity.sql  ✅ applied
│   │   │   ├── 004_purge_empty_text.sql   ✅ applied
│   │   │   ├── 005_dequeue_add_source_url.sql ✅ applied
│   │   │   ├── 006_fix_dequeue_html.sql   ✅ applied (sesi 5)
│   │   │   └── 007_entity_expansion_schema.sql ⏳ dibuat, belum run
│   │   └── seeds/
│   │       ├── 01_political_entities.sql  ✅ 18 tokoh (production)
│   │       ├── 02_scraping_configs.sql    ✅ 23 configs (production)
│   │       ├── 03_entities_comprehensive.sql  ⏳ 50+ tokoh (belum run)
│   │       └── 04_scraping_configs_expanded.sql ⏳ 47 configs (belum run)
│   └── nlp-worker/
│       ├── cli_test.py                    ✅ working (dummy model)
│       ├── sentiment_model.py             ✅ 2-stage gated pipeline v2 (sesi 8)
│       ├── test_sentiment_model.py        ✅ test relevancy gate v2 (sesi 8)
│       ├── export_sentiment_ground_truth.py ✅ stratified export (sesi 8)
│       ├── export_relevancy_review.py      ✅ kandidat relevancy (sesi 8)
│       ├── eval_metrics.py                ✅ P/R/F1 evaluator (sesi 8)
│       ├── GROUND_TRUTH_EVAL_GUIDE.md     ✅ panduan evaluasi (sesi 8)
│       ├── requirements.txt              ← requests, trafilatura, supabase, transformers
│       ├── entity_discovery/
│       │   ├── auto_discover.py           ← auto-discovery 3 sumber (sesi 6)
│       │   ├── requirements.txt
│       │   └── README.md
│       └── README.md
├── infra/
│   └── supabase/
│       ├── config.toml
│       └── functions/rss-ingestion/
│           └── index.ts                  ✅ deployed
├── .github/
│   └── workflows/
│       └── trigger-ingestion.yml         ✅ AKTIF — run #20-#26 sukses
├── docs/
│   ├── architecture.md
│   ├── workflow.drawio
│   └── internal/
│       └── HANDOFF_STATUS.md             ← FILE INI
├── .env.example
└── README.md
```

---

## ⚠️ PERINGATAN UNTUK AI LAIN (Claude/GLM)

1. **JANGAN recreate trigger `set_raw_texts_month` / `set_sentiment_scores_month` / `trg_set_partition_month()`.
   Mereka sudah di-DROP karena unreliable di partitioned table. Partition key diisi eksplisit
   di RPC. Kalau Anda baca `schema.sql` dan lihat trigger-nya, itu KODE LAMA — JANGAN apply.**

2. **JANGAN run `schema.sql` di production.** File itu untuk setup awal saja.
   Gunakan `migration_*.sql` untuk perubahan incremental.

3. **JANGAN tambah RLS policy ke `mv_dashboard_summary`.** Materialized views tidak mendukung
   RLS (error 42809). Pakai `GRANT SELECT TO anon` saja.

4. **`get_entity_ranking` sudah di-clean** — overload versi lama (tanpa `p_min_confidence`)
   sudah di-drop. Tinggal 1 versi dengan 3 args.

5. **Supabase auto-inject secrets:** `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, dan
   `SUPABASE_ANON_KEY` tersedia otomatis di Edge Function. Tidak perlu set manual.
   Satu-satunya custom secret: `CRON_SECRET`.

6. **`match_entities()` sekarang gabungkan `title + text`.** RSS feed sering kirim body kosong,
   jadi entity matching DAN sentiment prediction harus scan title+text gabungan. Jangan
   revert ke hanya scan `text`.

7. **Lapis 2 (2-stage scraping) butuh dependency Python: `trafilatura` + `requests`.**
   Kalau ImportError saat run CLI → `pip install trafilatura requests`. Fungsi `enrich_if_needed()`
   otomatis fallback ke title-only kalau library tidak ada (tidak crash).

8. **`dequeue_nlp_batch` sekarang return kolom `source_url`.** Migration `005_dequeue_add_source_url.sql`
   WAJIB di-run sebelum Lapis 2 bisa fetch full body. Tanpa `source_url`, enrich tidak bisa follow gnews redirect.

---

## 🔍 Cara verifikasi cepat (kapan saja)

```sql
-- Pipeline health check
SELECT status, COUNT(*) FROM raw_texts GROUP BY status;
SELECT COUNT(*) FROM pgmq.q_nlp_processing_queue;
SELECT COUNT(*) FROM sentiment_scores;
SELECT queue_name FROM pgmq.meta WHERE queue_name = 'nlp_processing_queue';
```

```powershell
# Manual trigger function
$CRON_SECRET = "<secret-anda>"
$ANON_KEY = "<anon-key>"
curl.exe -X POST `
  -H "Authorization: Bearer $ANON_KEY" `
  -H "x-cron-secret: $CRON_SECRET" `
  "supabase.co/functions/v1/rss-ingestion"
```

```powershell
# CLI testing (set env dulu)
$env:SUPABASE_URL = "supabase.co"
$env:SUPABASE_SERVICE_ROLE_KEY = "eyJ..."
python cli_test.py stats
python cli_test.py inspect
python cli_test.py batch 10
```
