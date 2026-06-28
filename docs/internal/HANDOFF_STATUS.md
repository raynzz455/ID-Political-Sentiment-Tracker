# HANDOFF STATUS — ID-Sentiment-Tracker

> **Tgl update:** 2026-06-28 (sesi 5)
> **Project Ref:** `bawvxti***` (lihat Supabase Dashboard)
> **Repo:** `raynzz455/ID-Political-Sentiment-Tracker.git`
> **Status:** ✅ Pipeline L2→L3→L4 end-to-end proven. HTML clean. Entity match working.
> Menunggu: IndoBERT ONNX (4b), GitHub Actions (2a), alias fix Prabowo.

Dokumen ini adalah **single source of truth** untuk sinkronisasi antar asisten AI
(GLM/ZCode ↔ Claude). Setiap perubahan production DB atau code WAJIB update dokumen ini.

---

> **Tgl update:** 2026-06-26 (sesi 4)
> **Project Ref:** `bawvxti***` (lihat Supabase Dashboard)
> **Repo:** `raynzz455/ID-Political-Sentiment-Tracker.git`
> **Status:** ✅ Arsitektur 3-lapis didesain. Lapis 2 (2-stage scraping) terimplementasi. Repo di-restructure ke monorepo.

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

## 📋 APA YANG MASIH HARUS DILAKUKAN

### URGENT (sebelum apapun)
| # | Tugas | Status |
|---|---|---|
| U1 | ~~Bersihkan queue orphan~~ | ✅ Done (sesi 3) |
| U2 | ~~Re-ingest data via curl~~ | ✅ Done (sesi 3) |
| U3 | ~~Debug `dequeue_nlp_batch` RPC error~~ | ✅ Fixed (sesi 3, queue orphan cause) |
| U4 | Verifikasi entity match rate setelah title+text fix | ⏳ **lanjutkan** |

### Layer 4 — NLP Worker
| # | Tugas | Status |
|---|---|---|
| 4a | ~~Fix CLI tool error~~ | ✅ Done (sesi 3 — queue drain + match fix) |
| 4b | Ganti `predict_sentiment()` ke IndoBERT ONNX | ⏳ |
| 4c | Validasi akurasi (target 85-90% di domain news) | ⏳ |
| 4d | Build production worker (poll queue, not CLI) | ⏳ |

### Layer 2 — Ingestion improvement
| # | Tugas | Status |
|---|---|---|
| 2a | Setup GitHub Actions scheduler (cron 30 min) | ⏳ |
| 2b | Fix Issue A: gnews concurrency limit | ⏳ |
| 2c | Fix Issue B: parser mismatch 5 feed | ⏳ |
| 2d | Fix Issue D: `last_run_at` always update | ⏳ |

### Historical data (Layer baru — belum dimulai)
| # | Tugas | Status |
|---|---|---|
| H1 | Desain archive scraper (Detik/Kompas/Tempo per-tanggal) | ⏳ |
| H2 | Tambah RPC `get_yearly_sentiment` (breakdown per-tahun) | ⏳ |
| H3 | Implementasi hotline tokoh (dynamic priority) | ⏳ |
| H4 | Backfill data 2019-2025 via archive scraper | ⏳ |

### Layer 6 — Frontend (teman user)
| # | Tugas | Status |
|---|---|---|
| 6a | Inisialisasi Next.js + Supabase client | ⏳ |
| 6b | Dashboard ranking tokoh + foto | ⏳ |
| 6c | Time-series per tokoh (per-tahun) | ⏳ |
| 6d | Highlight berita positif/negatif per tokoh | ⏳ |

---

## 📋 KEPUTUSAN ARSITEKTURAL

| # | Keputusan | Status |
|---|---|---|
| 1 | Free-tier only (Supabase + HF Spaces + Vercel + GitHub Actions) | ✅ Final |
| 2 | RSS-only Lapis 1 (23 feed aktif) | ✅ Active |
| 3 | Evaluasi distribusi SEBELUM commit ke ONNX | ✅ — distribusi belum bisa dievaluasi (dummy model) |
| 4 | Historical: archive scraper / GDELT, bukan Kaggle/Wayback | ✅ Final, belum diimplementasi |
| 5 | Hotline tokoh: dynamic priority scraping | ✅ Desain done, belum diimplementasi |
| 6 | CLI dulu sebelum production worker | ✅ CLI proven, siap ke ONNX |

---
## 🗺️ Urutan eksekusi yang disarankan (DIUPDATE)

```
SEKARANG (URGENT)
  → U1: Bersihkan queue orphan
  → U2: Re-ingest data (curl)
  → U3: Debug RPC dequeue error
  → U4: CLI tool jalan + lihat distribusi real
      ↓
  EVALUASI DISTRIBUSI
  → Kalau >80% neutral → tambah sumber opini (Issue Keputusan 3)
  → Kalau reasonable → lanjut ONNX model
      ↓
  NLP MODEL
  → 4b: IndoBERT ONNX replace dummy
  → 4c: Validasi 85-90% target
  → 4d: Production worker
      ↓
  OTOMASI + HISTORIS
  → 2a: GitHub Actions cron
  → H1-H2: Archive scraper + yearly RPC
  → H3: Hotline tokoh
      ↓
  FRONTEND
  → 6a-6d: Dashboard (teman user kerjakan)
```
## 🗺️ Urutan eksekusi recommended (By Claude)

```
SEKARANG (< 5 menit total)
  → FIX-1: SQL fix alias Prabowo (30 detik)
  → 2a: GitHub Actions 3 secrets (2 menit)
      ↓
SESI BERIKUTNYA
  → 4b: IndoBERT ONNX (ganti dummy predict_sentiment)
  → 4c: Validasi distribusi real
  → 4d: Production worker di HF Spaces
      ↓
PARALEL (bisa dikerjakan sambil ONNX running)
  → 2c: Fix parser mismatch 5 feed
  → H1-H4: Archive scraper + backfill
  → L3: YouTube comments
      ↓
SETELAH DATA CUKUP
  → 6a-6d: Frontend Next.js
```

---

## 🔐 Secrets & Credentials

| Secret | Lokasi | Status |
|---|---|---|
| `CRON_SECRET` | Supabase Dashboard → Edge Functions → Secrets | ✅ Set |
| `SUPABASE_SERVICE_ROLE_KEY` | Auto-injected ke Edge Function | ✅ |
| `SUPABASE_URL` | Auto-injected | ✅ |
| `SUPABASE_EDGE_FUNCTION_URL` | GitHub Actions Secrets | ⏳ BELUM |
| `SUPABASE_ANON_KEY` | GitHub Actions Secrets | ⏳ BELUM |
| `CRON_SECRET` | GitHub Actions Secrets | ⏳ BELUM |

---

## 📁 Struktur repo (current)

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
│   │   │   └── 006_fix_dequeue_html.sql   ✅ applied (sesi 5)
│   │   └── seeds/
│   │       ├── 01_political_entities.sql  ✅ 18 tokoh
│   │       └── 02_scraping_configs.sql    ✅ 23 configs
│   └── nlp-worker/
│       ├── cli_test.py                    ✅ working (dummy model)
│       ├── requirements.txt              ← requests, trafilatura, supabase
│       └── README.md
├── infra/
│   └── supabase/
│       ├── config.toml
│       └── functions/rss-ingestion/
│           └── index.ts                  ✅ deployed (cleanText fix)
├── .github/
│   └── workflows/
│       └── trigger-ingestion.yml         ✅ ready — secrets belum diset
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
