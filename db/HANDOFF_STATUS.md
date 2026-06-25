# HANDOFF STATUS — ID-Sentiment-Tracker

> **Tgl update:** 2026-06-25 (sesi 3)
> **Project Ref:** `bawvxtivogcuwvqdqoae`
> **Repo:** `raynzz455/ID-Political-Sentiment-Tracker.git`
> **Status:** ⚠️ DB sudah re-ingest (data kembali). Queue drain working. BUG: text kosong dari RSS → entity matching 0.

Dokumen ini adalah **single source of truth** untuk sinkronisasi antar asisten AI
(GLM/ZCode ↔ Claude). Setiap perubahan production DB atau code WAJIB update dokumen ini.

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
- Penyebab diduga: `schema_final_v2.sql` di-run ulang (DROP TABLE CASCADE)
- **PEMBELAJARAN:** JANGAN run `schema_final_v2.sql` ulang. Gunakan `migration_*.sql` saja.

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

**File referensi:** `db/migration_fix_partition_key.sql`

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

---

## 🔐 Secrets & Credential (RAHASIA — jangan commit)

| Secret | Lokasi | Catatan |
|---|---|---|
| `CRON_SECRET` | Dashboard → Edge Functions → Secrets | String hex 64-char |
| `SUPABASE_SERVICE_ROLE_KEY` | Dashboard → Settings → API | Auto-injected ke edge function |
| `SUPABASE_URL` | Auto-injected | `https://bawvxtivogcuwvqdqoae.supabase.co` |

**GitHub Actions Secrets (BELUM setup):**
| Secret | Value |
|---|---|
| `SUPABASE_EDGE_FUNCTION_URL` | `https://bawvxtivogcuwvqdqoae.supabase.co/functions/v1/rss-ingestion` |
| `SUPABASE_ANON_KEY` | `eyJ...` (anon key) |
| `CRON_SECRET` | (sama dengan di Edge Function) |

---

## 📁 Lokasi file penting

```
Bentar lagi di grebek/
├── supabase/
│   ├── config.toml                          ← dari `supabase init`
│   └── functions/rss-ingestion/index.ts     ← Edge Function Layer 2 (CRON_SECRET + enqueue)
├── db/
│   ├── schema_final_v2.sql                 ← ⚠️ JANGAN RUN ULANG — HAPUS DATA
│   ├── migration_pgmq_queue.sql             ← queue + RPC enqueue/dequeue/ack
│   ├── migration_fix_partition_key.sql      ← HOTFIX ingested_month (sudah applied)
│   ├── migration_allow_null_entity.sql     ← ALTER entity_id DROP NOT NULL (belum di-run)
│   ├── HANDOFF_STATUS.md                   ← FILE INI (single source of truth)
│   └── seed/
│       ├── 01_political_entities.sql        ← 18+ tokoh politik + foto
│       └── 02_scraping_configs.sql          ← 23 RSS configs
├── nlp-worker/
│   ├── cli_test.py                          ← CLI testing tool (dummy model, FIXED sesi 3)
│   ├── requirements.txt                     ← pip dependencies
│   └── README.md
├── ingestion/
│   ├── trigger-ingestion.yml               ← GitHub Actions (dengan CRON_SECRET + jitter)
│   └── README2.md
├── frontend/
│   └── README.md
├── docs/
│   ├── architecture.md                      ← dari ai.md (aturan PDP, 6-layer)
│   └── workflow.drawio
└── .env.example
```

---

## ⚠️ PERINGATAN UNTUK AI LAIN (Claude/GLM)

1. **JANGAN recreate trigger `set_raw_texts_month` / `set_sentiment_scores_month` / `trg_set_partition_month()`.
   Mereka sudah di-DROP karena unreliable di partitioned table. Partition key diisi eksplisit
   di RPC. Kalau Anda baca `schema_final_v2.sql` dan lihat trigger-nya, itu KODE LAMA — JANGAN apply.**

2. **JANGAN run `schema_final_v2.sql` di production.** File itu untuk setup awal saja.
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
  "https://bawvxtivogcuwvqdqoae.supabase.co/functions/v1/rss-ingestion"
```

```powershell
# CLI testing (set env dulu)
$env:SUPABASE_URL = "https://bawvxtivogcuwvqdqoae.supabase.co"
$env:SUPABASE_SERVICE_ROLE_KEY = "eyJ..."
python cli_test.py stats
python cli_test.py inspect
python cli_test.py batch 10
```
