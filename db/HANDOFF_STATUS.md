# HANDOFF STATUS — ID-Sentiment-Tracker

> **Tgl update:** 2026-06-25 (sesi 2)
> **Project Ref:** `bawvxtivogcuwvqdqoae`
> **Repo:** `raynzz455/ID-Political-Sentiment-Tracker.git`
> **Status:** ⚠️ Layer 1-3 pernah working, tapi DB saat ini KOSONK. Layer 4-6 belum dimulai.

Dokumen ini adalah **single source of truth** untuk sinkronisasi antar asisten AI
(GLM/ZCode ↔ Claude). Setiap perubahan production DB atau code WAJIB update dokumen ini.

---

## 🚨 STATUS TERKINI — DB KOSONK (2026-06-25 sesi 2)

### Apa yang terjadi
1. Sesi 1: Pipeline berhasil — curl return `total_inserted: 50, enqueued: 200`. Tempo 50, CNN 100 (dedup), Republika 15 (dedup). Data masuk, queue terisi.
2. Sesi 2: `python cli_test.py stats` return `raw_texts TOTAL 0, sentiment_scores total 0`. **SEMUA DATA HILANG.**

### Penyebab diduga
Antara sesi 1 dan 2, kemungkinan:
- `schema_final_v2.sql` di-run ulang → file itu diawali `DROP TABLE IF EXISTS raw_texts CASCADE` → semua data terhapus
- Atau query DROP/ALTER lain yang menghapus data

**PEMBELAJARAN:** JANGAN run `schema_final_v2.sql` ulang di production. File itu untuk setup awal saja.
Setelah DB jalan, gunakan `migration_*.sql` untuk perubahan incremental.

### Konsekuensi saat ini
- `raw_texts` kosong → tidak ada data untuk diproses NLP
- `pgmq.q_nlp_processing_queue` berisi ~200 message **orphan** (menunjuk `raw_text_id` yang sudah hilang)
- CLI tool `cli_test.py` error saat panggil `dequeue_nlp_batch` — kemungkinan karena message orphan menyebabkan LEFT JOIN ke `raw_texts` return NULL lalu jsonb parse gagal

### Yang harus dilakukan URGENT
1. **Bersihkan queue orphan:**
   ```sql
   SELECT pgmq.purge('nlp_processing_queue');
   -- atau kalau purge tidak ada:
   DELETE FROM pgmq.q_nlp_processing_queue;
   ```
2. **Re-ingest data:**
   ```powershell
   $CRON_SECRET = "<secret-anda>"
   $ANON_KEY = "<anon-key>"
   curl.exe -X POST `
     -H "Authorization: Bearer $ANON_KEY" `
     -H "x-cron-secret: $CRON_SECRET" `
     "https://bawvxtivogcuwvqdqoae.supabase.co/functions/v1/rss-ingestion"
   ```
3. **Verifikasi data kembali:**
   ```sql
   SELECT status, COUNT(*) FROM raw_texts GROUP BY status;
   SELECT COUNT(*) FROM pgmq.q_nlp_processing_queue;
   ```
4. **JANGAN run `schema_final_v2.sql` lagi** — semua DROP TABLE akan menghapus ulang data

### Error RPC `dequeue_nlp_batch` saat testing CLI
```
APIError: {'message': 'invalid input syntax for type json', 'code': '22P02',
           'details': 'Token "(" is invalid.'}
```
**Penyebab diduga:** Function `dequeue_nlp_batch` di `migration_pgmq_queue.sql` melakukan
`LEFT JOIN raw_texts rt ON rt.id = ((m->'message')->>'raw_text_id')::UUID` — saat
message orphan (raw_text_id tidak ada di raw_texts), LEFT JOIN return NULL. Kemudian
`jsonb_array_elements(v_msgs)` gagal parse output dari `pgmq.read` karena queue corruption.

**Debug yang perlu dilakukan:**
```sql
-- Test A: panggil function langsung di SQL Editor
SELECT * FROM dequeue_nlp_batch(60, 5);

-- Test B: cek apakah ada overload (seperti kasus get_entity_ranking)
SELECT proname, pg_get_function_arguments(oid)
FROM pg_proc WHERE proname = 'dequeue_nlp_batch';

-- Test C: cek isi queue
SELECT msg_id, vt, enqueued_at FROM pgmq.q_nlp_processing_queue LIMIT 5;
```
Kalau Test A error juga → function perlu di-rewrite. Kalau Test A sukses → masalah di Python client.

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

### Issue E — `dequeue_nlp_batch` RPC error (BARU)
`invalid input syntax for type json` saat dipanggil dari Python client.
Bersamaan dengan raw_texts TOTAL 0 (data hilang). Kemungkinan terkait queue orphan.
Perlu debug: cek Test A/B/C di bagian "Yang harus dilakukan URGENT" di atas.

---

## 📋 APA YANG MASIH HARUS DILAKUKAN

### URGENT (sebelum apapun)
| # | Tugas | Status |
|---|---|---|
| U1 | Bersihkan queue orphan (`DELETE FROM pgmq.q_nlp_processing_queue`) | ⏳ |
| U2 | Re-ingest data via curl edge function | ⏳ |
| U3 | Debug `dequeue_nlp_batch` RPC error (Test A/B/C di SQL Editor) | ⏳ |
| U4 | Verifikasi CLI tool bisa peek queue + proses item | ⏳ |

### Layer 4 — NLP Worker
| # | Tugas | Status |
|---|---|---|
| 4a | Fix CLI tool error + test distribusi dummy model | ⏳ |
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
│   ├── HANDOFF_STATUS.md                   ← FILE INI (single source of truth)
│   └── seed/
│       ├── 01_political_entities.sql        ← 18+ tokoh politik + foto
│       └── 02_scraping_configs.sql          ← 23 RSS configs
├── nlp-worker/
│   ├── cli_test.py                          ← CLI testing tool (dummy model, error saat ini)
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
