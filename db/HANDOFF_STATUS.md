# HANDOFF STATUS — ID-Sentiment-Tracker

> **Tgl:** 2026-06-25
> **Project Ref:** `bawvxtivogcuwvqdqoae`
> **Status:** ✅ Layer 1-3 WORKING. ✅ Schema 100% verified ready untuk Layer 4 & 6.

Dokumen ini adalah **single source of truth** untuk sinkronisasi antar asisten AI
(GLM/ZCode ↔ Claude). Setiap perubahan production DB atau code WAJIB update dokumen ini.

---

## 🎯 SCHEMA READINESS — VERIFIED (2026-06-25)

| Komponen | Status | Evidence |
|---|---|---|
| Tabel + partisi | ✅ | Dump: 6 partisi 2026-06/07/08 (raw + sentiment) |
| RLS policies | ✅ 12 policies | UU PDP compliant, raw_texts/sentiment blocked anon |
| RPC functions | ✅ 13 functions | Termasuk NLP worker + frontend query endpoints |
| MV `mv_dashboard_summary` | ✅ ADA | Frontend bisa query agregat |
| Trigger partition key | ✅ FIXED | Dihapus, RPC isi `ingested_month` eksplisit |
| Pipeline ingestion | ✅ WORKING | `total_inserted: 50, enqueued: 200` |

**Verdict: Schema siap untuk Layer 4 (NLP worker) & Layer 6 (frontend).**

### ⚠️ Satu cleanup tersisa (opsional, 1 baris)
Ada 2 versi `get_entity_ranking` (overloading ambigu di PostgREST).
Drop versi lama tanpa `p_min_confidence`:
```sql
DROP FUNCTION IF EXISTS get_entity_ranking(integer, integer);
```

---

## ✅ Yang SUDAH BISA (Working)

### Layer 1-3: RSS → DB → Queue — FULLY OPERATIONAL

Output curl terakhir (2026-06-25):
```json
{
  "ok": true,
  "total_inserted": 50,
  "enqueued": 200,
  "summary": {
    "tempo_nasional":       { "items_parsed": 50,  "inserted": 50, "duplicates": 0   },
    "republika_politik":    { "items_parsed": 15,  "inserted": 0,  "duplicates": 15  },
    "cnnindonesia_nasional":{ "items_parsed": 100, "inserted": 0,  "duplicates": 100 }
  }
}
```

**Interpretasi:**
- `tempo`: 50 artikel baru masuk ✓
- `cnnindonesia + republika`: 115 artikel sudah ada sebelumnya (dari run pertama) →
  ter-deduplikasi dengan benar (dedup via `raw_text_hashes` bekerja)
- `enqueued: 200`: pgmq queue terisi, NLP worker siap consume

### Yang sudah ter-setup
- ✅ Supabase project `bawvxtivogcuwvqdqoae` aktif
- ✅ Extension `pgmq` enabled
- ✅ Schema inti + seeder (political_entities + scraping_configs)
- ✅ Edge Function `rss-ingestion` deployed (`--no-verify-jwt`)
- ✅ Secret `CRON_SECRET` set di Dashboard
- ✅ RPC `batch_insert_raw_texts` + `enqueue_pending_raw_texts` working
- ✅ PGMQ queue `nlp_processing_queue` aktif + terisi
- ✅ Partisi `raw_texts_2026_06/07/08` dan `sentiment_scores_2026_06/07/08` dibuat

---

## 🔧 BUG KRITIKAL YANG SUDAH DIOBATI — JANGAN DIROLLBACK

### Bug: `ingested_month = NULL` → insert gagal ke partitioned table
**Root cause** (sudah dikonfirmasi production):
PostgreSQL melakukan **partition routing SEBELUM BEFORE INSERT trigger** fire. Saat
`ingested_month` NULL saat INSERT, PG langsung throw `23514 "no partition found"`.
Trigger tidak pernah sempat mengisinya, meskipun trigger function-nya benar.

**Fix yang diterapkan:**
1. DROP trigger `set_raw_texts_month` + `set_sentiment_scores_month` + function
   `trg_set_partition_month()` — trigger di partitioned table unreliable, JANGAN recreate.
2. RPC `batch_insert_raw_texts` mengisi `ingested_at` + `ingested_month` secara eksplisit:
   ```sql
   v_now   TIMESTAMPTZ := NOW();
   v_month DATE := date_trunc('month', NOW())::date;
   -- INSERT ... VALUES (..., v_now, v_month)
   ```
3. RPC `insert_sentiment_score` mengisi `scored_at` + `scored_month` eksplisit (sama).
4. `GRANT EXECUTE` ke `service_role` untuk kedua function — SECURITY DEFINER tetap butuh grant.

**File referensi:** `db/migration_fix_partition_key.sql`

### Bug: Pernyataan `supabase init` + struktur folder
- Folder `supabase/` harus di **root repo** (bukan di dalam `ingestion/`).
- `supabase init` membuat `config.toml` di `supabase/config.toml`.
- Deploy dari root repo: `supabase functions deploy rss-ingestion --no-verify-jwt`.

---

## ⚠️ MASALAH YANG MASIH ADA (Known Issues, Prioritas Rendah)

### Issue A — Google News RSS return 503 (semua gnews_* feed)
**Gejala:** 14 feed `gnews_*` + 7 feed `gnews_site_*` semua `[FETCH_ERROR] HTTP 503`.
**Penyebab:** Google News rate-limit karena 21 feed di-fetch **concurrent** (`Promise.allSettled`)
dari IP Supabase Edge yang sama. Dianggap bot spam.
**Fix (belum diterapkan):** Batasi concurrency gnews. Misal batch 3-3 dengan delay 2 detik,
atau pakai `Promise.allSettled` tapi sequential per gnews group.
**Dampak:** Tidak blocking — 9 general RSS feed (Tempo/CNN/Republika/Detik dll) sudah cukup.

### Issue B — 5 feed general return 0 item (parser mismatch)
**Gejala:** `detik_politik`, `antara_nasional`, `kompas_nasional`, `liputan6_politik`,
`jpnn_nasional` parse sukses (tidak 403) tapi `items_parsed: 0`.
**Penyebab:** Parser XML regex di `index.ts` tidak match struktur RSS publisher berikut.
Kemungkinan: tag `<item>` di-wrapped di dalam namespace (mis. media namespace), atau
strukturnya beda (mis. Atom feed `<entry>` bukan RSS `<item>`).
**Fix (belum diterapkan):** Update parser di `ingestion/supabase/functions/rss-ingestion/index.ts`.

### Issue C — Tribunnews HTTP 403 Forbidden
**Gejala:** Feed di-blokir publisher.
**Penyebab:** Tribunnews (jaringan Kompas Gramedia) sering blokir User-Agent non-browser.
**Fix:** Tidak ada quick fix selain rotasi User-Agent (grey area TOS). Skip dulu.

### Issue D — `last_run_at` tidak ter-update
**Gejala:** Kolom `scraping_configs.last_run_at` tetap NULL walau function jalan.
**Penyebab:** Di `index.ts`, `last_run_at` hanya di-update di dalam blok `insertBatch()`.
Kalau feed return 0 item, update di-skip. Design flaw.
**Fix:** Pindah update `last_run_at` ke luar conditional, jalankan untuk semua feed aktif.

---

## 📋 APA YANG MASIH HARUS DILAKUKAN (Prioritas tinggi → rendah)

| # | Tugas | Layer | File/lokasi | Status |
|---|--- |---|---|---|
| 1 | Setup GitHub Actions scheduler (cron tiap 30 menit) | 2 | `.github/workflows/trigger-ingestion.yml` | ⏳ Belum |
| 2 | Build NLP Worker (Python + ONNX) — dequeue queue, IndoBERT inference | 4 | `nlp-worker/` | ⏳ Belum |
| 3 | Test dequeue end-to-end (worker → sentiment_scores terisi) | 4-5 | DB + worker | ⏳ Belum |
| 4 | Build Next.js dashboard | 6 | `frontend/` | ⏳ Belum |
| AI | Fix Issue A (gnews 503) — concurrency limit | 2 | `index.ts` | ⏳ Belum |
| AI | Fix Issue B (parser mismatch detik/kompas/dll) | 2 | `index.ts` | ⏳ belum |
| AI | Fix Issue D (`last_run_at` selalu update) | 2 | `index.ts` + DB | ⏳ Belum |

---

## 🗺️ Urutan eksekusi yang disarankan

```
Sekarang        → Setup GitHub Actions (otorisasi 30-min cron)
                ↓
Setelah itu     → Bangun NLP Worker (test dequeue dari queue dulu, tanpa model dulu)
                ↓
                → Connect NLP Worker ke IndoBERT ONNX
                ↓
                → Build frontend (dashboard)
```

---

## 🔐 Secrets & Credential (RAHASIA — jangan commit)

| Secret | Lokasi | Catatan |
|---|---|---|
| `CRON_SECRET` | Dashboard → Edge Functions → Secrets | String hex 64-char |
| `SUPABASE_SERVICE_ROLE_KEY` | Dashboard → Settings → API | Auto-injected ke edge function, tidak perlu set manual |
| `SUPABASE_URL` | Auto-injected | `https://bawvxtivogcuwvqdqoae.supabase.co` |

**GitHub Actions Secrets (belum setup):**
| Secret | Value |
|---|--- working |
| `SUPABASE_EDGE_FUNCTION_URL` | `https://bawvxtivogcuwvqdqoae.supb.co/functions/v1/rss-ingestion` |
| `SUPABASE_ANON_KEY` | `eyJ...` (anon key) |
| `CRON_SECRET` | (sama dengan di Edge Function) |

---

## 📁 Lokasi file penting

```
Bentar lagi di grebek/
├── supabase/
│   ├── config.toml
│   └── functions/rss-ingestion/index.ts   ← Edge Function (Layer 2)
├── db/
│   ├── schema_final_v2.sql                ← schema inti (partitioned, RLS, RPC, MV)
│   ├── migration_pgmq_queue.sql           ← queue + RPC enqueue/dequeue/ack
│   ├── migration_fix_partition_key.sql    ← HOTFIX ingested_month explicit
│   ├── HANDOFF_STATUS.md                  ← FILE INI
│   └── seed/
│       ├── 01_political_entities.sql      ← 18+ tokoh politik
│       HANDOFF_STATUS.md (file ini)        ← 23 RSS configs
├── ingestion/
│   ├── trigger-ingestion.yml              ← GitHub Actions workflow (template)
│   └── README2.md
└── docs/
    ├── architecture.md
    └── workflow.drawio
```

---

## 🔍 Cara verifikasi cepat (kapan saja)

```sql
-- Pipeline health check
SELECT status, COUNT(*) FROM raw_texts GROUP BY status;        -- 'queued' harus tumbuh
SELECT COUNT(*) FROM pgmq.q_nlp_processing_queue;              -- queue depth
SELECT config_name, last_run_at FROM scraping_configs ORDER BY last_run_at DESC;
```

```powershell
# Manual trigger function
$CRON_SECRET = "<nilai-anda>"
$ANON_KEY = "<anon-key>"
curl.exe -X POST `
  -H "Authorization: Bearer $ANON_KEY" `
  -H "x-cron-secret: $CRON_SECRET" `
  "https://bowvxtivogcuwvqdqoae.supabase.co/functions/v1/rss-ingestion"
```
