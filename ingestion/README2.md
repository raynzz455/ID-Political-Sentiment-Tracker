# Layer 1 & 2 — RSS Ingestion

> **Layer 1** = sumber data (RSS feeds, dikonfigurasi di `scraping_configs`)  
> **Layer 2** = Edge Function Deno yang fetch, parse, dan insert ke Supabase

---

## Arsitektur ingestion

```
GitHub Actions (cron tiap 30 menit)
  └─► POST /functions/v1/rss-ingestion       ← Edge Function (Deno, Supabase)
        ├── SELECT scraping_configs WHERE is_active = true
        ├── FETCH semua RSS feed (concurrent, network I/O bebas dari CPU limit)
        ├── PARSE <item> tags (no dependency, pure regex/string)
        └── RPC batch_insert_raw_texts()      ← insert ke raw_texts + dedup
              └── raw_texts.status = 'pending' ← NLP worker picks up next
```

---

## Setup (urutan wajib)

### 1. Deploy ke Supabase

```bash
# Install Supabase CLI jika belum
npm install -g supabase

# Login
supabase login

# Link ke project
supabase link --project-ref <project-ref>
# project-ref ada di: Supabase Dashboard → Settings → General → Reference ID

# Deploy Edge Function
supabase functions deploy rss-ingestion --no-verify-jwt
# --no-verify-jwt: kita pakai anon key + Authorization header, tidak perlu JWT user
```

### 2. Set environment variables di Supabase

Di Supabase Dashboard → Edge Functions → `rss-ingestion` → Secrets, tambahkan:

| Secret | Value |
|--------|-------|
| `SUPABASE_URL` | `https://<project>.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | `eyJ...` (service_role key, **bukan** anon) |

> ⚠️ **JANGAN** pakai anon key di Edge Function untuk write — raw_texts punya RLS block anon.

### 3. Seed database

Jalankan di Supabase SQL Editor, **urutan wajib**:

```sql
-- a. Pastikan schema sudah ter-apply (db/schema_final_v2.sql)
-- b. Insert tokoh politik
-- (paste isi seed/01_political_entities.sql)

-- c. Insert RSS configs
-- (paste isi seed/02_scraping_configs.sql)
```

### 4. Test Edge Function

```bash
# Test via curl (gunakan anon key untuk invoke, bukan service_role)
curl -X POST \
  -H "Authorization: Bearer <anon-key>" \
  -H "Content-Type: application/json" \
  "https://<project>.supabase.co/functions/v1/rss-ingestion"

# Expected response:
# {
#   "ok": true,
#   "total_inserted": 47,
#   "summary": {
#     "detik_politik": { "items_parsed": 20, "inserted": 18, "duplicates": 2 },
#     ...
#   }
# }
```

### 5. Setup GitHub Actions scheduler

Di repo GitHub → Settings → Secrets and variables → Actions → New secret:

| Secret | Value |
|--------|-------|
| `SUPABASE_EDGE_FUNCTION_URL` | `https://<project>.supabase.co/functions/v1/rss-ingestion` |
| `SUPABASE_ANON_KEY` | `eyJ...` (anon key, aman untuk expose ke Actions) |

Setelah itu workflow di `.github/workflows/trigger-ingestion.yml` otomatis jalan tiap 30 menit.

Manual trigger: GitHub → Actions → RSS Ingestion Trigger → Run workflow.

---

## Monitoring

```sql
-- Cek status ingestion terbaru
SELECT config_name, last_run_at, is_active
FROM scraping_configs
ORDER BY last_run_at DESC NULLS LAST;

-- Cek raw_texts pending (belum diproses NLP)
SELECT status, COUNT(*) FROM raw_texts GROUP BY status;

-- Cek dedup rate (berapa yang ke-skip per hari)
SELECT
    date_trunc('hour', first_seen) AS hour,
    COUNT(*) AS total_hashes
FROM raw_text_hashes
WHERE first_seen > NOW() - INTERVAL '24 hours'
GROUP BY 1 ORDER BY 1;
```

---

## Troubleshooting

| Gejala | Kemungkinan penyebab | Solusi |
|--------|---------------------|--------|
| `total_inserted` selalu 0 | Semua artikel duplikat (normal kalau cron jalan sering) | Normal — cron berikutnya akan fetch artikel baru |
| HTTP 500 dari Edge Function | Service role key salah / scraping_configs kosong | Cek secret di Supabase Dashboard |
| RSS feed return 0 items | URL feed berubah / feed down | Verifikasi URL: `curl -s "<url>" \| head -100` |
| Edge Function timeout | Terlalu banyak feed aktif | Nonaktifkan beberapa feed: `UPDATE scraping_configs SET is_active = false WHERE config_name = '...'` |
| `duplicate_count` = 100% | Wajar untuk Google News RSS (artikel sama muncul di banyak query) | Tidak perlu tindakan — dedup bekerja benar |

---

## Struktur file

```
ingestion/
├── README.md                              ← file ini
├── supabase/
│   └── functions/
│       └── rss-ingestion/
│           └── index.ts                   ← Edge Function Deno
└── seed/
    ├── 01_political_entities.sql          ← 18 tokoh politik
    └── 02_scraping_configs.sql            ← 9 general RSS + 14 Google News RSS
```

---

## Next: Layer 4 (NLP Worker)

Setelah Layer 2 berjalan dan `raw_texts` mulai terisi dengan `status = 'pending'`, Layer 4 (NLP Worker Python di Hugging Face Spaces) akan:

1. Poll `raw_texts WHERE status = 'pending' LIMIT 32`
2. Run IndoBERT inference (micro-batch 16-32)
3. Insert hasil ke `sentiment_scores` via `insert_sentiment_score()` RPC
4. Update `raw_texts.status = 'processed'`
