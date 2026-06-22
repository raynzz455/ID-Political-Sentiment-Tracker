# ID-Sentiment-Tracker

Time-series political sentiment analysis pipeline untuk konteks Indonesia. 100% free-tier compliant.

> **Status:** 🚧 Starter / scaffold — siap untuk development kolaboratif.

---

## 🎯 Tujuan

Dashboard publik yang melacak sentimen (positif / netral / negatif) terhadap tokoh politik Indonesia, berbasis data RSS. Tanpa login, patuh [UU PDP](https://jdih.kominfo.go.id/produk_hukum/undangan/8c4d18ca103e4c78a06b1418bdbb3c9a).

## 💰 Constraint Anggaran: FREE TIER ONLY

| Komponen | Layanan | Tier |
|---|---|---|
| Database + Queue + Storage | **Supabase** | Free (500MB DB, 1GB storage) |
| NLP Worker (IndoBERT) | **Hugging Face Spaces** | Free CPU |
| Frontend | **Vercel** | Free |
| Sumber data | **RSS feeds** | Gratis (Detik, Kompas, Google News RSS) |

❌ **Tidak pakai:** Twitter/X API (bayar), YouTube Data API (kuota berbayar), Play Store scraper.

---

## 🏗️ Arsitektur (6 Layer)

```
RSS Sources → Edge Function (Deno) → pgmq Queue → NLP Worker (Python) → PostgreSQL → Next.js API
   L1              L2                    L3             L4                 L5            L6
```

Lihat detail lengkap: [`docs/architecture.md`](docs/architecture.md) dan diagram [`docs/workflow.drawio`](docs/workflow.drawio).

## 📁 Struktur Repo

```
ID-Sentiment/
├── db/             # Skema SQL (Supabase)
│   └── schema_final_v2.sql
├── ingestion/      # Layer 2: Edge Function (Deno/TS) — fetch & parse RSS  ← TODO
├── nlp-worker/     # Layer 4: Python (FastAPI + ONNX) — IndoBERT inference ← TODO
├── frontend/       # Layer 6: Next.js dashboard                          ← TODO
└── docs/           # Arsitektur + diagram workflow
```

---

## 🚀 Quick Start

### 1. Setup Database (5 menit)

1. Buat project baru di [supabase.com](https://supabase.com) (free tier)
2. Buka **SQL Editor** → paste seluruh isi [`db/schema_final_v2.sql`](db/schema_final_v2.sql) → **Run**
3. Aktifkan extension **pgmq** via Dashboard → Database → Extensions
4. Simpan credential dari *Project Settings → API*:
   - `Project URL` → `SUPABASE_URL`
   - `anon public` key → `SUPABASE_ANON_KEY`
   - `service_role` key → `SUPABASE_SERVICE_ROLE_KEY` (⚠️ rahasia, jangan expose ke frontend)

### 2. Test apakah DB jalan

```bash
# Pakai anon key, coba query MV (harus return data kosong / array, BUKAN error 406)
curl "https://<your-project>.supabase.co/rest/v1/mv_dashboard_summary?select=*&limit=1" \
  -H "apikey: <anon-key>"
```

### 3. Mulai develop tiap layer

Buka folder masing-masing, ikuti `README.md` di dalamnya:
- [`ingestion/README.md`](ingestion/README.md) — Deno Edge Function untuk RSS
- [`nlp-worker/README.md`](nlp-worker/README.md) — Python IndoBERT worker
- [`frontend/README.md`](frontend/README.md) — Next.js dashboard

---

## 🛡️ Aturan Penting (UU PDP / Security)

Kontributor WAJIB mematuhi aturan berikut. Lihat detail di [`docs/architecture.md`](docs/architecture.md).

1. **NO PII**: tidak ada kolom username / author_id / profile_url.
2. **No Raw Text Exposure**: frontend DILARANG menampilkan body artikel hasil scrape. Hanya headline + link + thumbnail + skor sentiment (via tabel `entity_highlights`).
3. **Service Role Key** hanya untuk Edge Function & NLP worker — **jangan pernah** dipakai di Next.js client-side.
4. **Tabel terlarang untuk anon** (`raw_texts`, `sentiment_scores`, `raw_text_hashes`): akses `anon` akan return error (by design).

---

## 🔑 Environment Variables

Salin `.env.example` ke `.env` di tiap subfolder yang membutuhkan:

```env
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_ANON_KEY=eyJ...        # aman untuk frontend
SUPABASE_SERVICE_ROLE_KEY=eyJ...# RAHASIA — hanya backend
```

---

## 📜 License

MIT — bebas dipakai untuk tujuan edukasi & non-komersial. Gunakan data hasil scrape dengan bertanggung jawab terhadap sumber asli.
