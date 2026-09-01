# Layer 6 — Frontend (Media Sentiment ID Dashboard)

> 🔗 **Frontend repository:** [mhmdrazkaa/Sentiment-Tracker/political-sentiment-tracker-fe](https://github.com/mhmdrazkaa/Sentiment-Tracker/tree/main/political-sentiment-tracker-fe)

Frontend sudah diimplementasi di repository terpisah (repo kolaborasi). Folder `apps/web/` di repo ini berfungsi sebagai **pointer** — bukan copy source code.

## 🚀 Cara Setup Frontend (Local Development)

```bash
# Clone repo frontend
git clone https://github.com/mhmdrazkaa/Sentiment-Tracker.git
cd Sentiment-Tracker/political-sentiment-tracker-fe

# Install dependencies
bun install  # atau npm install

# Setup environment variables
cp .env.example .env.local
# Isi .env.local dengan credentials Supabase (lihat .env.example)

# Run dev server
bun run dev  # atau npm run dev
# Buka http://localhost:3000
```

## 📊 Data Source

Frontend mengakses database Supabase berikut via API routes (server-side):

| Tabel/View | Akses | Kegunaan |
|---|---|---|
| `political_entities` | ✅ SELECT (anon RLS) | Master data tokoh politik |
| `entity_highlights` | ✅ SELECT (anon RLS) | Cache headline berita + sentiment |
| `mv_dashboard_summary` | ✅ SELECT (anon RLS) | Agregat 90 hari untuk trend chart |
| `raw_texts` | ❌ Blocked (UU PDP) | Teks artikel — TIDAK di-expose |
| `sentiment_scores` | ❌ Blocked (RLS) | Raw scores — akses via API route (server-side) |

## 🔌 API Routes

Frontend punya 7 API endpoints yang auto-fallback ke mock data jika Supabase unavailable:

- `GET /api/politicians` — daftar tokoh aktif
- `GET /api/sentiment/overview?period=30d` — statistik agregat
- `GET /api/sentiment/trend?entityId=<uuid>&period=30d` — trend per tokoh
- `GET /api/sentiment/headlines?limit=5` — berita terbaru dengan NLP + gambar
- `GET /api/sentiment/live-feed?limit=20` — live feed artikel
- `GET /api/sentiment/alerts` — alert spike/drop sentimen
- `GET /api/sentiment/head-to-head?a=<uuid>&b=<uuid>` — komparasi 2 tokoh

## 🎨 Tech Stack

- Next.js 16 (App Router, Server Components)
- TypeScript 5 + Tailwind CSS 4 + shadcn/ui
- Recharts (charts) + Framer Motion (animations)
- Playfair Display (headings) + Inter (body)
- Vitest + React Testing Library (103 tests)
- Vercel (deployment)

## 📁 Struktur

```
political-sentiment-tracker-fe/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── page.tsx            # / (About)
│   │   ├── dashboard/page.tsx  # /dashboard
│   │   ├── figures/page.tsx    # /figures
│   │   └── api/sentiment/      # API routes (7 endpoints)
│   ├── components/             # by-feature (about/, dashboard/, figures/, layout/, shared/)
│   ├── data/                   # static data (fallback mock)
│   ├── types/sentiment.ts      # TypeScript types
│   ├── lib/                    # supabase-client, supabase-server, cache, mock-data
│   └── hooks/use-sentiment.ts  # fetch hooks dengan auto-fallback
├── tests/                      # 103 tests (Vitest)
├── .env.example                # env vars template
├── DATABASE-SCHEMA.md          # schema documentation
└── LOCAL-DEV-GUIDE.md          # setup guide
```

## 🔄 Auto-Fallback Strategy

Frontend otomatis switch antara production (Supabase) dan mock mode:

1. Jika `NEXT_PUBLIC_SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` di-set → **Production mode** (query Supabase)
2. Jika belum di-set → **Mock mode** (pakai data dari `src/data/`)
3. Jika Supabase error/data kosong → **Auto-fallback** ke mock data

Tidak perlu ubah kode — toggle otomatis berdasarkan env vars.

## 📝 Referensi

- Schema database: [`packages/db/schema.sql`](../../packages/db/schema.sql)
- Aturan keamanan: [`docs/architecture.md`](../../docs/architecture.md)
- Frontend repo: [github.com/mhmdrazkaa/Sentiment-Tracker](https://github.com/mhmdrazkaa/Sentiment-Tracker/tree/main/political-sentiment-tracker-fe)
