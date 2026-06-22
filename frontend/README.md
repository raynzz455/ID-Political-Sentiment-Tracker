# Layer 6 — Frontend (Next.js Dashboard)

> 🚧 **Belum diimplementasi.** Folder ini adalah placeholder.

## Tujuan

Dashboard publik (tanpa login) yang menampilkan:
- Ranking tokoh politik + foto
- Time-series sentimen per tokoh
- Highlight berita positif/negatif per tokoh (headline + thumbnail, **bukan body**)
- Head-to-head perbandingan

## Tech Stack

- **Framework:** Next.js 14+ (App Router)
- **Styling:** Tailwind CSS
- **Charts:** Recharts atau Chart.js
- **Data fetching:** Supabase JS client (Server Components)

## Aturan wajib (UU PDP / Security)

1. ⚠️ **Pakai `anon` key** saja — JANGAN pernah `service_role` di client.
2. ⚠️ **DILARANG** menampilkan body artikel (`raw_texts.text`). Hanya tampilkan data dari `entity_highlights`.
3. ⚠️ Jangan query `raw_texts` / `sentiment_scores` langsung dari frontend — akan error 406 (by design RLS).

## Pola query yang diizinkan

```typescript
// lib/supabase.ts — pakai NEXT_PUBLIC_* (anon, aman expose)
import { createClient } from "@supabase/supabase-js";

export const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
);
```

```typescript
// 1. Daftar tokoh + foto
const { data: entities } = await supabase
  .from("political_entities")
  .select("id, canonical_name, photo_url, entity_type")
  .eq("is_active", true);

// 2. Agregat cepat (dari materialized view)
const { data: summary } = await supabase
  .from("mv_dashboard_summary")
  .select("*")
  .eq("entity_id", entityId);

// 3. Time-series
const { data: timeseries } = await supabase.rpc("get_sentiment_timeseries", {
  p_entity_id: entityId,
});

// 4. Ranking
const { data: ranking } = await supabase.rpc("get_entity_ranking", {
  p_days: 7,
});

// 5. Highlight berita (headline + thumbnail, AMAN)
const { data: highlights } = await supabase.rpc("get_entity_highlights", {
  p_entity_id: entityId,
  p_polarity: "positive",  // atau "negative", atau null untuk keduanya
});
```

## Foto tokoh (bucket "politik")

```typescript
// Foto self-hosted di Supabase Storage
const { data } = supabase.storage
  .from("politik")
  .getPublicUrl("jokowi.jpg");

// Foto berita = HOTLINK langsung dari entity_highlights.image_url
<img src={highlight.image_url} />  // URL eksternal, bukan bucket
```

## Struktur rekomendasi

```
frontend/
├── app/
│   ├── page.tsx                 # Home: ranking + overview
│   ├── tokoh/[id]/page.tsx      # Detail tokoh: timeseries + highlights
│   └── layout.tsx
├── components/
│   ├── RankingTable.tsx
│   ├── SentimentChart.tsx
│   └── HighlightCard.tsx
├── lib/supabase.ts
└── package.json
```

## Inisialisasi (untuk mulai)

```bash
cd frontend
npx create-next-app@latest . --typescript --tailwind --app
npm install @supabase/supabase-js recharts
cp ../.env.example .env.local   # isi NEXT_PUBLIC_* key
```

## Referensi

- RPC definitions: [`../db/schema_final_v2.sql`](../db/schema_final_v2.sql) blok #11 (timeseries), #11d (highlights), #12 (MV)
- Aturan keamanan: [`../docs/architecture.md`](../docs/architecture.md) — "STRICT LEGAL & SECURITY CONSTRAINTS"
