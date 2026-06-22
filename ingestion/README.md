# Layer 2 — Ingestion (RSS → DB → Queue)

> 🚧 **Belum diimplementasi.** Folder ini adalah placeholder.

## Tujuan

Fetch RSS feeds (Detik, Kompas, Google News RSS), parse XML, deduplikasi, lalu:
1. Insert ke `raw_texts` via RPC `batch_insert_raw_texts(p_items JSONB)`
2. Pindahkan item `pending` → pgmq queue `nlp_processing_queue`

## Tech Stack

- **Runtime:** Deno (Supabase Edge Function, TypeScript)
- **Library:** native `fetch` + DOMParser (sudah built-in Deno)
- **Schedule:** Supabase **pg_cron** memanggil edge function tiap 30 menit

## Data flow

```
RSS XML
  → parse <item>: title, link, pubDate, <media:thumbnail>
  → SHA-256 text_hash (DB handle dedup via raw_text_hashes table)
  → RPC: batch_insert_raw_texts([{source, source_id, title, source_url,
                                   image_url, text, published_at}])
  → move pending items to pgmq queue
```

## Field mapping RSS → DB

| RSS field | DB column (`raw_texts`) |
|---|---|
| `<title>` | `title` |
| `<link>` | `source_url` |
| `<description>` / `<content:encoded>` | `text` (PRIVATE) |
| `<media:thumbnail url="...">` / `<enclosure>` | `image_url` (hotlink) |
| `<guid>` / `<link>` | `source_id` |
| `<pubDate>` | `published_at` |

## Skeleton code (untuk mulai)

```typescript
// index.ts — Supabase Edge Function (Deno)
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

Deno.serve(async (req) => {
  const supabase = createClient(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!   // service_role: bisa INSERT ke raw_texts
  );

  const feeds = (Deno.env.get("RSS_FEEDS") ?? "").split(",").filter(Boolean);
  const items = [];

  for (const url of feeds) {
    const xml = await (await fetch(url)).text();
    // TODO: parse XML → push to items[]
  }

  // Insert (dedup otomatis di DB via raw_text_hashes)
  const { data, error } = await supabase.rpc("batch_insert_raw_texts", {
    p_items: items,
  });

  // TODO: move pending items → pgmq queue
  return Response.json({ inserted: data, error });
});
```

## Aturan

- ⚠️ Pakai **service_role key** (bukan anon) — `raw_texts` diblokir untuk anon.
- ⚠️ Jangan pernah kirim `ingested_month` di INSERT — diisi otomatis oleh trigger.
- ⚠️ `text` (body artikel) WAJIB ada — itu yang akan dinilai NLP worker.

## Referensi

- Skema: [`../db/schema_final_v2.sql`](../db/schema_final_v2.sql) blok #9 (RPC `batch_insert_raw_texts`)
- Arsitektur: [`../docs/architecture.md`](../docs/architecture.md) — Layer 2 & 3
