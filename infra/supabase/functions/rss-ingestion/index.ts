/**
 * ID-Political-Sentiment-Tracker — RSS Ingestion Edge Function
 * Layer 2: Fetch → Parse → Normalize → Insert to raw_texts via RPC
 *
 * Deploy: supabase functions deploy rss-ingestion
 * Trigger: HTTP POST (called by GitHub Actions cron)
 *
 * CPU budget (Supabase free): 150ms.
 * Network I/O does NOT count toward CPU — fetching multiple RSS feeds is safe.
 */

import { createClient, SupabaseClient } from 'https://esm.sh/@supabase/supabase-js@2'

// ─────────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────────

interface RSSConfig {
  id: string
  entity_id: string | null
  source_type: 'rss_news' | 'google_news_rss'
  config_name: string
  url: string
}

/** Matches batch_insert_raw_texts(p_items JSONB) parameter */
interface BatchItem {
  source: string
  source_id: string
  title: string | null
  source_url: string | null
  image_url: string | null
  text: string                      // body artikel (PRIVATE — RLS blocks anon)
  metadata: Record<string, unknown>
  published_at: string | null
}

interface RPCResult {
  inserted_count: number
  duplicate_count: number
}

interface FeedSummary {
  items_parsed: number
  inserted: number
  duplicates: number
  error?: string
}

// ─────────────────────────────────────────────────────────────────
// XML PARSING (no dependency — avoids cold-start from CDN)
// ─────────────────────────────────────────────────────────────────

/** Extract first occurrence of <tag>…</tag>, handles CDATA */
function extractTag(xml: string, tag: string): string | null {
  const re = [
    new RegExp(`<${tag}[^>]*><!\\[CDATA\\[([\\s\\S]*?)\\]\\]><\\/${tag}>`, 'i'),
    new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`, 'i'),
  ]
  for (const pattern of re) {
    const m = xml.match(pattern)
    if (m?.[1]?.trim()) return m[1].trim()
  }
  return null
}

/** Extract attribute value from a self-closing or opening tag */
function extractAttr(xml: string, tag: string, attr: string): string | null {
  const m = xml.match(new RegExp(`<${tag}[^>]*\\s${attr}=["']([^"']+)["']`, 'i'))
  return m?.[1] ?? null
}

/** Strip HTML tags, decode common entities, collapse whitespace */
function cleanText(html: string): string {
  return html
    // Step 1: decode HTML entities dulu
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&nbsp;/g, ' ')
    // Step 2: BARU strip HTML tags
    .replace(/<[^>]+>/g, ' ')
    // Step 3: collapse whitespace
    .replace(/\s+/g, ' ')
    .trim()
}

/**
 * Parse <item> elements from RSS/Atom XML.
 * Handles: standard RSS 2.0, Google News RSS, media:content/enclosure images.
 */
function parseItems(xml: string, configName: string): BatchItem[] {
  const items: BatchItem[] = []
  const itemRe = /<item[^>]*>([\s\S]*?)<\/item>/gi
  let match: RegExpExecArray | null

  while ((match = itemRe.exec(xml)) !== null) {
    const raw = match[1]

    const title      = extractTag(raw, 'title')
    const link       = extractTag(raw, 'link') ?? extractAttr(raw, 'link', 'href')
    // guid is the canonical unique ID per item in RSS — fallback to link
    const guid       = extractTag(raw, 'guid') ?? link
    const pubDateRaw = extractTag(raw, 'pubDate') ?? extractTag(raw, 'dc:date') ?? extractTag(raw, 'updated')

    // Prefer content:encoded (full body) over description (snippet)
    const rawContent = extractTag(raw, 'content:encoded') ?? extractTag(raw, 'description') ?? ''
    let text = cleanText(rawContent)

    // FALLBACK: Google News & banyak feed general hanya kirim <title>,
    // body kosong. Headline politik Indonesia cukup ekspresif untuk NLP,
    // jadi pakai title sebagai text saat body kosong.
    // (Lebih baik headline-only daripada artikel hilang sama sekali.)
    if (text.length < 20 && title) {
      text = title
    }

    // Image priority: enclosure → media:content → media:thumbnail → og from description
    const imageUrl =
      extractAttr(raw, 'enclosure', 'url') ??
      extractAttr(raw, 'media:content', 'url') ??
      extractAttr(raw, 'media:thumbnail', 'url') ??
      null

    // Reject items without usable text or unique ID
    if (!guid)         continue
    if (text.length < 20) continue   // headline-only stubs add no NLP value

    let publishedAt: string | null = null
    if (pubDateRaw) {
      const d = new Date(pubDateRaw)
      publishedAt = isNaN(d.getTime()) ? null : d.toISOString()
    }

    items.push({
      source:       configName,
      source_id:    guid,
      title:        title ?? null,
      source_url:   link  ?? null,
      image_url:    imageUrl,
      text,
      metadata:     { raw_pub_date: pubDateRaw ?? null },
      published_at: publishedAt,
    })
  }

  return items
}

// ─────────────────────────────────────────────────────────────────
// FETCH + PARSE ONE FEED
// ─────────────────────────────────────────────────────────────────

async function fetchAndParse(cfg: RSSConfig): Promise<BatchItem[]> {
  try {
    const res = await fetch(cfg.url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; ID-Sentiment-Tracker/1.0; +https://github.com/raynzz455/ID-Political-Sentiment-Tracker)',
        'Accept': 'application/rss+xml, application/xml, text/xml, */*',
      },
      signal: AbortSignal.timeout(12_000),   // 12s network timeout
    })

    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`)

    const xml = await res.text()
    const items = parseItems(xml, cfg.config_name)
    console.log(`[PARSE] ${cfg.config_name}: ${items.length} items`)
    return items

  } catch (err) {
    console.error(`[FETCH_ERROR] ${cfg.config_name}: ${(err as Error).message}`)
    return []
  }
}

// ─────────────────────────────────────────────────────────────────
// INSERT via RPC + UPDATE last_run_at
// ─────────────────────────────────────────────────────────────────

const CHUNK_SIZE = 50   // keep payload < 100KB per RPC call

async function insertBatch(
  supabase: SupabaseClient,
  cfg: RSSConfig,
  items: BatchItem[],
): Promise<FeedSummary> {
  let totalInserted  = 0
  let totalDuplicate = 0

  for (let i = 0; i < items.length; i += CHUNK_SIZE) {
    const chunk = items.slice(i, i + CHUNK_SIZE)

    const { data, error } = await supabase.rpc('batch_insert_raw_texts', {
      p_items: chunk,
    })

    if (error) {
      console.error(`[RPC_ERROR] ${cfg.config_name} chunk ${i}: ${error.message}`)
      return {
        items_parsed: items.length,
        inserted:     totalInserted,
        duplicates:   totalDuplicate,
        error:        error.message,
      }
    }

    const row = (data as RPCResult[])?.[0]
    totalInserted  += row?.inserted_count  ?? 0
    totalDuplicate += row?.duplicate_count ?? 0
  }

  // Update last_run_at regardless of insert count (even if all dupes)
  await supabase
    .from('scraping_configs')
    .update({ last_run_at: new Date().toISOString() })
    .eq('id', cfg.id)

  return {
    items_parsed: items.length,
    inserted:     totalInserted,
    duplicates:   totalDuplicate,
  }
}

// ─────────────────────────────────────────────────────────────────
// MAIN HANDLER
// ─────────────────────────────────────────────────────────────────

Deno.serve(async (req: Request) => {
  // Health check — used by GitHub Actions to verify the function is alive.
  // Kept public (no secret) because it leaks nothing and only returns status.
  if (req.method === 'GET') {
    return new Response(JSON.stringify({ status: 'ok' }), {
      headers: { 'Content-Type': 'application/json' },
    })
  }

  if (req.method !== 'POST') {
    return new Response('Method Not Allowed', { status: 405 })
  }

  // ── SECURITY GATE ──────────────────────────────────────────────
  // Endpoint is deployed with --no-verify-jwt and reachable publicly.
  // Anon key is public (designed for browsers), so Authorization alone
  // is NOT enough — anyone could hammer this and burn quota / get us
  // rate-limited by Detik/Kompas. Require a shared secret header.
  const cronSecret = Deno.env.get('CRON_SECRET')
  if (!cronSecret) {
    console.error('[FATAL] CRON_SECRET env var not set — refusing to run insecure')
    return new Response('server misconfigured', { status: 500 })
  }
  if (req.headers.get('x-cron-secret') !== cronSecret) {
    return new Response('unauthorized', { status: 401 })
  }
  // ────────────────────────────────────────────────────────────────

  const supabase = createClient(
    Deno.env.get('SUPABASE_URL')!,
    // MUST use service_role key — anon cannot write raw_texts
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!,
  )

  // ── 1. Load active configs ──
  const { data: configs, error: cfgErr } = await supabase
    .from('scraping_configs')
    .select('id, entity_id, source_type, config_name, url')
    .eq('is_active', true)

  if (cfgErr) {
    console.error('[CONFIG_ERROR]', cfgErr.message)
    return new Response(JSON.stringify({ error: cfgErr.message }), { status: 500 })
  }

  if (!configs?.length) {
    return new Response(JSON.stringify({ ok: true, message: 'no active configs' }), {
      headers: { 'Content-Type': 'application/json' },
    })
  }

  // ── 2. Fetch all feeds concurrently ──
  const fetchResults = await Promise.allSettled(
    (configs as RSSConfig[]).map(fetchAndParse)
  )

  // ── 3. Insert each feed's items concurrently ──
  const summary: Record<string, FeedSummary> = {}

  await Promise.allSettled(
    (configs as RSSConfig[]).map(async (cfg, idx) => {
      const result = fetchResults[idx]
      if (result.status === 'rejected' || !result.value?.length) {
        summary[cfg.config_name] = {
          items_parsed: 0, inserted: 0, duplicates: 0,
          error: result.status === 'rejected' ? String(result.reason) : undefined,
        }
        return
      }
      summary[cfg.config_name] = await insertBatch(supabase, cfg, result.value)
    })
  )

  const totalInserted = Object.values(summary).reduce((a, s) => a + s.inserted, 0)
  console.log(`[DONE] total_inserted=${totalInserted}`, JSON.stringify(summary))

  // ── 4. Enqueue newly-inserted rows to PGMQ (Layer 3 buffer) ──
  // NLP worker dequeues from pgmq instead of polling raw_texts directly.
  // This RPC atomically flips status 'pending'→'queued' AND enqueues.
  // Failure here is non-fatal: rows stay 'pending' and the next run
  // (or the enqueue RPC's own retry) will pick them up.
  let enqueued = 0
  if (totalInserted > 0) {
    const { data: enqData, error: enqErr } = await supabase.rpc('enqueue_pending_raw_texts')
    if (enqErr) {
      console.error('[ENQUEUE_ERROR]', enqErr.message)
    } else {
      enqueued = (enqData as { enqueued_count: number }[])?.[0]?.enqueued_count ?? 0
      console.log(`[ENQUEUE] ${enqueued} rows pushed to nlp_processing_queue`)
    }
  }

  return new Response(
    JSON.stringify({ ok: true, total_inserted: totalInserted, enqueued, summary }),
    { headers: { 'Content-Type': 'application/json' } },
  )
})
