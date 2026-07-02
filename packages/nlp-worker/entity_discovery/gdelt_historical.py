"""
gdelt_historical.py — ID-Political-Sentiment-Tracker (v3)
==========================================================
Perubahan dari v2:
1. HAPUS sourcelang filter — penyebab return 0
2. Query otomatis dari aliases entitas di DB
3. Multiple query per entitas (canonical + alias utama) → dedup hasil
4. Domain filter JAUH lebih luas (50+ media Indonesia)
5. Quarterly chunking sebagai opsi (4 request/tahun vs 12)

Usage:
    python gdelt_historical.py --entity "Airlangga Hartarto" --from 2021 --to 2025 --dry-run
    python gdelt_historical.py --entity "Joko Widodo" --from 2019 --to 2023 --quarterly
    python gdelt_historical.py --hotline --top 5 --from 2022 --to 2024 --dry-run
"""

import os
import sys
import time
import random
import hashlib
import argparse
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

try:
    import httpx
except ImportError:
    print("[ERROR] pip install httpx python-dateutil")
    sys.exit(1)

try:
    from supabase import create_client, Client
except ImportError:
    print("[ERROR] pip install supabase")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

GDELT_API     = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_DELAY   = 10.0   # base delay antar periode
GDELT_JITTER  = 3.0    # random 0-3 detik tambahan
GDELT_RETRIES = 3
GDELT_BACKOFF = 45.0   # tunggu setelah 429
CHUNK_SIZE    = 50
MIN_YEAR      = 2015

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SERVICE_KEY  = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

# Domain media Indonesia — diperluas ke 50+ domain
TARGET_DOMAINS: set[str] = {
    # Tier 1 nasional
    "detik.com", "kompas.com", "tempo.co", "republika.co.id",
    "antaranews.com", "cnnindonesia.com", "liputan6.com",
    "tribunnews.com", "jpnn.com", "medcom.id", "beritasatu.com",
    "viva.co.id", "sindonews.com", "okezone.com", "suara.com",
    "merdeka.com", "inews.id", "kumparan.com", "tirto.id",
    # Ekonomi & bisnis
    "bisnis.com", "katadata.co.id", "kontan.co.id", "investor.id",
    # Tier 2 nasional
    "rmol.id", "jawapos.com", "pojoksatu.id", "genpi.co",
    # TV / media digital
    "metrotvnews.com", "tvone.co.id", "kompastv.com",
    # Media regional yang cover politik nasional
    "harianjogja.com", "solopos.com", "suaramerdeka.com",
    # Portal & english
    "thejakartapost.com", "en.tempo.co",
}


# ─────────────────────────────────────────────────────────────
# QUERY BUILDER — otomatis dari aliases di DB
# ─────────────────────────────────────────────────────────────

def build_queries(canonical_name: str, aliases: list[str]) -> list[str]:
    """
    Bangun list query dari canonical name + aliases entitas.
    Paling spesifik → paling luas.
    Caller coba satu per satu dan dedup hasilnya.
    """
    queries = []

    # Query 1: canonical name (paling reliable)
    queries.append(canonical_name)

    # Query 2: alias terpendek yang bermakna (biasanya panggilan populer)
    useful = [
        a for a in (aliases or [])
        if 4 <= len(a) <= 15
        and a.lower() != canonical_name.lower()
    ]
    useful.sort(key=len)
    if useful:
        queries.append(useful[0])

    # Query 3: nama depan + "politik Indonesia" (broad fallback)
    parts = canonical_name.split()
    if len(parts) >= 2:
        first = parts[0]
        if len(first) > 4 and first not in queries:
            queries.append(f"{first} politik Indonesia")

    return queries


def get_client() -> Client:
    if not SUPABASE_URL or not SERVICE_KEY:
        print("[ERROR] Set SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY")
        sys.exit(1)
    return create_client(SUPABASE_URL, SERVICE_KEY)


def load_aliases(sb: Client, canonical_name: str) -> list[str]:
    res = sb.table("political_entities") \
            .select("aliases") \
            .eq("canonical_name", canonical_name) \
            .limit(1) \
            .execute()
    if res.data:
        return res.data[0].get("aliases") or []
    return []


# ─────────────────────────────────────────────────────────────
# GDELT FETCH — satu periode, satu query, dengan retry
# ─────────────────────────────────────────────────────────────

def fetch_gdelt(
    query: str,
    start: datetime,
    end: datetime,
    http: httpx.Client,
    label: str = "",
) -> list[dict]:
    params = {
        "query":         query,       # TIDAK pakai sourcelang — itu penyebab 0 result
        "mode":          "artlist",
        "maxrecords":    250,
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime":   end.strftime("%Y%m%d%H%M%S"),
        "format":        "json",
        "sort":          "DateDesc",
    }

    for attempt in range(1, GDELT_RETRIES + 1):
        try:
            r = http.get(GDELT_API, params=params, timeout=30)

            if r.status_code == 429:
                wait = GDELT_BACKOFF * attempt + random.uniform(0, 5)
                print(f"  [429] {label} attempt {attempt}/{GDELT_RETRIES} "
                      f"— tunggu {wait:.0f}s ...")
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r.json().get("articles") or []

        except httpx.HTTPStatusError as e:
            if attempt < GDELT_RETRIES:
                wait = GDELT_BACKOFF * attempt
                print(f"  [HTTP {e.response.status_code}] {label} "
                      f"retry {attempt} dalam {wait:.0f}s ...")
                time.sleep(wait)
            else:
                print(f"  [SKIP] {label}: semua retry habis")
                return []

        except Exception as e:
            print(f"  [ERROR] {label}: {e}")
            return []

    return []


def filter_indonesian(articles: list[dict]) -> list[dict]:
    """Filter ke domain Indonesia — lebih reliable dari filter di GDELT query."""
    result = []
    for a in articles:
        domain = (a.get("domain") or "").lower().strip()
        if any(domain == d or domain.endswith("." + d) for d in TARGET_DOMAINS):
            result.append(a)
    return result


# ─────────────────────────────────────────────────────────────
# NORMALIZE
# ─────────────────────────────────────────────────────────────

def normalize(article: dict, entity_name: str) -> dict | None:
    url   = (article.get("url")   or "").strip()
    title = (article.get("title") or "").strip()

    if not url or not title or len(title) < 8:
        return None

    seendate_raw = article.get("seendate") or ""
    published_at = None
    if seendate_raw:
        try:
            published_at = datetime.strptime(
                seendate_raw, "%Y%m%dT%H%M%SZ"
            ).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            pass

    domain    = (article.get("domain") or "").lower()
    source_id = hashlib.sha256(url.encode()).hexdigest()[:32]

    return {
        "source":       f"gdelt_{domain.replace('.','_').replace('-','_')}",
        "source_id":    source_id,
        "title":        title,
        "source_url":   url,
        "image_url":    None,
        "text":         title,    # judul saja dari GDELT
        "metadata": {
            "gdelt_entity":   entity_name,
            "gdelt_domain":   domain,
            "gdelt_seendate": seendate_raw,
            "data_source":    "gdelt_historical",
        },
        "published_at": published_at,
    }


# ─────────────────────────────────────────────────────────────
# INSERT
# ─────────────────────────────────────────────────────────────

def insert_batch(sb: Client, items: list[dict], dry_run: bool) -> tuple[int, int]:
    if not items or dry_run:
        return len(items) if dry_run else 0, 0

    total_ins = total_dup = 0
    for i in range(0, len(items), CHUNK_SIZE):
        try:
            res = sb.rpc("batch_insert_raw_texts",
                         {"p_items": items[i: i + CHUNK_SIZE]}).execute()
            row = (res.data or [{}])[0]
            total_ins += row.get("inserted_count", 0)
            total_dup += row.get("duplicate_count", 0)
        except Exception as e:
            print(f"  [INSERT_ERROR] chunk {i}: {e}")
    return total_ins, total_dup


# ─────────────────────────────────────────────────────────────
# HOTLINE
# ─────────────────────────────────────────────────────────────

def get_hotline_entities(sb: Client, top_n: int) -> list[dict]:
    res = sb.table("hotline_tokoh") \
            .select("id, canonical_name") \
            .gt("mention_count_7d", 0) \
            .limit(top_n) \
            .execute()
    if res.data:
        return res.data

    print("  [HOTLINE] Tidak ada data → fallback ke entitas aktif")
    res2 = sb.table("political_entities") \
             .select("id, canonical_name") \
             .eq("is_active", True) \
             .limit(top_n) \
             .execute()
    return res2.data or []


# ─────────────────────────────────────────────────────────────
# FETCH SATU ENTITAS
# ─────────────────────────────────────────────────────────────

def fetch_entity(
    sb: Client,
    entity_name: str,
    year_from: int,
    year_to: int,
    dry_run: bool = False,
    quarterly: bool = False,
) -> dict:
    now     = datetime.now(timezone.utc)
    aliases = load_aliases(sb, entity_name)
    queries = build_queries(entity_name, aliases)

    print(f"\n{'='*62}")
    print(f"FETCH  : {entity_name}")
    print(f"Range  : {year_from}-01 → {year_to}-{now.month:02d}")
    print(f"Queries: {queries}")
    print(f"Chunk  : {'quarterly (3 bulan)' if quarterly else 'monthly'}")
    print(f"Dry run: {dry_run}")
    print(f"{'='*62}")

    http_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":          "application/json, text/html, */*",
        "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
    }

    total_fetched = total_inserted = total_dup = 0
    chunk_months  = 3 if quarterly else 1

    with httpx.Client(headers=http_headers, timeout=30) as http:
        cursor    = datetime(year_from, 1, 1, tzinfo=timezone.utc)
        end_limit = datetime(year_to, now.month, 1, tzinfo=timezone.utc)

        while cursor < end_limit:
            period_end = (cursor + relativedelta(months=chunk_months)
                          - relativedelta(seconds=1))
            if period_end.replace(tzinfo=timezone.utc) > end_limit:
                period_end = end_limit - relativedelta(seconds=1)

            seen_ids:  set[str]  = set()
            all_items: list[dict] = []

            for q_idx, query in enumerate(queries):
                label = (f"{cursor.strftime('%Y-%m')} "
                         f"[q{q_idx+1}: '{query[:30]}']")

                raw = fetch_gdelt(query, cursor, period_end, http, label)
                id_articles = filter_indonesian(raw)

                new = 0
                for a in id_articles:
                    item = normalize(a, entity_name)
                    if item and item["source_id"] not in seen_ids:
                        seen_ids.add(item["source_id"])
                        all_items.append(item)
                        new += 1

                print(f"  {cursor.strftime('%Y-%m')}"
                      f"{'-'+period_end.strftime('%m') if quarterly else ''}"
                      f" [{q_idx+1}/{len(queries)}]"
                      f" total={len(raw):3d} ID={len(id_articles):3d}"
                      f" new={new:3d}  '{query[:30]}'")

                # Delay antar query dalam periode yang sama
                if q_idx < len(queries) - 1:
                    time.sleep(GDELT_DELAY / 2 + random.uniform(0, 1.5))

            if all_items:
                ins, dup = insert_batch(sb, all_items, dry_run)
                total_fetched  += len(all_items)
                total_inserted += ins
                total_dup      += dup

            # Delay antar periode
            time.sleep(GDELT_DELAY + random.uniform(0, GDELT_JITTER))
            cursor += relativedelta(months=chunk_months)

    print(f"\n{'─'*40}")
    print(f"SELESAI: {entity_name}")
    print(f"  Valid (deduped)  : {total_fetched:,}")
    if dry_run:
        print(f"  [DRY RUN] Tidak di-insert")
    else:
        print(f"  Inserted         : {total_inserted:,}")
        print(f"  Duplikat (skip)  : {total_dup:,}")
    print(f"{'─'*40}")

    return {
        "entity":   entity_name,
        "fetched":  total_fetched,
        "inserted": total_inserted,
        "dup":      total_dup,
    }


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GDELT Historical Fetcher v3 — auto-alias, wider domain"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--entity", "-e", type=str,
                       help='Canonical name sesuai DB. Contoh: "Airlangga Hartarto"')
    mode.add_argument("--hotline", action="store_true",
                       help="Auto top-N dari hotline_tokoh view")

    parser.add_argument("--year",      type=int)
    parser.add_argument("--from",     dest="year_from", type=int, default=2022)
    parser.add_argument("--to",       dest="year_to",
                         type=int, default=datetime.now(timezone.utc).year)
    parser.add_argument("--top",      type=int, default=5)
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--quarterly", action="store_true",
                         help="Chunk per kuartal (lebih cepat, mungkin lewatkan peak bulan)")

    args = parser.parse_args()

    if args.year_from < MIN_YEAR:
        print(f"[ERROR] GDELT mulai dari {MIN_YEAR}")
        sys.exit(1)

    year_from = args.year or args.year_from
    year_to   = args.year or args.year_to

    sb = get_client()

    if args.hotline:
        print(f"Mode: HOTLINE — top-{args.top} dari hotline_tokoh")
        entities = get_hotline_entities(sb, args.top)
        if not entities:
            print("[ERROR] Tidak ada entitas.")
            sys.exit(1)
        entity_names = [e["canonical_name"] for e in entities]
        print(f"Tokoh: {entity_names}\n")
    else:
        entity_names = [args.entity]

    summaries = []
    for name in entity_names:
        s = fetch_entity(sb, name, year_from, year_to,
                         dry_run=args.dry_run, quarterly=args.quarterly)
        summaries.append(s)

    print(f"\n{'='*62}\nRINGKASAN AKHIR\n{'='*62}")
    for s in summaries:
        line = f"  {s['entity']:35s} → {s['fetched']:4d}"
        if not args.dry_run:
            line += f"  ins={s['inserted']:4d}  dup={s['dup']:4d}"
        print(line)

    total = sum(s["fetched"] for s in summaries)
    print(f"  {'TOTAL':35s} → {total:4d}")
    if args.dry_run:
        print("  [DRY RUN] Tidak di-insert")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()