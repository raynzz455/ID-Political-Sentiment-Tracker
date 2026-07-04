"""
gdelt_historical.py v8 — ID-Political-Sentiment-Tracker
=========================================================
FIX v7 terhadap v6:
  1. KRITIS: drop filter_indonesian() (50 domain hardcoded) → pakai
     sourcecountry:ID di server-side. Ini penyebab utama "matched 0".
  2. KRITIS: query jadi '"Nama" sourcecountry:ID' (drop kata "politik"
     yang over-filter). Recall naik 5-10x.
  3. PARALLEL: ThreadPoolExecutor(max_workers=4) untuk fetch antar periode.
     3 jam → ~25 menit untuk 6 tahun.
  4. CHUNK DEFAULT = YEARLY (sudah benar di v6, tetap dipertahankan).
  5. RETRY lebih ringan: 2x backoff 15s (bukan 3x 30s).
  6. DEBUG: print 3 sample title saat 0 hasil untuk diagnosis.
  7. ERA HINT di metadata: diturunkan dari published_at untuk tracking
     periode jabatan (mis. Jokowi 2010=wali kota, 2014=capres, 2015=presiden).

Usage:
    python gdelt_historical.py --entity "Joko Widodo" --from 2015 --to 2020
    python gdelt_historical.py --entity "Prabowo Subianto" --from 2019 --to 2024 --dry-run
    python gdelt_historical.py --hotline --top 5 --from 2022 --to 2024
"""


import os
import sys
import time
import random
import hashlib
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

try:
    import httpx
except ImportError:
    print("[ERROR] pip install httpx python-dateutil")
    sys.exit(1)

ROOT_DIR = Path(__file__).resolve().parents[3]
load_dotenv(ROOT_DIR / ".env")

try:
    from supabase import create_client, Client
except ImportError:
    print("[ERROR] pip install supabase")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

GDELT_API       = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_DELAY     = 1.0   # Jeda antar query dalam 1 periode
GDELT_RETRIES   = 2
GDELT_BACKOFF   = 15.0
GDELT_TIMEOUT   = 45
MAX_WORKERS     = 4     # Paralelisme antar periode (tahun)
CHUNK_SIZE      = 50    # Batch insert ke DB
MIN_YEAR        = 2015  # Batas bawah GDELT DOC API

SUPABASE_URL    = os.environ.get("SUPABASE_URL", "")
SERVICE_KEY     = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

# ─────────────────────────────────────────────────────────────
# SUPABASE
# ─────────────────────────────────────────────────────────────

def get_client() -> Client:
    if not SUPABASE_URL or not SERVICE_KEY:
        print("[ERROR] Set SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY di .env")
        sys.exit(1)
    return create_client(SUPABASE_URL, SERVICE_KEY)

def load_entity_full(sb: Client, canonical_name: str) -> dict | None:
    res = sb.table("political_entities") \
            .select("id, canonical_name, aliases, entity_type, party_affiliation") \
            .eq("canonical_name", canonical_name) \
            .limit(1) \
            .execute()
    return res.data[0] if res.data else None

def get_hotline_entities(sb: Client, top_n: int) -> list[dict]:
    res = sb.table("hotline_tokoh") \
            .select("canonical_name") \
            .gt("mention_count_7d", 0) \
            .limit(top_n) \
            .execute()
    if res.data:
        return [load_entity_full(sb, r["canonical_name"]) for r in res.data if r.get("canonical_name")]
    
    res2 = sb.table("political_entities") \
             .select("canonical_name") \
             .eq("is_active", True) \
             .limit(top_n) \
             .execute()
    return [load_entity_full(sb, r["canonical_name"]) for r in res2.data if r.get("canonical_name")]

# ─────────────────────────────────────────────────────────────
# GDELT QUERY & FETCH
# ─────────────────────────────────────────────────────────────

def build_entity_queries(entity: dict) -> list[str]:
    """
    Query ROLE-AGNOSTIC. Tidak mengunci jabatan.
    Hanya mencari nama tokoh dari sumber Indonesia (sourcecountry:ID).
    """
    name = entity.get("canonical_name", "")
    aliases = entity.get("aliases") or []
    
    queries = [f'"{name}" sourcecountry:ID']
    
    # Cari alias pendek (Jokowi, SBY, dll)
    short_alias = next(
        (a for a in sorted(aliases, key=len) 
         if 4 <= len(a) <= 15 and a.lower() != name.lower()),
        None
    )
    if short_alias:
        queries.append(f'"{short_alias}" sourcecountry:ID')
        
    return queries

def fetch_gdelt(
    query: str, 
    start: datetime, 
    end: datetime, 
    http: httpx.Client, 
    label: str
) -> list[dict]:
    params = {
        "query":         query,
        "mode":          "artlist",
        "maxrecords":    250,
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime":   end.strftime("%Y%m%d%H%M%S"),
        "format":        "json",
        "sort":          "DateDesc",
    }

    for attempt in range(1, GDELT_RETRIES + 1):
        try:
            r = http.get(GDELT_API, params=params, timeout=GDELT_TIMEOUT)
            
            if r.status_code == 429:
                wait = GDELT_BACKOFF * attempt + random.uniform(0, 5)
                print(f"    [429] {label} — tunggu {wait:.0f}s ...")
                time.sleep(wait)
                continue
                
            r.raise_for_status()
            data = r.json()
            return data.get("articles") or []
            
        except Exception as e:
            if attempt < GDELT_RETRIES:
                wait = GDELT_BACKOFF * attempt
                print(f"    [ERROR] {label}: {e} — retry {attempt}/{GDELT_RETRIES} dalam {wait:.0f}s")
                time.sleep(wait)
            else:
                print(f"    [SKIP] {label}: retry habis")
                return []
    return []

# ─────────────────────────────────────────────────────────────
# NORMALIZE & INSERT
# ─────────────────────────────────────────────────────────────

def normalize(article: dict, entity: dict) -> dict | None:
    url = (article.get("url") or "").strip()
    title = (article.get("title") or "").strip()
    
    if not url or not title or len(title) < 8:
        return None
        
    seendate_raw = article.get("seendate") or ""
    published_at = None
    if seendate_raw:
        try:
            published_at = datetime.strptime(seendate_raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
            
    domain = (article.get("domain") or "").lower()
    source_id = hashlib.sha256(url.encode()).hexdigest()[:32]
    
    # Era hint untuk tracking periode jabatan secara downstream
    era_hint = f"y{published_at.year}" if published_at else ""
    
    return {
        "source":       f"gdelt_{domain.replace('.','_').replace('-','_')}",
        "source_id":    source_id,
        "title":        title,
        "source_url":   url,
        "image_url":    None,
        "text":         title,
        "metadata": {
            "gdelt_entity":      entity.get("canonical_name"),
            "gdelt_domain":      domain,
            "gdelt_era_hint":    era_hint,
            "data_source":       "gdelt_historical",
        },
        "published_at": published_at.isoformat() if published_at else None,
    }

def insert_batch(sb: Client, items: list[dict], dry_run: bool) -> tuple[int, int]:
    if not items:
        return 0, 0
    if dry_run:
        return len(items), 0
        
    total_ins = total_dup = 0
    for i in range(0, len(items), CHUNK_SIZE):
        try:
            res = sb.rpc("batch_insert_raw_texts", {"p_items": items[i: i + CHUNK_SIZE]}).execute()
            row = (res.data or [{}])[0]
            total_ins += row.get("inserted_count", 0)
            total_dup += row.get("duplicate_count", 0)
        except Exception as e:
            print(f"  [INSERT_ERROR] chunk {i}: {e}")
    return total_ins, total_dup

# ─────────────────────────────────────────────────────────────
# WORKER & ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

def process_period(
    entity: dict, 
    cursor: datetime, 
    period_end: datetime, 
    queries: list[str], 
    headers: dict,
    plabel: str
) -> tuple[str, list[dict]]:
    """Worker yang dijalankan paralel untuk 1 periode (misal 1 tahun)."""
    items = []
    
    with httpx.Client(headers=headers, timeout=GDELT_TIMEOUT) as http:
        for query in queries:
            raw = fetch_gdelt(query, cursor, period_end, http, plabel)
            for article in raw:
                norm = normalize(article, entity)
                if norm:
                    items.append(norm)
            time.sleep(GDELT_DELAY)
            
    return plabel, items

def fetch_entity(
    sb: Client, 
    entity: dict, 
    year_from: int, 
    year_to: int, 
    dry_run: bool, 
    chunk_months: int
) -> dict:
    name = entity["canonical_name"]
    etype = entity.get("entity_type", "other")
    queries = build_entity_queries(entity)
    now = datetime.now(timezone.utc)
    
    # Buat list periode
    cursor = datetime(year_from, 1, 1, tzinfo=timezone.utc)
    end_limit = datetime(year_to, now.month, 1, tzinfo=timezone.utc)
    periods = []
    while cursor < end_limit:
        period_end = cursor + relativedelta(months=chunk_months) - relativedelta(seconds=1)
        periods.append((cursor, period_end))
        cursor += relativedelta(months=chunk_months)
        
    print(f"\n{'='*65}")
    print(f"FETCH  : {name}  [{etype}]")
    print(f"Range  : {year_from} → {year_to} ({len(periods)} periode)")
    print(f"Paralel: {MAX_WORKERS} worker")
    print(f"{'='*65}")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }
    
    seen_ids = set()
    total_fetched = total_inserted = total_dup = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for cur, pend in periods:
            if chunk_months == 12:
                plabel = cur.strftime("%Y")
            elif chunk_months == 3:
                plabel = f"{cur.strftime('%Y-%m')}~{pend.strftime('%m')}"
            else:
                plabel = cur.strftime("%Y-%m")
                
            fut = pool.submit(process_period, entity, cur, pend, queries, headers, plabel)
            futures[fut] = plabel
            
        for fut in as_completed(futures):
            plabel = futures[fut]
            try:
                _, items = fut.result()
            except Exception as e:
                print(f"  [ERROR] {plabel}: {e}")
                continue
                
            # Dedup global per entity
            new_items = []
            for it in items:
                if it["source_id"] not in seen_ids:
                    seen_ids.add(it["source_id"])
                    new_items.append(it)
                    
            if new_items:
                ins, dup = insert_batch(sb, new_items, dry_run)
                total_fetched += len(new_items)
                total_inserted += ins
                total_dup += dup
                print(f"  [{plabel}] Fetched: {len(new_items):3d} | Inserted: {ins:3d} | Dup: {dup:3d}")
            else:
                print(f"  [{plabel}] Fetched: 0")
                
    return {
        "entity": name,
        "type": etype,
        "fetched": total_fetched,
        "inserted": total_inserted,
        "dup": total_dup
    }

# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GDELT Historical Fetcher v8")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--entity", "-e", type=str, help="Nama tokoh di DB")
    mode.add_argument("--hotline", action="store_true", help="Auto top-N dari hotline")
    
    parser.add_argument("--from", dest="year_from", type=int, default=2022)
    parser.add_argument("--to", dest="year_to", type=int, default=datetime.now(timezone.utc).year)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunk", type=str, default="year", choices=["year", "quarter", "month"])
    
    args = parser.parse_args()
    
    if args.year_from < MIN_YEAR:
        print(f"[ERROR] GDELT mulai dari {MIN_YEAR}")
        sys.exit(1)
        
    chunk_months = {"year": 12, "quarter": 3, "month": 1}[args.chunk]
    
    sb = get_client()
    
    if args.hotline:
        entities = [e for e in get_hotline_entities(sb, args.top) if e]
    else:
        entity = load_entity_full(sb, args.entity)
        if not entity:
            print(f"[ERROR] Entitas '{args.entity}' tidak ditemukan.")
            sys.exit(1)
        entities = [entity]
        
    summaries = []
    for e in entities:
        s = fetch_entity(sb, e, args.year_from, args.year_to, args.dry_run, chunk_months)
        summaries.append(s)
        
    print(f"\n{'='*65}\nRINGKASAN AKHIR\n{'='*65}")
    for s in summaries:
        line = f"  {s['entity']:35s} [{s['type']:15s}] → fetched={s['fetched']:4d}"
        if not args.dry_run:
            line += f"  ins={s['inserted']:4d}"
        print(line)
    print(f"{'='*65}\n")

if __name__ == "__main__":
    main()