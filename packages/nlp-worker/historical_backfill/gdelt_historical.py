"""
gdelt_radar.py — Autonomous GDELT Radar & Backfill
=====================================================
Menggabungkan Fase 1 (Entity Intelligence) & Fase 2 (Targeted Ingestion).

Cara Kerja:
  1. AMBIL TARGET:
     - Hotline: Top 5 tokoh yang sedang ramai (diutamakan, cari 7 hari terakhir).
     - Cold Start: 5 tokoh aktif yang datanya masih sedikit (cari 90 hari terakhir).
  2. FETCH GDELT:
     - Query: '"Nama Tokoh" sourcecountry:ID' (Role-Agnostic, anti-jebakan bahasa).
  3. URL VALIDATION:
     - Cek HTTP GET setiap URL. Jika 404/403 (mati), langsung dibuang.
     - Memastikan NLP Worker (trafilatura) tidak gagal saat memproses.
  4. INSERT:
     - Hanya URL hidup yang masuk ke pgmq (antrian NLP).

Usage:
    python gdelt_radar.py
"""

import os
import sys
import time
import hashlib
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv

try:
    import httpx
except ImportError:
    print("[ERROR] pip install httpx"); sys.exit(1)

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

try:
    from supabase import create_client, Client
except ImportError:
    print("[ERROR] pip install supabase"); sys.exit(1)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SERVICE_KEY  = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
GDELT_API    = "https://api.gdeltproject.org/api/v2/doc/doc"

# Domain media asing yang sering menulis Indonesia dalam Bahasa Inggris
ENGLISH_DOMAINS = ["thejakartapost.com", "reuters.com", "bloomberg.com", "apnews.com"]

def get_client() -> Client:
    if not SUPABASE_URL or not SERVICE_KEY:
        print("[ERROR] Set SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY")
        sys.exit(1)
    return create_client(SUPABASE_URL, SERVICE_KEY)

def get_target_entities(sb: Client) -> list[dict]:
    """Kumpulkan tokoh Hotline (7 hari) dan Cold Start (90 hari)."""
    entities = []
    
    # 1. Hotline (Tokoh Viral)
    res_hot = sb.table("hotline_tokoh") \
                .select("canonical_name, political_entities(id, canonical_name, aliases)") \
                .limit(5) \
                .execute()
    for r in (res_hot.data or []):
        pe = r.get("political_entities")
        if pe:
            pe["radar_mode"] = "hotline_7d"
            entities.append(pe)
            
    # 2. Cold Start (Tokoh aktif tapi mention 30 hari = 0)
    res_cold = sb.table("political_entities") \
                 .select("id, canonical_name, aliases") \
                 .eq("is_active", True) \
                 .eq("mention_count_30d", 0) \
                 .limit(5) \
                 .execute()
    for r in (res_cold.data or []):
        r["radar_mode"] = "cold_start_90d"
        entities.append(r)
        
    # Deduplikasi kalau ada tokoh yang muncul di keduanya
    seen_ids = set()
    unique_entities = []
    for e in entities:
        if e["id"] not in seen_ids:
            seen_ids.add(e["id"])
            unique_entities.append(e)
            
    return unique_entities

def is_url_alive(url: str, http: httpx.Client) -> bool:
    """Cek apakah URL mati (404/403) atau hidup (200)."""
    try:
        # Pakai GET karena banyak media blokir HEAD request
        r = http.get(url, timeout=5, follow_redirects=True)
        return r.status_code == 200
    except Exception:
        return False

def is_english_source(domain: str) -> bool:
    if not domain: return False
    if domain.startswith("en."): return True
    for ed in ENGLISH_DOMAINS:
        if ed in domain: return True
    return False

def fetch_gdelt(query: str, start: datetime, end: datetime, http: httpx.Client) -> list[dict]:
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": 100,
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        "format": "json",
        "sort": "DateDesc",
    }
    try:
        r = http.get(GDELT_API, params=params, timeout=15)
        if r.status_code == 200 and "application/json" in r.headers.get("content-type", ""):
            return r.json().get("articles") or []
    except Exception:
        pass
    return []

def insert_batch(sb: Client, items: list[dict]) -> int:
    if not items: return 0
    try:
        res = sb.rpc("batch_insert_raw_texts", {"p_items": items}).execute()
        return (res.data or [{}])[0].get("inserted_count", 0)
    except Exception as e:
        print(f"  [INSERT_ERROR] {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Autonomous GDELT Radar")
    args = parser.parse_args()

    sb = get_client()
    entities = get_target_entities(sb)
    
    if not entities:
        print("[INFO] Tidak ada target tokoh. Keluar.")
        return

    print(f"📡 GDELT Radar Aktif untuk {len(entities)} Tokoh")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }
    
    with httpx.Client(headers=headers, timeout=15) as http:
        for e in entities:
            name = e["canonical_name"]
            mode = e["radar_mode"]
            
            if mode == "hotline_7d":
                start_date = datetime.now(timezone.utc) - timedelta(days=7)
            else:
                start_date = datetime.now(timezone.utc) - timedelta(days=90)
            end_date = datetime.now(timezone.utc)
            
            print(f"\n[{mode.upper()}] Mencari berita untuk: {name}...")
            
            query = f'"{name}" sourcecountry:ID'
            raw_articles = fetch_gdelt(query, start_date, end_date, http)
            
            valid_items = []
            for a in raw_articles:
                url = a.get("url", "")
                title = a.get("title", "")
                domain = a.get("domain", "").lower()
                
                if not url or len(title) < 10: continue
                if is_english_source(domain): continue
                
                # VALIDASI URL MATI
                if not is_url_alive(url, http):
                    print(f"  ❌ URL Mati (Skip): {title[:40]}...")
                    continue
                
                source_id = hashlib.sha256(url.encode()).hexdigest()[:32]
                text_hash = hashlib.sha256(title.encode()).hexdigest()
                
                # Parse Tanggal
                pub_date = None
                if a.get("seendate"):
                    try:
                        pub_date = datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
                    except ValueError:
                        pass
                
                valid_items.append({
                    "source": f"gdelt_{domain.replace('.','_').replace('-','_')}",
                    "source_id": source_id,
                    "title": title,
                    "source_url": url,
                    "image_url": None,
                    "text": title,
                    "text_hash": text_hash,
                    "metadata": {
                        "gdelt_entity": name,
                        "data_source": f"gdelt_radar_{mode}",
                        "gdelt_era_hint": f"y{end_date.year}"
                    },
                    "published_at": pub_date
                })
                print(f"  ✅ URL Valid: {title[:40]}...")
            
            if valid_items:
                inserted = insert_batch(sb, valid_items)
                print(f"  -> {inserted} artikel baru dimasukkan ke antrian.")
            else:
                print(f"  -> Tidak ada artikel valid untuk {name}.")
            
            time.sleep(3) # Jeda anti rate-limit GDELT antar tokoh

    print("\n✅ GDELT Radar Selesai.")

if __name__ == "__main__":
    main()