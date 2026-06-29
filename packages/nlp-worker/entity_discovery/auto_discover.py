"""
Entity Auto-Discovery — ID-Sentiment-Tracker
=============================================
Temukan tokoh politik baru secara otomatis dari 3 sumber:

1. Wikipedia API (id.wikipedia.org) — daftar politisi Indonesia
2. Title scan — nama yang sering muncul di raw_texts tapi belum di DB
3. Google News validation — validasi relevansi via hit count

Semua kandidat masuk ke entity_candidates (status='pending').
Kandidat yang qualified (confidence >= 0.8) di-promote otomatis
via RPC auto_promote_candidates().

Usage:
    python entity_discovery/auto_discover.py --source all
    python entity_discovery/auto_discover.py --source wikipedia
    python entity_discovery/auto_discover.py --source title_scan
    python entity_discovery/auto_discover.py --promote-only

Env vars:
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
"""

import os
import re
import sys
import time
import json
import argparse
import unicodedata
from datetime import datetime, timezone

try:
    import httpx
except ImportError:
    print("[ERROR] pip install httpx")
    sys.exit(1)

try:
    from supabase import create_client, Client
except ImportError:
    print("[ERROR] pip install supabase")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SERVICE_KEY  = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

# Wikipedia categories untuk politisi Indonesia
WIKIPEDIA_CATEGORIES = [
    "Kategori:Presiden_Indonesia",
    "Kategori:Wakil_Presiden_Indonesia",
    "Kategori:Menteri_Indonesia",
    "Kategori:Gubernur_di_Indonesia",
    "Kategori:Anggota_DPR_RI",
    "Kategori:Politisi_Indonesia",
    "Kategori:Ekonom_Indonesia",
    "Kategori:Jurnalis_Indonesia",
]

# Pola nama yang DIKECUALIKAN (false positive umum)
EXCLUDE_PATTERNS = [
    r'^(Menteri|Gubernur|Presiden|Wakil|Ketua|Sekretaris|Direktur|Kepala)\s+\w+$',
    r'\b(Indonesia|Nasional|Republik|Negara|Pemerintah)\b',
    r'^\d',  # mulai angka
]

# Minimum mention di title agar masuk kandidat
MIN_TITLE_MENTIONS = 3

# Threshold confidence untuk auto-promote
AUTO_PROMOTE_CONFIDENCE = 0.80
AUTO_PROMOTE_MIN_MENTIONS = 3
AUTO_PROMOTE_MIN_GNEWS = 2

DELAY = 1.5  # detik antar request


# ─────────────────────────────────────────────────────────────
# SUPABASE CLIENT
# ─────────────────────────────────────────────────────────────

def get_client() -> Client:
    if not SUPABASE_URL or not SERVICE_KEY:
        print("[ERROR] Set SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY")
        sys.exit(1)
    return create_client(SUPABASE_URL, SERVICE_KEY)


# ─────────────────────────────────────────────────────────────
# HELPER: load nama yang sudah ada di DB (untuk skip)
# ─────────────────────────────────────────────────────────────

def load_existing_names(sb: Client) -> set[str]:
    """Return set nama canonical + semua alias yang sudah ada di DB."""
    res = sb.table("political_entities") \
            .select("canonical_name, aliases") \
            .execute()

    existing = set()
    for row in res.data or []:
        existing.add(row["canonical_name"].lower().strip())
        for alias in row.get("aliases") or []:
            existing.add(alias.lower().strip())
    return existing


def load_existing_candidates(sb: Client) -> set[str]:
    """Return nama kandidat yang sudah ada di entity_candidates."""
    res = sb.table("entity_candidates") \
            .select("detected_name") \
            .execute()
    return {r["detected_name"].lower().strip() for r in res.data or []}


def is_excluded(name: str) -> bool:
    """True kalau nama harus dikecualikan (false positive)."""
    if len(name) < 5:
        return True
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            return True
    return False


# ─────────────────────────────────────────────────────────────
# SOURCE 1: WIKIPEDIA API
# ─────────────────────────────────────────────────────────────

def fetch_wikipedia_category(category: str, client: httpx.Client) -> list[dict]:
    """
    Fetch anggota dari satu kategori Wikipedia Indonesia.
    Return list {name, wikipedia_url, snippet}.
    """
    results = []
    params = {
        "action":  "query",
        "list":    "categorymembers",
        "cmtitle": category,
        "cmlimit": 200,
        "cmnamespace": 0,  # artikel saja, bukan subkategori
        "format":  "json",
    }

    try:
        r = client.get(
            "https://id.wikipedia.org/w/api.php",
            params=params,
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()

        members = data.get("query", {}).get("categorymembers", [])
        for m in members:
            name = m.get("title", "").strip()
            if name and not is_excluded(name):
                results.append({
                    "name": name,
                    "wikipedia_url": f"https://id.wikipedia.org/wiki/{name.replace(' ', '_')}",
                    "snippet": None,
                })

    except Exception as e:
        print(f"  [WIKI_ERROR] {category}: {e}")

    return results


def run_wikipedia_discovery(sb: Client) -> int:
    """
    Fetch politisi dari Wikipedia, masukkan yang baru ke entity_candidates.
    Return jumlah kandidat baru.
    """
    print("\n[WIKIPEDIA] Fetching dari Wikipedia Indonesia...")
    existing_names     = load_existing_names(sb)
    existing_candidates = load_existing_candidates(sb)
    new_candidates = 0

    with httpx.Client(
        headers={"User-Agent": "ID-Sentiment-Tracker/1.0 (research project)"},
    ) as client:
        for category in WIKIPEDIA_CATEGORIES:
            print(f"  → {category}")
            members = fetch_wikipedia_category(category, client)

            for m in members:
                name_lower = m["name"].lower().strip()

                # Skip kalau sudah ada di DB atau kandidat
                if name_lower in existing_names:
                    continue
                if name_lower in existing_candidates:
                    continue

                # Determine suggested type dari nama kategori
                # FIX: 'politician' tidak ada di CHECK constraint entity_type.
                #       Default 'other' (valid), override berdasarkan kategori.
                suggested_type = "other"
                if "Presiden" in category:
                    suggested_type = "president"
                elif "Wakil_Presiden" in category:
                    suggested_type = "vp"
                elif "Menteri" in category:
                    suggested_type = "minister"
                elif "Gubernur" in category:
                    suggested_type = "governor"
                elif "Ekonom" in category:
                    suggested_type = "academic"
                elif "Jurnalis" in category:
                    suggested_type = "journalist"

                # Confidence dari Wikipedia = 0.7 (belum divalidasi GNews)
                current_year = datetime.now(timezone.utc).year
                try:
                    sb.table("entity_candidates").insert({
                        "detected_name":    m["name"],
                        "normalized_name":  m["name"].lower().strip(),
                        "detection_source": "wikipedia",
                        "wikipedia_url":    m["wikipedia_url"],
                        "suggested_type":   suggested_type,
                        "confidence_score": 0.70,
                        "last_seen_year":   current_year,
                        "is_within_5_years": True,   # FIX: kolom biasa, diisi Python
                        "status":           "pending",
                    }).execute()

                    existing_candidates.add(name_lower)
                    new_candidates += 1

                except Exception:
                    pass  # duplicate, skip

            time.sleep(DELAY)

    print(f"[WIKIPEDIA] {new_candidates} kandidat baru ditambahkan")
    return new_candidates


# ─────────────────────────────────────────────────────────────
# SOURCE 2: TITLE SCAN dari raw_texts
# ─────────────────────────────────────────────────────────────

def extract_name_candidates_from_titles(titles: list[str]) -> dict[str, int]:
    """
    Ekstrak kemungkinan nama orang dari judul artikel.
    Heuristik: 2-3 kata berurutan yang diawali huruf kapital.
    Return {nama: jumlah_muncul}.
    """
    # Pattern: 2-3 Kata Berkapital berurutan (nama Indonesia biasanya 2-3 kata)
    name_pattern = re.compile(
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
    )

    # Indonesian stop words yang tidak perlu dihitung
    stop_titles = {
        "Jawa Barat", "Jawa Tengah", "Jawa Timur", "Sumatera Utara",
        "Dewan Perwakilan", "Mahkamah Konstitusi", "Komisi Pemberantasan",
        "Badan Nasional", "Kementerian", "Indonesia Maju", "Partai Politik",
        "Peraturan Pemerintah", "Undang Undang", "Kepala Daerah",
        "Pemilihan Umum", "Komite Nasional",
    }

    counts: dict[str, int] = {}
    for title in titles:
        matches = name_pattern.findall(title)
        for match in matches:
            if match not in stop_titles and not is_excluded(match):
                counts[match] = counts.get(match, 0) + 1

    return counts


def run_title_scan(sb: Client) -> int:
    """
    Scan judul artikel di raw_texts, cari nama yang sering muncul
    tapi belum ada di political_entities.
    """
    print("\n[TITLE_SCAN] Scanning judul artikel di raw_texts...")
    existing_names      = load_existing_names(sb)
    existing_candidates = load_existing_candidates(sb)

    # Ambil semua title dari raw_texts
    res = sb.table("raw_texts").select("title").execute()
    titles = [r["title"] for r in res.data if r.get("title")]
    print(f"  → {len(titles)} judul artikel di-scan")

    name_counts = extract_name_candidates_from_titles(titles)

    # Filter: hanya yang muncul >= MIN_TITLE_MENTIONS
    qualified = {
        name: count for name, count in name_counts.items()
        if count >= MIN_TITLE_MENTIONS
           and name.lower() not in existing_names
           and name.lower() not in existing_candidates
    }

    print(f"  → {len(qualified)} nama baru terdeteksi (min {MIN_TITLE_MENTIONS} muncul)")

    new_candidates = 0
    current_year = datetime.now(timezone.utc).year
    for name, count in sorted(qualified.items(), key=lambda x: -x[1])[:100]:
        # Ambil sample titles
        sample = [t for t in titles if name in t][:3]

        try:
            sb.table("entity_candidates").insert({
                "detected_name":    name,
                "normalized_name":  name.lower().strip(),
                "detection_source": "title_scan",
                "mention_count":    count,
                "sample_titles":    sample,
                "suggested_type":   "other",
                "confidence_score": min(0.5 + (count * 0.05), 0.75),
                "last_seen_year":   current_year,
                "is_within_5_years": True,   # FIX: kolom biasa, diisi Python
                "status":           "pending",
            }).execute()

            existing_candidates.add(name.lower())
            new_candidates += 1

        except Exception:
            # Update mention count kalau sudah ada
            try:
                sb.table("entity_candidates") \
                  .update({"mention_count": count, "last_updated": "now()"}) \
                  .eq("detected_name", name) \
                  .execute()
            except Exception:
                pass

    print(f"[TITLE_SCAN] {new_candidates} kandidat baru ditambahkan")
    return new_candidates


# ─────────────────────────────────────────────────────────────
# VALIDASI: Google News hit count
# ─────────────────────────────────────────────────────────────

def count_gnews_hits(name: str, client: httpx.Client) -> int:
    """
    Hitung berapa artikel Google News yang menyebut nama ini + kata 'politik'.
    Return hit count (0 = tidak relevan secara politik).
    """
    query = f'"{name}" politik indonesia'
    url = (
        f"https://news.google.com/rss/search"
        f"?q={query.replace(' ', '+')}&hl=id&gl=ID&ceid=ID:id"
    )

    try:
        r = client.get(url, timeout=10, follow_redirects=True)
        if r.status_code != 200:
            return 0
        # Count <item> tags
        return r.text.count("<item>")
    except Exception:
        return 0


def run_gnews_validation(sb: Client, limit: int = 50) -> int:
    """
    Validasi kandidat pending via Google News.
    Update gnews_hit_count + tingkatkan confidence score.
    """
    print(f"\n[GNEWS_VALIDATE] Validasi {limit} kandidat pending...")

    res = sb.table("entity_candidates") \
            .select("id, detected_name, confidence_score") \
            .eq("status", "pending") \
            .eq("gnews_hit_count", 0) \
            .limit(limit) \
            .execute()

    candidates = res.data or []
    validated = 0

    with httpx.Client(
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "id-ID,id;q=0.9",
        }
    ) as client:
        for c in candidates:
            hits = count_gnews_hits(c["detected_name"], client)
            print(f"  {c['detected_name']:30s} → {hits:3d} hits")

            # Update confidence berdasarkan hits
            base_conf = c.get("confidence_score", 0.5)
            if hits >= 10:
                new_conf = min(base_conf + 0.2, 0.95)
            elif hits >= 5:
                new_conf = min(base_conf + 0.1, 0.90)
            elif hits >= 2:
                new_conf = min(base_conf + 0.05, 0.85)
            elif hits == 0:
                new_conf = max(base_conf - 0.2, 0.1)  # turunkan confidence
            else:
                new_conf = base_conf

            sb.table("entity_candidates").update({
                "gnews_hit_count":  hits,
                "confidence_score": new_conf,
                "last_updated":     datetime.now(timezone.utc).isoformat(),
            }).eq("id", c["id"]).execute()

            validated += 1
            time.sleep(DELAY)

    print(f"[GNEWS_VALIDATE] {validated} kandidat divalidasi")
    return validated


# ─────────────────────────────────────────────────────────────
# AUTO-PROMOTE
# ─────────────────────────────────────────────────────────────

def run_auto_promote(sb: Client) -> int:
    """Promote kandidat qualified ke political_entities via RPC."""
    print("\n[AUTO_PROMOTE] Promoting qualified candidates...")

    res = sb.rpc("auto_promote_candidates", {
        "p_min_confidence": AUTO_PROMOTE_CONFIDENCE,
        "p_min_mentions":   AUTO_PROMOTE_MIN_MENTIONS,
        "p_min_gnews_hits": AUTO_PROMOTE_MIN_GNEWS,
    }).execute()

    promoted = res.data or []
    for p in promoted:
        print(f"  ✅ {p['promoted_name']}")

    print(f"[AUTO_PROMOTE] {len(promoted)} entitas baru dipromote")
    return len(promoted)


# ─────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────

def print_report(sb: Client):
    """Print status entity_candidates saat ini."""
    print(f"\n{'='*60}")
    print("ENTITY CANDIDATES REPORT")
    print(f"{'='*60}")

    # Status breakdown
    res = sb.table("entity_candidates").select("status").execute()
    from collections import Counter
    counts = Counter(r["status"] for r in res.data or [])
    for status, count in counts.most_common():
        print(f"  {status:12s} {count:4d}")
    print(f"  {'TOTAL':12s} {len(res.data):4d}")

    # Top pending dengan confidence tinggi
    print("\nTop pending (confidence tinggi, belum di-promote):")
    res2 = sb.table("entity_candidates") \
             .select("detected_name, confidence_score, mention_count, gnews_hit_count") \
             .eq("status", "pending") \
             .order("confidence_score", desc=True) \
             .limit(10) \
             .execute()

    for r in res2.data or []:
        print(
            f"  {r['detected_name']:30s} "
            f"conf={r['confidence_score']:.2f} "
            f"mentions={r['mention_count']} "
            f"gnews={r['gnews_hit_count']}"
        )

    # Entity total
    res3 = sb.table("political_entities").select("entity_type").execute()
    type_counts = Counter(r["entity_type"] for r in res3.data or [])
    print(f"\npolitical_entities total: {len(res3.data)}")
    for t, c in type_counts.most_common():
        print(f"  {t:20s} {c:3d}")

    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────
# CLI MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Entity Auto-Discovery — ID-Sentiment-Tracker"
    )
    parser.add_argument(
        "--source",
        choices=["all", "wikipedia", "title_scan"],
        default="all",
        help="Sumber discovery (default: all)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Jalankan Google News validation setelah discovery",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Auto-promote qualified candidates setelah validation",
    )
    parser.add_argument(
        "--promote-only",
        action="store_true",
        help="Hanya jalankan auto-promote (skip discovery)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Tampilkan report status kandidat",
    )
    parser.add_argument(
        "--validate-limit",
        type=int, default=50,
        help="Berapa kandidat yang divalidasi via GNews (default: 50)",
    )
    args = parser.parse_args()

    sb = get_client()

    if args.promote_only:
        run_auto_promote(sb)
        print_report(sb)
        return

    # Discovery
    if args.source in ("all", "wikipedia"):
        run_wikipedia_discovery(sb)

    if args.source in ("all", "title_scan"):
        run_title_scan(sb)

    # Validation
    if args.validate or args.source == "all":
        run_gnews_validation(sb, limit=args.validate_limit)

    # Promote
    if args.promote or args.source == "all":
        run_auto_promote(sb)

    # Report
    print_report(sb)


if __name__ == "__main__":
    main()
