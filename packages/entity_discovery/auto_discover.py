"""
Entity Auto-Discovery v2 — ID-Sentiment-Tracker
=================================================
Murni analisis database. Membuang sampah Wikipedia & Regex naif.

Cara Kerja:
  1. Scan judul artikel yang statusnya 'processed'.
  2. Ekstrak nama dengan heuristik kata berkapital.
  3. Filter konteks: Harus ada kata "politik/presiden/menteri/dll" di judul.
  4. Filter entitas: Buang instansi/daerah (BPK, Jawa Barat, dll).
  5. Validasi Google News (hitung <item> di RSS) & update confidence.
  6. PROMOTE: Masuk political_entities + AUTO-CREATE scraping_configs.

Schedule:
  Cron GitHub Actions (Mingguan, jam 02:00 UTC)
"""

import os
import re
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[3]
load_dotenv(ROOT_DIR / ".env")

try:
    import httpx
    from supabase import create_client, Client
except ImportError:
    print("[ERROR] pip install httpx supabase"); sys.exit(1)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SERVICE_KEY  = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

# Filter kata yang sering kapital tapi bukan nama orang
NON_PERSON_KEYWORDS = {
    'indonesia', 'jakarta', 'jawa', 'sumatera', 'sulawesi', 'kalimantan', 'bali',
    'pemerintah', 'kementerian', 'dpr', 'mpr', 'bawaslu', 'kpu', 'golkar', 'pdip',
    'partai', 'koalisi', 'oposisi', 'kabinet', 'pemilu', 'pilkada', 'republik',
    'pers', 'media', 'tv', 'kompas', 'detik', 'tribun', 'rakyat', 'umum', 'nasional'
}

CONTEXT_KEYWORDS = {
    'politik', 'presiden', 'wapres', 'menteri', 'gubernur', 'bupati', 'walikota',
    'capres', 'cawapres', 'ketua', 'sekjen', 'partai', 'pemilu', 'pilkada'
}

def get_client() -> Client:
    if not SUPABASE_URL or not SERVICE_KEY:
        print("[ERROR] Set SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY")
        sys.exit(1)
    return create_client(SUPABASE_URL, SERVICE_KEY)

def is_likely_person(name: str) -> bool:
    """Filter cerdas: buang nama tempat, instansi, atau kata tunggal."""
    words = name.split()
    if len(words) < 2: return False # Nama orang Indonesia minimal 2 kata
    
    for w in words.lower():
        if w in NON_PERSON_KEYWORDS:
            return False
    return True

def extract_name_candidates(titles: list[str]) -> dict[str, int]:
    """Ekstrak nama dengan konteks politik di judul."""
    name_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b')
    counts = {}
    
    for title in titles:
        if not title: continue
        # Hanya scan judul yang mengandung konteks politik
        if not any(k in title.lower() for k in CONTEXT_KEYWORDS):
            continue
            
        matches = name_pattern.findall(title)
        for match in matches:
            if is_likely_person(match):
                counts[match] = counts.get(match, 0) + 1
    return counts

def run_title_scan(sb: Client) -> int:
    print("\n[TITLE_SCAN] Menganalisis judul berita...")
    res = sb.table("raw_texts").select("title").eq("status", "processed").limit(5000).execute()
    titles = [r["title"] for r in res.data if r.get("title")]
    
    name_counts = extract_name_candidates(titles)
    qualified = {n: c for n, c in name_counts.items() if c >= 3} # Minimal 3x muncul

    existing = sb.table("political_entities").select("canonical_name").execute()
    existing_names = {r["canonical_name"].lower() for r in existing.data}

    new_candidates = 0
    for name, count in qualified.items():
        if name.lower() in existing_names: continue
        
        base_conf = min(0.5 + (count * 0.1), 0.8)
        
        try:
            sb.table("entity_candidates").upsert({
                "detected_name": name,
                "normalized_name": name.lower(),
                "detection_source": "title_scan",
                "mention_count": count,
                "confidence_score": base_conf,
                "status": "pending"
            }, on_conflict="detected_name").execute()
            new_candidates += 1
        except Exception:
            pass
            
    print(f"[TITLE_SCAN] {new_candidates} kandidat baru ditemukan.")
    return new_candidates

def run_gnews_validation(sb: Client):
    print("\n[GNEWS] Validasi kandidat...")
    res = sb.table("entity_candidates") \
            .select("id, detected_name, mention_count") \
            .eq("status", "pending") \
            .limit(50) \
            .execute()
            
    with httpx.Client(headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}) as client:
        for c in res.data or []:
            name = c["detected_name"]
            url = f"https://news.google.com/rss/search?q=%22{name.replace(' ', '+')}%22+politik&hl=id&gl=ID&ceid=ID:id"
            try:
                r = client.get(url, timeout=10)
                hits = r.text.count("<item>") if r.status_code == 200 else 0
                
                # Update confidence berdasarkan hits Google News
                base_conf = min(0.5 + (c.get("mention_count", 0) * 0.1), 0.8)
                if hits >= 10:
                    final_conf = min(base_conf + 0.15, 0.95)
                elif hits >= 5:
                    final_conf = min(base_conf + 0.10, 0.90)
                elif hits >= 3:
                    final_conf = min(base_conf + 0.05, 0.85)
                else:
                    final_conf = base_conf
                    
                sb.table("entity_candidates").update({
                    "gnews_hit_count": hits,
                    "confidence_score": final_conf
                }).eq("id", c["id"]).execute()
            except Exception:
                pass
            time.sleep(1.5)

def run_auto_promote_and_config(sb: Client) -> int:
    """Promote tokoh + OTOMATIS buat RSS config-nya."""
    print("\n[PROMOTE] Mempromosikan kandidat dan membuat RSS Config...")
    
    # Ambil kandidat yang memenuhi syarat (Confidence >= 0.85 & GNews >= 3)
    res = sb.table("entity_candidates") \
            .select("id, detected_name") \
            .eq("status", "pending") \
            .gte("confidence_score", 0.85) \
            .gte("gnews_hit_count", 3) \
            .execute()
            
    promoted_count = 0
    for c in res.data or []:
        name = c["detected_name"]
        
        # 1. Upsert ke political_entities (on_conflict mencegah error duplikat)
        pe_res = sb.table("political_entities").upsert({
            "canonical_name": name,
            "entity_type": "other",
            "is_active": True,
            "auto_discovered": True
        }, on_conflict="canonical_name").execute()
        
        if not pe_res.data: continue
        entity_id = pe_res.data[0]["id"]
        
        # 2. AUTO-CREATE Scraping Config (Upsert agar idempotent)
        config_name = f"gnews_{name.lower().replace(' ', '_')}"
        gnews_url = f'https://news.google.com/rss/search?q=%22{name.replace(" ", "+")}%22&hl=id&gl=ID&ceid=ID:id'
        
        sb.table("scraping_configs").upsert({
            "entity_id": entity_id,
            "source_type": "google_news_rss",
            "config_name": config_name,
            "url": gnews_url,
            "is_active": True
        }, on_conflict="config_name").execute()
        
        # 3. Update status kandidat
        sb.table("entity_candidates").update({
            "status": "approved", 
            "promoted_entity_id": entity_id
        }).eq("id", c["id"]).execute()
        
        print(f"  ✅ {name} dipromosikan & RSS dibuat.")
        promoted_count += 1

    return promoted_count

def main():
    sb = get_client()
    run_title_scan(sb)
    run_gnews_validation(sb)
    count = run_auto_promote_and_config(sb)
    if count == 0:
        print("\n[TITLE_SCAN] Tidak ada kandidat yang memenuhi syarat promosi minggu ini.")
    print("\n✅ Auto-Discovery Selesai.")

if __name__ == "__main__":
    main()