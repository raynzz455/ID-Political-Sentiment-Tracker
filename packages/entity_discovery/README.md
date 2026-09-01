# Entity Expansion — ID-Sentiment-Tracker

Ekspansi entitas dari 18 → 50+ tokoh + sistem auto-discovery.

## Urutan eksekusi

```
1. packages/db/migrations/007_entity_expansion_schema.sql
   → Ekspansi schema (entity_type baru, kolom baru, tabel entity_candidates)

2. packages/db/seeds/03_entities_comprehensive.sql
   → Seed 50+ tokoh (presiden historis, kabinet, pengamat, influencer)

3. packages/db/seeds/04_scraping_configs_expanded.sql
   → Google News RSS untuk semua entitas baru

4. packages/nlp-worker/entity_discovery/auto_discover.py
   → Auto-discovery via Wikipedia + title scan + GNews validation
```

## Auto-discovery

```powershell
# Jalankan semua sumber + validasi + promote otomatis
python entity_discovery/auto_discover.py --source all

# Wikipedia saja
python entity_discovery/auto_discover.py --source wikipedia

# Scan title artikel yang sudah ada di DB
python entity_discovery/auto_discover.py --source title_scan

# Hanya promote kandidat yang sudah qualified
python entity_discovery/auto_discover.py --promote-only

# Lihat report kandidat pending
python entity_discovery/auto_discover.py --report
```

## Kriteria auto-promote

Kandidat dipromote otomatis ke `political_entities` kalau:
- `confidence_score >= 0.80`
- `mention_count >= 3` (muncul di >=3 artikel)
- `gnews_hit_count >= 2` (tervalidasi di Google News)
- `is_within_5_years = true` (masih relevan 5 tahun terakhir)

Kandidat yang tidak lolos tetap di `entity_candidates` dengan status `pending`
untuk review manual.
