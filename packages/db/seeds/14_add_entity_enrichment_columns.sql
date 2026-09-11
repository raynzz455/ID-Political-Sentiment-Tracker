-- ============================================================
-- MIGRATION v2: Add bio + enrichment_status to political_entities
-- FIXED: Match actual production schema (photo_url, not image_url)
--
-- Jalankan di Supabase SQL Editor
-- ============================================================

-- ✅ Kolom yang SUDAH ADA (tidak perlu ditambah):
--   photo_url text              — foto entity
--   wikipedia_id_url text       — Wikipedia Indonesia URL
--   wikipedia_en_url text       — Wikipedia English URL
--   auto_discovered boolean     — flag auto-discovery
--   discovery_source text       — sumber discovery
--   discovery_confidence real   — confidence score

-- ❌ Kolom yang PERLU ditambah:
ALTER TABLE political_entities 
ADD COLUMN IF NOT EXISTS bio TEXT;

ALTER TABLE political_entities 
ADD COLUMN IF NOT EXISTS enrichment_status TEXT DEFAULT 'pending';
-- Values: 'pending', 'enriched', 'failed', 'not_found'

COMMENT ON COLUMN political_entities.bio IS 'Deskripsi singkat entity (dari Wikipedia Indonesia)';
COMMENT ON COLUMN political_entities.enrichment_status IS 'Status enrichment: pending/enriched/failed/not_found';

-- Index untuk query entities yang belum di-enrich
CREATE INDEX IF NOT EXISTS idx_political_entities_enrichment_status 
ON political_entities(enrichment_status) 
WHERE enrichment_status = 'pending';

-- ============================================================
-- VERIFIKASI: Cek entities yang belum punya photo_url atau bio
-- ============================================================
SELECT 
  COUNT(*) as total,
  COUNT(*) FILTER (WHERE photo_url IS NOT NULL) as has_photo,
  COUNT(*) FILTER (WHERE photo_url IS NULL) as missing_photo,
  COUNT(*) FILTER (WHERE bio IS NOT NULL) as has_bio,
  COUNT(*) FILTER (WHERE bio IS NULL) as missing_bio,
  COUNT(*) FILTER (WHERE wikipedia_id_url IS NOT NULL) as has_wiki_id,
  COUNT(*) FILTER (WHERE wikipedia_id_url IS NULL) as missing_wiki_id
FROM political_entities;
