-- ============================================================
-- MIGRATION: Add image_url + bio columns to political_entities
-- Jalankan di Supabase SQL Editor
-- ============================================================

-- Add image_url column (foto/gambar entity dari Wikipedia)
ALTER TABLE political_entities 
ADD COLUMN IF NOT EXISTS image_url TEXT;

-- Add bio column (deskripsi singkat dari Wikipedia)
ALTER TABLE political_entities 
ADD COLUMN IF NOT EXISTS bio TEXT;

-- Add wikipedia_url column (link ke Wikipedia page)
ALTER TABLE political_entities 
ADD COLUMN IF NOT EXISTS wikipedia_url TEXT;

-- Add discovered_at column (kapan entity di-auto-discover)
ALTER TABLE political_entities 
ADD COLUMN IF NOT EXISTS discovered_at TIMESTAMPTZ DEFAULT NULL;

-- Add enrichment_status column (tracking enrichment progress)
ALTER TABLE political_entities 
ADD COLUMN IF NOT EXISTS enrichment_status TEXT DEFAULT 'pending';
-- Values: 'pending', 'enriched', 'failed', 'not_found'

-- Update comment
COMMENT ON COLUMN political_entities.image_url IS 'URL foto/gambar entity (dari Wikipedia)';
COMMENT ON COLUMN political_entities.bio IS 'Deskripsi singkat entity (dari Wikipedia)';
COMMENT ON COLUMN political_entities.wikipedia_url IS 'Link ke halaman Wikipedia';
COMMENT ON COLUMN political_entities.enrichment_status IS 'Status enrichment: pending/enriched/failed/not_found';

-- ============================================================
-- INDEX untuk query entities yang belum di-enrich
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_political_entities_enrichment_status 
ON political_entities(enrichment_status) 
WHERE enrichment_status = 'pending';
