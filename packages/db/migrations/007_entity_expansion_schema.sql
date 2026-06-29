-- ============================================================
-- 007_entity_expansion_schema.sql (FIXED — Ponytail audit)
-- ============================================================
-- Ekspansi schema political_entities untuk support:
-- 1. Kategori lebih luas (pengamat, influencer, historis, dll)
-- 2. Hotness scoring (seberapa sering disebut)
-- 3. Tabel entity_candidates sebagai staging auto-discovery
--
-- FIX vs versi Claude:
--   Bug #1: `is_within_5_years BOOLEAN GENERATED ALWAYS AS (... NOW())`
--           PostgreSQL tolak karena NOW() volatile (not immutable).
--           FIX: jadikan kolom biasa, compute di Python (auto_discover.py).
--
-- PRE-REQUISITE: 009_pre_expansion_unique_constraints.sql WAJIB di-run dulu.
-- Idempotent. Safe to re-run.
-- ============================================================

-- ─────────────────────────────────────────────────────────────
-- STEP 1: Ekspansi entity_type CHECK constraint
-- ─────────────────────────────────────────────────────────────

ALTER TABLE political_entities
  DROP CONSTRAINT IF EXISTS political_entities_entity_type_check;

ALTER TABLE political_entities
  ADD CONSTRAINT political_entities_entity_type_check
  CHECK (entity_type IN (
    -- Pejabat eksekutif
    'president', 'vp', 'minister', 'former_minister',
    -- Legislatif
    'legislator',
    -- Kepala daerah
    'governor', 'mayor',
    -- Partai
    'party', 'party_official',
    -- Non-pejabat tapi aktif di politik
    'commentator',
    'influencer',
    'academic',
    'journalist',
    -- Tokoh historis
    'former_official',
    'other'
  ));

-- ─────────────────────────────────────────────────────────────
-- STEP 2: Tambah kolom baru ke political_entities
-- ─────────────────────────────────────────────────────────────

ALTER TABLE political_entities
  ADD COLUMN IF NOT EXISTS era TEXT[] DEFAULT '{}',
  ADD COLUMN IF NOT EXISTS birth_year SMALLINT,
  ADD COLUMN IF NOT EXISTS active_since_year SMALLINT,
  ADD COLUMN IF NOT EXISTS last_relevant_year SMALLINT,
  ADD COLUMN IF NOT EXISTS mention_count_7d  INTEGER DEFAULT 0,
  ADD COLUMN IF NOT EXISTS mention_count_30d INTEGER DEFAULT 0,
  ADD COLUMN IF NOT EXISTS last_mentioned_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS auto_discovered   BOOLEAN DEFAULT false,
  ADD COLUMN IF NOT EXISTS discovery_source  TEXT,
  ADD COLUMN IF NOT EXISTS discovery_confidence REAL DEFAULT 1.0,
  ADD COLUMN IF NOT EXISTS wikipedia_id_url  TEXT,
  ADD COLUMN IF NOT EXISTS wikipedia_en_url  TEXT;

-- ─────────────────────────────────────────────────────────────
-- STEP 3: Tabel entity_candidates (staging auto-discovery)
-- FIX: is_within_5_years jadi kolom biasa (bukan GENERATED) karena
--       NOW() tidak immutable. Computed di Python saat insert.
-- ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS entity_candidates (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  detected_name     TEXT NOT NULL UNIQUE,
  normalized_name   TEXT,
  detection_source  TEXT NOT NULL
    CHECK (detection_source IN ('wikipedia', 'title_scan', 'ner', 'manual')),
  mention_count     INTEGER DEFAULT 0,
  gnews_hit_count   INTEGER DEFAULT 0,
  sample_titles     TEXT[] DEFAULT '{}',
  wikipedia_url     TEXT,
  wikipedia_snippet TEXT,
  suggested_type    TEXT DEFAULT 'other',
  suggested_aliases TEXT[] DEFAULT '{}',
  status            TEXT DEFAULT 'pending'
    CHECK (status IN ('pending', 'approved', 'rejected', 'duplicate')),
  confidence_score  REAL DEFAULT 0.5,
  promoted_entity_id UUID REFERENCES political_entities(id),
  last_seen_year    SMALLINT,
  -- FIX: kolom biasa, diisi Python (NOW() tidak immutable untuk GENERATED)
  is_within_5_years BOOLEAN DEFAULT true,
  first_detected    TIMESTAMPTZ DEFAULT NOW(),
  last_updated      TIMESTAMPTZ DEFAULT NOW(),
  reviewed_at       TIMESTAMPTZ,
  notes             TEXT
);

CREATE INDEX IF NOT EXISTS idx_candidates_status
  ON entity_candidates (status, confidence_score DESC);

CREATE INDEX IF NOT EXISTS idx_candidates_active
  ON entity_candidates (is_within_5_years, mention_count DESC)
  WHERE status = 'pending';

-- ─────────────────────────────────────────────────────────────
-- STEP 4: Function auto_promote kandidat yang qualified
-- FIX: suggested_type default 'other' (bukan 'politician' yang tidak
--       ada di CHECK constraint).
-- ─────────────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION auto_promote_candidates(
  p_min_confidence  REAL    DEFAULT 0.8,
  p_min_mentions    INTEGER DEFAULT 3,
  p_min_gnews_hits  INTEGER DEFAULT 2
)
RETURNS TABLE(promoted_name TEXT, entity_id UUID)
LANGUAGE plpgsql SECURITY DEFINER AS $$
DECLARE
  v_candidate RECORD;
  v_entity_id UUID;
  v_safe_type TEXT;
BEGIN
  FOR v_candidate IN
    SELECT *
    FROM entity_candidates
    WHERE status = 'pending'
      AND is_within_5_years = true
      AND confidence_score >= p_min_confidence
      AND mention_count    >= p_min_mentions
      AND gnews_hit_count  >= p_min_gnews_hits
    ORDER BY confidence_score DESC, mention_count DESC
    LIMIT 50
  LOOP
    -- FIX: pastikan suggested_type valid sebelum insert
    v_safe_type := CASE
      WHEN v_candidate.suggested_type IN (
        'president','vp','minister','former_minister','legislator',
        'governor','mayor','party','party_official','commentator',
        'influencer','academic','journalist','former_official','other'
      ) THEN v_candidate.suggested_type
      ELSE 'other'
    END;

    INSERT INTO political_entities (
      canonical_name, aliases, entity_type,
      auto_discovered, discovery_source, discovery_confidence,
      wikipedia_id_url, is_active, era
    ) VALUES (
      v_candidate.detected_name,
      v_candidate.suggested_aliases,
      v_safe_type,
      true,
      v_candidate.detection_source,
      v_candidate.confidence_score,
      v_candidate.wikipedia_url,
      true,
      ARRAY['Post-Reformasi']
    )
    ON CONFLICT (canonical_name) DO NOTHING
    RETURNING id INTO v_entity_id;

    IF v_entity_id IS NOT NULL THEN
      UPDATE entity_candidates
      SET status             = 'approved',
          promoted_entity_id = v_entity_id,
          reviewed_at        = NOW()
      WHERE id = v_candidate.id;

      RETURN QUERY SELECT v_candidate.detected_name, v_entity_id;
    ELSE
      UPDATE entity_candidates
      SET status = 'duplicate'
      WHERE id = v_candidate.id;
    END IF;
  END LOOP;
END;
$$;

GRANT EXECUTE ON FUNCTION auto_promote_candidates(REAL, INTEGER, INTEGER) TO service_role;

-- ─────────────────────────────────────────────────────────────
-- STEP 5: Function update hotness score harian
-- ─────────────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION refresh_entity_hotness()
RETURNS VOID
LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  UPDATE political_entities pe
  SET
    mention_count_7d = (
      SELECT COUNT(DISTINCT ss.raw_text_id)
      FROM sentiment_scores ss
      WHERE ss.entity_id = pe.id
        AND ss.scored_at > NOW() - INTERVAL '7 days'
    ),
    mention_count_30d = (
      SELECT COUNT(DISTINCT ss.raw_text_id)
      FROM sentiment_scores ss
      WHERE ss.entity_id = pe.id
        AND ss.scored_at > NOW() - INTERVAL '30 days'
    ),
    last_mentioned_at = (
      SELECT MAX(ss.scored_at)
      FROM sentiment_scores ss
      WHERE ss.entity_id = pe.id
    );
END;
$$;

GRANT EXECUTE ON FUNCTION refresh_entity_hotness() TO service_role;

SELECT cron.schedule(
  'refresh-entity-hotness',
  '0 2 * * *',
  $$ SELECT refresh_entity_hotness() $$
);

-- ─────────────────────────────────────────────────────────────
-- STEP 6: View hotline_tokoh (siapa yang sedang ramai)
-- ─────────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW hotline_tokoh AS
SELECT
  id,
  canonical_name,
  entity_type,
  party_affiliation,
  era,
  mention_count_7d,
  mention_count_30d,
  last_mentioned_at,
  auto_discovered,
  CASE
    WHEN mention_count_7d  >= 20 THEN 'viral'
    WHEN mention_count_7d  >= 10 THEN 'hot'
    WHEN mention_count_7d  >= 3  THEN 'active'
    WHEN mention_count_30d >= 1  THEN 'moderate'
    ELSE 'quiet'
  END AS hotness_label
FROM political_entities
WHERE is_active = true
ORDER BY mention_count_7d DESC, mention_count_30d DESC;

GRANT SELECT ON hotline_tokoh TO anon, authenticated, service_role;

-- ─────────────────────────────────────────────────────────────
-- STEP 7: RLS untuk entity_candidates
-- ─────────────────────────────────────────────────────────────

ALTER TABLE entity_candidates ENABLE ROW LEVEL SECURITY;

CREATE POLICY "svc candidates all"
  ON entity_candidates FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "anon block candidates"
  ON entity_candidates FOR SELECT TO anon USING (false);

-- Verifikasi
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'political_entities'
ORDER BY ordinal_position;
