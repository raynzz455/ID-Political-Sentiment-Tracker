-- ============================================================
-- DASHBOARD RPC FUNCTIONS — untuk frontend API routes
-- Jalankan di Supabase SQL Editor
-- ============================================================

-- 1. get_dashboard_summary — overview untuk dashboard page
-- Returns: total entities, total articles, avg sentiment, trending entities
CREATE OR REPLACE FUNCTION get_dashboard_summary()
RETURNS jsonb
LANGUAGE sql
AS $$
SELECT jsonb_build_object(
  'total_entities', (SELECT COUNT(*) FROM political_entities WHERE is_active = true),
  'total_articles', (SELECT COUNT(*) FROM raw_texts WHERE status = 'processed'),
  'total_sentiments', (SELECT COUNT(*) FROM sentiment_scores),
  'positive_count', (SELECT COUNT(*) FROM sentiment_scores WHERE label = 'positive'),
  'negative_count', (SELECT COUNT(*) FROM sentiment_scores WHERE label = 'negative'),
  'neutral_count', (SELECT COUNT(*) FROM sentiment_scores WHERE label = 'neutral'),
  'trending_entities', (
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
      'id', pe.id,
      'name', pe.canonical_name,
      'photo_url', pe.photo_url,
      'mention_count_7d', pe.mention_count_7d,
      'mention_count_30d', pe.mention_count_30d,
      'party', pe.party_affiliation,
      'position', pe.position
    )), '[]'::jsonb)
    FROM political_entities pe
    WHERE pe.is_active = true AND pe.mention_count_7d > 0
    ORDER BY pe.mention_count_7d DESC
    LIMIT 10
  )
);
$$;

-- 2. get_entity_sentiment_timeline — sentiment per entity over time
-- Parameters: p_entity_id, p_days (default 30)
CREATE OR REPLACE FUNCTION get_entity_sentiment_timeline(
  p_entity_id uuid,
  p_days integer DEFAULT 30
)
RETURNS jsonb
LANGUAGE sql
AS $$
SELECT COALESCE(jsonb_agg(jsonb_build_object(
  'date', DATE(scored_at),
  'label', label,
  'confidence', confidence,
  'score_positive', score_positive,
  'score_negative', score_negative,
  'score_neutral', score_neutral,
  'title', (SELECT title FROM raw_texts WHERE id = ss.raw_text_id LIMIT 1)
)), '[]'::jsonb)
FROM sentiment_scores ss
WHERE ss.entity_id = p_entity_id
  AND ss.scored_at >= NOW() - (p_days || ' days')::interval
ORDER BY ss.scored_at DESC;
$$;

-- 3. get_entity_highlights — featured articles per entity
-- Parameters: p_entity_id (optional, NULL = all), p_limit
CREATE OR REPLACE FUNCTION get_entity_highlights(
  p_entity_id uuid DEFAULT NULL,
  p_limit integer DEFAULT 20
)
RETURNS jsonb
LANGUAGE sql
AS $$
SELECT COALESCE(jsonb_agg(jsonb_build_object(
  'id', eh.id,
  'entity_id', eh.entity_id,
  'entity_name', (SELECT canonical_name FROM political_entities WHERE id = eh.entity_id),
  'entity_photo', (SELECT photo_url FROM political_entities WHERE id = eh.entity_id),
  'polarity', eh.polarity,
  'title', eh.title,
  'source_url', eh.source_url,
  'source_name', eh.source_name,
  'image_url', eh.image_url,
  'label', eh.label,
  'confidence', eh.confidence,
  'published_at', eh.published_at,
  'curated_at', eh.curated_at
)), '[]'::jsonb)
FROM entity_highlights eh
WHERE (p_entity_id IS NULL OR eh.entity_id = p_entity_id)
ORDER BY eh.confidence DESC, eh.published_at DESC NULLS LAST
LIMIT p_limit;
$$;

-- 4. get_entities_list — list semua entities dengan stats
-- Parameters: p_limit, p_offset (untuk pagination)
CREATE OR REPLACE FUNCTION get_entities_list(
  p_limit integer DEFAULT 50,
  p_offset integer DEFAULT 0
)
RETURNS jsonb
LANGUAGE sql
AS $$
SELECT COALESCE(jsonb_agg(jsonb_build_object(
  'id', pe.id,
  'name', pe.canonical_name,
  'aliases', pe.aliases,
  'entity_type', pe.entity_type,
  'party', pe.party_affiliation,
  'position', pe.position,
  'photo_url', pe.photo_url,
  'bio', pe.bio,
  'is_active', pe.is_active,
  'mention_count_7d', pe.mention_count_7d,
  'mention_count_30d', pe.mention_count_30d,
  'last_mentioned_at', pe.last_mentioned_at,
  'wikipedia_id_url', pe.wikipedia_id_url
)), '[]'::jsonb)
FROM (
  SELECT *
  FROM political_entities
  WHERE is_active = true
  ORDER BY mention_count_7d DESC NULLS LAST, canonical_name ASC
  LIMIT p_limit
  OFFSET p_offset
) pe;
$$;

-- 5. get_sentiment_distribution — untuk pie chart dashboard
-- Parameters: p_days (default 30)
CREATE OR REPLACE FUNCTION get_sentiment_distribution(
  p_days integer DEFAULT 30
)
RETURNS jsonb
LANGUAGE sql
AS $$
SELECT jsonb_build_object(
  'positive', COUNT(*) FILTER (WHERE label = 'positive'),
  'negative', COUNT(*) FILTER (WHERE label = 'negative'),
  'neutral', COUNT(*) FILTER (WHERE label = 'neutral'),
  'total', COUNT(*)
)
FROM sentiment_scores
WHERE scored_at >= NOW() - (p_days || ' days')::interval;
$$;

-- ============================================================
-- VERIFIKASI: Test semua functions
-- ============================================================
-- SELECT get_dashboard_summary();
-- SELECT get_entity_sentiment_timeline('00000000-0000-0000-0000-000000000000'::uuid, 30);
-- SELECT get_entity_highlights(NULL, 10);
-- SELECT get_entities_list(10, 0);
-- SELECT get_sentiment_distribution(30);
