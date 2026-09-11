-- ============================================================
-- DASHBOARD RPC FUNCTIONS v2 — untuk frontend API routes
-- FIXED: performance + dependency on bio column
--
-- PRASYARAT: Run 14_add_entity_enrichment_columns.sql SEBELUM file ini!
-- (karena get_entities_list reference kolom bio)
--
-- Jalankan di Supabase SQL Editor
-- ============================================================

-- 1. get_dashboard_summary — overview untuk dashboard page
-- Returns: total entities, total articles, sentiment counts, trending entities
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
-- FIXED v2: Use JOIN instead of correlated subquery (faster on partitioned tables)
-- Parameters: p_entity_id, p_days (default 30)
CREATE OR REPLACE FUNCTION get_entity_sentiment_timeline(
  p_entity_id uuid,
  p_days integer DEFAULT 30
)
RETURNS jsonb
LANGUAGE sql
AS $$
SELECT COALESCE(jsonb_agg(jsonb_build_object(
  'date', DATE(ss.scored_at),
  'label', ss.label,
  'confidence', ss.confidence,
  'score_positive', ss.score_positive,
  'score_negative', ss.score_negative,
  'score_neutral', ss.score_neutral,
  'title', rt.title,
  'source_url', rt.source_url,
  'published_at', rt.published_at
)), '[]'::jsonb)
FROM sentiment_scores ss
LEFT JOIN raw_texts rt ON rt.id = ss.raw_text_id
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
  'entity_name', pe.canonical_name,
  'entity_photo', pe.photo_url,
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
LEFT JOIN political_entities pe ON pe.id = eh.entity_id
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

-- 6. BONUS: get_entity_detail — detail 1 entity untuk profile page
-- Parameters: p_entity_id
CREATE OR REPLACE FUNCTION get_entity_detail(
  p_entity_id uuid
)
RETURNS jsonb
LANGUAGE sql
AS $$
SELECT jsonb_build_object(
  'id', pe.id,
  'name', pe.canonical_name,
  'aliases', pe.aliases,
  'entity_type', pe.entity_type,
  'party', pe.party_affiliation,
  'position', pe.position,
  'photo_url', pe.photo_url,
  'bio', pe.bio,
  'era', pe.era,
  'birth_year', pe.birth_year,
  'is_active', pe.is_active,
  'mention_count_7d', pe.mention_count_7d,
  'mention_count_30d', pe.mention_count_30d,
  'last_mentioned_at', pe.last_mentioned_at,
  'wikipedia_id_url', pe.wikipedia_id_url,
  'wikipedia_en_url', pe.wikipedia_en_url,
  'sentiment_summary', (
    SELECT jsonb_build_object(
      'positive', COUNT(*) FILTER (WHERE label = 'positive'),
      'negative', COUNT(*) FILTER (WHERE label = 'negative'),
      'neutral', COUNT(*) FILTER (WHERE label = 'neutral'),
      'total', COUNT(*),
      'avg_confidence', COALESCE(AVG(confidence), 0)
    )
    FROM sentiment_scores
    WHERE entity_id = p_entity_id
  ),
  'recent_highlights', (
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
      'title', eh.title,
      'polarity', eh.polarity,
      'source_url', eh.source_url,
      'source_name', eh.source_name,
      'image_url', eh.image_url,
      'confidence', eh.confidence,
      'published_at', eh.published_at
    )), '[]'::jsonb)
    FROM entity_highlights eh
    WHERE eh.entity_id = p_entity_id
    ORDER BY eh.published_at DESC NULLS LAST
    LIMIT 5
  )
)
FROM political_entities pe
WHERE pe.id = p_entity_id;
$$;

-- ============================================================
-- VERIFIKASI: Test semua functions
-- ============================================================
-- SELECT get_dashboard_summary();
-- SELECT get_entity_sentiment_timeline('00000000-0000-0000-0000-000000000000'::uuid, 30);
-- SELECT get_entity_highlights(NULL, 10);
-- SELECT get_entities_list(10, 0);
-- SELECT get_sentiment_distribution(30);
-- SELECT get_entity_detail('00000000-0000-0000-0000-000000000000'::uuid);
