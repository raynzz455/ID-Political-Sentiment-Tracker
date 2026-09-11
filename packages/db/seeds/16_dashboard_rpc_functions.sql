-- ============================================================
-- DASHBOARD RPC FUNCTIONS v3 — SECURITY DEFINER + fixes
--
-- CHANGES v3:
--   1. Add SECURITY DEFINER to functions that read RLS-blocked tables
--      (sentiment_scores, raw_texts). Without this, anon browser calls
--      return 0/empty because RLS blocks access.
--   2. Add SET search_path = public (security best practice for SECURITY DEFINER)
--   3. Resolve duplicate get_entity_highlights (drop old, create new)
--   4. Add get_entity_daily_sentiment (aggregated daily, for line charts)
--   5. Add get_entities_comparison (head-to-head, for compare page)
--
-- PRASYARAT: Run 14_add_entity_enrichment_columns.sql SEBELUM file ini!
--
-- Jalankan di Supabase SQL Editor
-- ============================================================

-- ============================================================
-- DROP old functions first (resolve duplicates/conflicts)
-- ============================================================
DROP FUNCTION IF EXISTS get_dashboard_summary();
DROP FUNCTION IF EXISTS get_entity_sentiment_timeline(uuid, integer);
DROP FUNCTION IF EXISTS get_entity_highlights(uuid, integer);
DROP FUNCTION IF EXISTS get_entities_list(integer, integer);
DROP FUNCTION IF EXISTS get_sentiment_distribution(integer);
DROP FUNCTION IF EXISTS get_entity_detail(uuid);
DROP FUNCTION IF EXISTS get_entity_daily_sentiment(uuid, integer);
DROP FUNCTION IF EXISTS get_entities_comparison(uuid[], integer);

-- ============================================================
-- 1. get_dashboard_summary — overview untuk dashboard page
-- SECURITY DEFINER: reads sentiment_scores + raw_texts (RLS-blocked for anon)
-- ============================================================
CREATE OR REPLACE FUNCTION get_dashboard_summary()
RETURNS jsonb
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
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

-- ============================================================
-- 2. get_entity_sentiment_timeline — per-article sentiment over time
-- SECURITY DEFINER: reads sentiment_scores + raw_texts
-- ============================================================
CREATE OR REPLACE FUNCTION get_entity_sentiment_timeline(
  p_entity_id uuid,
  p_days integer DEFAULT 30
)
RETURNS jsonb
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
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

-- ============================================================
-- 3. get_entity_highlights — featured articles
-- NO SECURITY DEFINER needed: entity_highlights is anon-readable
-- ============================================================
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

-- ============================================================
-- 4. get_entities_list — paginated entity list
-- NO SECURITY DEFINER needed: political_entities is anon-readable
-- ============================================================
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

-- ============================================================
-- 5. get_sentiment_distribution — pie chart
-- SECURITY DEFINER: reads sentiment_scores
-- ============================================================
CREATE OR REPLACE FUNCTION get_sentiment_distribution(
  p_days integer DEFAULT 30
)
RETURNS jsonb
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
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
-- 6. get_entity_detail — full entity profile page
-- SECURITY DEFINER: reads sentiment_scores
-- ============================================================
CREATE OR REPLACE FUNCTION get_entity_detail(
  p_entity_id uuid
)
RETURNS jsonb
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
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
-- 7. NEW: get_entity_daily_sentiment — aggregated daily for line charts
-- Returns: {date, positive, negative, neutral, total, net_score}
-- net_score = avg(score_positive - score_negative)
-- ============================================================
CREATE OR REPLACE FUNCTION get_entity_daily_sentiment(
  p_entity_id uuid,
  p_days integer DEFAULT 30
)
RETURNS jsonb
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
SELECT COALESCE(jsonb_agg(jsonb_build_object(
  'date', d.day,
  'positive', COUNT(*) FILTER (WHERE ss.label = 'positive'),
  'negative', COUNT(*) FILTER (WHERE ss.label = 'negative'),
  'neutral', COUNT(*) FILTER (WHERE ss.label = 'neutral'),
  'total', COUNT(*),
  'net_score', COALESCE(AVG(ss.score_positive - ss.score_negative), 0)
)), '[]'::jsonb)
FROM generate_series(
  DATE(NOW() - (p_days || ' days')::interval),
  DATE(NOW()),
  '1 day'
) AS d(day)
LEFT JOIN sentiment_scores ss 
  ON DATE(ss.scored_at) = d.day 
  AND ss.entity_id = p_entity_id
GROUP BY d.day
ORDER BY d.day ASC;
$$;

-- ============================================================
-- 8. NEW: get_entities_comparison — head-to-head compare
-- Compare multiple entities' daily sentiment side by side
-- ============================================================
CREATE OR REPLACE FUNCTION get_entities_comparison(
  p_entity_ids uuid[],
  p_days integer DEFAULT 30
)
RETURNS jsonb
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
SELECT COALESCE(jsonb_agg(jsonb_build_object(
  'entity_id', pe.id,
  'entity_name', pe.canonical_name,
  'photo_url', pe.photo_url,
  'daily_data', (
    SELECT COALESCE(jsonb_agg(jsonb_build_object(
      'date', d.day,
      'positive', COUNT(*) FILTER (WHERE ss.label = 'positive'),
      'negative', COUNT(*) FILTER (WHERE ss.label = 'negative'),
      'neutral', COUNT(*) FILTER (WHERE ss.label = 'neutral'),
      'total', COUNT(*),
      'net_score', COALESCE(AVG(ss.score_positive - ss.score_negative), 0)
    )), '[]'::jsonb)
    FROM generate_series(
      DATE(NOW() - (p_days || ' days')::interval),
      DATE(NOW()),
      '1 day'
    ) AS d(day)
    LEFT JOIN sentiment_scores ss 
      ON DATE(ss.scored_at) = d.day 
      AND ss.entity_id = pe.id
    GROUP BY d.day
    ORDER BY d.day ASC
  )
)), '[]'::jsonb)
FROM political_entities pe
WHERE pe.id = ANY(p_entity_ids);
$$;

-- ============================================================
-- GRANT EXECUTE to anon and authenticated
-- ============================================================
GRANT EXECUTE ON FUNCTION get_dashboard_summary() TO anon, authenticated;
GRANT EXECUTE ON FUNCTION get_entity_sentiment_timeline(uuid, integer) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION get_entity_highlights(uuid, integer) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION get_entities_list(integer, integer) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION get_sentiment_distribution(integer) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION get_entity_detail(uuid) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION get_entity_daily_sentiment(uuid, integer) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION get_entities_comparison(uuid[], integer) TO anon, authenticated;

-- ============================================================
-- VERIFIKASI: Test semua functions
-- ============================================================
-- SELECT get_dashboard_summary();
-- SELECT get_entity_sentiment_timeline('00000000-0000-0000-0000-000000000000'::uuid, 30);
-- SELECT get_entity_highlights(NULL, 10);
-- SELECT get_entities_list(10, 0);
-- SELECT get_sentiment_distribution(30);
-- SELECT get_entity_detail('00000000-0000-0000-0000-000000000000'::uuid);
-- SELECT get_entity_daily_sentiment('00000000-0000-0000-0000-000000000000'::uuid, 30);
-- SELECT get_entities_comparison(ARRAY['uuid1'::uuid, 'uuid2'::uuid], 30);

-- ============================================================
-- 9. NEW: search_entities — text search di canonical_name + aliases
-- Untuk fitur search di frontend
-- ============================================================
CREATE OR REPLACE FUNCTION search_entities(
  p_query text,
  p_limit integer DEFAULT 20
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
  'mention_count_7d', pe.mention_count_7d,
  'mention_count_30d', pe.mention_count_30d,
  'last_mentioned_at', pe.last_mentioned_at
)), '[]'::jsonb)
FROM political_entities pe
WHERE pe.is_active = true
  AND (
    pe.canonical_name ILIKE '%' || p_query || '%'
    OR EXISTS (SELECT 1 FROM unnest(pe.aliases) AS alias WHERE alias ILIKE '%' || p_query || '%')
    OR pe.party_affiliation ILIKE '%' || p_query || '%'
    OR pe.position ILIKE '%' || p_query || '%'
  )
ORDER BY 
  CASE 
    WHEN pe.canonical_name ILIKE p_query || '%' THEN 1  -- exact start match
    WHEN pe.canonical_name ILIKE '%' || p_query || '%' THEN 2  -- contains
    ELSE 3  -- alias match
  END,
  pe.mention_count_7d DESC NULLS LAST
LIMIT p_limit;
$$;

-- ============================================================
-- 10. NEW: get_entities_filtered — filter + pagination + total count
-- Untuk entity list page dengan filter dan pagination
-- ============================================================
CREATE OR REPLACE FUNCTION get_entities_filtered(
  p_limit integer DEFAULT 50,
  p_offset integer DEFAULT 0,
  p_entity_type text DEFAULT NULL,
  p_party text DEFAULT NULL,
  p_sort_by text DEFAULT 'trending'  -- 'trending', 'name', 'recent'
)
RETURNS jsonb
LANGUAGE sql
AS $$
SELECT jsonb_build_object(
  'entities', COALESCE(jsonb_agg(jsonb_build_object(
    'id', pe.id,
    'name', pe.canonical_name,
    'aliases', pe.aliases,
    'entity_type', pe.entity_type,
    'party', pe.party_affiliation,
    'position', pe.position,
    'photo_url', pe.photo_url,
    'bio', pe.bio,
    'mention_count_7d', pe.mention_count_7d,
    'mention_count_30d', pe.mention_count_30d,
    'last_mentioned_at', pe.last_mentioned_at
  )), '[]'::jsonb),
  'total', COUNT(*) OVER (),
  'limit', p_limit,
  'offset', p_offset
)
FROM (
  SELECT *
  FROM political_entities
  WHERE is_active = true
    AND (p_entity_type IS NULL OR entity_type = p_entity_type)
    AND (p_party IS NULL OR party_affiliation ILIKE '%' || p_party || '%')
  ORDER BY 
    CASE p_sort_by
      WHEN 'name' THEN canonical_name
      ELSE ''
    END ASC,
    CASE p_sort_by
      WHEN 'recent' THEN COALESCE(last_mentioned_at, '1970-01-01'::timestamptz)
      ELSE '1970-01-01'::timestamptz
    END DESC,
    mention_count_7d DESC NULLS LAST
  LIMIT p_limit
  OFFSET p_offset
) pe;
$$;

-- ============================================================
-- 11. NEW: get_feed — global feed of latest highlights
-- Untuk feed page (all entities, latest articles)
-- ============================================================
CREATE OR REPLACE FUNCTION get_feed(
  p_limit integer DEFAULT 30,
  p_offset integer DEFAULT 0,
  p_polarity text DEFAULT NULL  -- filter: 'positive', 'negative', NULL=all
)
RETURNS jsonb
LANGUAGE sql
AS $$
SELECT jsonb_build_object(
  'items', COALESCE(jsonb_agg(jsonb_build_object(
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
    'score_positive', eh.score_positive,
    'score_negative', eh.score_negative,
    'published_at', eh.published_at,
    'curated_at', eh.curated_at
  )), '[]'::jsonb),
  'total', COUNT(*) OVER (),
  'limit', p_limit,
  'offset', p_offset
)
FROM entity_highlights eh
LEFT JOIN political_entities pe ON pe.id = eh.entity_id
WHERE (p_polarity IS NULL OR eh.polarity = p_polarity)
ORDER BY eh.published_at DESC NULLS LAST
LIMIT p_limit
OFFSET p_offset;
$$;

-- GRANT new functions
GRANT EXECUTE ON FUNCTION search_entities(text, integer) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION get_entities_filtered(integer, integer, text, text, text) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION get_feed(integer, integer, text) TO anon, authenticated;

-- VERIFIKASI:
-- SELECT search_entities('Prabowo', 10);
-- SELECT get_entities_filtered(20, 0, 'president', NULL, 'trending');
-- SELECT get_feed(30, 0, NULL);
-- SELECT get_feed(10, 0, 'negative');
