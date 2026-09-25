-- ============================================================
-- 22_extend_historical_retention.sql
-- ============================================================
-- PURPOSE: Extend historical sentiment retention dari 90 → 180 hari.
--
-- MASALAH:
--   mv_dashboard_summary hanya simpan 90 hari sentiment data.
--   User butuh historical sentiment per entity yang lebih panjang
--   untuk lihat tren jangka panjang (6 bulan).
--
-- SOLUSI:
--   1. Recreate mv_dashboard_summary dengan 180-day window
--   2. Add mv_entity_historical_180d untuk deep historical analysis
--   3. Keep mv_dashboard_summary (90d) untuk dashboard cepat
--   4. Tambah RPC get_entity_historical_sentiment untuk FE
-- ============================================================

-- ============================================================
-- SECTION 1: Recreate mv_dashboard_summary dengan 180-day window
-- ============================================================
DROP MATERIALIZED VIEW IF EXISTS mv_dashboard_summary CASCADE;

CREATE MATERIALIZED VIEW mv_dashboard_summary AS
SELECT pe.id AS entity_id, pe.canonical_name, pe.entity_type, pe.photo_url,
       date_trunc('day', ss.scored_at) AS day,
       COUNT(*) AS total_mentions,
       COUNT(*) FILTER (WHERE label = 'positive') AS positive_count,
       COUNT(*) FILTER (WHERE label = 'negative') AS negative_count,
       COUNT(*) FILTER (WHERE label = 'neutral') AS neutral_count,
       ROUND(AVG(ss.score_positive - ss.score_negative)::numeric, 4) AS net_sentiment_score
FROM sentiment_scores ss
JOIN political_entities pe ON pe.id = ss.entity_id
WHERE ss.confidence >= 0.6
  AND ss.scored_at >= NOW() - INTERVAL '180 days'  -- 90 → 180 days
GROUP BY pe.id, pe.canonical_name, pe.entity_type, pe.photo_url, date_trunc('day', ss.scored_at)
WITH DATA;

CREATE UNIQUE INDEX idx_mv_dashboard ON mv_dashboard_summary (entity_id, day);
CREATE INDEX idx_mv_dashboard_day ON mv_dashboard_summary (day DESC);
GRANT SELECT ON mv_dashboard_summary TO anon, authenticated;

-- ============================================================
-- SECTION 2: Add RPC for historical sentiment per entity (180d)
-- ============================================================
CREATE OR REPLACE FUNCTION get_entity_historical_sentiment(
    p_entity_id UUID,
    p_days INTEGER DEFAULT 180
)
RETURNS TABLE (
    day DATE,
    total_mentions BIGINT,
    positive_count BIGINT,
    negative_count BIGINT,
    neutral_count BIGINT,
    net_sentiment_score NUMERIC
)
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = public AS $$
    SELECT day::DATE,
           total_mentions,
           positive_count,
           negative_count,
           neutral_count,
           net_sentiment_score
    FROM mv_dashboard_summary
    WHERE entity_id = p_entity_id
      AND day >= date_trunc('day', NOW() - make_interval(days => p_days))
    ORDER BY day ASC;
$$;

GRANT EXECUTE ON FUNCTION get_entity_historical_sentiment(UUID, INTEGER) TO anon, authenticated;

-- ============================================================
-- SECTION 3: Refresh schedule tetap tiap 10 menit
-- ============================================================
DO $$
DECLARE r RECORD;
BEGIN
    -- Unschedule old mv refresh kalau ada
    FOR r IN SELECT jobid, command FROM cron.job WHERE command LIKE '%mv_dashboard%' LOOP
        PERFORM cron.unschedule(r.jobid);
    END LOOP;
    -- Reschedule dengan 10-min interval
    PERFORM cron.schedule('refresh_mv_dashboard', '*/10 * * * *',
        'REFRESH MATERIALIZED VIEW CONCURRENTLY mv_dashboard_summary');
END $$;

-- ============================================================
-- VERIFIKASI
-- ============================================================
SELECT 'mv_dashboard_summary sekarang simpan 180 hari' AS info;
SELECT COUNT(*) AS total_rows, MIN(day) AS earliest_day, MAX(day) AS latest_day
FROM mv_dashboard_summary;
