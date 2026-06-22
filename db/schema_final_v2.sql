-- ==========================================
-- FINAL SCHEMA v2: ID-Political-Sentiment-Tracker
-- 100% FREE-TIER COMPLIANT
--   - Supabase Free (500MB DB, pgmq, pg_cron)
--   - Hugging Face Spaces Free (NLP worker)
--   - RSS-only sources (no paid APIs: no X/Twitter, no YouTube Data API, no Play Store)
-- Corrections vs v1:
--   #1 sha256()       -> digest() + pgcrypto
--   #2 partitions     -> auto-create current + 2 ahead (v1 only had 2024, expired)
--   #3 MV RLS         -> enabled for anon (v1 wrongly assumed auto-expose)
--   #4 PROCEDURE      -> FUNCTION so PostgREST RPC works
--   #5 partition prune-> filter on scored_month (key) not just scored_at
--   #6 global dedup   -> separate raw_text_hashes table (cross-month safe)
--   #7 MV rolling 90d -> bounded growth, fast refresh
--   #8 pg_cron refresh-> non-CONCURRENT (transaction-safe on all pg_cron)
--   #9 retention      -> drop partitions > 6 months (free-tier storage budget)
--   #10 confidence    -> parameterized everywhere
-- ==========================================

-- 0. EXTENSIONS & CLEANUP
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- digest() for SHA-256
CREATE EXTENSION IF NOT EXISTS pg_cron;
-- pgmq: enable via Supabase Dashboard > Database > Extensions, OR:
-- CREATE EXTENSION IF NOT EXISTS pgmq;

-- Drop routines FIRST. CRITICAL: v1 created batch_insert_raw_texts as a
-- PROCEDURE; v2 needs it as a FUNCTION. PostgreSQL refuses to change the
-- "routine kind" via CREATE OR REPLACE, so we must DROP it explicitly.
-- DROP ROUTINE is kind-agnostic (kills PROCEDURE or FUNCTION).
DROP ROUTINE IF EXISTS batch_insert_raw_texts(JSONB) CASCADE;
DROP ROUTINE IF EXISTS batch_insert_raw_texts(JSONB, INTEGER, INTEGER) CASCADE;
DROP ROUTINE IF EXISTS insert_sentiment_score(UUID, UUID, TEXT, REAL, REAL, REAL, REAL, TEXT, TEXT) CASCADE;
DROP ROUTINE IF EXISTS get_sentiment_timeseries(UUID, TIMESTAMPTZ, TIMESTAMPTZ, REAL) CASCADE;
DROP ROUTINE IF EXISTS get_entity_ranking(INTEGER, INTEGER, REAL) CASCADE;
DROP ROUTINE IF EXISTS create_monthly_partitions() CASCADE;
DROP ROUTINE IF EXISTS drop_old_partitions(INTEGER) CASCADE;
DROP ROUTINE IF EXISTS trg_set_partition_month() CASCADE;

DROP MATERIALIZED VIEW IF EXISTS mv_dashboard_summary CASCADE;
DROP TABLE IF EXISTS sentiment_scores CASCADE;
DROP TABLE IF EXISTS raw_texts CASCADE;
DROP TABLE IF EXISTS raw_text_hashes CASCADE;
DROP TABLE IF EXISTS scraping_configs CASCADE;
DROP TABLE IF EXISTS political_entities CASCADE;


-- ==========================================
-- 1. MASTER TOKOH POLITIK (PUBLIC READ, INCLUDE FOTO)
-- ==========================================
CREATE TABLE political_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    canonical_name TEXT NOT NULL,
    aliases TEXT[] NOT NULL DEFAULT '{}',          -- NLP entity matching
    entity_type TEXT NOT NULL CHECK (entity_type IN ('president','vp','minister','legislator','party','governor','other')),
    party_affiliation TEXT,
    position TEXT,
    photo_url TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);


-- ==========================================
-- 2. CONFIG RSS (FREE SOURCES ONLY)
-- ==========================================
CREATE TABLE scraping_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id UUID REFERENCES political_entities(id) ON DELETE CASCADE,
    -- RSS-only: no twitter_api, no youtube_api, no play_store (all cost money/quota)
    source_type TEXT NOT NULL CHECK (source_type IN ('rss_news','google_news_rss')),
    config_name TEXT NOT NULL,
    url TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    last_run_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);


-- ==========================================
-- 3. GLOBAL DEDUP TABLE (NON-PARTITIONED)
--    Memecahkan masalah v1: duplikat lolos antar-bulan karena
--    index dedup v1 terikat ke ingested_month (partisi key).
-- ==========================================
CREATE TABLE raw_text_hashes (
    text_hash TEXT PRIMARY KEY,
    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ==========================================
-- 4. RAW TEXTS (PARTITIONED BY MONTH, NO PK IN CREATE TABLE)
-- ==========================================
CREATE TABLE raw_texts (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    title TEXT,                                  -- headline (public-safe, NOT body)
    source_url TEXT,                             -- link ke artikel asli (public-safe)
    image_url TEXT,                              -- HOTLINK thumbnail dari RSS (tidak di-host di bucket)
    text TEXT NOT NULL,                          -- body artikel: PRIVATE, RLS blocks anon (UU PDP)
    text_hash TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','queued','processing','processed','failed','skipped')),
    published_at TIMESTAMPTZ,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    ingested_month DATE NOT NULL
) PARTITION BY RANGE (ingested_month);


-- ==========================================
-- 5. SENTIMENT SCORES (PARTITIONED BY MONTH, NO PK IN CREATE TABLE)
-- ==========================================
CREATE TABLE sentiment_scores (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    raw_text_id UUID NOT NULL,                 -- app-level integrity (composite PK on raw_texts)
    entity_id UUID REFERENCES political_entities(id) ON DELETE CASCADE,
    aspect TEXT,
    score_negative REAL NOT NULL CHECK (score_negative BETWEEN 0 AND 1),
    score_neutral REAL NOT NULL CHECK (score_neutral BETWEEN 0 AND 1),
    score_positive REAL NOT NULL CHECK (score_positive BETWEEN 0 AND 1),
    label TEXT NOT NULL CHECK (label IN ('negative','neutral','positive')),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    model_version TEXT NOT NULL DEFAULT 'indobert-v1',
    scored_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    scored_month DATE NOT NULL
) PARTITION BY RANGE (scored_month);


-- ==========================================
-- 6. PARTITION MANAGEMENT (must run BEFORE any insert)
--    Creates current month + 2 ahead. Buffer of 2 protects against a single
--    missed cron run (v1 buffer was 0 for current month -> all inserts failed).
-- ==========================================
CREATE OR REPLACE FUNCTION create_monthly_partitions()
RETURNS VOID LANGUAGE plpgsql AS $$
DECLARE
    d DATE := date_trunc('month', NOW())::date;
    m TEXT;
    start_d DATE;
    end_d DATE;
    i INT;
BEGIN
    FOR i IN 0..2 LOOP
        start_d := (d + make_interval(months => i))::date;
        end_d   := (d + make_interval(months => i + 1))::date;
        m       := to_char(start_d, 'YYYY_MM');

        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS raw_texts_%s PARTITION OF raw_texts FOR VALUES FROM (%L) TO (%L)',
            m, start_d, end_d);

        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS sentiment_scores_%s PARTITION OF sentiment_scores FOR VALUES FROM (%L) TO (%L)',
            m, start_d, end_d);
    END LOOP;
END;
$$;

-- !!! RUN ONCE AT SETUP — creates partitions for the current quarter !!!
SELECT create_monthly_partitions();


-- ==========================================
-- 7. TRIGGER (AUTO-FILL PARTITION KEY)
--    Never pass ingested_month / scored_month in INSERT.
-- ==========================================
CREATE OR REPLACE FUNCTION trg_set_partition_month()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_TABLE_NAME = 'raw_texts' THEN
        NEW.ingested_month := date_trunc('month', NEW.ingested_at);
    ELSIF TG_TABLE_NAME = 'sentiment_scores' THEN
        NEW.scored_month := date_trunc('month', NEW.scored_at);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE PLPGSQL;

CREATE TRIGGER set_raw_texts_month
BEFORE INSERT OR UPDATE OF ingested_at ON raw_texts
FOR EACH ROW EXECUTE FUNCTION trg_set_partition_month();

CREATE TRIGGER set_sentiment_scores_month
BEFORE INSERT OR UPDATE OF scored_at ON sentiment_scores
FOR EACH ROW EXECUTE FUNCTION trg_set_partition_month();


-- ==========================================
-- 8. INDEXES (REPLACE PRIMARY KEY ON PARTITIONED TABLES)
--    BRIN used where possible -> tiny on free-tier 500MB budget.
-- ==========================================
-- raw_texts: composite unique id (acts as PK), dedup handled by raw_text_hashes now
CREATE UNIQUE INDEX idx_raw_texts_pk      ON raw_texts (id, ingested_month);
CREATE INDEX        idx_raw_status         ON raw_texts (status) WHERE status IN ('pending','queued');
CREATE INDEX        idx_raw_published      ON raw_texts (published_at DESC);
CREATE INDEX        idx_raw_ingested_brin  ON raw_texts USING BRIN (ingested_at);   -- cheap time scan

-- sentiment_scores
CREATE UNIQUE INDEX idx_sentiment_scores_pk ON sentiment_scores (id, scored_month);
CREATE INDEX        idx_scores_entity_time  ON sentiment_scores (entity_id, scored_at DESC);
CREATE INDEX        idx_scores_raw_text     ON sentiment_scores (raw_text_id);
CREATE INDEX        idx_scores_month_brin   ON sentiment_scores USING BRIN (scored_at);


-- ==========================================
-- 9. INGESTION RPC (FUNCTION, NOT PROCEDURE -> PostgREST-callable)
--    Global dedup via raw_text_hashes ON CONFLICT (cross-month safe).
-- ==========================================
CREATE OR REPLACE FUNCTION batch_insert_raw_texts(p_items JSONB)
RETURNS TABLE(inserted_count INTEGER, duplicate_count INTEGER)
LANGUAGE plpgsql SECURITY DEFINER AS $$
DECLARE
    v_item JSONB;
    v_hash TEXT;
    v_new  TEXT;
    ins INT := 0;
    dup INT := 0;
BEGIN
    FOR v_item IN SELECT * FROM jsonb_array_elements(p_items) LOOP
        -- FIX #1: digest() not sha256()
        v_hash := encode(digest((v_item->>'text')::bytea, 'sha256'), 'hex');

        -- Atomic global dedup: insert hash, skip if exists (works across all months)
        INSERT INTO raw_text_hashes (text_hash) VALUES (v_hash)
        ON CONFLICT (text_hash) DO NOTHING
        RETURNING text_hash INTO v_new;

        IF v_new IS NOT NULL THEN
            INSERT INTO raw_texts (source, source_id, title, source_url, image_url, text, text_hash, metadata, published_at)
            VALUES (
                v_item->>'source',
                v_item->>'source_id',
                NULLIF(v_item->>'title', ''),         -- headline (public-safe via highlight)
                NULLIF(v_item->>'source_url', ''),    -- link (public-safe via highlight)
                NULLIF(v_item->>'image_url', ''),     -- hotlink thumbnail (public-safe)
                v_item->>'text',                      -- body: PRIVATE, RLS blocks anon
                v_hash,
                v_item->'metadata',
                NULLIF(v_item->>'published_at', '')::timestamptz
            );
            ins := ins + 1;
        ELSE
            dup := dup + 1;
        END IF;
    END LOOP;
    RETURN QUERY VALUES (ins, dup);
END;
$$;


-- ==========================================
-- 10. NLP RESULT INSERT (FUNCTION for worker convenience)
-- ==========================================
CREATE OR REPLACE FUNCTION insert_sentiment_score(
    p_raw_text_id UUID,
    p_entity_id   UUID,
    p_label       TEXT,
    p_neg REAL, p_neu REAL, p_pos REAL,
    p_confidence  REAL,
    p_aspect      TEXT DEFAULT NULL,
    p_model_version TEXT DEFAULT 'indobert-v1'
) RETURNS UUID
LANGUAGE plpgsql SECURITY DEFINER AS $$
DECLARE v_id UUID;
BEGIN
    INSERT INTO sentiment_scores
        (raw_text_id, entity_id, aspect, score_negative, score_neutral, score_positive,
         label, confidence, model_version)
    VALUES
        (p_raw_text_id, p_entity_id, p_aspect, p_neg, p_neu, p_pos,
         p_label, p_confidence, p_model_version)
    RETURNING id INTO v_id;
    RETURN v_id;
END;
$$;


-- ==========================================
-- 11. API RPC FOR NEXT.JS (with partition pruning)
--     FIX #5: filter on scored_month (partition key) so Postgres prunes partitions.
-- ==========================================
CREATE OR REPLACE FUNCTION get_sentiment_timeseries(
    p_entity_id      UUID,
    p_start_date     TIMESTAMPTZ DEFAULT NOW() - INTERVAL '30 days',
    p_end_date       TIMESTAMPTZ DEFAULT NOW(),
    p_min_confidence REAL        DEFAULT 0.7
) RETURNS TABLE (
    period         TIMESTAMPTZ,
    label          TEXT,
    count          BIGINT,
    percentage     REAL,
    avg_confidence REAL
)
LANGUAGE sql STABLE AS $$
    SELECT
        date_trunc('day', scored_at) AS period,
        label,
        COUNT(*) AS count,
        ROUND(COUNT(*) * 100.0 / NULLIF(SUM(COUNT(*)) OVER (PARTITION BY date_trunc('day', scored_at)), 0), 2)::REAL AS percentage,
        ROUND(AVG(confidence)::numeric, 4)::REAL AS avg_confidence
    FROM sentiment_scores
    WHERE entity_id = p_entity_id
      AND scored_at BETWEEN p_start_date AND p_end_date
      AND scored_month BETWEEN date_trunc('month', p_start_date)::date
                           AND date_trunc('month', p_end_date)::date   -- enables pruning
      AND confidence >= p_min_confidence
    GROUP BY 1, 2
    ORDER BY 1, 2;
$$;


CREATE OR REPLACE FUNCTION get_entity_ranking(
    p_days           INTEGER DEFAULT 7,
    p_min_mentions   INTEGER DEFAULT 50,
    p_min_confidence REAL    DEFAULT 0.7        -- FIX #10: parameterized (was hardcoded)
) RETURNS TABLE (
    entity_id      UUID,
    entity_name    TEXT,
    entity_type    TEXT,
    photo_url      TEXT,
    total_mentions BIGINT,
    positive_pct   REAL,
    negative_pct   REAL,
    net_sentiment  REAL
)
LANGUAGE sql STABLE AS $$
    SELECT
        pe.id AS entity_id,
        pe.canonical_name AS entity_name,
        pe.entity_type,
        pe.photo_url,
        COUNT(*) AS total_mentions,
        ROUND(COUNT(*) FILTER (WHERE label = 'positive') * 100.0 / COUNT(*), 2)::REAL AS positive_pct,
        ROUND(COUNT(*) FILTER (WHERE label = 'negative') * 100.0 / COUNT(*), 2)::REAL AS negative_pct,
        ROUND((AVG(score_positive) - AVG(score_negative))::numeric, 4)::REAL AS net_sentiment
    FROM sentiment_scores ss
    JOIN political_entities pe ON pe.id = ss.entity_id
    WHERE ss.scored_at > NOW() - make_interval(days => p_days)
      AND ss.scored_month >= date_trunc('month', NOW() - make_interval(days => p_days))::date  -- pruning
      AND ss.confidence >= p_min_confidence
    GROUP BY pe.id, pe.canonical_name, pe.entity_type, pe.photo_url
    HAVING COUNT(*) >= p_min_mentions
    ORDER BY net_sentiment DESC;
$$;


-- ==========================================
-- 11b. ENTITY HIGHLIGHTS (PUBLIC, CURATED, NO RAW TEXT)
--      Solusi untuk kebutuhan: dashboard tokoh butuh "headline + foto + skor"
--      dengan PEMBATASAN UU PDP: body artikel (raw_texts.text) TIDAK boleh
--      diekspos ke publik. Tabel ini hanya menyimpan metadata yang aman:
--        - title      (headline)
--        - source_url (link)
--        - image_url  (hotlink thumbnail, BUKAN bucket)
--        - label + confidence (skor sentiment)
--      Isinya dipilih otomatis oleh refresh_entity_highlights() tiap 15 menit.
-- ==========================================
DROP TABLE IF EXISTS entity_highlights CASCADE;
CREATE TABLE entity_highlights (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id       UUID NOT NULL REFERENCES political_entities(id) ON DELETE CASCADE,
    raw_text_id     UUID NOT NULL,                 -- app-level ref (composite PK on raw_texts)
    polarity        TEXT NOT NULL CHECK (polarity IN ('positive','negative')),
    title           TEXT,                          -- headline (public-safe)
    source_url      TEXT,                          -- link (public-safe)
    source_name     TEXT,                          -- e.g. "Detik", "Kompas"
    image_url       TEXT,                          -- HOTLINK dari RSS (tidak di-host di bucket)
    label           TEXT NOT NULL CHECK (label IN ('positive','neutral','negative')),
    confidence      REAL NOT NULL,
    score_positive  REAL NOT NULL,
    score_negative  REAL NOT NULL,
    published_at    TIMESTAMPTZ,
    curated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (entity_id, raw_text_id)               -- 1 artikel tidak masuk 2x untuk tokoh sama
);

CREATE INDEX idx_highlights_entity_pol ON entity_highlights (entity_id, polarity, published_at DESC);
CREATE INDEX idx_highlights_time        ON entity_highlights (curated_at DESC);


-- ==========================================
-- 11c. CURATION FUNCTION (cron 15 min)
--      Memilih TOP N artikel per tokoh per polaritas (positif & negatif)
--      berdasarkan confidence + recency. Insert-or-replace (UPSERT).
--      Hanya menyentuh kolom PUBLIC-SAFE; raw body tidak pernah disentuh.
-- ==========================================
CREATE OR REPLACE FUNCTION refresh_entity_highlights(p_top_n INTEGER DEFAULT 5)
RETURNS VOID LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
    -- Hapus highlight lama yang artikelnya sudah > 30 hari atau confidence turun
    DELETE FROM entity_highlights
    WHERE published_at < NOW() - INTERVAL '30 days';

    -- UPSERT top-N positif & negatif per tokoh (window function ranking)
    INSERT INTO entity_highlights (
        entity_id, raw_text_id, polarity, title, source_url, source_name,
        image_url, label, confidence, score_positive, score_negative, published_at
    )
    SELECT
        entity_id, raw_text_id, polarity, title, source_url, source_name,
        image_url, label, confidence, score_positive, score_negative, published_at
    FROM (
        SELECT
            ss.entity_id,
            ss.raw_text_id,
            CASE WHEN ss.label = 'positive' THEN 'positive' ELSE 'negative' END AS polarity,
            rt.title,
            rt.source_url,
            rt.source AS source_name,
            rt.image_url,
            ss.label,
            ss.confidence,
            ss.score_positive,
            ss.score_negative,
            rt.published_at,
            ROW_NUMBER() OVER (
                PARTITION BY ss.entity_id,
                             CASE WHEN ss.label = 'positive' THEN 'positive' ELSE 'negative' END
                ORDER BY ss.confidence DESC, rt.published_at DESC NULLS LAST
            ) AS rn
        FROM sentiment_scores ss
        JOIN raw_texts rt ON rt.id = ss.raw_text_id
        WHERE ss.confidence >= 0.7
          AND ss.label IN ('positive','negative')          -- netral tidak jadi highlight
          AND rt.published_at >= NOW() - INTERVAL '30 days'
          AND rt.title IS NOT NULL                          -- butuh headline
    ) ranked
    WHERE rn <= p_top_n
    ON CONFLICT (entity_id, raw_text_id) DO UPDATE
    SET confidence     = EXCLUDED.confidence,
        score_positive = EXCLUDED.score_positive,
        score_negative = EXCLUDED.score_negative,
        curated_at     = NOW();
END;
$$;


-- ==========================================
-- 11d. PUBLIC RPC: get highlights per tokoh
--      Frontend Next.js: supabase.rpc('get_entity_highlights', { p_entity_id, p_polarity })
-- ==========================================
CREATE OR REPLACE FUNCTION get_entity_highlights(
    p_entity_id UUID,
    p_polarity  TEXT DEFAULT NULL,        -- 'positive' | 'negative' | NULL (both)
    p_limit     INTEGER DEFAULT 10
) RETURNS TABLE (
    highlight_id   UUID,
    polarity       TEXT,
    title          TEXT,
    source_url     TEXT,
    source_name    TEXT,
    image_url      TEXT,                  -- hotlink URL (frontend render <img src>)
    label          TEXT,
    confidence     REAL,
    score_positive REAL,
    score_negative REAL,
    published_at   TIMESTAMPTZ
)
LANGUAGE sql STABLE SECURITY DEFINER AS $$
    SELECT
        id, polarity, title, source_url, source_name, image_url,
        label, confidence, score_positive, score_negative, published_at
    FROM entity_highlights
    WHERE entity_id = p_entity_id
      AND (p_polarity IS NULL OR polarity = p_polarity)
    ORDER BY polarity, published_at DESC
    LIMIT p_limit;
$$;


-- ==========================================
-- 12. MATERIALIZED VIEW (rolling 90 days -> bounded for free tier)
--     FIX #7: WHERE scored_at >= NOW() - 90 days keeps MV small & refresh fast.
-- ==========================================
CREATE MATERIALIZED VIEW mv_dashboard_summary AS
SELECT
    pe.id AS entity_id,
    pe.canonical_name,
    pe.entity_type,
    pe.photo_url,
    date_trunc('day', ss.scored_at) AS day,
    COUNT(*) AS total_mentions,
    COUNT(*) FILTER (WHERE label = 'positive') AS positive_count,
    COUNT(*) FILTER (WHERE label = 'negative') AS negative_count,
    COUNT(*) FILTER (WHERE label = 'neutral')  AS neutral_count,
    ROUND(AVG(ss.score_positive - ss.score_negative)::numeric, 4) AS net_sentiment_score
FROM sentiment_scores ss
JOIN political_entities pe ON pe.id = ss.entity_id
WHERE ss.confidence >= 0.6
  AND ss.scored_at >= NOW() - INTERVAL '90 days'
GROUP BY pe.id, pe.canonical_name, pe.entity_type, pe.photo_url, date_trunc('day', ss.scored_at)
WITH DATA;

CREATE UNIQUE INDEX idx_mv_dashboard ON mv_dashboard_summary (entity_id, day);


-- ==========================================
-- 13. RETENTION POLICY (free-tier storage budget)
--     Drops partitions older than N months. Default 6 months.
--     FIX #9: v1 had no retention -> storage grew unbounded.
-- ==========================================
CREATE OR REPLACE FUNCTION drop_old_partitions(p_keep_months INT DEFAULT 6)
RETURNS VOID LANGUAGE plpgsql AS $$
DECLARE
    cutoff DATE := date_trunc('month', NOW() - make_interval(months => p_keep_months))::date;
    r RECORD;
    pm DATE;
BEGIN
    FOR r IN
        SELECT inhrelid::regclass::text AS pname
        FROM pg_inherits
        WHERE inhparent IN ('raw_texts'::regclass, 'sentiment_scores'::regclass)
    LOOP
        BEGIN
            -- pname e.g. 'raw_texts_2024_05' -> '2024-05-01'
            pm := to_date(
                split_part(r.pname, '_', 3) || '-' || split_part(r.pname, '_', 4) || '-01',
                'YYYY-MM-DD'
            );
            IF pm < cutoff THEN
                EXECUTE format('DROP TABLE IF EXISTS %s', r.pname);
                RAISE NOTICE 'Dropped partition %', r.pname;
            END IF;
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'Skipped %: %', r.pname, SQLERRM;
        END;
    END LOOP;
END;
$$;


-- ==========================================
-- 14. PG_CRON JOBS
--     FIX #8: plain REFRESH (no CONCURRENTLY) — works inside pg_cron's
--     transaction wrapper on ALL versions. Lock is brief on a 90-day MV.
-- ==========================================
SELECT cron.schedule('refresh_mv_dashboard',  '*/10 * * * *',  'REFRESH MATERIALIZED VIEW mv_dashboard_summary');
SELECT cron.schedule('auto_create_partitions', '0 0 25 * *',    'SELECT create_monthly_partitions();');
SELECT cron.schedule('drop_old_partitions',    '0 1 1 * *',     'SELECT drop_old_partitions(6);');   -- monthly
SELECT cron.schedule('refresh_highlights',     '*/15 * * * *',  'SELECT refresh_entity_highlights(5);');  -- top-5 per polarity per tokoh


-- ==========================================
-- 15. ROW LEVEL SECURITY (UU PDP — no login, public dashboard)
-- ==========================================
ALTER TABLE raw_texts          ENABLE ROW LEVEL SECURITY;
ALTER TABLE sentiment_scores   ENABLE ROW LEVEL SECURITY;
ALTER TABLE political_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE scraping_configs   ENABLE ROW LEVEL SECURITY;
ALTER TABLE raw_text_hashes    ENABLE ROW LEVEL SECURITY;

-- !!! MATERIALIZED VIEW TIDAK MENDUKUNG RLS !!!
-- PostgREST mengekspos MV ke anon via GRANT, BUKAN policy. Tanpa baris ini,
-- supabase.from('mv_dashboard_summary').select() akan error untuk user publik.
GRANT SELECT ON mv_dashboard_summary TO anon, authenticated;

-- raw_texts: anon BLOCKED (contains scraped text — PDP risk)
CREATE POLICY "svc raw_texts all"    ON raw_texts          FOR ALL    TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "anon block raw_texts" ON raw_texts          FOR SELECT TO anon        USING (false);

-- sentiment_scores: anon BLOCKED
CREATE POLICY "svc scores all"    ON sentiment_scores      FOR ALL    TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "anon block scores" ON sentiment_scores      FOR SELECT TO anon        USING (false);

-- political_entities: anon READ (name + photo for dashboard)
CREATE POLICY "anon read entities" ON political_entities   FOR SELECT TO anon        USING (true);
CREATE POLICY "svc entities all"   ON political_entities   FOR ALL    TO service_role USING (true) WITH CHECK (true);

-- scraping_configs: anon BLOCKED (internal config)
CREATE POLICY "svc config all"     ON scraping_configs     FOR ALL    TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "anon block config"  ON scraping_configs     FOR SELECT TO anon        USING (false);

-- raw_text_hashes: anon BLOCKED (internal dedup)
CREATE POLICY "svc hashes all"     ON raw_text_hashes      FOR ALL    TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "anon block hashes"  ON raw_text_hashes      FOR SELECT TO anon        USING (false);

-- mv_dashboard_summary: anon READ via GRANT (above). MVs cannot have RLS policies.
-- (no CREATE POLICY here — RLS is not supported on materialized views)

-- entity_highlights: anon READ (public-safe: headline+link+thumbnail+skor, NO body text)
ALTER TABLE entity_highlights ENABLE ROW LEVEL SECURITY;
CREATE POLICY "svc highlights all" ON entity_highlights FOR ALL    TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "anon read highlights" ON entity_highlights FOR SELECT TO anon       USING (true);


-- ==========================================
-- 16. STORAGE POLICIES — bucket 'politik'
--     Bucket ini khusus FOTO TOKOH POLITIK (self-hosted, mis. hasil crop atau
--     gambar bebas hak). Thumbnail berita TIDAK di-host di sini — di-hotlink.
--     Anon: baca public. Service role: upload/update/delete (via NLP worker / admin).
--     NOTE: Storage policies pakai tabel storage.objects, scope ke bucket.id.
-- ==========================================
INSERT INTO storage.buckets (id, name, public)
VALUES ('politik', 'politik', true)
ON CONFLICT (id) DO UPDATE SET public = true;

-- Anon: baca semua object di bucket 'politik' (untuk render foto tokoh di dashboard)
CREATE POLICY "anon read politik bucket"
ON storage.objects FOR SELECT TO anon
USING (bucket_id = 'politik');

-- Service role: tulah penuh (upload/update/delete foto tokoh)
CREATE POLICY "svc write politik bucket"
ON storage.objects FOR ALL TO service_role
USING (bucket_id = 'politik')
WITH CHECK (bucket_id = 'politik');
