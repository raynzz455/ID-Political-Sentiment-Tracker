-- ==========================================================
-- ID POLITICAL SENTIMENT TRACKER - FINAL SCHEMA v3 (ULTIMATE MERGE)
-- Menggabungkan arsitektur v2 (Global Dedup, Auto-Partition, RLS, Highlights)
-- dengan kebutuhan pipeline v19 (Entity Contexts, Bulk RPC, Trigger Log).
-- 100% FREE-TIER COMPLIANT & UU PDP READY
-- ==========================================================

-- 0. EXTENSIONS & CLEANUP
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_cron;
CREATE EXTENSION IF NOT EXISTS pgmq;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Cleanup routines lama agar tidak bentrok
DROP ROUTINE IF EXISTS batch_insert_raw_texts(JSONB) CASCADE;
DROP ROUTINE IF EXISTS bulk_update_raw_texts(JSONB) CASCADE;
DROP ROUTINE IF EXISTS insert_sentiment_score(UUID, UUID, TEXT, REAL, REAL, REAL, REAL, TEXT, TEXT) CASCADE;
DROP ROUTINE IF EXISTS create_monthly_partitions() CASCADE;
DROP ROUTINE IF EXISTS trg_set_partition_month() CASCADE;
DROP ROUTINE IF EXISTS refresh_entity_highlights(INTEGER) CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_dashboard_summary CASCADE;
DROP TABLE IF EXISTS entity_highlights CASCADE;
DROP TABLE IF EXISTS sentiment_scores CASCADE;
DROP TABLE IF EXISTS raw_texts CASCADE;
DROP TABLE IF EXISTS entity_contexts CASCADE;
DROP TABLE IF EXISTS entity_mentions CASCADE;
DROP TABLE IF EXISTS article_entity_map CASCADE;
DROP TABLE IF EXISTS pipeline_runs CASCADE;
DROP TABLE IF EXISTS entity_candidates CASCADE;
DROP TABLE IF EXISTS raw_text_hashes CASCADE;
DROP TABLE IF EXISTS scraping_configs CASCADE;
DROP TABLE IF EXISTS political_entities CASCADE;

-- ==========================================
-- 1. MASTER DATA (PUBLIC READ, INCLUDE FOTO)
-- ==========================================
CREATE TABLE political_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    canonical_name TEXT NOT NULL UNIQUE,
    aliases TEXT[] NOT NULL DEFAULT '{}',
    entity_type TEXT NOT NULL CHECK (entity_type IN ('president','vp','minister','legislator','party','governor','other')),
    party_affiliation TEXT,
    position TEXT,
    photo_url TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE scraping_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id UUID REFERENCES political_entities(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL CHECK (source_type IN ('rss_news','google_news_rss')),
    config_name TEXT NOT NULL UNIQUE,
    url TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    last_run_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE entity_candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    detected_name TEXT NOT NULL UNIQUE,
    normalized_name TEXT,
    detection_source TEXT NOT NULL,
    mention_count INTEGER DEFAULT 0,
    gnews_hit_count INTEGER DEFAULT 0,
    sample_titles TEXT[] DEFAULT '{}',
    wikipedia_url TEXT,
    wikipedia_snippet TEXT,
    suggested_type TEXT DEFAULT 'other',
    suggested_aliases TEXT[] DEFAULT '{}',
    status TEXT DEFAULT 'pending',
    confidence_score REAL DEFAULT 0.5,
    promoted_entity_id UUID,
    last_seen_year SMALLINT,
    is_within_5_years BOOLEAN DEFAULT true,
    first_detected TIMESTAMPTZ DEFAULT NOW(),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    reviewed_at TIMESTAMPTZ,
    notes TEXT
);

-- ==========================================
-- 2. GLOBAL DEDUP TABLE (NON-PARTITIONED)
-- ==========================================
CREATE TABLE raw_text_hashes (
    text_hash TEXT PRIMARY KEY,
    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ==========================================
-- 3. RAW TEXTS (PARTITIONED BY MONTH)
-- ==========================================
CREATE TABLE raw_texts (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    title TEXT,
    source_url TEXT,
    image_url TEXT,
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','enriched','validated','queued','processing','processed','failed','skipped')),
    published_at TIMESTAMPTZ,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    ingested_month DATE NOT NULL,
    -- Kolom tambahan untuk pipeline v19
    entity_resolved_at TIMESTAMPTZ,
    pipeline_version TEXT,
    resolver_version TEXT,
    context_version TEXT,
    duplicate_of UUID,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    preprocessing_version TEXT,
    content_hash TEXT,
    canonical_url TEXT,
    resolved_domain TEXT,
    preprocessed_at TIMESTAMPTZ,
    context_extracted_at TIMESTAMPTZ,
    nlp_ready_at TIMESTAMPTZ,
    content_type TEXT DEFAULT 'UNKNOWN',
    recovery_attempts INTEGER DEFAULT 0,
    recovery_status TEXT DEFAULT 'pending'
) PARTITION BY RANGE (ingested_month);

-- ==========================================
-- 4. ENTITY PIPELINE TABLES (Layer 3.2 - 3.5)
-- ==========================================
CREATE TABLE entity_mentions (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    raw_text_id UUID,
    ingested_month DATE,
    entity_id UUID,
    mention_text TEXT,
    start_offset INTEGER,
    end_offset INTEGER,
    ner_model TEXT DEFAULT 'spacy_sm',
    ner_confidence REAL DEFAULT 1.0
);
CREATE INDEX idx_mentions_raw ON entity_mentions(raw_text_id, ingested_month);

CREATE TABLE entity_contexts (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    raw_text_id UUID,
    ingested_month DATE,
    entity_id UUID,
    context_text TEXT,
    context_version TEXT DEFAULT 'v1',
    metadata JSONB
);
CREATE INDEX idx_contexts_raw ON entity_contexts(raw_text_id, ingested_month);

CREATE TABLE article_entity_map (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    raw_text_id UUID,
    ingested_month DATE,
    entity_id UUID,
    is_main_entity BOOLEAN DEFAULT false,
    confidence REAL DEFAULT 1.0,
    resolver_source TEXT DEFAULT 'unknown'
);
CREATE INDEX idx_map_raw ON article_entity_map(raw_text_id, ingested_month);

-- ==========================================
-- 5. SENTIMENT SCORES (PARTITIONED BY MONTH)
-- ==========================================
CREATE TABLE sentiment_scores (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    raw_text_id UUID NOT NULL,
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
-- 6. PIPELINE LOGS
-- ==========================================
CREATE TABLE pipeline_runs (
    id UUID NOT NULL DEFAULT gen_random_uuid(),
    worker_name TEXT NOT NULL,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    duration_seconds REAL,
    articles_processed INTEGER DEFAULT 0,
    articles_succeeded INTEGER DEFAULT 0,
    articles_failed INTEGER DEFAULT 0,
    version TEXT,
    status TEXT DEFAULT 'running',
    notes TEXT
);

-- ==========================================
-- 7. ENTITY HIGHLIGHTS (PUBLIC CACHE FOR DASHBOARD/HOTLINE TOKOH)
-- ==========================================
CREATE TABLE entity_highlights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id UUID NOT NULL REFERENCES political_entities(id) ON DELETE CASCADE,
    raw_text_id UUID NOT NULL,
    polarity TEXT NOT NULL CHECK (polarity IN ('positive','negative')),
    title TEXT,
    source_url TEXT,
    source_name TEXT,
    image_url TEXT,
    label TEXT NOT NULL CHECK (label IN ('positive','neutral','negative')),
    confidence REAL NOT NULL,
    score_positive REAL NOT NULL,
    score_negative REAL NOT NULL,
    published_at TIMESTAMPTZ,
    curated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (entity_id, raw_text_id)
);
CREATE INDEX idx_highlights_entity_pol ON entity_highlights (entity_id, polarity, published_at DESC);

-- ==========================================
-- 8. AUTO-PARTITION MANAGEMENT
-- ==========================================
CREATE OR REPLACE FUNCTION create_monthly_partitions()
RETURNS VOID LANGUAGE plpgsql AS $$ DECLARE d DATE := date_trunc('month', NOW())::date; m TEXT; start_d DATE; end_d DATE; i INT;
BEGIN
    FOR i IN 0..2 LOOP
        start_d := (d + make_interval(months => i))::date;
        end_d := (d + make_interval(months => i + 1))::date;
        m := to_char(start_d, 'YYYY_MM');
        EXECUTE format('CREATE TABLE IF NOT EXISTS raw_texts_%s PARTITION OF raw_texts FOR VALUES FROM (%L) TO (%L)', m, start_d, end_d);
        EXECUTE format('CREATE TABLE IF NOT EXISTS sentiment_scores_%s PARTITION OF sentiment_scores FOR VALUES FROM (%L) TO (%L)', m, start_d, end_d);
        
        -- Tambahkan Foreign Key untuk tabel entity di setiap partisi baru
        EXECUTE format('ALTER TABLE entity_mentions ADD CONSTRAINT fk_mentions_%s FOREIGN KEY (raw_text_id, ingested_month) REFERENCES raw_texts_%s(id, ingested_month)', m, m);
        EXECUTE format('ALTER TABLE entity_contexts ADD CONSTRAINT fk_contexts_%s FOREIGN KEY (raw_text_id, ingested_month) REFERENCES raw_texts_%s(id, ingested_month)', m, m);
        EXECUTE format('ALTER TABLE article_entity_map ADD CONSTRAINT fk_map_%s FOREIGN KEY (raw_text_id, ingested_month) REFERENCES raw_texts_%s(id, ingested_month)', m, m);
    END LOOP;
END;
 $$;
SELECT create_monthly_partitions();

-- ==========================================
-- 9. TRIGGERS (AUTO-FILL PARTITION KEY & UPDATED_AT)
-- ==========================================
CREATE OR REPLACE FUNCTION trg_set_meta_fields()
RETURNS TRIGGER AS $$ BEGIN
    IF TG_TABLE_NAME = 'raw_texts' THEN
        NEW.ingested_month := date_trunc('month', NEW.ingested_at);
        NEW.updated_at = NOW();
    ELSIF TG_TABLE_NAME = 'sentiment_scores' THEN
        NEW.scored_month := date_trunc('month', NEW.scored_at);
    END IF;
    RETURN NEW;
END;
 $$ LANGUAGE PLPGSQL;

CREATE TRIGGER set_raw_texts_month BEFORE INSERT OR UPDATE OF ingested_at ON raw_texts FOR EACH ROW EXECUTE FUNCTION trg_set_meta_fields();
CREATE TRIGGER set_sentiment_scores_month BEFORE INSERT OR UPDATE OF scored_at ON sentiment_scores FOR EACH ROW EXECUTE FUNCTION trg_set_meta_fields();

-- ==========================================
-- 10. INDEXES
-- ==========================================
CREATE UNIQUE INDEX idx_raw_texts_pk ON raw_texts (id, ingested_month);
CREATE INDEX idx_raw_status ON raw_texts (status) WHERE status IN ('pending','queued','enriched','validated');
CREATE INDEX idx_raw_ingested_brin ON raw_texts USING BRIN (ingested_at);

CREATE UNIQUE INDEX idx_sentiment_scores_pk ON sentiment_scores (id, scored_month);
CREATE INDEX idx_scores_entity_time ON sentiment_scores (entity_id, scored_at DESC);
CREATE INDEX idx_scores_raw_text ON sentiment_scores (raw_text_id);

-- ==========================================
-- 11. RPC FUNCTIONS (INGESTION, PIPELINE, DASHBOARD)
-- ==========================================

-- A. Ingestion (Global Dedup via raw_text_hashes)
CREATE OR REPLACE FUNCTION batch_insert_raw_texts(p_items JSONB)
RETURNS TABLE(inserted_count INTEGER, duplicate_count INTEGER)
LANGUAGE plpgsql SECURITY DEFINER AS $$ DECLARE v_item JSONB; v_hash TEXT; v_new TEXT; ins INT := 0; dup INT := 0;
BEGIN
    FOR v_item IN SELECT * FROM jsonb_array_elements(p_items) LOOP
        v_hash := encode(digest((v_item->>'text')::bytea, 'sha256'), 'hex');
        INSERT INTO raw_text_hashes (text_hash) VALUES (v_hash) ON CONFLICT (text_hash) DO NOTHING RETURNING text_hash INTO v_new;
        IF v_new IS NOT NULL THEN
            INSERT INTO raw_texts (source, source_id, title, source_url, image_url, text, text_hash, metadata, published_at)
            VALUES (v_item->>'source', v_item->>'source_id', NULLIF(v_item->>'title', ''), NULLIF(v_item->>'source_url', ''), NULLIF(v_item->>'image_url', ''), v_item->>'text', v_hash, v_item->'metadata', NULLIF(v_item->>'published_at', '')::timestamptz);
            ins := ins + 1;
        ELSE
            dup := dup + 1;
        END IF;
    END LOOP;
    RETURN QUERY VALUES (ins, dup);
END;
 $$;

-- B. Bulk Update (Untuk Python Worker Enricher/Preprocessing v19)
CREATE OR REPLACE FUNCTION bulk_update_raw_texts(p_updates JSONB)
RETURNS VOID LANGUAGE plpgsql SECURITY DEFINER AS $$ BEGIN
    UPDATE public.raw_texts AS r
    SET 
        text = COALESCE(x.text, r.text), status = COALESCE(x.status, r.status), content_type = COALESCE(x.content_type, r.content_type),
        metadata = COALESCE(r.metadata || x.metadata, r.metadata), recovery_attempts = COALESCE(x.recovery_attempts, r.recovery_attempts),
        recovery_status = COALESCE(x.recovery_status, r.recovery_status), resolved_domain = COALESCE(x.resolved_domain, r.resolved_domain),
        canonical_url = COALESCE(x.canonical_url, r.canonical_url), content_hash = COALESCE(x.content_hash, r.content_hash),
        processed_at = COALESCE(x.processed_at, r.processed_at), pipeline_version = COALESCE(x.pipeline_version, r.pipeline_version),
        resolver_version = COALESCE(x.resolver_version, r.resolver_version), context_version = COALESCE(x.context_version, r.context_version),
        preprocessed_at = COALESCE(x.preprocessed_at, r.preprocessed_at), context_extracted_at = COALESCE(x.context_extracted_at, r.context_extracted_at),
        nlp_ready_at = COALESCE(x.nlp_ready_at, r.nlp_ready_at), duplicate_of = COALESCE(x.duplicate_of, r.duplicate_of)
    FROM jsonb_populate_recordset(null::public.raw_texts, p_updates) AS x
    WHERE r.id = x.id;
END;
 $$;

-- C. NLP Result Insert (Idempotent)
CREATE OR REPLACE FUNCTION insert_sentiment_score(
    p_raw_text_id UUID, p_entity_id UUID, p_label TEXT, p_neg REAL, p_neu REAL, p_pos REAL, 
    p_confidence REAL, p_aspect TEXT DEFAULT NULL, p_model_version TEXT DEFAULT 'indobert-v1'
) RETURNS VOID AS $$ BEGIN
    DELETE FROM sentiment_scores WHERE raw_text_id = p_raw_text_id AND (entity_id = p_entity_id OR (entity_id IS NULL AND p_entity_id IS NULL));
    INSERT INTO sentiment_scores (raw_text_id, entity_id, label, score_negative, score_neutral, score_positive, confidence, aspect, model_version)
    VALUES (p_raw_text_id, p_entity_id, p_label, p_neg, p_neu, p_pos, p_confidence, p_aspect, p_model_version);
END;
 $$ LANGUAGE plpgsql;

-- D. Dashboard Hotline Tokoh (Refresh & Get)
CREATE OR REPLACE FUNCTION refresh_entity_highlights(p_top_n INTEGER DEFAULT 5)
RETURNS VOID LANGUAGE plpgsql SECURITY DEFINER AS $$ BEGIN
    DELETE FROM entity_highlights WHERE published_at < NOW() - INTERVAL '30 days';
    INSERT INTO entity_highlights (entity_id, raw_text_id, polarity, title, source_url, source_name, image_url, label, confidence, score_positive, score_negative, published_at)
    SELECT entity_id, raw_text_id, polarity, title, source_url, source_name, image_url, label, confidence, score_positive, score_negative, published_at
    FROM (
        SELECT ss.entity_id, ss.raw_text_id, CASE WHEN ss.label = 'positive' THEN 'positive' ELSE 'negative' END AS polarity,
               rt.title, rt.source_url, rt.source AS source_name, rt.image_url, ss.label, ss.confidence, ss.score_positive, ss.score_negative, rt.published_at,
               ROW_NUMBER() OVER (PARTITION BY ss.entity_id, CASE WHEN ss.label = 'positive' THEN 'positive' ELSE 'negative' END ORDER BY ss.confidence DESC, rt.published_at DESC NULLS LAST) AS rn
        FROM sentiment_scores ss JOIN raw_texts rt ON rt.id = ss.raw_text_id
        WHERE ss.confidence >= 0.7 AND ss.label IN ('positive','negative') AND rt.published_at >= NOW() - INTERVAL '30 days' AND rt.title IS NOT NULL
    ) ranked WHERE rn <= p_top_n
    ON CONFLICT (entity_id, raw_text_id) DO UPDATE SET confidence = EXCLUDED.confidence, score_positive = EXCLUDED.score_positive, score_negative = EXCLUDED.score_negative, curated_at = NOW();
END;
 $$;

CREATE OR REPLACE FUNCTION get_entity_highlights(p_entity_id UUID, p_polarity TEXT DEFAULT NULL, p_limit INTEGER DEFAULT 10)
RETURNS TABLE (highlight_id UUID, polarity TEXT, title TEXT, source_url TEXT, source_name TEXT, image_url TEXT, label TEXT, confidence REAL, score_positive REAL, score_negative REAL, published_at TIMESTAMPTZ)
LANGUAGE sql STABLE SECURITY DEFINER AS $$     SELECT id, polarity, title, source_url, source_name, image_url, label, confidence, score_positive, score_negative, published_at
    FROM entity_highlights WHERE entity_id = p_entity_id AND (p_polarity IS NULL OR polarity = p_polarity)
    ORDER BY polarity, published_at DESC LIMIT p_limit;
 $$;

-- ==========================================
-- 12. MATERIALIZED VIEW & RETENTION
-- ==========================================
CREATE MATERIALIZED VIEW mv_dashboard_summary AS
SELECT pe.id AS entity_id, pe.canonical_name, pe.entity_type, pe.photo_url, date_trunc('day', ss.scored_at) AS day,
       COUNT(*) AS total_mentions, COUNT(*) FILTER (WHERE label = 'positive') AS positive_count,
       COUNT(*) FILTER (WHERE label = 'negative') AS negative_count, COUNT(*) FILTER (WHERE label = 'neutral') AS neutral_count,
       ROUND(AVG(ss.score_positive - ss.score_negative)::numeric, 4) AS net_sentiment_score
FROM sentiment_scores ss JOIN political_entities pe ON pe.id = ss.entity_id
WHERE ss.confidence >= 0.6 AND ss.scored_at >= NOW() - INTERVAL '90 days'
GROUP BY pe.id, pe.canonical_name, pe.entity_type, pe.photo_url, date_trunc('day', ss.scored_at)
WITH DATA;
CREATE UNIQUE INDEX idx_mv_dashboard ON mv_dashboard_summary (entity_id, day);

CREATE OR REPLACE FUNCTION drop_old_partitions(p_keep_months INT DEFAULT 6)
RETURNS VOID LANGUAGE plpgsql AS $$ DECLARE cutoff DATE := date_trunc('month', NOW() - make_interval(months => p_keep_months))::date; r RECORD; pm DATE;
BEGIN
    FOR r IN SELECT inhrelid::regclass::text AS pname FROM pg_inherits WHERE inhparent IN ('raw_texts'::regclass, 'sentiment_scores'::regclass) LOOP
        BEGIN
            pm := to_date(split_part(r.pname, '_', 3) || '-' || split_part(r.pname, '_', 4) || '-01', 'YYYY-MM-DD');
            IF pm < cutoff THEN EXECUTE format('DROP TABLE IF EXISTS %s', r.pname); RAISE NOTICE 'Dropped partition %', r.pname; END IF;
        EXCEPTION WHEN OTHERS THEN RAISE NOTICE 'Skipped %: %', r.pname, SQLERRM; END;
    END LOOP;
END;
 $$;

-- ==========================================
-- 13. PG_CRON JOBS
-- ==========================================
SELECT cron.schedule('refresh_mv_dashboard',  '*/10 * * * *',  'REFRESH MATERIALIZED VIEW mv_dashboard_summary');
SELECT cron.schedule('auto_create_partitions', '0 0 25 * *',    'SELECT create_monthly_partitions();');
SELECT cron.schedule('drop_old_partitions',    '0 1 1 * *',     'SELECT drop_old_partitions(6);');
SELECT cron.schedule('refresh_highlights',     '*/15 * * * *',  'SELECT refresh_entity_highlights(5);');

-- ==========================================
-- 14. ROW LEVEL SECURITY (UU PDP)
-- ==========================================
ALTER TABLE raw_texts ENABLE ROW LEVEL SECURITY;
ALTER TABLE sentiment_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE political_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE scraping_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE raw_text_hashes ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_highlights ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_contexts ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_mentions ENABLE ROW LEVEL SECURITY;
ALTER TABLE article_entity_map ENABLE ROW LEVEL SECURITY;

GRANT SELECT ON mv_dashboard_summary TO anon, authenticated;

CREATE POLICY "svc raw_texts all" ON raw_texts FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "anon block raw_texts" ON raw_texts FOR SELECT TO anon USING (false);

CREATE POLICY "svc scores all" ON sentiment_scores FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "anon block scores" ON sentiment_scores FOR SELECT TO anon USING (false);

CREATE POLICY "anon read entities" ON political_entities FOR SELECT TO anon USING (true);
CREATE POLICY "svc entities all" ON political_entities FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "svc config all" ON scraping_configs FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "anon block config" ON scraping_configs FOR SELECT TO anon USING (false);

CREATE POLICY "svc hashes all" ON raw_text_hashes FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "anon block hashes" ON raw_text_hashes FOR SELECT TO anon USING (false);

CREATE POLICY "svc highlights all" ON entity_highlights FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "anon read highlights" ON entity_highlights FOR SELECT TO anon USING (true);

CREATE POLICY "svc contexts all" ON entity_contexts FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "anon block contexts" ON entity_contexts FOR SELECT TO anon USING (false);

-- ==========================================
-- 15. STORAGE POLICIES — bucket 'politik'
-- ==========================================
INSERT INTO storage.buckets (id, name, public) VALUES ('politik', 'politik', true) ON CONFLICT (id) DO UPDATE SET public = true;
CREATE POLICY "anon read politik bucket" ON storage.objects FOR SELECT TO anon USING (bucket_id = 'politik');
CREATE POLICY "svc write politik bucket" ON storage.objects FOR ALL TO service_role USING (bucket_id = 'politik') WITH CHECK (bucket_id = 'politik');