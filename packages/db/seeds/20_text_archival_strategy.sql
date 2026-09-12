-- ============================================================
-- 20_text_archival_strategy.sql
-- ============================================================
-- PURPOSE: Long-term text archival untuk kontrol ukuran DB.
--          Jalan otomatis via pg_cron + maintenance worker.
--
-- KONSEP:
--   Setelah artikel status='processed' DAN highlight sudah di-curate:
--     - raw_texts.text (full article body, up to 20KB) → NULL
--     - entity_contexts.context_text (context span, ~364 chars) → NULL
--   Karena:
--     - Dashboard FE HANYA baca entity_highlights (title+url+scores cached)
--     - raw_texts.text di-block RLS untuk anon (UU PDP)
--     - Sentiment sudah tersimpan di sentiment_scores
--
--   Yang TETAP disimpan:
--     - title, source_url, published_at (metadata ringan)
--     - text_hash (untuk dedup)
--     - sentiment_scores (analisis)
--     - entity_highlights (cache dashboard)
--     - entity_mentions offsets (audit trail)
--
--   Re-process path (kalau model upgrade):
--     1. enricher_worker re-fetch source_url → dapat text baru
--     2. text_hash beda (kalau source berubah) → artikel baru
--     3. text_hash sama → skip (sudah diproses)
--
-- SCHEDULE:
--   - pg_cron harian: archive text processed >30 hari
--   - Maintenance worker mingguan: archive + VACUUM
-- ============================================================

-- ============================================================
-- RPC: archive_old_texts(p_days INT DEFAULT 30)
-- NULL-kan text & context_text untuk artikel processed > p_days hari.
-- Aman dipanggil tiap hari (idempotent, batch UPDATE).
-- ============================================================
CREATE OR REPLACE FUNCTION archive_old_texts(p_days INT DEFAULT 30)
RETURNS JSONB
LANGUAGE plpgsql SECURITY DEFINER AS $$
DECLARE
    v_text_count INT := 0;
    v_ctx_count INT := 0;
    v_metadata_bytes_before BIGINT;
    v_metadata_bytes_after BIGINT;
BEGIN
    -- 1. Archive raw_texts.text
    SELECT COUNT(*) INTO v_text_count
    FROM raw_texts
    WHERE status = 'processed'
      AND ingested_at < NOW() - (p_days || ' days')::INTERVAL
      AND text IS NOT NULL
      AND text != '';

    IF v_text_count > 0 THEN
        UPDATE raw_texts
        SET text = NULL,
            metadata = metadata || jsonb_build_object(
                'text_archived_at', NOW(),
                'text_archived_reason', 'auto_archive_' || p_days || 'd'
            )
        WHERE status = 'processed'
          AND ingested_at < NOW() - (p_days || ' days')::INTERVAL
          AND text IS NOT NULL
          AND text != '';
    END IF;

    -- 2. Archive entity_contexts.context_text
    SELECT COUNT(*) INTO v_ctx_count
    FROM entity_contexts ec
    JOIN raw_texts rt ON rt.id = ec.raw_text_id
    WHERE rt.status = 'processed'
      AND rt.ingested_at < NOW() - (p_days || ' days')::INTERVAL
      AND ec.context_text IS NOT NULL;

    IF v_ctx_count > 0 THEN
        UPDATE entity_contexts
        SET context_text = NULL,
            metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object(
                'context_archived_at', NOW()
            )
        FROM raw_texts rt
        WHERE entity_contexts.raw_text_id = rt.id
          AND rt.status = 'processed'
          AND rt.ingested_at < NOW() - (p_days || ' days')::INTERVAL
          AND entity_contexts.context_text IS NOT NULL;
    END IF;

    RETURN jsonb_build_object(
        'archived_text_rows', v_text_count,
        'archived_context_rows', v_ctx_count,
        'threshold_days', p_days,
        'finished_at', NOW()
    );
END;
$$;

GRANT EXECUTE ON FUNCTION archive_old_texts(INT) TO service_role;


-- ============================================================
-- RPC: get_storage_breakdown()
-- Breakdown ukuran per tabel + estimasi archive potential.
-- Untuk monitoring & decision making.
-- ============================================================
CREATE OR REPLACE FUNCTION get_storage_breakdown()
RETURNS JSONB
LANGUAGE sql STABLE SECURITY DEFINER AS $$
    SELECT jsonb_build_object(
        'db_size_bytes', pg_database_size(current_database()),
        'db_size_pretty', pg_size_pretty(pg_database_size(current_database())),
        'table_sizes', (
            SELECT jsonb_agg(jsonb_build_object(
                'table', relname,
                'total_size_bytes', pg_total_relation_size(c.oid),
                'total_size_pretty', pg_size_pretty(pg_total_relation_size(c.oid)),
                'data_size_bytes', pg_relation_size(c.oid),
                'index_size_bytes', pg_total_relation_size(c.oid) - pg_relation_size(c.oid),
                'approx_rows', c.reltuples::bigint
            ) ORDER BY pg_total_relation_size(c.oid) DESC)
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public' AND c.relkind IN ('r','p','m')
        ),
        'archive_potential', jsonb_build_object(
            -- Rows yang BISA di-archive (processed >30d, text masih ada)
            'archivable_raw_texts', (
                SELECT COUNT(*) FROM raw_texts
                WHERE status='processed'
                  AND ingested_at < NOW() - INTERVAL '30 days'
                  AND text IS NOT NULL AND text != ''
            ),
            'archivable_contexts', (
                SELECT COUNT(*) FROM entity_contexts ec
                JOIN raw_texts rt ON rt.id = ec.raw_text_id
                WHERE rt.status='processed'
                  AND rt.ingested_at < NOW() - INTERVAL '30 days'
                  AND ec.context_text IS NOT NULL
            )
        ),
        'captured_at', NOW()
    );
$$;

GRANT EXECUTE ON FUNCTION get_storage_breakdown() TO service_role;


-- ============================================================
-- SCHEDULE: Auto-archive tiap hari jam 03:00 UTC (10:00 WIB)
-- ============================================================
-- Idempotent — kalau tidak ada row baru yang processed, no-op.
DO $$
DECLARE r RECORD;
BEGIN
    -- Unschedule versi lama kalau ada
    FOR r IN SELECT jobid, jobname FROM cron.job WHERE jobname = 'archive_old_texts' LOOP
        PERFORM cron.unschedule(r.jobid);
    END LOOP;

    -- Schedule daily auto-archive
    PERFORM cron.schedule(
        'archive_old_texts',
        '0 3 * * *',  -- tiap hari 03:00 UTC
        'SELECT archive_old_texts(30);'
    );

    RAISE NOTICE '✅ Scheduled: archive_old_texts(30) tiap hari 03:00 UTC';
END $$;


-- ============================================================
-- VERIFIKASI: cek setelah install
-- ============================================================
SELECT 'Storage breakdown (current):' AS info;
SELECT * FROM get_storage_breakdown();
