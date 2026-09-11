-- ============================================================
-- 18_cleanup_rpc_functions.sql
-- ============================================================
-- RPC functions untuk cleanup terprogram (dipanggil dari
-- packages/db/maintenance_worker.py via GitHub Action weekly).
--
-- Kenapa RPC, bukan SQL script langsung?
--   - GitHub Action hanya punya SUPABASE_URL + SERVICE_ROLE_KEY
--     (REST API / PostgREST). Tidak bisa jalankan SQL ad-hoc.
--   - Dengan bungkus logic di RPC SECURITY DEFINER, worker cukup
--     panggil sb.rpc("run_weekly_cleanup").
--   - VACUUM tetap harus dijalankan via koneksi langsung (psycopg2)
--     atau Supabase SQL Editor — tidak bisa via RPC.
-- ============================================================

-- ============================================================
-- RPC 1: run_weekly_cleanup()
-- Heavy dedup + orphan cleanup. Aman dipanggil mingguan.
-- Returns JSONB summary: {dups_deleted, orphans_deleted, ...}
-- ============================================================
CREATE OR REPLACE FUNCTION run_weekly_cleanup()
RETURNS JSONB
LANGUAGE plpgsql SECURITY DEFINER AS $$
DECLARE
    v_result JSONB;
    v_dup_ids UUID[];
    v_dup_count INT := 0;
    v_orphan_mentions INT := 0;
    v_orphan_contexts INT := 0;
    v_orphan_map INT := 0;
    v_orphan_scores INT := 0;
    v_orphan_highlights INT := 0;
    v_hashes_deleted INT := 0;
    v_runs_deleted INT := 0;
    v_candidates_deleted INT := 0;
BEGIN
    -- 1. Identifikasi duplikat (CTE → array)
    --    Keep: status='processed' dulu, lalu ingested_at DESC (terbaru)
    WITH dup_ids AS (
        -- content_hash dupes
        SELECT id FROM (
            SELECT id, content_hash,
                   ROW_NUMBER() OVER (PARTITION BY content_hash
                       ORDER BY CASE WHEN status='processed' THEN 0 ELSE 1 END,
                                ingested_at DESC) AS rn
            FROM raw_texts WHERE content_hash IS NOT NULL AND content_hash != ''
        ) t WHERE rn > 1
        UNION
        -- text_hash dupes
        SELECT id FROM (
            SELECT id, text_hash,
                   ROW_NUMBER() OVER (PARTITION BY text_hash
                       ORDER BY CASE WHEN status='processed' THEN 0 ELSE 1 END,
                                ingested_at DESC) AS rn
            FROM raw_texts WHERE text_hash IS NOT NULL
        ) t WHERE rn > 1
        UNION
        -- title dupes (hanya yang belum processed)
        SELECT id FROM (
            SELECT id, lower(title) AS nt,
                   ROW_NUMBER() OVER (PARTITION BY lower(title)
                       ORDER BY CASE WHEN status='processed' THEN 0 ELSE 1 END,
                                ingested_at DESC) AS rn
            FROM raw_texts
            WHERE title IS NOT NULL AND title != '' AND status != 'processed'
        ) t WHERE rn > 1
        UNION
        -- failed snippets (no text, too many attempts)
        SELECT id FROM raw_texts
        WHERE content_type='SNIPPET' AND (text IS NULL OR text='')
          AND recovery_attempts >= 5 AND status IN ('failed','skipped')
    )
    SELECT array_agg(id) INTO v_dup_ids FROM dup_ids;

    IF v_dup_ids IS NOT NULL THEN
        v_dup_count := array_length(v_dup_ids, 1);

        -- 2. Hapus child rows DULU (FK NO ACTION, no cascade)
        DELETE FROM entity_highlights WHERE raw_text_id = ANY(v_dup_ids);
        DELETE FROM sentiment_scores WHERE raw_text_id = ANY(v_dup_ids);
        DELETE FROM article_entity_map WHERE raw_text_id = ANY(v_dup_ids);
        DELETE FROM entity_contexts WHERE raw_text_id = ANY(v_dup_ids);
        DELETE FROM entity_mentions WHERE raw_text_id = ANY(v_dup_ids);

        -- 3. Hapus parent
        DELETE FROM raw_texts WHERE id = ANY(v_dup_ids);
    END IF;

    -- 4. Orphan cleanup (child rows untuk raw_texts yang sudah di-drop)
    DELETE FROM entity_mentions
    WHERE raw_text_id IS NOT NULL
      AND NOT EXISTS (SELECT 1 FROM raw_texts rt WHERE rt.id = entity_mentions.raw_text_id);
    GET DIAGNOSTICS v_orphan_mentions = ROW_COUNT;

    DELETE FROM entity_contexts
    WHERE raw_text_id IS NOT NULL
      AND NOT EXISTS (SELECT 1 FROM raw_texts rt WHERE rt.id = entity_contexts.raw_text_id);
    GET DIAGNOSTICS v_orphan_contexts = ROW_COUNT;

    DELETE FROM article_entity_map
    WHERE raw_text_id IS NOT NULL
      AND NOT EXISTS (SELECT 1 FROM raw_texts rt WHERE rt.id = article_entity_map.raw_text_id);
    GET DIAGNOSTICS v_orphan_map = ROW_COUNT;

    DELETE FROM sentiment_scores
    WHERE NOT EXISTS (SELECT 1 FROM raw_texts rt WHERE rt.id = sentiment_scores.raw_text_id);
    GET DIAGNOSTICS v_orphan_scores = ROW_COUNT;

    DELETE FROM entity_highlights
    WHERE raw_text_id IS NOT NULL
      AND NOT EXISTS (SELECT 1 FROM raw_texts rt WHERE rt.id = entity_highlights.raw_text_id);
    GET DIAGNOSTICS v_orphan_highlights = ROW_COUNT;

    -- 5. raw_text_hashes cleanup
    DELETE FROM raw_text_hashes
    WHERE text_hash NOT IN (SELECT DISTINCT text_hash FROM raw_texts WHERE text_hash IS NOT NULL);
    GET DIAGNOSTICS v_hashes_deleted = ROW_COUNT;

    -- 6. pipeline_runs log (30-day retention)
    DELETE FROM pipeline_runs WHERE started_at < NOW() - INTERVAL '30 days';
    GET DIAGNOSTICS v_runs_deleted = ROW_COUNT;

    -- 7. stale entity_candidates
    DELETE FROM entity_candidates
    WHERE status='pending' AND first_detected < NOW() - INTERVAL '14 days';
    GET DIAGNOSTICS v_candidates_deleted = ROW_COUNT;

    -- 8. Refresh materialized view
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_dashboard_summary;

    v_result := jsonb_build_object(
        'dups_deleted', v_dup_count,
        'orphan_mentions', v_orphan_mentions,
        'orphan_contexts', v_orphan_contexts,
        'orphan_map', v_orphan_map,
        'orphan_scores', v_orphan_scores,
        'orphan_highlights', v_orphan_highlights,
        'hashes_deleted', v_hashes_deleted,
        'runs_deleted', v_runs_deleted,
        'candidates_deleted', v_candidates_deleted,
        'finished_at', NOW()
    );

    -- Log ke db_cleanup_log
    INSERT INTO db_cleanup_log (metric, value, notes)
    SELECT k, (v::text)::numeric, 'run_weekly_cleanup'
    FROM jsonb_each_text(v_result) AS t(k, v)
    WHERE k != 'finished_at';

    RETURN v_result;
END;
$$;


-- ============================================================
-- RPC 2: get_db_size_report()
-- Returns JSONB: ukuran DB + count per tabel (untuk monitoring)
-- ============================================================
CREATE OR REPLACE FUNCTION get_db_size_report()
RETURNS JSONB
LANGUAGE sql STABLE SECURITY DEFINER AS $$
    SELECT jsonb_build_object(
        'db_size_bytes', pg_database_size(current_database()),
        'db_size_pretty', pg_size_pretty(pg_database_size(current_database())),
        'tables', jsonb_build_object(
            'raw_texts', (SELECT COUNT(*) FROM raw_texts),
            'sentiment_scores', (SELECT COUNT(*) FROM sentiment_scores),
            'entity_mentions', (SELECT COUNT(*) FROM entity_mentions),
            'entity_contexts', (SELECT COUNT(*) FROM entity_contexts),
            'article_entity_map', (SELECT COUNT(*) FROM article_entity_map),
            'raw_text_hashes', (SELECT COUNT(*) FROM raw_text_hashes),
            'entity_highlights', (SELECT COUNT(*) FROM entity_highlights),
            'pipeline_runs', (SELECT COUNT(*) FROM pipeline_runs),
            'entity_candidates', (SELECT COUNT(*) FROM entity_candidates)
        ),
        'top_tables_by_size', (
            SELECT jsonb_agg(jsonb_build_object(
                'table', relname,
                'size_bytes', pg_total_relation_size(c.oid),
                'size_pretty', pg_size_pretty(pg_total_relation_size(c.oid))
            ) ORDER BY pg_total_relation_size(c.oid) DESC LIMIT 10)
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public' AND c.relkind IN ('r','p','m')
        ),
        'captured_at', NOW()
    );
$$;


-- ============================================================
-- RPC 3: drop_old_partitions_aggressive(p_months INT DEFAULT 4)
-- Wrapper RPC agar bisa dipanggil dari worker (drop_old_partitions
-- adalah VOID function, kita bungkus biar return status)
-- ============================================================
CREATE OR REPLACE FUNCTION drop_old_partitions_aggressive(p_months INT DEFAULT 4)
RETURNS JSONB
LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
    PERFORM drop_old_partitions(p_months);
    RETURN jsonb_build_object(
        'action', 'drop_old_partitions',
        'keep_months', p_months,
        'finished_at', NOW()
    );
END;
$$;


-- Grant akses untuk service_role (sudah punya via SECURITY DEFINER,
-- tapi eksplisit biar jelas)
GRANT EXECUTE ON FUNCTION run_weekly_cleanup() TO service_role;
GRANT EXECUTE ON FUNCTION get_db_size_report() TO service_role;
GRANT EXECUTE ON FUNCTION drop_old_partitions_aggressive(INT) TO service_role;
