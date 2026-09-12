-- ============================================================
-- 19_emergency_size_reduction.sql
-- ============================================================
-- PURPOSE: EMERGENCY cleanup saat Supabase project sudah PAUSED /
--          "exceeded usage limits" karena DB > 500MB.
--
-- KENAPA INI BEDA DARI seed 17?
--   seed 17 = maintenance berkala (dedup + orphan, aman, slow)
--   seed 19 = EMERGENCY (aggressive, priority: reclaim space CEPAT)
--
-- STRATEGI PRIORITAS (urutan = impact terbesar dulu):
--   1. NULL-kan raw_texts.text untuk artikel PROCESSED > 30 hari
--      → Ini SATU langkah yang reclaim PALING BESAR (50-70% space)
--      → Dashboard HANYA butuh entity_highlights (title+url+scores cached),
--        jadi text body aman di-NULL setelah diproses
--   2. Drop old partitions aggressive (keep 2 bulan, bukan 6)
--   3. Dedup + orphan cleanup (dari seed 17, dipanggil via RPC)
--   4. VACUUM FULL (reclaim physical space — butuh setelah DELETE)
--
-- BAHAYA & MITIGASI:
--   ⚠️  NULL-kan text = TIDAK BISA re-process artikel itu tanpa re-fetch URL.
--       Mitigasi: kita simpan source_url, jadi kalau perlu re-process,
--       enricher_worker bisa re-fetch dari URL asli.
--   ⚠️  VACUUM FULL lock tabel — jalankan saat traffic rendah.
--
-- CARA PAKAI:
--   1. Buka Supabase Dashboard → SQL Editor
--   2. Paste seluruh file ini → Run
--   3. Kalau project sudah PAUSED dan SQL Editor tidak bisa akses:
--      a. Cek apakah project di-grace period (biasanya 7 hari)
--      b. Kalau ya, Supabase biasanya masih allow SQL Editor untuk cleanup
--      c. Kalau tidak bisa sama sekali → upgrade ke Pro sementara
--         ($25/bulan), run cleanup, lalu pause/downgrade
-- ============================================================

-- ============================================================
-- SECTION 0: SNAPSHOT — ukuran sebelum emergency cleanup
-- ============================================================
-- Pastikan db_cleanup_log table ada (dibuat di seed 17)
CREATE TABLE IF NOT EXISTS db_cleanup_log (
    id BIGSERIAL PRIMARY KEY,
    run_at TIMESTAMPTZ DEFAULT NOW(),
    metric TEXT NOT NULL,
    value NUMERIC,
    notes TEXT
);

DO $$
DECLARE
    v_size_before TEXT;
    v_bytes_before NUMERIC;
BEGIN
    v_bytes_before := pg_database_size(current_database());
    v_size_before := pg_size_pretty(v_bytes_before);
    RAISE NOTICE '🚨 EMERGENCY SIZE REDUCTION';
    RAISE NOTICE '   DB size BEFORE: %', v_size_before;
    INSERT INTO db_cleanup_log (metric, value, notes)
    VALUES ('emergency_db_size_bytes_before', v_bytes_before, v_size_before);
END $$;

-- Cek top 10 tabel terbesar (diagnostik penting)
SELECT
    relname AS table_name,
    pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size,
    pg_size_pretty(pg_relation_size(c.oid)) AS data_size,
    n_live_tup AS approx_rows
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relkind IN ('r','p','m')
ORDER BY pg_total_relation_size(c.oid) DESC
LIMIT 10;


-- ============================================================
-- SECTION 1: NULL-kan raw_texts.text untuk artikel PROCESSED > 30 hari
-- ============================================================
-- INI LANGKAH PALING EFEKTIF. raw_texts.text bisa sampai 20.000 char
-- per artikel (~5-20KB). Ribuan artikel processed = ratusan MB.
--
-- Setelah artikel status='processed':
--   - sentiment_scores sudah ada (analisis selesai)
--   - entity_highlights sudah di-curate (cache untuk dashboard)
--   - entity_contexts sudah diekstrak (context span)
--   - raw_texts.text TIDAK DIPAKAI LAGI oleh dashboard (RLS block anon)
--
-- Jadi NULL-kan text body = safe untuk production FE.
-- source_url tetap disimpan → bisa re-fetch kalau perlu re-process.

DO $$
DECLARE
    v_null_count INT := 0;
BEGIN
    -- Hitung dulu berapa row yang akan di-NULL
    SELECT COUNT(*) INTO v_null_count
    FROM raw_texts
    WHERE status = 'processed'
      AND ingested_at < NOW() - INTERVAL '30 days'
      AND text IS NOT NULL
      AND text != '';

    RAISE NOTICE '';
    RAISE NOTICE '=== SECTION 1: TEXT ARCHIVAL (NULL) ===';
    RAISE NOTICE '  Articles processed >30d dengan text: %', v_null_count;
    RAISE NOTICE '  Estimasi space yang di-reclaim: ~% MB',
        (v_null_count * 5 / 1000);  -- asumsi avg 5KB/article

    -- Lakukan NULL (batch untuk hindari lock terlalu lama)
    UPDATE raw_texts
    SET text = NULL,
        metadata = metadata || jsonb_build_object(
            'text_archived_at', NOW(),
            'text_archived_reason', 'emergency_size_reduction'
        )
    WHERE status = 'processed'
      AND ingested_at < NOW() - INTERVAL '30 days'
      AND text IS NOT NULL
      AND text != '';

    RAISE NOTICE '  ✓ Text di-NULL untuk % articles', v_null_count;
    INSERT INTO db_cleanup_log (metric, value, notes)
    VALUES ('emergency_text_nulled', v_null_count, 'processed >30d');
END $$;


-- ============================================================
-- SECTION 2: NULL-kan entity_contexts.context_text untuk artikel lama
-- ============================================================
-- context_text avg 364 chars, tapi ribuan rows = MB.
-- Setelah sentiment scored, context hanya dipakai untuk audit/re-debug.
DO $$
DECLARE
    v_ctx_count INT := 0;
BEGIN
    SELECT COUNT(*) INTO v_ctx_count
    FROM entity_contexts ec
    JOIN raw_texts rt ON rt.id = ec.raw_text_id
    WHERE rt.status = 'processed'
      AND rt.ingested_at < NOW() - INTERVAL '30 days'
      AND ec.context_text IS NOT NULL;

    RAISE NOTICE '';
    RAISE NOTICE '=== SECTION 2: CONTEXT TEXT ARCHIVAL ===';
    RAISE NOTICE '  Contexts to NULL: %', v_ctx_count;

    UPDATE entity_contexts
    SET context_text = NULL,
        metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object(
            'context_archived_at', NOW()
        )
    FROM raw_texts rt
    WHERE entity_contexts.raw_text_id = rt.id
      AND rt.status = 'processed'
      AND rt.ingested_at < NOW() - INTERVAL '30 days'
      AND entity_contexts.context_text IS NOT NULL;

    RAISE NOTICE '  ✓ Context text di-NULL untuk % rows', v_ctx_count;
    INSERT INTO db_cleanup_log (metric, value, notes)
    VALUES ('emergency_context_nulled', v_ctx_count, 'processed >30d');
END $$;


-- ============================================================
-- SECTION 3: DROP OLD PARTITIONS AGGRESSIVE (keep 2 bulan)
-- ============================================================
-- Default schema: drop_old_partitions(6) = keep 6 bulan.
-- Emergency: keep hanya 2 bulan. Artinya artikel >2 bulan DIHAPUS TOTAL.
--
-- ⚠️  INI MENGHAPUS DATA. Pastikan:
--   - sentiment_scores untuk artikel tsb SUDAH ada (kalau belum, hilang)
--   - entity_highlights sudah di-curate (ini survive karena non-partitioned)
--   - Kalau mau keep data sentiment, skip section ini dan andalkan section 1-2
DO $$
DECLARE v_dropped TEXT;
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '=== SECTION 3: DROP OLD PARTITIONS (keep 2 months) ===';
    -- Panggil drop_old_partitions dengan 2 bulan
    PERFORM drop_old_partitions(2);
    RAISE NOTICE '  ✓ Partitions older than 2 months dropped';
    INSERT INTO db_cleanup_log (metric, value, notes)
    VALUES ('emergency_partitions_dropped', 1, 'keep_months=2');
END $$;


-- ============================================================
-- SECTION 4: RUN WEEKLY CLEANUP RPC (dedup + orphan + hashes)
-- ============================================================
-- Panggil RPC dari seed 18 (pastikan sudah di-install).
-- Kalau belum install, jalankan seed 18 dulu, atau skip section ini.
DO $$
DECLARE v_result JSONB;
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '=== SECTION 4: DEDUP + ORPHAN CLEANUP (RPC) ===';
    BEGIN
        v_result := run_weekly_cleanup();
        RAISE NOTICE '  Duplicates deleted: %', v_result->>'dups_deleted';
        RAISE NOTICE '  Orphan scores deleted: %', v_result->>'orphan_scores';
        RAISE NOTICE '  Hashes deleted: %', v_result->>'hashes_deleted';
        INSERT INTO db_cleanup_log (metric, value, notes)
        VALUES ('emergency_rpc_cleanup', 1, v_result::text);
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE '  ⚠️  RPC run_weekly_cleanup() tidak tersedia: %', SQLERRM;
        RAISE NOTICE '      Jalankan seed 18_cleanup_rpc_functions.sql dulu, atau';
        RAISE NOTICE '      jalankan manual dedup dari seed 17 section 1-3.';
    END;
END $$;


-- ============================================================
-- SECTION 5: VACUUM ANALYZE (reclaim space dari UPDATE/DELETE)
-- ============================================================
-- VACUUM ANALYZE tidak bisa dijalankan dalam transaction block (DO $$).
-- Tapi VACUUM (tanpa FULL) bisa dijalankan via RPC atau pg_cron.
-- Untuk emergency, jalankan MANUAL setelah script ini:
--
--   VACUUM (ANALYZE) raw_texts;
--   VACUUM (ANALYZE) sentiment_scores;
--   VACUUM (ANALYZE) entity_mentions;
--   VACUUM (ANALYZE) entity_contexts;
--   VACUUM (ANALYZE) article_entity_map;
--   VACUUM (ANALYZE) raw_text_hashes;
--   VACUUM (ANALYZE) entity_highlights;
--   VACUUM (ANALYZE) pipeline_runs;
--
-- Kalau masih belum turun cukup, VACUUM FULL (lock tabel, tapi reclaim 100%):
--   VACUUM FULL raw_texts;
--   VACUUM FULL entity_contexts;
--   VACUUM FULL raw_text_hashes;
--
-- ⚠️  VACUUM FULL lock tabel — jalankan saat tidak ada worker running.
-- ============================================================

DO $$
DECLARE
    v_size_after TEXT;
    v_bytes_after NUMERIC;
    v_bytes_before NUMERIC;
    v_reclaimed NUMERIC;
BEGIN
    v_bytes_after := pg_database_size(current_database());
    v_size_after := pg_size_pretty(v_bytes_after);

    SELECT value INTO v_bytes_before
    FROM db_cleanup_log
    WHERE metric = 'emergency_db_size_bytes_before'
    ORDER BY run_at DESC LIMIT 1;

    v_reclaimed := v_bytes_before - v_bytes_after;

    RAISE NOTICE '';
    RAISE NOTICE '========================================';
    RAISE NOTICE '✅ EMERGENCY CLEANUP SELESAI (pre-VACUUM)';
    RAISE NOTICE '========================================';
    RAISE NOTICE '  DB size BEFORE: %', pg_size_pretty(v_bytes_before);
    RAISE NOTICE '  DB size AFTER:  %', v_size_after;
    RAISE NOTICE '  Space reclaimed: % (pre-VACUUM)',
        pg_size_pretty(GREATEST(v_reclaimed, 0)::BIGINT);
    RAISE NOTICE '';
    RAISE NOTICE '  ⚠️  Space fisik baru di-reclaim setelah VACUUM.';
    RAISE NOTICE '      Jalankan VACUUM (ANALYZE) manual (lihat Section 5).';
    RAISE NOTICE '      Kalau masih >500MB, jalankan VACUUM FULL.';
    RAISE NOTICE '';

    INSERT INTO db_cleanup_log (metric, value, notes) VALUES
        ('emergency_db_size_bytes_after', v_bytes_after, v_size_after),
        ('emergency_space_reclaimed_pre_vacuum',
         GREATEST(v_reclaimed, 0),
         pg_size_pretty(GREATEST(v_reclaimed, 0)::BIGINT));
END $$;

-- Final: top tables after cleanup
SELECT
    relname AS table_name,
    pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size,
    n_live_tup AS approx_rows
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND c.relkind IN ('r','p','m')
ORDER BY pg_total_relation_size(c.oid) DESC
LIMIT 10;
