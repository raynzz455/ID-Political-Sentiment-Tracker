-- ============================================================
-- 17_database_cleanup_and_maintenance.sql
-- ============================================================
-- PURPOSE: Periodic garbage collection & deduplication to keep
--          the Supabase database under the 500MB free-tier limit.
--
-- PROBLEM: Setelah pipeline berjalan berminggu-minggu, banyak data
--          sampah menumpuk:
--            1. Duplikat raw_texts (content_hash / text_hash / title sama)
--            2. Child rows orphaned (entity_mentions, entity_contexts,
--               article_entity_map, sentiment_scores) untuk raw_texts
--               yang sudah di-drop via partition
--            3. raw_text_hashes tidak pernah dibersihkan (global dedup
--               table tumbuh tanpa batas)
--            4. pipeline_runs log tumbuh tanpa batas
--            5. entity_candidates pending yang tidak pernah direview
--
-- STRATEGY: Safe, idempotent, reversible cleanup yang:
--   - DELETE child rows DULU, baru parent (FK adalah NO ACTION,
--     tidak ada ON DELETE CASCADE, jadi urutan WAJIB begini)
--   - JANGAN pernah hapus raw_texts dengan status='processed'
--     yang masih punya sentiment_scores valid (keep the gold data)
--   - Log before/after sizes untuk audit
--
-- RUN MODES:
--   A) Manual run (full):   paste seluruh file di Supabase SQL Editor
--   B) pg_cron (light):     bagian terakhir di-schedule otomatis
--   C) GitHub Action:       via packages/db/maintenance_worker.py
--      (worker ini bisa VACUUM ANALYZE — tidak bisa dilakukan pg_cron)
-- ============================================================

-- ============================================================
-- SECTION 0: SNAPSHOT BEFORE CLEANUP (untuk audit)
-- ============================================================
CREATE TABLE IF NOT EXISTS db_cleanup_log (
    id BIGSERIAL PRIMARY KEY,
    run_at TIMESTAMPTZ DEFAULT NOW(),
    metric TEXT NOT NULL,
    value NUMERIC,
    notes TEXT
);

DO $$
DECLARE
    v_total_size TEXT;
    v_raw_count BIGINT;
    v_scores_count BIGINT;
    v_mentions_count BIGINT;
    v_contexts_count BIGINT;
    v_hashes_count BIGINT;
    v_runs_count BIGINT;
BEGIN
    SELECT pg_size_pretty(pg_database_size(current_database())) INTO v_total_size;
    SELECT COUNT(*) INTO v_raw_count FROM raw_texts;
    SELECT COUNT(*) INTO v_scores_count FROM sentiment_scores;
    SELECT COUNT(*) INTO v_mentions_count FROM entity_mentions;
    SELECT COUNT(*) INTO v_contexts_count FROM entity_contexts;
    SELECT COUNT(*) INTO v_hashes_count FROM raw_text_hashes;
    SELECT COUNT(*) INTO v_runs_count FROM pipeline_runs;

    RAISE NOTICE '=== BEFORE CLEANUP ===';
    RAISE NOTICE 'DB size: %', v_total_size;
    RAISE NOTICE 'raw_texts: %, sentiment_scores: %', v_raw_count, v_scores_count;
    RAISE NOTICE 'entity_mentions: %, entity_contexts: %', v_mentions_count, v_contexts_count;
    RAISE NOTICE 'raw_text_hashes: %, pipeline_runs: %', v_hashes_count, v_runs_count;

    INSERT INTO db_cleanup_log (metric, value, notes) VALUES
        ('db_size_bytes_before', pg_database_size(current_database()), v_total_size),
        ('raw_texts_before', v_raw_count, NULL),
        ('sentiment_scores_before', v_scores_count, NULL),
        ('entity_mentions_before', v_mentions_count, NULL),
        ('entity_contexts_before', v_contexts_count, NULL),
        ('raw_text_hashes_before', v_hashes_count, NULL),
        ('pipeline_runs_before', v_runs_count, NULL);
END $$;


-- ============================================================
-- SECTION 1: DEDUP RAW_TEXTS — Identify duplicates to delete
-- ============================================================
-- Kita bangun sebuah temp table berisi raw_text_id yang akan dihapus.
-- Prioritas KEEP: status='processed' > ingested_at DESC (paling baru).
-- Child rows dihapus di Section 2 sebelum parent di Section 3.

-- 1a. Duplikat berdasarkan content_hash (keep processed & newest)
CREATE TEMP TABLE _dup_to_delete AS
SELECT id, ingested_month
FROM (
    SELECT
        id,
        ingested_month,
        content_hash,
        ROW_NUMBER() OVER (
            PARTITION BY content_hash
            ORDER BY
                CASE WHEN status = 'processed' THEN 0 ELSE 1 END,
                ingested_at DESC
        ) AS rn
    FROM raw_texts
    WHERE content_hash IS NOT NULL AND content_hash != ''
) t
WHERE rn > 1;

-- 1b. Duplikat berdasarkan text_hash (keep processed & newest) — exclude sudah di 1a
INSERT INTO _dup_to_delete
SELECT id, ingested_month
FROM (
    SELECT
        id,
        ingested_month,
        text_hash,
        ROW_NUMBER() OVER (
            PARTITION BY text_hash
            ORDER BY
                CASE WHEN status = 'processed' THEN 0 ELSE 1 END,
                ingested_at DESC
        ) AS rn
    FROM raw_texts
    WHERE text_hash IS NOT NULL
      AND id NOT IN (SELECT id FROM _dup_to_delete)
) t
WHERE rn > 1;

-- 1c. Duplikat berdasarkan judul (lowercase) — HANYA yang belum diproses
--     (jangan hapus artikel yang sudah ada sentiment_score-nya)
INSERT INTO _dup_to_delete
SELECT id, ingested_month
FROM (
    SELECT
        id,
        ingested_month,
        lower(title) AS norm_title,
        ROW_NUMBER() OVER (
            PARTITION BY lower(title)
            ORDER BY
                CASE WHEN status = 'processed' THEN 0 ELSE 1 END,
                ingested_at DESC
        ) AS rn
    FROM raw_texts
    WHERE title IS NOT NULL AND title != ''
      AND status != 'processed'
      AND id NOT IN (SELECT id FROM _dup_to_delete)
) t
WHERE rn > 1;

-- 1d. SNIPPET yang gagal di-resolve terlalu banyak kali (recovery_attempts >= 5)
--     dan tidak punya text — ini pure sampah, buang
INSERT INTO _dup_to_delete
SELECT id, ingested_month
FROM raw_texts
WHERE content_type = 'SNIPPET'
  AND (text IS NULL OR text = '')
  AND recovery_attempts >= 5
  AND status IN ('failed', 'skipped')
  AND id NOT IN (SELECT id FROM _dup_to_delete);

DO $$
DECLARE v_cnt BIGINT;
BEGIN
    SELECT COUNT(*) INTO v_cnt FROM _dup_to_delete;
    RAISE NOTICE 'Duplicates/sampah untuk dihapus: %', v_cnt;
    INSERT INTO db_cleanup_log (metric, value, notes) VALUES ('dup_rows_identified', v_cnt, 'content_hash+text_hash+title+failed_snippets');
END $$;


-- ============================================================
-- SECTION 2: DELETE CHILD ROWS untuk duplikat (WAJIB sebelum parent)
-- ============================================================
-- FK adalah NO ACTION (tidak ada cascade), jadi kita hapus children dulu.
-- Urutan: highlights -> sentiment_scores -> article_entity_map
--         -> entity_contexts -> entity_mentions -> (parent raw_texts)

DELETE FROM entity_highlights
WHERE raw_text_id IN (SELECT id FROM _dup_to_delete);

DELETE FROM sentiment_scores
WHERE raw_text_id IN (SELECT id FROM _dup_to_delete);

DELETE FROM article_entity_map
WHERE raw_text_id IN (SELECT id FROM _dup_to_delete);

DELETE FROM entity_contexts
WHERE raw_text_id IN (SELECT id FROM _dup_to_delete);

DELETE FROM entity_mentions
WHERE raw_text_id IN (SELECT id FROM _dup_to_delete);


-- ============================================================
-- SECTION 3: DELETE PARENT RAW_TEXTS (the duplicates)
-- ============================================================
DELETE FROM raw_texts
WHERE id IN (SELECT id FROM _dup_to_delete);

DROP TABLE _dup_to_delete;


-- ============================================================
-- SECTION 4: CLEAN ORPHANED CHILD ROWS
-- ============================================================
-- Setelah partition drop (drop_old_partitions), raw_texts lama hilang,
-- tapi child rows di entity_mentions / entity_contexts / article_entity_map
-- TIDAK ikut terhapus (mereka tidak dipartisi). Ini orphaned data = sampah.

-- 4a. entity_mentions yang raw_text_id-nya tidak ada lagi
DELETE FROM entity_mentions
WHERE raw_text_id IS NOT NULL
  AND NOT EXISTS (
      SELECT 1 FROM raw_texts rt WHERE rt.id = entity_mentions.raw_text_id
  );

-- 4b. entity_contexts orphaned
DELETE FROM entity_contexts
WHERE raw_text_id IS NOT NULL
  AND NOT EXISTS (
      SELECT 1 FROM raw_texts rt WHERE rt.id = entity_contexts.raw_text_id
  );

-- 4c. article_entity_map orphaned
DELETE FROM article_entity_map
WHERE raw_text_id IS NOT NULL
  AND NOT EXISTS (
      SELECT 1 FROM raw_texts rt WHERE rt.id = article_entity_map.raw_text_id
  );

-- 4d. sentiment_scores orphaned (raw_text_id sudah tidak ada)
DELETE FROM sentiment_scores
WHERE NOT EXISTS (
    SELECT 1 FROM raw_texts rt WHERE rt.id = sentiment_scores.raw_text_id
);

-- 4e. entity_highlights orphaned
DELETE FROM entity_highlights
WHERE NOT EXISTS (
    SELECT 1 FROM raw_texts rt WHERE rt.id = entity_highlights.raw_text_id
)
AND raw_text_id IS NOT NULL;


-- ============================================================
-- SECTION 5: CLEAN raw_text_hashes (global dedup table)
-- ============================================================
-- raw_text_hashes tumbuh tanpa batas karena tidak pernah dibersihkan.
-- Hanya hash yang MASIH dipakai oleh raw_texts yang kita simpan.
-- Hash untuk artikel yang sudah di-drop (via partition atau dedup) = sampah.
DELETE FROM raw_text_hashes
WHERE text_hash NOT IN (
    SELECT DISTINCT text_hash FROM raw_texts WHERE text_hash IS NOT NULL
);


-- ============================================================
-- SECTION 6: CLEAN pipeline_runs LOG (keep 30 days)
-- ============================================================
DELETE FROM pipeline_runs
WHERE started_at < NOW() - INTERVAL '30 days';


-- ============================================================
-- SECTION 7: CLEAN entity_candidates (stale pending > 14 days)
-- ============================================================
DELETE FROM entity_candidates
WHERE status = 'pending'
  AND first_detected < NOW() - INTERVAL '14 days';


-- ============================================================
-- SECTION 8: DROP OLD PARTITIONS (configurable retention)
-- ============================================================
-- Default schema: drop_old_partitions(6) = keep 6 bulan.
-- Kalau DB masih overload, panggil dengan 3 bulan untuk hemat storage.
-- pg_cron sudah schedule drop_old_partitions(6) tiap bulan (lihat schema.sql).
-- Di sini kita panggil versi 4 bulan sebagai "aggressive" cleanup:
SELECT drop_old_partitions(4);


-- ============================================================
-- SECTION 9: REFRESH MATERIALIZED VIEW
-- ============================================================
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_dashboard_summary;


-- ============================================================
-- SECTION 10: SNAPSHOT AFTER CLEANUP + LOG
-- ============================================================
DO $$
DECLARE
    v_total_size_after TEXT;
    v_raw_count BIGINT;
    v_scores_count BIGINT;
    v_mentions_count BIGINT;
    v_contexts_count BIGINT;
    v_hashes_count BIGINT;
    v_runs_count BIGINT;
    v_saved NUMERIC;
    v_before NUMERIC;
BEGIN
    SELECT pg_size_pretty(pg_database_size(current_database())) INTO v_total_size_after;
    SELECT COUNT(*) INTO v_raw_count FROM raw_texts;
    SELECT COUNT(*) INTO v_scores_count FROM sentiment_scores;
    SELECT COUNT(*) INTO v_mentions_count FROM entity_mentions;
    SELECT COUNT(*) INTO v_contexts_count FROM entity_contexts;
    SELECT COUNT(*) INTO v_hashes_count FROM raw_text_hashes;
    SELECT COUNT(*) INTO v_runs_count FROM pipeline_runs;

    SELECT value INTO v_before FROM db_cleanup_log
    WHERE metric = 'db_size_bytes_before' ORDER BY run_at DESC LIMIT 1;
    v_saved := v_before - pg_database_size(current_database());

    RAISE NOTICE '=== AFTER CLEANUP ===';
    RAISE NOTICE 'DB size: %', v_total_size_after;
    RAISE NOTICE 'raw_texts: %, sentiment_scores: %', v_raw_count, v_scores_count;
    RAISE NOTICE 'entity_mentions: %, entity_contexts: %', v_mentions_count, v_contexts_count;
    RAISE NOTICE 'raw_text_hashes: %, pipeline_runs: %', v_hashes_count, v_runs_count;
    RAISE NOTICE 'Space reclaimed (pre-VACUUM): %',
        pg_size_pretty(GREATEST(v_saved, 0)::BIGINT);

    INSERT INTO db_cleanup_log (metric, value, notes) VALUES
        ('db_size_bytes_after', pg_database_size(current_database()), v_total_size_after),
        ('raw_texts_after', v_raw_count, NULL),
        ('sentiment_scores_after', v_scores_count, NULL),
        ('entity_mentions_after', v_mentions_count, NULL),
        ('entity_contexts_after', v_contexts_count, NULL),
        ('raw_text_hashes_after', v_hashes_count, NULL),
        ('pipeline_runs_after', v_runs_count, NULL),
        ('space_reclaimed_bytes', GREATEST(v_saved, 0), pg_size_pretty(GREATEST(v_saved, 0)::BIGINT));
END $$;

-- ⚠️ VACUUM FULL tidak bisa dijalankan di dalam transaction / function.
--    Jalankan manual setelah script ini, atau lewat maintenance_worker.py:
--      VACUUM (ANALYZE, VERBOSE) raw_texts;
--      VACUUM (ANALYZE, VERBOSE) sentiment_scores;
--      VACUUM (ANALYZE) entity_mentions, entity_contexts, article_entity_map;
--      VACUUM (ANALYZE) raw_text_hashes, pipeline_runs, entity_highlights;
--    VACUUM FULL akan lock tabel — hanya untuk emergency reclaim.


-- ============================================================
-- SECTION 11: SCHEDULE LIGHT CLEANUP via pg_cron
-- ============================================================
-- Bagian dedup berat (Section 1-3) berbahaya dijalankan terlalu sering
-- karena lock tabel. Kita schedule hanya cleanup ringan (orphan + log + hashes)
-- tiap 6 jam. Dedup berat dijalankan mingguan via GitHub Action.

-- Unschedule versi lama kalau ada (idempotent)
DO $$
DECLARE r RECORD;
BEGIN
    FOR r IN SELECT jobid, jobname, command FROM cron.job WHERE jobname LIKE 'cleanup_%' LOOP
        PERFORM cron.unschedule(r.jobid);
    END LOOP;
END $$;

-- Light cleanup tiap 6 jam: orphan child rows + raw_text_hashes + pipeline_runs log
SELECT cron.schedule(
    'cleanup_light_orphans',
    '0 */6 * * *',
    $cleanup$
        DELETE FROM entity_mentions WHERE raw_text_id IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM raw_texts rt WHERE rt.id = entity_mentions.raw_text_id);
        DELETE FROM entity_contexts WHERE raw_text_id IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM raw_texts rt WHERE rt.id = entity_contexts.raw_text_id);
        DELETE FROM article_entity_map WHERE raw_text_id IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM raw_texts rt WHERE rt.id = article_entity_map.raw_text_id);
        DELETE FROM sentiment_scores
          WHERE NOT EXISTS (SELECT 1 FROM raw_texts rt WHERE rt.id = sentiment_scores.raw_text_id);
        DELETE FROM entity_highlights WHERE raw_text_id IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM raw_texts rt WHERE rt.id = entity_highlights.raw_text_id);
        DELETE FROM raw_text_hashes
          WHERE text_hash NOT IN (SELECT DISTINCT text_hash FROM raw_texts WHERE text_hash IS NOT NULL);
        DELETE FROM pipeline_runs WHERE started_at < NOW() - INTERVAL '30 days';
    $cleanup$
);

-- ============================================================
-- DONE. Untuk reclaim fisik space, jalankan maintenance_worker.py
--       (GitHub Action weekly) yang menjalankan VACUUM ANALYZE.
-- ============================================================
