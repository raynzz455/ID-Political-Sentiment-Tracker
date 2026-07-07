-- ============================================================
-- MIGRATION 010: Support enricher_worker.py status + fix pgmq permission
-- ============================================================
-- 2 masalah yang difix:
--   1. CHECK constraint raw_texts.status tidak ada 'enriched'/'dead_link'
--      → enricher_worker.py gagal UPDATE status
--   2. Function bulk_enqueue_enriched() butuh SECURITY DEFINER + GRANT
--      supaya bisa akses schema pgmq (permission denied 42501)
--
-- Idempotent. Safe to re-run.
-- ============================================================

-- ─── STEP 1: Cek apakah ada trigger yang menyentuh pgmq saat status berubah ───
-- Ini untuk diagnosis — bukan mengubah data
SELECT
    tgname AS trigger_name,
    tgrelid::regclass AS table_name,
    pg_get_triggerdef(oid) AS definition
FROM pg_trigger
WHERE tgrelid = 'raw_texts'::regclass
  AND NOT tgisinternal;


-- ─── STEP 2: Update CHECK constraint status ───────────────────
ALTER TABLE raw_texts DROP CONSTRAINT IF EXISTS raw_texts_status_check;

ALTER TABLE raw_texts
ADD CONSTRAINT raw_texts_status_check
CHECK (status = ANY (ARRAY[
    'pending'::text,
    'enriched'::text,
    'queued'::text,
    'processing'::text,
    'processed'::text,
    'dead_link'::text,
    'failed'::text,
    'skipped'::text
]));


-- ─── STEP 3: Drop function lama kalau ada (signature berubah) ─
DROP FUNCTION IF EXISTS bulk_enqueue_enriched() CASCADE;


-- ─── STEP 4: Recreate dengan SECURITY DEFINER + pgmq qualified ─
-- SECURITY DEFINER = function jalan sebagai OWNER (postgres), bukan caller.
-- Ini yang fix "permission denied for schema pgmq" — karena postgres
-- pasti punya akses pgmq, sedangkan service_role via PostgREST tidak.
CREATE OR REPLACE FUNCTION bulk_enqueue_enriched()
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pgmq
AS $$
DECLARE
    v_count INTEGER;
BEGIN
    WITH ids_to_enqueue AS (
        SELECT id FROM raw_texts
        WHERE status = 'enriched'
        LIMIT 500
        FOR UPDATE SKIP LOCKED
    ),
    enqueue_actions AS (
        SELECT pgmq.send(
            'nlp_processing_queue',
            json_build_object('raw_text_id', id)::jsonb
        )
        FROM ids_to_enqueue
    )
    UPDATE raw_texts
    SET status = 'queued'
    WHERE id IN (SELECT id FROM ids_to_enqueue);

    GET DIAGNOSTICS v_count = ROW_COUNT;
    RETURN v_count;
END;
$$;


-- ─── STEP 5: Grant agar service_role bisa panggil function ini ─
GRANT EXECUTE ON FUNCTION bulk_enqueue_enriched() TO service_role;


-- ─── STEP 6: GRANT USAGE pgmq schema ke service_role (defense in depth) ─
-- Ini handle kasus kalau ada trigger lain yang juga akses pgmq.
-- PostgREST service_role butuh USAGE di schema untuk panggil function pgmq.*
GRANT USAGE ON SCHEMA pgmq TO service_role, authenticated;


-- ─── STEP 7: Reschedule cron (drop dulu kalau sudah ada, hindari dup) ─
SELECT cron.unschedule('auto-enqueue-enriched-job') WHERE EXISTS (
    SELECT 1 FROM cron.job WHERE jobname = 'auto-enqueue-enriched-job'
);

SELECT cron.schedule(
    'auto-enqueue-enriched-job',
    '*/5 * * * *',
    $$SELECT bulk_enqueue_enriched();$$
);


-- ─── STEP 8: Verifikasi ───────────────────────────────────────
SELECT
    'constraint' AS check_type,
    conname AS name
FROM pg_constraint
WHERE conname = 'raw_texts_status_check'
UNION ALL
SELECT
    'function',
    proname
FROM pg_proc
WHERE proname = 'bulk_enqueue_enriched'
UNION ALL
SELECT
    'cron',
    jobname
FROM cron.job
WHERE jobname = 'auto-enqueue-enriched-job';
