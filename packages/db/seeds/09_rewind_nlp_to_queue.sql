-- ============================================================
-- 09_rewind_nlp_to_queue.sql
-- Rewind semua artikel yang sudah masuk NLP kembali ke antrian
-- ============================================================
-- Efek: Semua artikel yang status='processed' akan kembali ke 'queued'
--       nlp_ready_at tetap (sudah enqueue sebelumnya)
--       Sentiment scores TIDAK dihapus (untuk audit)
--       
-- Setelah run: python main.py run-nlp --all
-- ============================================================

-- STEP 1: Reset artikel yang sudah 'processed' kembali ke 'queued'
UPDATE raw_texts
SET 
    status = 'queued',
    processed_at = NULL,
    pipeline_version = NULL
WHERE status = 'processed';

-- STEP 2: Bersihkan sentiment_scores lama (opsional)
-- Hapus jika ingin fresh start, comment jika ingin keep audit trail
TRUNCATE TABLE sentiment_scores;

-- STEP 3: Re-enqueue semua artikel 'queued' ke PGMQ
-- (Run via Python: python main.py run-worker readiness --limit 5000)
-- Atau manual via RPC:
DO $$
DECLARE
    r RECORD;
    count INTEGER := 0;
BEGIN
    FOR r IN
        SELECT id FROM raw_texts
        WHERE status = 'queued'
        ORDER BY ingested_at DESC
    LOOP
        BEGIN
            PERFORM enqueue_nlp_message(r.id);
            count := count + 1;
        EXCEPTION WHEN OTHERS THEN
            -- Skip jika sudah ada di queue
            NULL;
        END IF;
    END LOOP;
    RAISE NOTICE 'Re-enqueued % articles to PGMQ', count;
END $$;

-- STEP 4: Verifikasi
SELECT 
    status,
    COUNT(*) as total
FROM raw_texts
WHERE status IN ('queued', 'processed', 'failed')
GROUP BY status;

-- Cek PGMQ queue
SELECT COUNT(*) as queue_count FROM pgmq.q_nlp_processing_queue;

-- ============================================================
-- SETELAH RUN SQL INI:
--   1. git pull origin main (dapatkan code v16 terbaru)
--   2. python main.py run-nlp --all
-- ============================================================
