-- ============================================================
-- 13_run_context_for_5000.sql
-- Reset articles yang entity_resolved tapi context belum di-extract
-- Target: capai 5,000+ contexts → 5,000+ dataset rows
-- ============================================================

-- STEP 1: Reset articles yang entity_resolved TAPI context_extracted_at NULL
-- (Entity berhasil, tapi context worker belum jalan / gagal)
UPDATE raw_texts
SET 
    context_extracted_at = NULL,
    nlp_ready_at = NULL
WHERE entity_resolved_at IS NOT NULL
  AND context_extracted_at IS NULL
  AND status = 'validated';

-- STEP 2: Reset articles yang context_extracted TAPI nlp_ready_at NULL
-- (Context berhasil, tapi nlp_readiness belum jalan / gagal)
UPDATE raw_texts
SET nlp_ready_at = NULL
WHERE context_extracted_at IS NOT NULL
  AND nlp_ready_at IS NULL
  AND status = 'validated';

-- STEP 3: Reset articles yang sudah 'processed' TAPI context kosong
-- (Processed tanpa context → hanya fallback sentiment, tidak berguna untuk dataset)
UPDATE raw_texts
SET 
    status = 'validated',
    context_extracted_at = NULL,
    nlp_ready_at = NULL,
    processed_at = NULL
WHERE status = 'processed'
  AND id NOT IN (SELECT DISTINCT raw_text_id FROM entity_contexts);

-- STEP 4: Reset articles yang 'queued' tapi belum di-enqueue ke PGMQ
-- (Stuck di queued status tapi tidak ada di queue)
UPDATE raw_texts
SET 
    status = 'validated',
    nlp_ready_at = NULL
WHERE status = 'queued'
  AND nlp_ready_at IS NULL;

-- STEP 5: VERIFIKASI — berapa articles siap untuk context_worker
SELECT '=== ARTICLES SIAP UNTUK CONTEXT_WORKER ===' as info;

SELECT 
    CASE 
        WHEN entity_resolved_at IS NOT NULL AND context_extracted_at IS NULL THEN 'Ready for Context'
        WHEN context_extracted_at IS NOT NULL AND nlp_ready_at IS NULL THEN 'Ready for Readiness'
        WHEN nlp_ready_at IS NOT NULL AND status = 'queued' THEN 'Ready for NLP'
        WHEN status = 'processed' THEN 'Already Processed'
        ELSE 'Other'
    END as pipeline_stage,
    COUNT(*) as total
FROM raw_texts
WHERE status = 'validated'
  AND entity_resolved_at IS NOT NULL
GROUP BY 1
ORDER BY 1;

-- ============================================================
-- SETELAH RUN SQL INI:
--   1. python main.py run-worker context --limit 100 --max-total 5000
--      → Akan extract context untuk ~5,000+ articles
--   2. python main.py run-worker readiness --limit 100
--      → Enqueue ke PGMQ
--   3. python main.py run-nlp --all
--      → Process sentiment
--   4. python devtools/dataset/export_finetune_dataset.py --limit 10000
--      → Export 5,000+ rows
-- ============================================================
