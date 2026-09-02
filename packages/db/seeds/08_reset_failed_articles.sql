-- ============================================================
-- reset_failed_articles.sql
-- Reset artikel yang gagal agar bisa masuk layer lagi
-- ============================================================
-- Jalankan di Supabase SQL Editor
-- ============================================================

-- STEP 1: Reset artikel yang gagal di NLP READINESS (pgmq_enqueue_failed)
--         Artikel SUDAH punya context valid, cuma gagal enqueue ke PGMQ
--         → Cukup reset nlp_ready_at, status kembali validated
UPDATE raw_texts
SET 
    status = 'validated',
    nlp_ready_at = NULL,
    pipeline_version = NULL
WHERE status = 'failed'
  AND metadata->>'fail_reason' = 'pgmq_enqueue_failed';

-- STEP 2: Reset artikel yang gagal di NLP READINESS (no_valid_context)
--         Artikel punya text tapi context tidak valid/kosong
--         → Reset nlp_ready_at + context_extracted_at (biar context worker jalan lagi)
UPDATE raw_texts
SET 
    status = 'validated',
    nlp_ready_at = NULL,
    context_extracted_at = NULL
WHERE status = 'failed'
  AND metadata->>'fail_reason' = 'nlp_ready_no_valid_context';

-- STEP 3: Reset artikel yang skipped (duplicate_title_at_gate)
--         Artikel di-skip karena judul duplikat
--         → Reset nlp_ready_at, status kembali validated
UPDATE raw_texts
SET 
    status = 'validated',
    nlp_ready_at = NULL,
    duplicate_of = NULL
WHERE status = 'skipped'
  AND metadata->>'fail_reason' = 'duplicate_title_at_gate';

-- STEP 4: Reset artikel yang gagal di ENTITY RESOLUTION
--         resolver_version = 'failed_no_entity' (entity tidak ditemukan)
--         → Reset entity_resolved_at + context_extracted_at + nlp_ready_at
UPDATE raw_texts
SET 
    status = 'validated',
    entity_resolved_at = NULL,
    context_extracted_at = NULL,
    nlp_ready_at = NULL
WHERE resolver_version = 'failed_no_entity';

-- STEP 5: Reset artikel yang entity_resolved_at di-set TAPI context_extracted_at NULL
--         (Entity berhasil, tapi context worker belum jalan)
UPDATE raw_texts
SET 
    context_extracted_at = NULL,
    nlp_ready_at = NULL
WHERE entity_resolved_at IS NOT NULL
  AND context_extracted_at IS NULL
  AND status = 'validated';

-- ============================================================
-- VERIFIKASI: Cek hasil reset
-- ============================================================
SELECT 
    status,
    COUNT(*) as total,
    COUNT(*) FILTER (WHERE entity_resolved_at IS NULL) as entity_null,
    COUNT(*) FILTER (WHERE context_extracted_at IS NULL) as context_null,
    COUNT(*) FILTER (WHERE nlp_ready_at IS NULL) as nlp_ready_null
FROM raw_texts
WHERE status IN ('validated', 'failed', 'skipped')
GROUP BY status
ORDER BY status;

-- ============================================================
-- STATISTIK: Artikel yang siap untuk masing-masing layer
-- ============================================================
SELECT 'Ready for Entity Resolution' as layer,
       COUNT(*) as total
FROM raw_texts
WHERE status = 'validated'
  AND entity_resolved_at IS NULL
  AND text IS NOT NULL
  AND text != ''

UNION ALL

SELECT 'Ready for Context Worker' as layer,
       COUNT(*) as total
FROM raw_texts
WHERE status = 'validated'
  AND entity_resolved_at IS NOT NULL
  AND context_extracted_at IS NULL

UNION ALL

SELECT 'Ready for NLP Readiness' as layer,
       COUNT(*) as total
FROM raw_texts
WHERE status = 'validated'
  AND context_extracted_at IS NOT NULL
  AND nlp_ready_at IS NULL;
