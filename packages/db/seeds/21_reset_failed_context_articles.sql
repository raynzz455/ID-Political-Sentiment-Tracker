-- ============================================================
-- 21_reset_failed_context_articles.sql
-- ============================================================
-- PURPOSE: Reset articles yang sudah ditandai context_extracted_at
--          TAPI tidak punya entity_contexts (false-processed).
--
-- MASALAH:
--   Context worker v18 (sebelum fix) menandai SEMUA artikel sebagai
--   context_extracted_at = NOW(), termasuk artikel dengan 0 entity_mentions.
--   Akibatnya: artikel tsb tidak akan pernah diproses ulang, meskipun
--   entity resolution sudah diperbaiki.
--
-- SOLUSI:
--   1. Cari artikel yang context_extracted_at IS NOT NULL tapi tidak
--      punya entity_contexts
--   2. Reset context_extracted_at = NULL untuk artikel tsb
--   3. Juga reset entity_resolved_at = NULL untuk artikel yang punya
--      resolver_version = 'failed_no_entity' (entity resolution gagal)
--
-- SETELAH RUN SCRIPT INI:
--   1. Re-run entity resolution: python main.py run-worker entity --limit 200
--   2. Re-run context worker: python main.py run-worker context --limit 200
-- ============================================================

-- ============================================================
-- SECTION 1: DIAGNOSTIK — lihat situasi sebelum reset
-- ============================================================
SELECT '=== DIAGNOSTIK SEBELUM RESET ===' AS info;

SELECT
    CASE
        WHEN context_extracted_at IS NOT NULL AND EXISTS (
            SELECT 1 FROM entity_contexts ec WHERE ec.raw_text_id = raw_texts.id
        ) THEN 'context_extracted + has_contexts (OK)'
        WHEN context_extracted_at IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM entity_contexts ec WHERE ec.raw_text_id = raw_texts.id
        ) THEN 'context_extracted + NO contexts (BUG — needs reset)'
        WHEN context_extracted_at IS NULL AND entity_resolved_at IS NOT NULL THEN 'resolved but not context-extracted (pending)'
        WHEN entity_resolved_at IS NULL THEN 'not resolved yet'
        ELSE 'other'
    END AS category,
    COUNT(*) AS total
FROM raw_texts
WHERE status = 'validated'
  AND content_type != 'SNIPPET'
  AND text IS NOT NULL AND text != ''
GROUP BY 1
ORDER BY 1;


-- ============================================================
-- SECTION 2: RESET context_extracted_at untuk artikel tanpa contexts
-- ============================================================
-- Artikel ini ditandai "sudah di-extract context" tapi sebenarnya
-- tidak ada context yang dibuat (0 entity_mentions atau offset gagal).
-- Reset supaya context worker bisa pick up lagi.
DO $$
DECLARE
    v_reset_count INT := 0;
BEGIN
    UPDATE raw_texts
    SET context_extracted_at = NULL
    WHERE context_extracted_at IS NOT NULL
      AND NOT EXISTS (
          SELECT 1 FROM entity_contexts ec WHERE ec.raw_text_id = raw_texts.id
      )
      AND status = 'validated'
      AND content_type != 'SNIPPET'
      AND text IS NOT NULL AND text != '';
    
    GET DIAGNOSTICS v_reset_count = ROW_COUNT;
    RAISE NOTICE 'Reset context_extracted_at untuk % articles (had contexts=0)', v_reset_count;
END $$;


-- ============================================================
-- SECTION 3: RESET entity_resolved_at untuk artikel failed_no_entity
-- ============================================================
-- Artikel ini ditandai "sudah di-resolve" tapi resolver_version=
-- 'failed_no_entity' (entity resolution tidak menemukan entity sama sekali).
-- Reset supaya entity resolution worker bisa pick up lagi dengan
-- LIGHTWEIGHT_MODE=0 (Stanza aktif).
DO $$
DECLARE
    v_reset_count INT := 0;
BEGIN
    UPDATE raw_texts
    SET entity_resolved_at = NULL,
        resolver_version = NULL
    WHERE resolver_version = 'failed_no_entity'
      AND status = 'validated'
      AND content_type != 'SNIPPET'
      AND text IS NOT NULL AND text != '';
    
    GET DIAGNOSTICS v_reset_count = ROW_COUNT;
    RAISE NOTICE 'Reset entity_resolved_at untuk % articles (failed_no_entity)', v_reset_count;
END $$;


-- ============================================================
-- SECTION 4: RESET entity_resolved_at untuk artikel dengan 0 mentions
-- ============================================================
-- Artikel yang entity_resolved_at IS NOT NULL tapi tidak punya
-- entity_mentions sama sekali (entah lightweight mode atau bug).
-- Reset supaya bisa re-resolve dengan Stanza aktif.
DO $$
DECLARE
    v_reset_count INT := 0;
BEGIN
    UPDATE raw_texts
    SET entity_resolved_at = NULL,
        resolver_version = NULL
    WHERE entity_resolved_at IS NOT NULL
      AND NOT EXISTS (
          SELECT 1 FROM entity_mentions em WHERE em.raw_text_id = raw_texts.id
      )
      AND status = 'validated'
      AND content_type != 'SNIPPET'
      AND text IS NOT NULL AND text != '';
    
    GET DIAGNOSTICS v_reset_count = ROW_COUNT;
    RAISE NOTICE 'Reset entity_resolved_at untuk % articles (0 mentions)', v_reset_count;
END $$;


-- ============================================================
-- SECTION 5: DIAGNOSTIK — lihat situasi setelah reset
-- ============================================================
SELECT '=== DIAGNOSTIK SETELAH RESET ===' AS info;

SELECT
    CASE
        WHEN entity_resolved_at IS NULL THEN 'pending entity resolution (ready for re-run)'
        WHEN entity_resolved_at IS NOT NULL AND context_extracted_at IS NULL THEN 'resolved, pending context extraction'
        WHEN context_extracted_at IS NOT NULL THEN 'fully processed (has contexts)'
        ELSE 'other'
    END AS category,
    COUNT(*) AS total
FROM raw_texts
WHERE status = 'validated'
  AND content_type != 'SNIPPET'
  AND text IS NOT NULL AND text != ''
GROUP BY 1
ORDER BY 1;

-- ============================================================
-- SETELAH RUN SCRIPT INI:
--   1. python main.py run-worker entity --limit 200
--      (dengan LIGHTWEIGHT_MODE=0 — Stanza aktif, sudah di-set di workflow)
--   2. python main.py run-worker context --limit 200
--      (akan pick up artikel yang sudah di-reset)
--   3. Atau trigger workflow 2b lalu 2c via GitHub Actions UI
-- ============================================================
