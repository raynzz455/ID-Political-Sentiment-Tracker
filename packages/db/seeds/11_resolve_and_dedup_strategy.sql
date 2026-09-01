-- ============================================================
-- 11_resolve_and_dedup_strategy.sql
-- Strategi: Resolve 80K snippets → 20K+ fulltext + anti-duplikasi
-- ============================================================
-- DATA TERBARU DARI DB:
--   Processed: 2,430 articles (punya sentiment scores)
--   Scores: 3,892 total (positive=1,379, negative=709, neutral=1,804)
--   Snippet articles: ~80K+ (text kosong, butuh resolver)
--   Fulltext articles: ~10K+ (sudah punya text)
-- ============================================================

-- ============================================================
-- STEP 1: Reset SNIPPET yang FAILED → PENDING (siap di-resolve)
-- ============================================================

UPDATE raw_texts
SET 
    status = 'pending',
    recovery_status = 'pending',
    recovery_attempts = 0
WHERE content_type = 'SNIPPET'
  AND status = 'failed'
  AND source_url IS NOT NULL
  AND source_url != '';

-- ============================================================
-- STEP 2: Reset SNIPPET yang SKIPPED → PENDING
-- ============================================================

UPDATE raw_texts
SET 
    status = 'pending',
    recovery_status = 'pending',
    recovery_attempts = 0,
    duplicate_of = NULL
WHERE content_type = 'SNIPPET'
  AND status = 'skipped'
  AND source_url IS NOT NULL
  AND source_url != '';

-- ============================================================
-- STEP 3: JANGAN RESET articles yang sudah PROCESSED
-- (2,430 articles sudah punya sentiment scores — jangan buang)
-- ============================================================

-- ============================================================
-- STEP 4: DEDUP — hapus duplikat yang sudah ada
-- ============================================================

-- 4a: Hapus duplikat berdasarkan content_hash (keep newest)
DELETE FROM raw_texts
WHERE id NOT IN (
    SELECT DISTINCT ON (content_hash) id
    FROM raw_texts
    WHERE content_hash IS NOT NULL
    AND content_hash != ''
    ORDER BY content_hash, ingested_at DESC
)
AND content_hash IS NOT NULL
AND content_hash != '';

-- 4b: Hapus duplikat berdasarkan text_hash (keep newest)
DELETE FROM raw_texts
WHERE id NOT IN (
    SELECT DISTINCT ON (text_hash) id
    FROM raw_texts
    ORDER BY text_hash, ingested_at DESC
)
AND text_hash IS NOT NULL;

-- 4c: Hapus duplikat berdasarkan judul yang sama (keep newest, yang sudah processed)
DELETE FROM raw_texts
WHERE id NOT IN (
    SELECT DISTINCT ON (lower(title)) id
    FROM raw_texts
    WHERE title IS NOT NULL AND title != ''
    ORDER BY lower(title), 
             CASE WHEN status = 'processed' THEN 0 ELSE 1 END,
             ingested_at DESC
)
AND title IS NOT NULL
AND title != ''
AND status != 'processed';  -- JANGAN hapus yang sudah processed

-- ============================================================
-- STEP 5: VERIFIKASI
-- ============================================================

SELECT '=== SETELAH RESET ===' as info;

SELECT 
    CASE 
        WHEN content_type = 'SNIPPET' AND status = 'pending' THEN 'Snippet siap resolve'
        WHEN content_type = 'SNIPPET' AND status = 'validated' THEN 'Snippet validated (text ada)'
        WHEN content_type = 'FULLTEXT' AND status = 'validated' THEN 'Fulltext validated'
        WHEN content_type = 'FULLTEXT' AND status = 'queued' THEN 'Fulltext queued (siap NLP)'
        WHEN content_type = 'FULLTEXT' AND status = 'processed' THEN 'Fulltext processed (ada sentiment)'
        ELSE 'Other: ' || content_type || '/' || status
    END as category,
    COUNT(*) as total
FROM raw_texts
GROUP BY 1
ORDER BY 1;

-- ============================================================
-- EKSPEKTASI:
--   ~80K snippet → resolver → ~24K fulltext (30% success)
--   ~10K existing fulltext
--   Total: ~34K fulltext
--   After dedup: ~20-25K unique fulltext
--   After validation: ~20K+ masuk NLP pipeline
-- ============================================================

-- ============================================================
-- SETELAH RUN SQL INI:
--   1. git pull origin main
--   2. Run resolver: python main.py run-worker gnews_resolver --limit 200 --max-total 5000
--   3. Run prep: python main.py run-prep --limit 100 --max-total 5000
--   4. Run NLP: python main.py run-nlp --all
-- ============================================================
