-- ============================================================
-- 12_reset_for_gnews_local_resolver.sql
-- Reset snippet articles agar siap untuk gnews_local_resolver (Playwright)
-- ============================================================
-- gnews_local_resolver query:
--   .or_(status.eq.failed, and(status.eq.enriched, content_type.eq.SNIPPET))
--   .lt(recovery_attempts, MAX_RECOVERY_RETRY)
--   .gte(ingested_at, 30_days_ago)
--
-- Artinya resolver ambil:
--   1. status=FAILED (snippet yang gagal enrich)
--   2. status=ENRICHED + content_type=SNIPPET (snippet yang berhasil enrich tapi text pendek)
--   3. recovery_attempts < MAX_RECOVERY_RETRY (belum di-retry terlalu banyak)
--   4. ingested within 30 days
-- ============================================================

-- ============================================================
-- STEP 1: Reset SNIPPET yang FAILED → tetap FAILED tapi recovery_attempts=0
-- (Resolver ambil status=FAILED, jadi tetap FAILED tapi reset retry count)
-- ============================================================

UPDATE raw_texts
SET 
    recovery_attempts = 0,
    recovery_status = 'pending'
WHERE content_type = 'SNIPPET'
  AND status = 'failed'
  AND source_url IS NOT NULL
  AND source_url != '';

-- ============================================================
-- STEP 2: Reset SNIPPET yang SKIPPED → status=ENRICHED
-- (Resolver ambil ENRICHED + SNIPPET, jadi ubah skipped → enriched)
-- ============================================================

UPDATE raw_texts
SET 
    status = 'enriched',
    recovery_attempts = 0,
    recovery_status = 'pending',
    duplicate_of = NULL
WHERE content_type = 'SNIPPET'
  AND status = 'skipped'
  AND source_url IS NOT NULL
  AND source_url != '';

-- ============================================================
-- STEP 3: JANGAN RESET articles yang sudah PROCESSED
-- (2,430 articles sudah punya sentiment scores — jangan ganggu)
-- ============================================================

-- ============================================================
-- STEP 4: DEDUP — hapus duplikat yang sudah ada di DB
-- ============================================================

-- 4a: Hapus duplikat content_hash (keep yang processed/validated)
DELETE FROM raw_texts a
WHERE a.content_hash IS NOT NULL
  AND a.content_hash != ''
  AND a.id NOT IN (
    SELECT DISTINCT ON (content_hash) id
    FROM raw_texts
    WHERE content_hash IS NOT NULL AND content_hash != ''
    ORDER BY content_hash,
             CASE WHEN status = 'processed' THEN 0
                  WHEN status = 'queued' THEN 1
                  WHEN status = 'validated' THEN 2
                  ELSE 3 END,
             ingested_at DESC
  );

-- 4b: Hapus duplikat text_hash (keep yang processed/validated)
DELETE FROM raw_texts a
WHERE a.text_hash IS NOT NULL
  AND a.id NOT IN (
    SELECT DISTINCT ON (text_hash) id
    FROM raw_texts
    WHERE text_hash IS NOT NULL
    ORDER BY text_hash,
             CASE WHEN status = 'processed' THEN 0
                  WHEN status = 'queued' THEN 1
                  WHEN status = 'validated' THEN 2
                  ELSE 3 END,
             ingested_at DESC
  );

-- 4c: Hapus duplikat title (keep yang processed, jangan hapus processed)
DELETE FROM raw_texts a
WHERE a.title IS NOT NULL
  AND a.title != ''
  AND a.status != 'processed'
  AND a.id NOT IN (
    SELECT DISTINCT ON (lower(title)) id
    FROM raw_texts
    WHERE title IS NOT NULL AND title != ''
    ORDER BY lower(title),
             CASE WHEN status = 'processed' THEN 0
                  WHEN status = 'queued' THEN 1
                  WHEN status = 'validated' THEN 2
                  ELSE 3 END,
             ingested_at DESC
  );

-- ============================================================
-- STEP 5: VERIFIKASI
-- ============================================================

SELECT '=== STATUS SETELAH RESET ===' as info;

SELECT 
    content_type || ' / ' || status as category,
    COUNT(*) as total
FROM raw_texts
GROUP BY content_type, status
ORDER BY 1;

-- ============================================================
-- EKSPEKTASI:
--   ~68K snippet (failed, recovery_attempts=0) → siap resolver
--   ~8K snippet (skipped→enriched, recovery_attempts=0) → siap resolver
--   Total siap resolver: ~76K articles
--   × 30% success rate = ~23K new fulltext
--   + 10K existing fulltext = ~33K total
--   After dedup: ~20-25K unique → TARGET 20K TERCAPAI ✅
-- ============================================================

-- ============================================================
-- SETELAH RUN SQL INI:
--   1. git pull origin main
--   2. cd devtools/recovery
--   3. python gnews_local_resolver.py --limit 50 --max-total 10000
--      (Run bertahap, 10K per session)
--   4. Setelah resolve: python main.py run-prep --limit 100 --max-total 5000
--   5. Setelah prep: python main.py run-nlp --all
-- ============================================================
