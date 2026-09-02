-- ============================================================
-- 10_resolve_snippets_for_20k.sql
-- Reset 68K snippet articles agar bisa di-resolve oleh gnews_resolver
-- Target: dapat 20K+ fulltext articles
-- ============================================================
-- STRATEGI:
--   1. Reset 68,349 SNIPPET (status=failed) → status=pending (siap di-resolve)
--   2. Reset 5,526 SNIPPET (status=validated) → tetap (sudah punya text, tapi snippet)
--   3. Jangan sentuh FULLTEXT yang sudah validated/queued/processed (8,082 + 1,872 + 474)
--   4. Cegah duplikasi: sebelum resolver fetch, cek content_hash di DB
-- ============================================================

-- ============================================================
-- STEP 1: Reset SNIPPET yang FAILED → PENDING (siap di-resolve)
-- Target: 68,349 articles
-- ============================================================

UPDATE raw_texts
SET 
    status = 'pending',
    recovery_status = 'pending',
    recovery_attempts = 0,
    metadata = jsonb_set(
        COALESCE(metadata, '{}'::jsonb),
        '{fail_reason}',
        '"pending_resolver"'
    )
WHERE content_type = 'SNIPPET'
  AND status = 'failed'
  AND source_url IS NOT NULL
  AND source_url != '';

-- ============================================================
-- STEP 2: Reset SNIPPET yang SKIPPED → PENDING
-- Target: ~12,533 articles (yang di-skip karena duplicate di readiness)
-- Tapi sekarang resolver akan cek duplikasi yang lebih ketat
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
-- (sudah punya sentiment scores, jangan buang)
-- ============================================================

-- 474 processed articles → biarkan, sudah ada sentiment scores

-- ============================================================
-- STEP 4: Tambah DEDUP GATE ke gnews_resolver_worker
-- Cek content_hash SEBELUM fetch HTTP (hemat bandwidth)
-- ============================================================

-- ============================================================
-- STEP 5: Set MAX_WORKERS = 1 di gnews_resolver (untuk Colab/GH Actions)
-- (Resolver butuh HTTP fetch, parallel bisa di-rate-limit)
-- ============================================================

-- ============================================================
-- VERIFIKASI: Potensi data setelah reset
-- ============================================================

SELECT 
    content_type,
    status,
    COUNT(*) as total
FROM raw_texts
WHERE source_url IS NOT NULL AND source_url != ''
GROUP BY content_type, status
ORDER BY content_type, status;

-- ============================================================
-- TOTAL POTENSI:
--   PENDING (siap resolver): ~80K+ snippet articles
--   VALIDATED (sudah ada text): ~8K fulltext + ~5.5K snippet
--   QUEUED (siap NLP): ~1.8K
--   PROCESSED (sudah ada sentiment): 474
--
-- EXPECTED RESULT setelah resolver:
--   - 80K snippet × ~30% success rate = ~24K fulltext articles
--   - Total fulltext: 24K + 8K existing = ~32K
--   - Yang masuk NLP: ~20K+ (setelah validation + dedup)
-- ============================================================

-- ============================================================
-- SETELAH RUN SQL INI:
--   1. git pull origin main (dapatkan gnews_resolver_worker yang updated)
--   2. python main.py run-worker gnews_resolver --limit 200 --max-total 5000
--   3. Setelah resolver selesai, run prep pipeline:
--      python main.py run-prep --limit 100 --max-total 5000
--   4. Setelah prep selesai, run NLP:
--      python main.py run-nlp --all
-- ============================================================
