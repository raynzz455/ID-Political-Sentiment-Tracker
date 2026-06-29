-- ============================================================
-- DIAGNOSTIC: Cek kondisi nyata DB sebelum apply apapun
-- ============================================================
-- JALANKAN INI DULU. Output-nya menentukan langkah selanjutnya.
-- Tidak mengubah data apa-apa. 100% read-only. Aman.
-- ============================================================

-- ─── 1. Dequeue RPC: versi mana yang aktif? ─────────────────
SELECT
    proname AS function_name,
    pg_get_function_arguments(oid) AS args,
    pg_get_function_result(oid) AS result_columns
FROM pg_proc
WHERE proname = 'dequeue_nlp_batch';
-- EXPECTED: 1 row, result = ...source text, source_url text
-- Kalau TIDAK ada source_url → migration 005 belum di-run


-- ─── 2. Queue depth: ada berapa message? ────────────────────
SELECT
    (SELECT COUNT(*) FROM pgmq.q_nlp_processing_queue) AS queue_size,
    (SELECT COUNT(*) FROM raw_texts WHERE status = 'pending') AS pending_raw,
    (SELECT COUNT(*) FROM raw_texts WHERE status = 'queued') AS queued_raw,
    (SELECT COUNT(*) FROM sentiment_scores) AS total_scores;
-- Kalau total_scores = 0 → pipeline belum pernah jalan end-to-end


-- ─── 3. Entity count: berapa tokoh di DB? ───────────────────
SELECT
    COUNT(*) AS total_entities,
    COUNT(*) FILTER (WHERE is_active) AS active_entities
FROM political_entities;
-- 18 = seed lama (belum ekspansi), 50+ = seed 03 sudah di-run


-- ─── 4. RSS configs: berapa feed aktif? ─────────────────────
SELECT
    COUNT(*) AS total_configs,
    COUNT(*) FILTER (WHERE is_active) AS active_configs
FROM scraping_configs;
-- 23 = config lama, 70+ = config 04 sudah di-run


-- ─── 5. Kolom baru 007: sudah ada? ──────────────────────────
SELECT column_name
FROM information_schema.columns
WHERE table_name = 'political_entities'
  AND column_name IN ('era', 'birth_year', 'mention_count_7d', 'wikipedia_id_url');
-- Kalau return 0 rows → migration 007 belum di-run
-- Kalau return 4 rows → migration 007 sudah di-run


-- ─── 6. Tabel entity_candidates: ada? ───────────────────────
SELECT
    EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'entity_candidates'
    ) AS candidates_table_exists,
    EXISTS (
        SELECT 1 FROM information_schema.views
        WHERE table_name = 'hotline_tokoh'
    ) AS hotline_view_exists;
-- Kalau false/false → migration 007 belum di-run


-- ─── 7. Test dequeue langsung (CRITICAL) ────────────────────
-- Ini query yang error "Token ( is invalid" kalau RPC masih rusak
SELECT * FROM dequeue_nlp_batch(5, 2);
-- EXPECTED: rows atau (0 rows), TANPA ERROR
-- Kalau error → migration 005 belum berhasil di-run
