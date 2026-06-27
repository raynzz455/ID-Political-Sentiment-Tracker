-- ============================================================
-- MIGRATION: Purge queue items yang text-nya kosong
-- ============================================================
-- Latar belakang:
--   gnews RSS feeds hanya mengirim <title>, body text kosong.
--   Item-item lama yang lolos ke queue sebelum fix index.ts
--   tidak bisa di-proses NLP dengan baik (text kosong = 0 NLP value).
--
--   Migration ini:
--   1. Cek jumlah queue items yang bermasalah
--   2. Hapus queue message (mark_raw_text_failed via RPC)
--   3. Update raw_texts status = 'failed' untuk cleanup
--
-- Idempotent. Safe to re-run.
-- ============================================================

-- ─── STEP 1: Diagnosa ──────────────────────────────────────────
-- Cek berapa queue message yang raw_text-nya text kosong
SELECT
    COUNT(*) AS total_orphan_messages,
    COUNT(DISTINCT rt.id) AS distinct_raw_texts_affected
FROM pgmq.q_nlp_processing_queue q
LEFT JOIN raw_texts rt
    ON rt.id = ((q.message)->>'raw_text_id')::uuid
WHERE COALESCE(rt.text, '') = ''
   OR rt.id IS NULL;


-- ─── STEP 2: Purge ─────────────────────────────────────────────
-- Hapus queue messages yang raw_text-nya text kosong.
-- (Gunakan DELETE langsung di tabel queue pgmq.)
DELETE FROM pgmq.q_nlp_processing_queue q
WHERE EXISTS (
    SELECT 1
    FROM raw_texts rt
    WHERE rt.id = ((q.message)->>'raw_text_id')::uuid
      AND COALESCE(rt.text, '') = ''
);


-- ─── STEP 3: Cleanup raw_texts yang kena-purge ─────────────────
-- Tandai raw_texts yang text-nya kosong sebagai 'failed' supaya:
--   - tidak re-enqueue (enqueue hanya ambil status='pending')
--   - bisa audit nanti (di-distinct dari data valid)
UPDATE raw_texts
SET status = 'failed'
WHERE COALESCE(text, '') = ''
  AND status = 'queued';


-- ─── STEP 4: Verifikasi ────────────────────────────────────────
SELECT
    (SELECT COUNT(*) FROM pgmq.q_nlp_processing_queue) AS queue_size,
    (SELECT COUNT(*) FROM raw_texts WHERE COALESCE(text, '') = '' AND status = 'failed') AS purged_raw_texts,
    (SELECT COUNT(*) FROM raw_texts WHERE COALESCE(text, '') = '' AND status IN ('pending', 'queued')) AS remaining_empty_text;
