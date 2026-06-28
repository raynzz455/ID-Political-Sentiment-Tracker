-- ============================================================
-- MIGRATION: Reset dedup + requeue untuk test Lapis 2
-- ============================================================
-- Latar belakang: queue kosong setelah beberapa kali test drain.
--                Re-ingest RSS gagal karena raw_text_hashes sudah
--                menyimpan hash → semua artikel di-anggap duplikat.
--
-- Migration ini:
--   1. Reset dedup (clear raw_text_hashes) supaya re-ingest bisa
--      insert ulang
--   2. Reset status raw_texts 'processed'/'failed' → 'pending'
--      supaya re-enqueue
--   3. Kosongkan queue dari message lama
--   4. Re-enqueue semua pending → queue terisi
--
-- Idempotent. Safe to re-run.
-- ============================================================

-- ─── STEP 1: Reset dedup table ───────────────────────────────
TRUNCATE TABLE raw_text_hashes;
-- Sekarang re-ingest RSS akan insert semua artikel sebagai "new"

-- ─── STEP 2: Reset raw_texts status → pending ────────────────
-- (kecuali 'failed' permanent, yang text-nya benar-benar kosong)
UPDATE raw_texts
SET status = 'pending'
WHERE status IN ('queued', 'processed');

-- ─── STEP 3: Kosongkan queue ─────────────────────────────────
DELETE FROM pgmq.q_nlp_processing_queue;

-- ─── STEP 4: Re-enqueue semua pending ────────────────────────
SELECT * FROM enqueue_pending_raw_texts(1000);

-- ─── STEP 5: Verifikasi ──────────────────────────────────────
SELECT
    (SELECT COUNT(*) FROM pgmq.q_nlp_processing_queue) AS queue_size,
    (SELECT COUNT(*) FROM raw_texts WHERE status = 'pending') AS still_pending,
    (SELECT COUNT(*) FROM raw_texts WHERE status = 'queued') AS now_queued;
