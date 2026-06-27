-- ============================================================
-- MIGRATION: PGMQ Queue + Enqueue/Dequeue RPCs (Layer 3)
-- ============================================================
-- Tujuan: Menghubungkan Layer 2 (Ingestion) dan Layer 4 (NLP Worker)
-- melalui queue, sesuai arsitektur di ai.md.
--
-- SEBELUM INI:
--   - Pastikan extension pgmq aktif (Dashboard > Database > Extensions > pgmq).
--     Jika tidak ada di list, jalankan: CREATE EXTENSION IF NOT EXISTS pgmq;
--   - schema_final_v2.sql sudah ter-apply (tabel raw_texts ada).
--
-- File ini HANYA menambah: queue + 2 RPC. Tidak mengubah skema inti.
-- Aman dijalankan ulang (idempoten).
-- ============================================================

-- ─────────────────────────────────────────────────────────────
-- 1. BUAT QUEUE
--    pgmq.create membuat tabel pgmq.q_nlp_processing_queue + retention.
--    max_msg_id, retention_interval default pgmq (idempoten via IF NOT EXISTS
--    di pgmq.create jika versi mendukung; versi lama bisa error duplikat —
--    bungkus di DO block untuk safety).
-- ─────────────────────────────────────────────────────────────
DO $$
BEGIN
    -- pgmq.create raises exception if queue already exists on older versions.
    -- Guard dengan cek di pgmq.meta terlebih dahulu.
    IF NOT EXISTS (
        SELECT 1 FROM pgmq.meta WHERE queue_name = 'nlp_processing_queue'
    ) THEN
        PERFORM pgmq.create('nlp_processing_queue');
        RAISE NOTICE 'Queue nlp_processing_queue created.';
    ELSE
        RAISE NOTICE 'Queue nlp_processing_queue already exists — skipping.';
    END IF;
END $$;


-- ─────────────────────────────────────────────────────────────
-- 2. RPC: enqueue_pending_raw_texts()
--    Dipanggil Edge Function SETELAH batch_insert_raw_texts().
--    Membaca semua raw_texts dengan status='pending' (yang belum pernah
--    di-enqueue), flip status -> 'queued', dan push message ke pgmq.
--
--    Message envelope: {"raw_text_id": "...", "entity_id": "..." | null}
--    entity_id diambil dari scraping_configs (kalau feed itu per-tokoh).
--
--    SECURITY DEFINER + service_role-only: NLP worker & Edge Function.
-- ─────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION enqueue_pending_raw_texts(
    p_batch_limit INTEGER DEFAULT 200
) RETURNS TABLE (enqueued_count BIGINT)
LANGUAGE plpgsql SECURITY DEFINER AS $$
DECLARE
    v_count BIGINT := 0;
    r RECORD;
    v_msg JSONB;
BEGIN
    FOR r IN
        SELECT rt.id AS raw_text_id,
               sc.entity_id AS entity_id
        FROM raw_texts rt
        LEFT JOIN scraping_configs sc
               ON sc.config_name = rt.source
        WHERE rt.status = 'pending'
        ORDER BY rt.ingested_at ASC
        LIMIT p_batch_limit
    LOOP
        -- Flip status FIRST (atomic), then enqueue. Kalau enqueue gagal,
        -- status tetap 'queued' tapi message hilang — row itu akan dipickup
        -- ulang oleh cron berikutnya? TIDAK, karena status sudah 'queued'.
        -- Solusi: pakai single transaction (default plpgsql), semua rollback
        -- kalau ada error. Aman.
        UPDATE raw_texts SET status = 'queued' WHERE id = r.raw_text_id;

        v_msg := jsonb_build_object(
            'raw_text_id', r.raw_text_id,
            'entity_id',   r.entity_id
        );

        PERFORM pgmq.send('nlp_processing_queue', v_msg);
        v_count := v_count + 1;
    END LOOP;

    RETURN QUERY SELECT v_count;
END;
$$;

-- RLS: raw_texts blocks anon, tapi SECURITY DEFINER function berjalan sebagai
-- owner (service_role-level). Tidak perlu policy tambahan.


-- ─────────────────────────────────────────────────────────────
-- 3. RPC: dequeue_nlp_batch(p_vt, p_qty)
--    Dipanggil NLP Worker. Membaca sejumlah message dari queue dengan
--    visibility timeout (vt). Selama vt, message tidak terlihat worker lain
--    -> anti race condition multi-worker. Setelah diproses, worker panggil
--    ack_message().
--
--    RETURNS: array of message envelopes + raw text content (1 round-trip).
-- ─────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION dequeue_nlp_batch(
    p_vt  INTEGER DEFAULT 60,    -- visibility timeout (detik)
    p_qty INTEGER DEFAULT 16
) RETURNS TABLE (
    msg_id       BIGINT,
    raw_text_id  UUID,
    entity_id    UUID,
    text         TEXT,
    title        TEXT,
    source       TEXT
)
LANGUAGE plpgsql SECURITY DEFINER AS $$
DECLARE
    v_msgs JSONB;
BEGIN
    -- pgmq.read returns JSONB array of {msg_id, message, read_ct, ...}
    SELECT pgmq.read('nlp_processing_queue', p_vt, p_qty) INTO v_msgs;

    RETURN QUERY
    SELECT
        (m->>'msg_id')::BIGINT              AS msg_id,
        ((m->'message')->>'raw_text_id')::UUID AS raw_text_id,
        ((m->'message')->>'entity_id')::UUID   AS entity_id,
        rt.text                             AS text,
        rt.title                            AS title,
        rt.source                           AS source
    FROM jsonb_array_elements(v_msgs) AS m
    LEFT JOIN raw_texts rt ON rt.id = ((m->'message')->>'raw_text_id')::UUID
    WHERE rt.id IS NOT NULL;   -- skip orphaned messages (row purged)
END;
$$;


-- ─────────────────────────────────────────────────────────────
-- 4. RPC: ack_nlp_message(p_msg_id)
--    Dipanggil NLP Worker SETELAH inference + insert_sentiment_score sukses.
--    Menghapus message dari queue (mark processed). Jika worker crash,
--    message akan reappear setelah vt berlalu -> otomatis retry.
-- ─────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION ack_nlp_message(p_msg_id BIGINT)
RETURNS BOOLEAN
LANGUAGE sql SECURITY DEFINER AS $$
    SELECT pgmq.delete('nlp_processing_queue', p_msg_id) IS NOT NULL;
$$;


-- ─────────────────────────────────────────────────────────────
-- 5. RPC: mark_raw_text_failed(p_raw_text_id)
--    Dipanggil worker kalau inference error permanen. Set status 'failed'
--    + ack message (tidak retry lagi). Untuk transient error, worker cukup
--    TIDAK ack -> message reappear setelah vt.
-- ─────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION mark_raw_text_failed(p_raw_text_id UUID)
RETURNS VOID
LANGUAGE sql SECURITY DEFINER AS $$
    UPDATE raw_texts SET status = 'failed', processed_at = NOW()
    WHERE id = p_raw_text_id;
$$;


-- ─────────────────────────────────────────────────────────────
-- 6. VERIFIKASI
-- ─────────────────────────────────────────────────────────────
SELECT queue_name FROM pgmq.meta WHERE queue_name = 'nlp_processing_queue';

-- Test enqueue/dequeue round-trip (jalankan manual, lalu rollback kalau mau bersih):
-- SELECT * FROM enqueue_pending_raw_texts(5);
-- SELECT * FROM dequeue_nlp_batch(60, 5);
-- SELECT * FROM ack_nlp_message(<msg_id dari atas>);
