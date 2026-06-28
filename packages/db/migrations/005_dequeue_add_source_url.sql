-- ============================================================
-- MIGRATION: Rewrite dequeue_nlp_batch — fix pgmq.read + add source_url
-- ============================================================
-- ROOT CAUSE bug "Token ( is invalid":
--   pgmq.read() return SETOF record (msg_id, read_ct, vt, message, ...),
--   BUKAN jsonb array. Pattern lama `SELECT pgmq.read(...) INTO v_msgs`
--   assign composite record ke variabel jsonb → text form "(123,1,...)"
--   → jsonb_array_elements() gagal parse "(", hence error.
--
-- FIX: pakai pgmq.read() langsung di FROM clause (table form),
--   akses kolom m.msg_id / m.message langsung. No intermediate jsonb.
--
-- Bonus: tambah kolom source_url (dibutuhkan Lapis 2 untuk scrape).
--
-- Idempotent. Safe to re-run.
-- ============================================================

-- Drop semua versi lama (signature berubah)
DROP FUNCTION IF EXISTS dequeue_nlp_batch(INTEGER, INTEGER) CASCADE;

CREATE OR REPLACE FUNCTION dequeue_nlp_batch(
    p_vt  INTEGER DEFAULT 60,    -- visibility timeout (detik)
    p_qty INTEGER DEFAULT 16
) RETURNS TABLE (
    msg_id       BIGINT,
    raw_text_id  UUID,
    entity_id    UUID,
    text         TEXT,
    title        TEXT,
    source       TEXT,
    source_url   TEXT
)
LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
    -- pgmq.read return SETOF record; akses kolom langsung di FROM.
    -- Tidak ada lagi intermediate jsonb variable.
    RETURN QUERY
    SELECT
        m.msg_id,
        (m.message->>'raw_text_id')::UUID AS raw_text_id,
        NULLIF(m.message->>'entity_id', '')::UUID AS entity_id,
        rt.text,
        rt.title,
        rt.source,
        rt.source_url
    FROM pgmq.read('nlp_processing_queue', p_vt, p_qty) AS m
    LEFT JOIN raw_texts rt ON rt.id = (m.message->>'raw_text_id')::UUID
    WHERE rt.id IS NOT NULL;
END;
$$;

-- Grant ulang (DROP CASCADE bisa revoke)
GRANT EXECUTE ON FUNCTION dequeue_nlp_batch(INTEGER, INTEGER) TO service_role;

-- Verifikasi signature + test
SELECT proname, pg_get_function_arguments(oid), pg_get_function_result(oid)
FROM pg_proc WHERE proname = 'dequeue_nlp_batch';

-- Test panggil langsung (harus return rows atau empty, TIDAK boleh error)
SELECT * FROM dequeue_nlp_batch(5, 3);
