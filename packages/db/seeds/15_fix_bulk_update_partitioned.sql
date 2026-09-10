-- ============================================================
-- MIGRATION: Fix bulk_update_raw_texts untuk partitioned tables
-- 
-- BUG: UPDATE raw_texts (parent) tidak affect child partitions.
-- Database menggunakan monthly partitioning:
--   raw_texts (parent, no data) → raw_texts_2026_06, _07, _08, _09, _10
--
-- Fix: Gunakan dynamic SQL untuk UPDATE langsung ke partition
-- berdasarkan ingested_month dari row yang akan di-update.
-- ============================================================

-- DROP function lama
DROP FUNCTION IF EXISTS bulk_update_raw_texts(jsonb);

-- CREATE function baru yang support partitioning
CREATE OR REPLACE FUNCTION bulk_update_raw_texts(p_updates jsonb)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    update_record record;
    partition_name text;
    target_month text;
BEGIN
    -- Iterate over each update record
    FOR update_record IN 
        SELECT * FROM jsonb_populate_recordset(null::public.raw_texts, p_updates)
    LOOP
        -- Determine partition name from ingested_month
        -- Partition naming: raw_texts_YYYY_MM
        IF update_record.ingested_month IS NOT NULL THEN
            target_month := to_char(update_record.ingested_month::date, 'YYYY_MM');
            partition_name := 'raw_texts_' || target_month;
        ELSE
            -- Fallback: try to find the row's partition
            SELECT to_char(ingested_month, 'YYYY_MM') INTO target_month
            FROM raw_texts WHERE id = update_record.id LIMIT 1;
            
            IF target_month IS NULL THEN
                -- Last resort: try current month partition
                target_month := to_char(NOW(), 'YYYY_MM');
            END IF;
            partition_name := 'raw_texts_' || target_month;
        END IF;
        
        -- Dynamic UPDATE ke partition yang benar
        EXECUTE format(
            'UPDATE %I SET 
                text = COALESCE($1, text),
                status = COALESCE($2, status),
                content_type = COALESCE($3, content_type),
                metadata = CASE WHEN $4 IS NOT NULL THEN metadata || $4 ELSE metadata END,
                recovery_attempts = COALESCE($5, recovery_attempts),
                recovery_status = COALESCE($6, recovery_status),
                resolved_domain = COALESCE($7, resolved_domain),
                canonical_url = COALESCE($8, canonical_url),
                content_hash = COALESCE($9, content_hash),
                processed_at = COALESCE($10, processed_at),
                pipeline_version = COALESCE($11, pipeline_version),
                resolver_version = COALESCE($12, resolver_version),
                context_version = COALESCE($13, context_version),
                preprocessed_at = COALESCE($14, preprocessed_at),
                context_extracted_at = COALESCE($15, context_extracted_at),
                nlp_ready_at = COALESCE($16, nlp_ready_at),
                duplicate_of = COALESCE($17, duplicate_of),
                preprocessing_version = COALESCE($18, preprocessing_version),
                entity_resolved_at = COALESCE($19, entity_resolved_at),
                updated_at = NOW()
            WHERE id = $20',
            partition_name
        )
        USING 
            update_record.text,
            update_record.status,
            update_record.content_type,
            update_record.metadata,
            update_record.recovery_attempts,
            update_record.recovery_status,
            update_record.resolved_domain,
            update_record.canonical_url,
            update_record.content_hash,
            update_record.processed_at,
            update_record.pipeline_version,
            update_record.resolver_version,
            update_record.context_version,
            update_record.preprocessed_at,
            update_record.context_extracted_at,
            update_record.nlp_ready_at,
            update_record.duplicate_of,
            update_record.preprocessing_version,
            update_record.entity_resolved_at,
            update_record.id;
    END LOOP;
END;
$$;

-- ============================================================
-- VERIFIKASI: Cek apakah function terbuat dengan benar
-- ============================================================
SELECT proname, prosrc LIKE '%partition_name%' as uses_partitioning
FROM pg_proc 
WHERE proname = 'bulk_update_raw_texts';

-- ============================================================
-- TEST: Update 1 row untuk verify partitioning works
-- ============================================================
-- SELECT bulk_update_raw_texts(
--   '[{"id": "<UUID_HERE>", "ingested_month": "2026-09-01", "status": "processed", "pipeline_version": "test"}]'::jsonb
-- );
-- 
-- Lalu cek:
-- SELECT id, status, pipeline_version FROM raw_texts_2026_09 WHERE id = '<UUID_HERE>';
