-- ============================================================
-- MIGRATION: Allow NULL entity_id in sentiment_scores
-- ============================================================
-- Tujuan: Pipeline testing — NLP CLI bisa insert score tanpa
--   entity match (entity_id = NULL). Ini berguna untuk:
--   1. Test end-to-end pipeline sebelum entity matching perfek
--   2. Analisis sentiment umum (non-entity-specific)
--   3. Data yang belum matched tetap tersimpan untuk re-processing
--
-- Catatan: entity_id tetap punya FK ke political_entities(id),
--   tapi NULL values diperbolehkan (ON DELETE CASCADE only applies
--   when entity_id is NOT NULL).
--
-- Idempotent. Safe to re-run.
-- ============================================================

ALTER TABLE sentiment_scores ALTER COLUMN entity_id DROP NOT NULL;

-- Verifikasi
SELECT column_name, is_nullable, data_type
FROM information_schema.columns
WHERE table_name = 'sentiment_scores' AND column_name = 'entity_id';
