-- ============================================================
-- MIGRATION: Pre-requisite UNIQUE constraints untuk ekspansi tokoh
-- ============================================================
-- Latar belakang:
--   File ekspansi Claude (03, 04) pakai ON CONFLICT (canonical_name)
--   dan ON CONFLICT (config_name), tapi tabel belum punya UNIQUE
--   constraint di kolom itu. Migration ini menambahkannya.
--
--   JUGA: dedup data yang sudah ada sebelum UNIQUE bisa di-add
--   (PostgreSQL tolak ADD UNIQUE kalau ada duplikat existing).
--
-- WAJIB di-run SEBELUM 007_entity_expansion_schema.sql.
-- Idempotent. Safe to re-run.
-- ============================================================

-- ─── STEP 1: Dedup political_entities (kalau ada) ────────────
-- Hapus row duplikat canonical_name, simpan yang paling baru (created_at).
DELETE FROM political_entities
WHERE id NOT IN (
    SELECT DISTINCT ON (canonical_name) id
    FROM political_entities
    ORDER BY canonical_name, created_at DESC
);

-- ─── STEP 2: Add UNIQUE constraint ke canonical_name ─────────
ALTER TABLE political_entities
  ADD CONSTRAINT political_entities_canonical_name_key UNIQUE (canonical_name);

-- ─── STEP 3: Dedup scraping_configs (kalau ada) ──────────────
DELETE FROM scraping_configs
WHERE id NOT IN (
    SELECT DISTINCT ON (config_name) id
    FROM scraping_configs
    ORDER BY config_name, created_at DESC
);

-- ─── STEP 4: Add UNIQUE constraint ke config_name ────────────
ALTER TABLE scraping_configs
  ADD CONSTRAINT scraping_configs_config_name_key UNIQUE (config_name);

-- ─── STEP 5: Verifikasi ──────────────────────────────────────
SELECT
    conname,
    conrelid::regclass AS table_name
FROM pg_constraint
WHERE conname IN ('political_entities_canonical_name_key',
                  'scraping_configs_config_name_key');
