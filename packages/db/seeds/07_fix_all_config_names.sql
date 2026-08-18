-- ============================================================
-- 07_fix_all_config_names.sql
-- Fix ALL truncated config_names based on entity_id lookup
-- ============================================================
-- BUG: config_name kehilangan huruf pertama tiap kata
--   gnews_rabowo_ubianto → gnews_prabowo_subianto
--   gnews_rick_hohir → gnews_erick_thohir
--   dll (50+ configs affected)
--
-- ROOT CAUSE: Python script yang generate 04_scraping_configs_expanded.sql
-- memotong huruf pertama tiap kata. SQL slug generation BENAR,
-- tapi data yang dihasilkan script Python-nya sudah rusak.
--
-- FIX: Update config_name berdasarkan canonical_name dari entity_id
-- ============================================================

DO $$
DECLARE
  r RECORD;
  v_correct_slug TEXT;
  v_updated_count INTEGER := 0;
  v_skipped_count INTEGER := 0;
BEGIN
  -- Loop semua configs yang punya entity_id
  FOR r IN
    SELECT sc.id, sc.config_name, sc.entity_id, pe.canonical_name
    FROM scraping_configs sc
    JOIN political_entities pe ON sc.entity_id = pe.id
    WHERE sc.source_type = 'google_news_rss'
      AND sc.entity_id IS NOT NULL
  LOOP
    -- Generate slug yang BENAR dari canonical_name
    v_correct_slug := 'gnews_' || lower(
      regexp_replace(
        regexp_replace(r.canonical_name, '\s+', '_', 'g'),
        '[^a-z0-9_]', '', 'g'
      )
    );

    -- Cek apakah config_name saat ini berbeda dari yang seharusnya
    IF r.config_name != v_correct_slug THEN
      -- Cek apakah slug yang benar sudah ada (duplicate check)
      IF NOT EXISTS (
        SELECT 1 FROM scraping_configs 
        WHERE config_name = v_correct_slug 
        AND id != r.id
      ) THEN
        -- Update config_name ke yang benar
        UPDATE scraping_configs
        SET config_name = v_correct_slug
        WHERE id = r.id;

        v_updated_count := v_updated_count + 1;
        RAISE NOTICE 'FIX: % → %', r.config_name, v_correct_slug;
      ELSE
        -- Slug sudah ada di config lain — delete duplicate
        DELETE FROM scraping_configs WHERE id = r.id;
        v_skipped_count := v_skipped_count + 1;
        RAISE NOTICE 'DEDUP: deleted duplicate % (slug % already exists)', 
          r.config_name, v_correct_slug;
      END IF;
    END IF;
  END LOOP;

  RAISE NOTICE '=== FIX COMPLETE ===';
  RAISE NOTICE 'Updated: % configs', v_updated_count;
  RAISE NOTICE 'Deduped (deleted duplicates): % configs', v_skipped_count;
END $$;

-- ============================================================
-- VERIFIKASI: Cek hasil setelah fix
-- ============================================================
SELECT 
  COUNT(*) AS total_configs,
  COUNT(*) FILTER (WHERE entity_id IS NOT NULL) AS with_entity,
  COUNT(*) FILTER (WHERE entity_id IS NULL) AS without_entity
FROM scraping_configs
WHERE source_type = 'google_news_rss';

-- Sample 20 config_names untuk verifikasi (harus TIDAK ada huruf pertama hilang)
SELECT 
  sc.config_name,
  pe.canonical_name,
  CASE 
    WHEN sc.config_name = 'gnews_' || lower(
      regexp_replace(
        regexp_replace(pe.canonical_name, '\s+', '_', 'g'),
        '[^a-z0-9_]', '', 'g'
      )
    ) THEN '✅ CORRECT'
    ELSE '❌ STILL BUGGED'
  END AS status
FROM scraping_configs sc
JOIN political_entities pe ON sc.entity_id = pe.id
WHERE sc.source_type = 'google_news_rss'
ORDER BY sc.config_name
LIMIT 30;
