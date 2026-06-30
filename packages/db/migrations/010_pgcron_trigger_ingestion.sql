-- ============================================================
-- 008_pgcron_trigger_ingestion.sql
-- Pindahkan trigger RSS ingestion dari GitHub Actions (delay 2-3 jam,
-- documented "best effort" behavior dari GitHub) ke pg_cron + pg_net
-- (presisi sampai 1 menit, native di Supabase, tidak perlu service luar)
--
-- WAJIB: enable pg_net dulu via Dashboard -> Database -> Extensions
-- (pg_cron seharusnya sudah aktif dari migration sebelumnya)
-- ============================================================

CREATE EXTENSION IF NOT EXISTS pg_net;

-- ─────────────────────────────────────────────────────────────
-- STEP 1: Simpan secrets ke Vault
-- JANGAN commit file ini setelah diisi value asli -- jalankan
-- manual sekali di SQL Editor, JANGAN masukkan value asli ke git.
-- ─────────────────────────────────────────────────────────────

-- Ganti <...> dengan value asli, jalankan SEKALI di SQL Editor:
--
-- SELECT vault.create_secret(
--   'https://bawvxtivogcuwvqdqoae.supabase.co/functions/v1/rss-ingestion',
--   'rss_ingestion_url'
-- );
-- SELECT vault.create_secret('<anon-key>', 'rss_ingestion_anon_key');
-- SELECT vault.create_secret('<cron-secret>', 'rss_ingestion_cron_secret');

-- Verifikasi tersimpan (tidak menampilkan value asli, cuma nama):
-- SELECT name, created_at FROM vault.secrets WHERE name LIKE 'rss_ingestion%';

-- ─────────────────────────────────────────────────────────────
-- STEP 2: Hapus job lama kalau pernah ada (idempotent)
-- ─────────────────────────────────────────────────────────────

SELECT cron.unschedule('trigger-rss-ingestion')
WHERE EXISTS (SELECT 1 FROM cron.job WHERE jobname = 'trigger-rss-ingestion');

-- ─────────────────────────────────────────────────────────────
-- STEP 3: Schedule via pg_cron + pg_net
-- Tiap 30 menit, presisi ~1 menit (vs GitHub Actions yang bisa
-- molor 2-3 jam berdasarkan observasi langsung di project ini)
-- ─────────────────────────────────────────────────────────────

SELECT cron.schedule(
  'trigger-rss-ingestion',
  '*/30 * * * *',
  $$
  SELECT net.http_post(
    url := (SELECT decrypted_secret FROM vault.decrypted_secrets
            WHERE name = 'rss_ingestion_url'),
    headers := jsonb_build_object(
      'Authorization', 'Bearer ' || (
        SELECT decrypted_secret FROM vault.decrypted_secrets
        WHERE name = 'rss_ingestion_anon_key'
      ),
      'x-cron-secret', (
        SELECT decrypted_secret FROM vault.decrypted_secrets
        WHERE name = 'rss_ingestion_cron_secret'
      ),
      'Content-Type', 'application/json'
    ),
    body := '{}'::jsonb,
    timeout_milliseconds := 60000   -- 60s, RSS fetch ke 24 sumber butuh waktu
  );
  $$
);

-- ─────────────────────────────────────────────────────────────
-- VERIFIKASI
-- ─────────────────────────────────────────────────────────────

-- Cek job terdaftar
SELECT jobid, jobname, schedule, active
FROM cron.job
WHERE jobname = 'trigger-rss-ingestion';

-- Cek riwayat eksekusi (tunggu beberapa menit setelah schedule dibuat)
SELECT jobname, status, return_message, start_time, end_time
FROM cron.job_run_details jrd
JOIN cron.job j ON j.jobid = jrd.jobid
WHERE j.jobname = 'trigger-rss-ingestion'
ORDER BY start_time DESC
LIMIT 10;

-- Cek response detail dari pg_net (body actual dari Edge Function)
-- Berguna untuk debug kalau status failed
SELECT id, status_code, content, created
FROM net._http_response
ORDER BY created DESC
LIMIT 5;

-- ─────────────────────────────────────────────────────────────
-- CATATAN
-- ─────────────────────────────────────────────────────────────
-- - pg_net response disimpan di unlogged table, hanya 6 jam terakhir
--   (cukup untuk debug langsung, bukan untuk audit jangka panjang)
-- - Supabase merekomendasikan max 8 cron job concurrent -- project ini
--   sudah punya beberapa (refresh MV, partman maintenance, entity hotness),
--   tambah 1 lagi untuk ingestion masih jauh dari limit
-- - GitHub Actions (.github/workflows/trigger-ingestion.yml) TIDAK perlu
--   dihapus -- ubah jadi backup/manual-only (lihat update workflow di bawah)
