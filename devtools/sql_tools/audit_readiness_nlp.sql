-- 1. Jumlah Artikel Siap NLP
SELECT COUNT(*) as nlp_ready_count
FROM raw_texts
WHERE nlp_ready_at IS NOT NULL;

-- 2. Distribusi Status (Pastikan tidak ada yang nyangkut)
SELECT status, COUNT(*)
FROM raw_texts
GROUP BY status;