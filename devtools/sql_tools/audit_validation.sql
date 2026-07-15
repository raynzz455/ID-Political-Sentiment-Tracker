-- Statistik Panjang Teks Fulltext
SELECT 
    MIN(LENGTH(text)) as min_len, 
    MAX(LENGTH(text)) as max_len, 
    AVG(LENGTH(text))::int as avg_len
FROM raw_texts 
WHERE content_type = 'FULLTEXT' AND LENGTH(text) > 0;

-- Ambil 5 Sampel Teks Fulltext (Baca manual bagian awalnya)
SELECT id, title, LEFT(text, 500) as text_preview
FROM raw_texts
WHERE content_type = 'FULLTEXT' AND LENGTH(text) > 500
ORDER BY RANDOM()
LIMIT 5;


-- 1. Distribusi Lolos vs Gagal
SELECT status, COUNT(*) 
FROM raw_texts 
WHERE status IN ('validated', 'failed') 
GROUP BY status;

-- 2. Alasan Kegagalan Validation
SELECT metadata->>'fail_reason' as reason, COUNT(*) 
FROM raw_texts 
WHERE status = 'failed' AND metadata ? 'fail_reason'
GROUP BY 1 
ORDER BY COUNT(*) DESC;