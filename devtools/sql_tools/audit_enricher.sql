-- 1. Distribusi Content Type & Status
SELECT content_type, status, COUNT(*) 
FROM raw_texts 
GROUP BY content_type, status;

-- 2. Statistik Panjang Teks Fulltext
SELECT 
    MIN(LENGTH(text)) as min_len, 
    MAX(LENGTH(text)) as max_len, 
    AVG(LENGTH(text))::int as avg_len
FROM raw_texts 
WHERE content_type = 'FULLTEXT' AND LENGTH(text) > 0;

-- 3. Sampel Teks Fulltext (Baca manual apakah isinya artikel utuh)
SELECT id, title, LEFT(text, 500) as text_preview
FROM raw_texts
WHERE content_type = 'FULLTEXT' AND LENGTH(text) > 500
ORDER BY RANDOM()
LIMIT 5;