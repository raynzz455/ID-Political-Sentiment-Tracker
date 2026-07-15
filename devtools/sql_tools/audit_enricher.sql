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

-- 1. PERBANDINGAN SUKSES VS GAGAL BERDASARKAN SUMBER URL
-- Membandingkan URL GNews (yang di-bypass jadi snippet) vs URL Asli (yang di-fetch trafilatura)
SELECT 
    CASE 
        WHEN source_url LIKE '%news.google.com%' THEN 'GNews (Bypass Snippet)'
        ELSE 'URL Asli (DDG/RSS Native)'
    END AS sumber_url,
    content_type,
    status,
    COUNT(*) as jumlah
FROM raw_texts
GROUP BY sumber_url, content_type, status
ORDER BY sumber_url, status DESC;

-- 2. RINCIAN KEGAGALAN URL ASLI (Kenapa DDG/RSS gagal di-fetch?)
-- Ini penting untuk tau apakah media memblokir kita (403) atau linknya memang mati (404)
SELECT 
    metadata->>'fail_reason' as alasan_gagal, 
    COUNT(*) as jumlah
FROM raw_texts
WHERE source_url NOT LIKE '%news.google.com%' 
  AND status = 'failed'
  AND metadata ? 'fail_reason'
GROUP BY 1
ORDER BY jumlah DESC;

-- 3. KUALITAS EKSTRAKSI TEKS URL ASLI (Apakah teks yang didapat utuh?)
-- Kita kelompokkan berdasarkan kategori panjang teks
SELECT 
    CASE 
        WHEN LENGTH(text) = 0 THEN '0. Kosong (Gagal Total)'
        WHEN LENGTH(text) < 500 THEN '1. Pendek (< 500 char)'
        WHEN LENGTH(text) < 1500 THEN '2. Sedang (500 - 1500 char)'
        WHEN LENGTH(text) >= 1500 THEN '3. Panjang (> 1500 char - Ideal)'
    END AS kategori_panjang,
    COUNT(*) as jumlah_artikel
FROM raw_texts
WHERE source_url NOT LIKE '%news.google.com%' 
  AND status = 'enriched'
GROUP BY kategori_panjang
ORDER BY kategori_panjang;