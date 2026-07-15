-- Ambil 5 Sampel Perbandingan Teks (Baca manual perbedaannya)
-- Catatan: Karena teks lama ditimpa teks bersih, kita cek metadata audit_stats
SELECT 
    id, 
    title, 
    LEFT(text, 400) as clean_text_preview,
    metadata->'audit_stats'->>'original_len' as original_len,
    metadata->'audit_stats'->>'clean_len' as clean_len,
    metadata->'audit_stats'->>'urls_emails_removed' as urls_removed
FROM raw_texts 
WHERE preprocessed_at IS NOT NULL 
ORDER BY RANDOM()
LIMIT 5;


