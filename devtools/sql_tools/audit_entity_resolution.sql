-- 1. Top 10 Tokoh yang Paling Sering Disebut
SELECT pe.canonical_name, COUNT(*) as mention_count
FROM entity_mentions em
JOIN political_entities pe ON em.entity_id = pe.id
GROUP BY pe.canonical_name
ORDER BY mention_count DESC
LIMIT 10;

-- 2. Cek Kesemburan Alias (Ambil 10 sampel teks mention)
SELECT mention_text, entity_id
FROM entity_mentions
LIMIT 10;