-- 1. Sampel Context (Baca manual apakah kalimatnya membahas tokoh tersebut)
SELECT 
    pe.canonical_name, 
    LEFT(ec.context_text, 300) as context_preview,
    ec.metadata->>'quality_score' as score
FROM entity_contexts ec
JOIN political_entities pe ON ec.entity_id = pe.id
ORDER BY RANDOM()
LIMIT 10;