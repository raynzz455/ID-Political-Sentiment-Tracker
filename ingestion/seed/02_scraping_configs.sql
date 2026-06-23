-- ============================================================
-- SEED: scraping_configs
-- Harus dijalankan SETELAH 01_political_entities.sql
--
-- Dua jenis config:
--   1. General RSS (entity_id = NULL) → NLP worker lakukan NER
--   2. Google News RSS per tokoh (entity_id = <uuid>) → entity attribution lebih akurat
--
-- Cara verify feed sebelum insert:
--   curl -s "https://rss.detik.com/index.php/detikpolitik" | head -50
-- ============================================================


-- ─────────────────────────────────────────────────────────────
-- BAGIAN 1: GENERAL NEWS RSS
-- entity_id = NULL → NLP melakukan NER untuk tentukan entitas
-- Ini sumber utama volume data
-- ─────────────────────────────────────────────────────────────

INSERT INTO scraping_configs
    (entity_id, source_type, config_name, url, is_active)
VALUES

-- Detik Politik — RSS 2.0 standar, update tiap ~15 menit
(NULL, 'rss_news', 'detik_politik',
 'https://rss.detik.com/index.php/detikpolitik', true),

-- Antara Nasional — kantor berita negara, sangat reliable
(NULL, 'rss_news', 'antara_nasional',
 'https://www.antaranews.com/rss/nasional.rss', true),

-- Kompas Nasional
(NULL, 'rss_news', 'kompas_nasional',
 'https://rss.kompas.com/nasional', true),

-- Republika Politik
(NULL, 'rss_news', 'republika_politik',
 'https://www.republika.co.id/rss/news/politik', true),

-- Liputan6 Politik
(NULL, 'rss_news', 'liputan6_politik',
 'https://www.liputan6.com/feeds/rss2/news/politik.xml', true),

-- JPNN Nasional (Jawa Pos Group)
(NULL, 'rss_news', 'jpnn_nasional',
 'https://www.jpnn.com/rss/nasional', true),

-- CNN Indonesia Nasional
(NULL, 'rss_news', 'cnnindonesia_nasional',
 'https://www.cnnindonesia.com/nasional/rss', true),

-- Tribun Nasional (high volume)
(NULL, 'rss_news', 'tribunnews_nasional',
 'https://www.tribunnews.com/rss/nasional', true),

-- Tempo.co Nasional
(NULL, 'rss_news', 'tempo_nasional',
 'https://rss.tempo.co/nasional', true)

ON CONFLICT DO NOTHING;


-- ─────────────────────────────────────────────────────────────
-- BAGIAN 2: GOOGLE NEWS RSS PER TOKOH
-- entity_id = <uuid> → langsung tahu tokoh mana yg dimaksud
-- Berguna untuk tokoh yang namanya ambigu di NER
-- ─────────────────────────────────────────────────────────────
-- Google News RSS format:
-- https://news.google.com/rss/search?q=<query>&hl=id&gl=ID&ceid=ID:id
-- Tidak ada auth, tidak ada rate limit ketat untuk query sederhana.
-- Snippet description ~100-150 kata — cukup untuk IndoBERT.

DO $$
DECLARE
    v_id   UUID;
    -- Array of [canonical_name, google_query]
    v_map  TEXT[][] := ARRAY[
        ARRAY['Prabowo Subianto',          'prabowo+subianto+politik'],
        ARRAY['Gibran Rakabuming Raka',    'gibran+rakabuming+wakil+presiden'],
        ARRAY['Sri Mulyani Indrawati',     'sri+mulyani+keuangan+negara'],
        ARRAY['Agus Harimurti Yudhoyono',  'AHY+demokrat+menteri'],
        ARRAY['Airlangga Hartarto',        'airlangga+hartarto+golkar'],
        ARRAY['Muhaimin Iskandar',         'cak+imin+pkb+menko'],
        ARRAY['Zulkifli Hasan',            'zulhas+pan+mendag'],
        ARRAY['Erick Thohir',              'erick+thohir+bumn'],
        ARRAY['Megawati Soekarnoputri',    'megawati+pdip+ketua+umum'],
        ARRAY['Anies Baswedan',            'anies+baswedan+politik'],
        ARRAY['Ganjar Pranowo',            'ganjar+pranowo+pdip'],
        ARRAY['Puan Maharani',             'puan+maharani+dpr'],
        ARRAY['Ridwan Kamil',              'ridwan+kamil+rk+politik'],
        ARRAY['Khofifah Indar Parawansa',  'khofifah+gubernur+jatim']
    ];
    v_pair TEXT[];
    v_slug TEXT;
BEGIN
    FOREACH v_pair SLICE 1 IN ARRAY v_map LOOP
        SELECT id INTO v_id
        FROM political_entities
        WHERE canonical_name = v_pair[1]
        LIMIT 1;

        IF v_id IS NOT NULL THEN
            -- Buat config_name slug: 'gnews_prabowo_subianto'
            v_slug := 'gnews_' || lower(regexp_replace(v_pair[1], '\s+', '_', 'g'));

            INSERT INTO scraping_configs
                (entity_id, source_type, config_name, url, is_active)
            VALUES (
                v_id,
                'google_news_rss',
                v_slug,
                'https://news.google.com/rss/search?q='
                    || v_pair[2]
                    || '&hl=id&gl=ID&ceid=ID:id',
                true
            )
            ON CONFLICT DO NOTHING;

            RAISE NOTICE 'Inserted Google News RSS config for: %', v_pair[1];
        ELSE
            RAISE WARNING 'Entity not found: % — skip', v_pair[1];
        END IF;
    END LOOP;
END $$;


-- ─────────────────────────────────────────────────────────────
-- VERIFIKASI
-- ─────────────────────────────────────────────────────────────
SELECT
    sc.config_name,
    sc.source_type,
    pe.canonical_name AS entity,
    sc.is_active
FROM scraping_configs sc
LEFT JOIN political_entities pe ON pe.id = sc.entity_id
ORDER BY sc.source_type, sc.config_name;
