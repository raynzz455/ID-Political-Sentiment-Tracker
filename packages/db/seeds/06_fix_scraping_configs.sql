-- ============================================================
-- 06_fix_scraping_configs.sql
-- Fix config_name yang terpotong + tambah configs untuk entitas baru
-- ============================================================
-- BUG: config_name kehilangan huruf pertama tiap kata
--   gnews_rabowo_ubianto → gnews_prabowo_subianto
--   gnews_rick_hohir → gnews_erick_thohir
--   dll
-- ============================================================

-- ============================================================
-- PART 1: FIX config_name yang terpotong (update existing)
-- ============================================================

DO $$
DECLARE
  r RECORD;
  v_new_slug TEXT;
  v_canonical TEXT;
BEGIN
  -- Loop semua configs yang punya entity_id
  FOR r IN
    SELECT sc.id, sc.config_name, sc.url, sc.entity_id, pe.canonical_name
    FROM scraping_configs sc
    LEFT JOIN political_entities pe ON sc.entity_id = pe.id
    WHERE sc.source_type = 'google_news_rss'
      AND sc.entity_id IS NOT NULL
      AND pe.canonical_name IS NOT NULL
  LOOP
    -- Generate slug yang BENAR dari canonical_name
    v_new_slug := 'gnews_' || lower(
      regexp_replace(
        regexp_replace(r.canonical_name, '\s+', '_', 'g'),
        '[^a-z0-9_]', '', 'g'
      )
    );

    -- Cek apakah config_name saat ini berbeda dari yang seharusnya
    IF r.config_name != v_new_slug THEN
      -- Update config_name ke yang benar
      UPDATE scraping_configs
      SET config_name = v_new_slug
      WHERE id = r.id;

      RAISE NOTICE 'FIX: % → %', r.config_name, v_new_slug;
    END IF;
  END LOOP;
END $$;

-- ============================================================
-- PART 2: Tambah scraping configs untuk entitas baru (dari 05_entities_expanded)
-- ============================================================

DO $$
DECLARE
  v_id   UUID;
  v_slug TEXT;
  v_map  TEXT[][] := ARRAY[
    -- Presiden & Wapres (yang belum ada config)
    ARRAY['Gibran Rakabuming Raka',     'gibran+rakabuming+wakil+presiden'],
    ARRAY['Megawati Soekarnoputri',     'megawati+pdip+ketua+umum'],
    ARRAY['Hamzah Haz',                 'hamzah+haz+wapres+politik'],
    ARRAY['Abdurrahman Wahid',          'gus+dur+abdurrahman+wahid'],
    ARRAY['Bacharuddin Jusuf Habibie',  'habibie+BJ+presiden+politik'],
    ARRAY['Try Sutrisno',               'try+sutrisno+jenderal+politik'],
    ARRAY['Soekarno',                   'soekarno+bung+karno+presiden'],
    ARRAY['Mohammad Hatta',             'bung+hatta+mohammad+hatta'],
    -- Kabinet Prabowo (yang belum ada)
    ARRAY['Hadi Tjahjanto',             'hadi+tjahjanto+menko+polhukam'],
    ARRAY['Zulkifli Hasan',             'zulhas+zulkifli+hasan+pan'],
    ARRAY['Budi Gunadi Sadikin',        'budi+gunadi+sadikin+menkes'],
    ARRAY['Sugiono',                    'sugiono+menlu+gerindra'],
    ARRAY['Andi Amran Sulaiman',        'amran+sulaiman+menteri+pertanian'],
    ARRAY['Budi Arie Setiadi',          'budi+arie+kominfo+menteri'],
    ARRAY['Dito Ariotedjo',             'dito+ariotedjo+menter+pemuda'],
    ARRAY['Pratikno',                   'pratikno+sesneg+menteri'],
    ARRAY['Nusron Wahid',               'nusron+wahid+menteri+agraria'],
    -- Tokoh Partai (yang belum ada)
    ARRAY['Yahya Cholil Staquf',        'yahya+cholil+staquf+PBNU'],
    ARRAY['Haedar Nashir',              'haedar+nashir+muhammadiyah'],
    ARRAY['Said Aqil Siroj',            'said+aqil+siroj+PBNU'],
    ARRAY['Din Syamsuddin',             'din+syamsuddin+muhammadiyah'],
    ARRAY['Anwar Abbas',                'anwar+abbas+muhammadiyah'],
    -- Pengamat & Komentator (yang belum ada)
    ARRAY['Bonar Hutahean',             'bonar+hutahean+komentator+politik'],
    ARRAY['Fadjroel Rakman',            'fadjroel+rakman+jubir+presiden'],
    ARRAY['Denni Nurjaman',             'denni+nurjaman+pengamat+politik'],
    ARRAY['Riefqi Muna',                'riefqi+muna+pengamat+politik'],
    ARRAY['Eep Saefulloh Fatah',        'eep+saefulloh+fatah+UGM+politik'],
    ARRAY['Burhanuddin Muhtadi',        'burhanuddin+muhtadi+indikator+politik'],
    -- Aktivis & Influencer
    ARRAY['Haris Azhar',                'haris+azhar+aktivis+HAM'],
    ARRAY['Fatia Maulidiyanti',         'fatia+maulidiyanti+aktivis+HAM'],
    ARRAY['Deddy Corbuzier',            'deddy+corbuzier+podcast+politik'],
    -- Akademisi
    ARRAY['Bhima Yudhistira',           'bhima+yudhistira+CORE+ekonom'],
    ARRAY['Enny Sri Hartati',           'enny+sri+hartati+ekonom+IEWPI'],
    ARRAY['Rhenald Kasali',             'rhenald+kasali+UI+pakar'],
    -- Bisnis-K Politik
    ARRAY['Hary Tanoesoedibjo',         'hary+tanoesoedibjo+HT+perindo'],
    ARRAY['Aburizal Bakrie',            'aburizal+bakrie+ical+golkar'],
    ARRAY['Rusdi Kirana',               'rusdi+kirana+PKB+bisnis'],
    -- Mantan Menteri/Pejabat
    ARRAY['Luhut Binsar Pandjaitan',    'luhut+pandjaitan+menko+marves'],
    ARRAY['Tito Karnavian',             'tito+karnavian+dagri+politik'],
    ARRAY['Yasonna Laoly',              'yasonna+laoly+menkumham'],
    ARRAY['Siti Nurbaya',               'siti+nurbaya+menlhk+lingkungan'],
    ARRAY['Retno Marsudi',              'retno+marsudi+menlu+diplomasi'],
    ARRAY['Erick Thohir',               'erick+thohir+bumn+menpora'],
    ARRAY['Ryamizard Ryacudu',          'ryamizard+ryacudu+menhan'],
    ARRAY['Tedjo Edhy Purdijatno',      'tedjo+edhy+menko+polhukam'],
    ARRAY['Moeldoko',                   'moeldoko+KSP+politik'],
    ARRAY['Gatot Nurmantyo',            'gatot+nurmantyo+TNI+politik'],
    ARRAY['Rachmat Gobel',              'rachmat+gobel+menkes+pdip'],
    -- Tokoh Lainnya
    ARRAY['Suharso Monoarfa',           'suharso+monoarfa+bappenas+PPP'],
    ARRAY['Muhammad Romahurmuziy',      'romahurmuziy+PPP+ketum'],
    ARRAY['Ahmad Riza Patria',          'ahmad+riza+patria+gerindra+wagub'],
    -- Gubernur (yang belum ada)
    ARRAY['Basuki Tjahaja Purnama',     'ahok+basuki+tjahaja+purnama'],
    ARRAY['Dedi Mulyadi',               'dedi+mulyadi+gubernur+jabar'],
    ARRAY['Pramono Anung',              'pramono+anung+gubernur+dki'],
    -- DPR (yang belum ada)
    ARRAY['Fadli Zon',                  'fadli+zon+gerindra+dpr'],
    ARRAY['Ahmad Doli Kurnia Tampubolon', 'doli+kurnia+tampubolon+golkar+dpr'],
    ARRAY['Bambang Wuryanto',           'bambang+wuryanto+pacul+pdip+dpr'],
    ARRAY['Maruarar Sirait',            'maruarar+sirait+pdip+dpr'],
    ARRAY['Utut Adianto',               'utut+adianto+pdip+dpr'],
    ARRAY['Sandiaga Salahudin Uno',     'sandiaga+uno+menparekraf+pan'],
    ARRAY['Trimedya Panjaitan',         'trimedya+panjaitan+pdip+dpr'],
    ARRAY['Zainudin Amali',             'zainudin+amali+golkar+menpora'],
    ARRAY['Yandri Susanto',             'yandri+susanto+pan+dpr']
  ];
  v_pair TEXT[];
BEGIN
  FOREACH v_pair SLICE 1 IN ARRAY v_map LOOP
    SELECT id INTO v_id
    FROM political_entities
    WHERE canonical_name = v_pair[1]
    LIMIT 1;

    IF v_id IS NOT NULL THEN
      v_slug := 'gnews_' || lower(
        regexp_replace(
          regexp_replace(v_pair[1], '\s+', '_', 'g'),
          '[^a-z0-9_]', '', 'g'
        )
      );

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
      ON CONFLICT (config_name) DO UPDATE SET
        entity_id  = EXCLUDED.entity_id,
        url        = EXCLUDED.url,
        is_active  = true;

      RAISE NOTICE 'Config: % → %', v_pair[1], v_slug;
    ELSE
      RAISE WARNING 'Entity tidak ditemukan: %', v_pair[1];
    END IF;
  END LOOP;
END $$;

-- ============================================================
-- PART 3: Verifikasi
-- ============================================================
SELECT
  source_type,
  COUNT(*) AS total,
  COUNT(*) FILTER (WHERE is_active) AS active,
  COUNT(*) FILTER (WHERE entity_id IS NOT NULL) AS with_entity
FROM scraping_configs
GROUP BY source_type;

SELECT COUNT(*) AS total_configs FROM scraping_configs;

-- Sample: cek config_name yang sudah benar (10 teratas)
SELECT config_name, url
FROM scraping_configs
WHERE source_type = 'google_news_rss'
  AND entity_id IS NOT NULL
ORDER BY config_name
LIMIT 10;
