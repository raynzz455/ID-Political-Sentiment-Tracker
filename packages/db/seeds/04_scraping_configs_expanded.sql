-- ============================================================
-- 04_scraping_configs_expanded.sql
-- Tambah Google News RSS untuk semua entitas baru
-- Jalankan SETELAH 03_entities_comprehensive.sql
-- ============================================================

DO $$
DECLARE
  v_id   UUID;
  v_slug TEXT;
  -- [canonical_name, google_query]
  v_map  TEXT[][] := ARRAY[
    -- Presiden & Wapres
    ARRAY['Joko Widodo',               'jokowi+joko+widodo+politik'],
    ARRAY['Susilo Bambang Yudhoyono',   'SBY+susilo+bambang+yudhoyono'],
    ARRAY['Jusuf Kalla',               'jusuf+kalla+JK+politik'],
    ARRAY['Ma''ruf Amin',              'maruf+amin+wapres+politik'],
    ARRAY['Boediono',                  'boediono+wapres+politik'],
    ARRAY['Abdurrahman Wahid',         'gus+dur+abdurrahman+wahid'],
    -- Kabinet Prabowo
    ARRAY['Prabowo Subianto',          'prabowo+subianto+presiden+politik'],
    ARRAY['Gibran Rakabuming Raka',    'gibran+rakabuming+wapres'],
    ARRAY['Sri Mulyani Indrawati',     'sri+mulyani+keuangan+negara'],
    ARRAY['Agus Harimurti Yudhoyono',  'AHY+agus+harimurti+yudhoyono'],
    ARRAY['Airlangga Hartarto',        'airlangga+hartarto+golkar'],
    ARRAY['Muhaimin Iskandar',         'cak+imin+muhaimin+iskandar'],
    ARRAY['Zulkifli Hasan',            'zulhas+zulkifli+hasan+pan'],
    ARRAY['Erick Thohir',              'erick+thohir+bumn'],
    ARRAY['Budi Gunadi Sadikin',       'budi+gunadi+sadikin+menkes'],
    ARRAY['Yusril Ihza Mahendra',      'yusril+ihza+mahendra+menko'],
    ARRAY['Bima Arya Sugiarto',        'bima+arya+wamendagri'],
    ARRAY['Sufmi Dasco Ahmad',         'sufmi+dasco+ahmad+ketua+dpr'],
    ARRAY['Bambang Soesatyo',          'bamsoet+bambang+soesatyo+mpr'],
    -- Partai & Legislatif
    ARRAY['Megawati Soekarnoputri',    'megawati+pdip+ketua+umum'],
    ARRAY['Puan Maharani',             'puan+maharani+pdip+dpr'],
    ARRAY['Hasto Kristiyanto',         'hasto+kristiyanto+sekjen+pdip'],
    ARRAY['Ahmad Syaikhu',             'ahmad+syaikhu+presiden+pks'],
    ARRAY['Surya Paloh',               'surya+paloh+nasdem'],
    ARRAY['Amien Rais',                'amien+rais+pan+politik'],
    -- Gubernur
    ARRAY['Anies Baswedan',            'anies+baswedan+politik'],
    ARRAY['Ridwan Kamil',              'ridwan+kamil+kang+emil+politik'],
    ARRAY['Ganjar Pranowo',            'ganjar+pranowo+pdip'],
    ARRAY['Khofifah Indar Parawansa',  'khofifah+gubernur+jatim'],
    ARRAY['Pramono Anung',             'pramono+anung+gubernur+dki'],
    ARRAY['Bobby Nasution',            'bobby+nasution+gubernur+sumut'],
    ARRAY['Dedi Mulyadi',              'dedi+mulyadi+gubernur+jabar'],
    -- Pengamat & Influencer
    ARRAY['Rocky Gerung',              'rocky+gerung+pengamat+politik'],
    ARRAY['Refly Harun',               'refly+harun+hukum+tata+negara'],
    ARRAY['Ferry Irwandi',             'ferry+irwandi+analis+politik'],
    ARRAY['Najwa Shihab',              'najwa+shihab+politik+wawancara'],
    ARRAY['Karni Ilyas',               'karni+ilyas+ILC+politik'],
    ARRAY['Rizal Ramli',               'rizal+ramli+ekonomi+politik'],
    ARRAY['Faisal Basri',              'faisal+basri+ekonom+politik'],
    ARRAY['Chatib Basri',              'chatib+basri+ekonom+menteri'],
    ARRAY['Ade Armando',               'ade+armando+pengamat+politik'],
    ARRAY['Budiman Sudjatmiko',        'budiman+sudjatmiko+politik'],
    -- Mantan Pejabat
    ARRAY['Mahfud MD',                 'mahfud+md+menko+polhukam'],
    ARRAY['Thomas Lembong',            'tom+lembong+menteri+perdagangan'],
    ARRAY['Wiranto',                   'wiranto+jenderal+menko'],
    ARRAY['Hatta Rajasa',              'hatta+rajasa+menko+pan'],
    ARRAY['Anas Urbaningrum',          'anas+urbaningrum+demokrat'],
    ARRAY['Setya Novanto',             'setya+novanto+setnov+golkar']
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

-- Verifikasi
SELECT
  source_type,
  COUNT(*) AS total,
  COUNT(*) FILTER (WHERE is_active) AS active
FROM scraping_configs
GROUP BY source_type;

SELECT COUNT(*) AS total_configs FROM scraping_configs;
