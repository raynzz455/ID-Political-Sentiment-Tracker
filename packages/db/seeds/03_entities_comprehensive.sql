-- ============================================================
-- 03_entities_comprehensive.sql
-- Seed komprehensif tokoh politik Indonesia
-- Jalankan SETELAH migration 007_entity_expansion_schema.sql
--
-- Kategori:
-- A. Presiden & Wapres semua era yang masih relevan
-- B. Kabinet Prabowo aktif (2024-2029)
-- C. Ketua & tokoh partai
-- D. Gubernur & kepala daerah strategis
-- E. Pengamat, influencer & tokoh media politik
-- F. Mantan pejabat yang masih diperbincangkan
-- ============================================================

INSERT INTO political_entities
  (canonical_name, aliases, entity_type, party_affiliation,
   position, era, is_active, birth_year, wikipedia_id_url)
VALUES

-- ════════════════════════════════════════════
-- A. PRESIDEN & WAPRES
-- ════════════════════════════════════════════

('Prabowo Subianto',
 ARRAY['Prabowo Subianto','Pak Prabowo','Presiden Prabowo',
       'Capres Prabowo','Bapak Prabowo','Menhan Prabowo'],
 'president','Gerindra','Presiden RI ke-8',
 ARRAY['Era Prabowo'],true,1951,
 'https://id.wikipedia.org/wiki/Prabowo_Subianto'),

('Gibran Rakabuming Raka',
 ARRAY['Gibran','Wapres Gibran','Mas Gibran',
       'Gibran Rakabuming','Wakil Presiden Gibran'],
 'vp','PSI','Wakil Presiden RI',
 ARRAY['Era Prabowo'],true,1987,
 'https://id.wikipedia.org/wiki/Gibran_Rakabuming_Raka'),

('Joko Widodo',
 ARRAY['Jokowi','Pak Jokowi','Presiden Jokowi','Bapak Jokowi',
       'Joko Widodo','Mantan Presiden Jokowi'],
 'former_official','PDI-P','Presiden RI ke-7 (2014-2024)',
 ARRAY['Era Jokowi'],false,1961,
 'https://id.wikipedia.org/wiki/Joko_Widodo'),

('Susilo Bambang Yudhoyono',
 ARRAY['SBY','Pak SBY','Presiden SBY',
       'Susilo Bambang Yudhoyono','Mantan Presiden SBY'],
 'former_official','Demokrat','Presiden RI ke-6 (2004-2014)',
 ARRAY['Era SBY'],false,1949,
 'https://id.wikipedia.org/wiki/Susilo_Bambang_Yudhoyono'),

('Megawati Soekarnoputri',
 ARRAY['Megawati','Bu Megawati','Ibu Megawati',
       'Megawati Soekarnoputri','Ketum PDIP','Bu Mega'],
 'party','PDI-P','Ketua Umum PDI-P / Presiden RI ke-5',
 ARRAY['Reformasi','Era SBY','Era Jokowi','Era Prabowo'],true,1947,
 'https://id.wikipedia.org/wiki/Megawati_Soekarnoputri'),

('Jusuf Kalla',
 ARRAY['JK','Pak JK','Jusuf Kalla','Wapres JK',
       'Muhammad Jusuf Kalla'],
 'former_official','Golkar','Wapres RI ke-10 dan ke-12',
 ARRAY['Era SBY','Era Jokowi'],false,1942,
 'https://id.wikipedia.org/wiki/Jusuf_Kalla'),

('Ma''ruf Amin',
 ARRAY['Ma''ruf Amin','Pak Ma''ruf','Wapres Ma''ruf',
       'KH Ma''ruf Amin','Kyai Ma''ruf'],
 'former_official','PKB','Wapres RI ke-13 (2019-2024)',
 ARRAY['Era Jokowi'],false,1943,
 'https://id.wikipedia.org/wiki/Ma%27ruf_Amin'),

('Boediono',
 ARRAY['Boediono','Pak Boediono','Wapres Boediono'],
 'former_official','Independen','Wapres RI ke-11 (2009-2014)',
 ARRAY['Era SBY'],false,1943,
 'https://id.wikipedia.org/wiki/Boediono'),

('Abdurrahman Wahid',
 ARRAY['Gus Dur','KH Abdurrahman Wahid','Gusdur',
       'Presiden Gus Dur'],
 'former_official','PKB','Presiden RI ke-4 (1999-2001)',
 ARRAY['Reformasi'],false,1940,
 'https://id.wikipedia.org/wiki/Abdurrahman_Wahid'),

-- ════════════════════════════════════════════
-- B. KABINET PRABOWO AKTIF (2024-2029)
-- ════════════════════════════════════════════

('Sri Mulyani Indrawati',
 ARRAY['Sri Mulyani','Bu Sri Mulyani','Menkeu Sri Mulyani',
       'Menteri Keuangan Sri Mulyani'],
 'minister','Independen','Menteri Keuangan RI',
 ARRAY['Era Jokowi','Era Prabowo'],true,1962,
 'https://id.wikipedia.org/wiki/Sri_Mulyani_Indrawati'),

('Agus Harimurti Yudhoyono',
 ARRAY['AHY','Mas AHY','Menteri AHY','Agus Harimurti',
       'Agus Yudhoyono','Ketum Demokrat AHY'],
 'minister','Demokrat','Menteri ATR/BPN',
 ARRAY['Era Prabowo'],true,1978,
 'https://id.wikipedia.org/wiki/Agus_Harimurti_Yudhoyono'),

('Airlangga Hartarto',
 ARRAY['Airlangga','Pak Airlangga','Menko Airlangga',
       'Airlangga Hartarto','Ketum Golkar'],
 'minister','Golkar','Menko Perekonomian',
 ARRAY['Era Jokowi','Era Prabowo'],true,1962,
 'https://id.wikipedia.org/wiki/Airlangga_Hartarto'),

('Muhaimin Iskandar',
 ARRAY['Cak Imin','Muhaimin Iskandar','Gus Imin',
       'Menko Muhaimin','Ketum PKB'],
 'minister','PKB','Menko Pemberdayaan Masyarakat',
 ARRAY['Era Jokowi','Era Prabowo'],true,1966,
 'https://id.wikipedia.org/wiki/Muhaimin_Iskandar'),

('Zulkifli Hasan',
 ARRAY['Zulhas','Pak Zulhas','Zulkifli Hasan',
       'Mendag Zulhas','Ketum PAN'],
 'minister','PAN','Menteri Perdagangan',
 ARRAY['Era Jokowi','Era Prabowo'],true,1962,
 'https://id.wikipedia.org/wiki/Zulkifli_Hasan'),

('Erick Thohir',
 ARRAY['Erick Thohir','Pak Erick','Menteri BUMN Erick'],
 'minister','Independen','Menteri BUMN',
 ARRAY['Era Jokowi','Era Prabowo'],true,1970,
 'https://id.wikipedia.org/wiki/Erick_Thohir'),

('Budi Gunadi Sadikin',
 ARRAY['Budi Gunadi','Menkes Budi','Budi Gunadi Sadikin',
       'Menteri Kesehatan Budi'],
 'minister','Independen','Menteri Kesehatan',
 ARRAY['Era Jokowi','Era Prabowo'],true,1964,
 'https://id.wikipedia.org/wiki/Budi_Gunadi_Sadikin'),

('Yusril Ihza Mahendra',
 ARRAY['Yusril','Pak Yusril','Menko Yusril',
       'Yusril Ihza','Yusril Mahendra'],
 'minister','PBB','Menko Hukum HAM',
 ARRAY['Reformasi','Era Prabowo'],true,1956,
 'https://id.wikipedia.org/wiki/Yusril_Ihza_Mahendra'),

('Bima Arya Sugiarto',
 ARRAY['Bima Arya','Wamendagri Bima','Bima Arya Sugiarto'],
 'minister','PAN','Wakil Menteri Dalam Negeri',
 ARRAY['Era Prabowo'],true,1972,
 'https://id.wikipedia.org/wiki/Bima_Arya_Sugiarto'),

('Sufmi Dasco Ahmad',
 ARRAY['Sufmi Dasco','Dasco','Pak Dasco',
       'Ketua DPR Dasco'],
 'legislator','Gerindra','Ketua DPR RI',
 ARRAY['Era Prabowo'],true,1969,
 'https://id.wikipedia.org/wiki/Sufmi_Dasco_Ahmad'),

('Bambang Soesatyo',
 ARRAY['Bamsoet','Bambang Soesatyo','Ketua MPR Bamsoet'],
 'legislator','Golkar','Ketua MPR RI',
 ARRAY['Era Jokowi','Era Prabowo'],true,1962,
 'https://id.wikipedia.org/wiki/Bambang_Soesatyo'),

-- ════════════════════════════════════════════
-- C. KETUA & TOKOH PARTAI
-- ════════════════════════════════════════════

('Puan Maharani',
 ARRAY['Puan Maharani','Bu Puan','Ketua DPR Puan',
       'Mba Puan Maharani'],
 'legislator','PDI-P','Ketua DPR RI periode 2019-2024',
 ARRAY['Era Jokowi','Era Prabowo'],true,1973,
 'https://id.wikipedia.org/wiki/Puan_Maharani'),

('Hasto Kristiyanto',
 ARRAY['Hasto','Pak Hasto','Sekjen PDIP Hasto',
       'Hasto Kristiyanto'],
 'party_official','PDI-P','Sekretaris Jenderal PDI-P',
 ARRAY['Era Jokowi','Era Prabowo'],true,1966,
 'https://id.wikipedia.org/wiki/Hasto_Kristiyanto'),

('Ahmad Syaikhu',
 ARRAY['Ahmad Syaikhu','Syaikhu','Presiden PKS',
       'Ketum PKS'],
 'party_official','PKS','Presiden PKS',
 ARRAY['Era Prabowo'],true,1965,
 'https://id.wikipedia.org/wiki/Ahmad_Syaikhu'),

('Surya Paloh',
 ARRAY['Surya Paloh','Pak Surya','Ketum Nasdem',
       'Surya Paloh Nasdem'],
 'party_official','Nasdem','Ketua Umum Nasdem',
 ARRAY['Era Jokowi','Era Prabowo'],true,1951,
 'https://id.wikipedia.org/wiki/Surya_Paloh'),

('Amien Rais',
 ARRAY['Amien Rais','Pak Amien','Prof Amien Rais'],
 'former_official','PAN','Ketua MPR (1999-2004) / Pendiri PAN',
 ARRAY['Reformasi','Era SBY'],false,1944,
 'https://id.wikipedia.org/wiki/Amien_Rais'),

-- ════════════════════════════════════════════
-- D. GUBERNUR & KEPALA DAERAH
-- ════════════════════════════════════════════

('Anies Baswedan',
 ARRAY['Anies','Pak Anies','Anies Baswedan',
       'Mas Anies','Gubernur Anies','Cagub Anies'],
 'former_official','Nasdem','Mantan Gubernur DKI Jakarta',
 ARRAY['Era Jokowi','Era Prabowo'],true,1969,
 'https://id.wikipedia.org/wiki/Anies_Baswedan'),

('Ridwan Kamil',
 ARRAY['Ridwan Kamil','Kang Emil','Gubernur Ridwan Kamil',
       'Gubernur Ridwan'],
 'former_official','Golkar','Mantan Gubernur Jawa Barat',
 ARRAY['Era Jokowi','Era Prabowo'],true,1971,
 'https://id.wikipedia.org/wiki/Ridwan_Kamil'),

('Ganjar Pranowo',
 ARRAY['Ganjar','Pak Ganjar','Mas Ganjar','Ganjar Pranowo',
       'Capres Ganjar'],
 'former_official','PDI-P','Mantan Gubernur Jawa Tengah',
 ARRAY['Era Jokowi','Era Prabowo'],true,1968,
 'https://id.wikipedia.org/wiki/Ganjar_Pranowo'),

('Khofifah Indar Parawansa',
 ARRAY['Khofifah','Bu Khofifah','Gubernur Khofifah',
       'Khofifah Indar'],
 'governor','PKB','Gubernur Jawa Timur',
 ARRAY['Era Jokowi','Era Prabowo'],true,1965,
 'https://id.wikipedia.org/wiki/Khofifah_Indar_Parawansa'),

('Pramono Anung',
 ARRAY['Pramono Anung','Pramono','Gubernur Pramono',
       'Pak Pramono Anung'],
 'governor','PDI-P','Gubernur DKI Jakarta',
 ARRAY['Era Prabowo'],true,1963,
 'https://id.wikipedia.org/wiki/Pramono_Anung'),

('Bobby Nasution',
 ARRAY['Bobby Nasution','Bobby','Gubernur Bobby',
       'Muhammad Bobby Afif Nasution'],
 'governor','Gerindra','Gubernur Sumatera Utara',
 ARRAY['Era Prabowo'],true,1991,
 'https://id.wikipedia.org/wiki/Bobby_Nasution'),

('Dedi Mulyadi',
 ARRAY['Dedi Mulyadi','Kang Dedi','Gubernur Dedi Mulyadi',
       'Dedi Mulyadi Jabar'],
 'governor','Golkar','Gubernur Jawa Barat',
 ARRAY['Era Prabowo'],true,1971,
 'https://id.wikipedia.org/wiki/Dedi_Mulyadi'),

-- ════════════════════════════════════════════
-- E. PENGAMAT, INFLUENCER & TOKOH MEDIA POLITIK
-- ════════════════════════════════════════════

('Rocky Gerung',
 ARRAY['Rocky Gerung','Pak Rocky','Rocky',
       'Filosof Rocky Gerung'],
 'commentator',NULL,'Filsuf / Pengamat Politik',
 ARRAY['Era Jokowi','Era Prabowo'],true,1959,
 'https://id.wikipedia.org/wiki/Rocky_Gerung'),

('Refly Harun',
 ARRAY['Refly Harun','Pak Refly','Refly',
       'Ahli Hukum Refly Harun'],
 'commentator',NULL,'Pakar Hukum Tata Negara',
 ARRAY['Era Jokowi','Era Prabowo'],true,1971,
 'https://id.wikipedia.org/wiki/Refly_Harun'),

('Ferry Irwandi',
 ARRAY['Ferry Irwandi','Pak Ferry','Ferry Irwandi Analis'],
 'commentator',NULL,'Analis Politik / Content Creator',
 ARRAY['Era Jokowi','Era Prabowo'],true,NULL,
 NULL),

('Najwa Shihab',
 ARRAY['Najwa Shihab','Mbak Nana','Najwa','Nana Shihab'],
 'journalist',NULL,'Jurnalis / Presenter',
 ARRAY['Era SBY','Era Jokowi','Era Prabowo'],true,1977,
 'https://id.wikipedia.org/wiki/Najwa_Shihab'),

('Karni Ilyas',
 ARRAY['Karni Ilyas','Pak Karni','Presenter ILC'],
 'journalist',NULL,'Jurnalis / Presenter ILC',
 ARRAY['Era SBY','Era Jokowi','Era Prabowo'],true,1952,
 'https://id.wikipedia.org/wiki/Karni_Ilyas'),

('Rizal Ramli',
 ARRAY['Rizal Ramli','Pak Rizal','Dr Rizal Ramli',
       'Menko Rizal Ramli'],
 'commentator',NULL,'Ekonom / Mantan Menko Ekonomi',
 ARRAY['Reformasi','Era Jokowi','Era Prabowo'],true,1954,
 'https://id.wikipedia.org/wiki/Rizal_Ramli'),

('Faisal Basri',
 ARRAY['Faisal Basri','Pak Faisal','Ekonom Faisal Basri',
       'Almarhum Faisal Basri'],
 'commentator',NULL,'Ekonom (almarhum 2024)',
 ARRAY['Reformasi','Era SBY','Era Jokowi'],false,1959,
 'https://id.wikipedia.org/wiki/Faisal_Basri'),

('Chatib Basri',
 ARRAY['Chatib Basri','Muhamad Chatib Basri','Menkeu Chatib'],
 'commentator','Independen','Ekonom / Mantan Menteri Keuangan',
 ARRAY['Era SBY','Era Jokowi','Era Prabowo'],true,1967,
 'https://id.wikipedia.org/wiki/Chatib_Basri'),

('Ade Armando',
 ARRAY['Ade Armando','Pak Ade Armando','Akademisi Ade'],
 'commentator','PSI','Akademisi / Pengamat Politik',
 ARRAY['Era Jokowi','Era Prabowo'],true,1965,
 'https://id.wikipedia.org/wiki/Ade_Armando'),

('Budiman Sudjatmiko',
 ARRAY['Budiman Sudjatmiko','Budiman','Pak Budiman'],
 'commentator','Independen','Aktivis / Politisi',
 ARRAY['Reformasi','Era SBY','Era Jokowi','Era Prabowo'],true,1970,
 'https://id.wikipedia.org/wiki/Budiman_Sudjatmiko'),

-- ════════════════════════════════════════════
-- F. MANTAN PEJABAT YANG MASIH DIPERBINCANGKAN
-- ════════════════════════════════════════════

('Mahfud MD',
 ARRAY['Mahfud MD','Pak Mahfud','Prof Mahfud',
       'Menko Mahfud','Mahfud Mohammad'],
 'former_minister','PKB','Mantan Menko Polhukam',
 ARRAY['Era Jokowi','Era Prabowo'],true,1957,
 'https://id.wikipedia.org/wiki/Mahfud_MD'),

('Thomas Lembong',
 ARRAY['Tom Lembong','Thomas Lembong','Pak Tom Lembong'],
 'former_minister','Independen','Mantan Menteri Perdagangan',
 ARRAY['Era Jokowi'],false,1971,
 'https://id.wikipedia.org/wiki/Thomas_Lembong'),

('Wiranto',
 ARRAY['Wiranto','Jenderal Wiranto','Menko Wiranto'],
 'former_official','Hanura','Mantan Panglima TNI / Menko Polhukam',
 ARRAY['Orde Baru','Reformasi','Era SBY','Era Jokowi'],false,1947,
 'https://id.wikipedia.org/wiki/Wiranto'),

('Hatta Rajasa',
 ARRAY['Hatta Rajasa','Pak Hatta','Menko Hatta'],
 'former_minister','PAN','Mantan Menko Perekonomian',
 ARRAY['Era SBY','Era Jokowi'],false,1953,
 'https://id.wikipedia.org/wiki/Hatta_Rajasa'),

('Anas Urbaningrum',
 ARRAY['Anas Urbaningrum','Pak Anas','Mantan Ketum Demokrat'],
 'former_official','Demokrat','Mantan Ketua Umum Demokrat',
 ARRAY['Era SBY','Era Jokowi'],false,1969,
 'https://id.wikipedia.org/wiki/Anas_Urbaningrum'),

('Setya Novanto',
 ARRAY['Setya Novanto','Pak Setnov','Setnov'],
 'former_official','Golkar','Mantan Ketua DPR RI',
 ARRAY['Era Jokowi'],false,1954,
 'https://id.wikipedia.org/wiki/Setya_Novanto')

ON CONFLICT (canonical_name) DO UPDATE SET
  aliases          = EXCLUDED.aliases,
  entity_type      = EXCLUDED.entity_type,
  position         = EXCLUDED.position,
  era              = EXCLUDED.era,
  birth_year       = EXCLUDED.birth_year,
  wikipedia_id_url = EXCLUDED.wikipedia_id_url,
  is_active        = EXCLUDED.is_active;

-- Verifikasi
SELECT entity_type, COUNT(*) AS total
FROM political_entities
GROUP BY entity_type ORDER BY total DESC;

SELECT COUNT(*) AS grand_total FROM political_entities;
