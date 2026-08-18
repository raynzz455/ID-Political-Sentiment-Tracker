-- ============================================================
-- SEED: political_entities
-- Tokoh politik aktif Indonesia (per 2025-2026)
-- Jalankan SEKALI setelah schema_final_v2.sql
-- ============================================================

INSERT INTO political_entities
    (canonical_name, aliases, entity_type, party_affiliation, position)
VALUES

-- ── EKSEKUTIF ──────────────────────────────────────────────────
(
    'Prabowo Subianto',
    ARRAY['Prabowo','Pak Prabowo','Presiden Prabowo','Capres Prabowo',
          'Prabowo Subianto','Bapak Prabowo','Menhan Prabowo'],
    'president', 'Gerindra', 'Presiden Republik Indonesia'
),
(
    'Gibran Rakabuming Raka',
    ARRAY['Gibran','Wapres Gibran','Mas Gibran','Gibran Rakabuming',
          'Gibran Raka','Wakil Presiden Gibran'],
    'vp', 'PSI', 'Wakil Presiden Republik Indonesia'
),

-- ── MENTERI PRIORITAS ──────────────────────────────────────────
(
    'Sri Mulyani Indrawati',
    ARRAY['Sri Mulyani','Bu Sri Mulyani','Menkeu Sri Mulyani',
          'Ibu Keuangan','Sri Mulyani Indrawati'],
    'minister', 'Independen', 'Menteri Keuangan RI'
),
(
    'Agus Harimurti Yudhoyono',
    ARRAY['AHY','Agus Yudhoyono','Mas AHY','Menteri AHY',
          'Agus Harimurti','Ketum Demokrat AHY'],
    'minister', 'Demokrat', 'Menteri ATR/BPN'
),
(
    'Airlangga Hartarto',
    ARRAY['Airlangga','Pak Airlangga','Menko Airlangga',
          'Airlangga Hartarto','Ketum Golkar'],
    'minister', 'Golkar', 'Menko Perekonomian'
),
(
    'Muhaimin Iskandar',
    ARRAY['Cak Imin','Muhaimin','Gus Imin','Menko Muhaimin',
          'Muhaimin Iskandar','Ketum PKB'],
    'minister', 'PKB', 'Menko Pemberdayaan Masyarakat'
),
(
    'Zulkifli Hasan',
    ARRAY['Zulhas','Pak Zulhas','Zulkifli Hasan',
          'Mendag Zulhas','Ketum PAN'],
    'minister', 'PAN', 'Menteri Perdagangan'
),
(
    'Erick Thohir',
    ARRAY['Erick Thohir','Erick','Menteri BUMN Erick',
          'Pak Erick','Erick Thohir BUMN'],
    'minister', 'Independen', 'Menteri BUMN'
),
(
    'Budi Gunadi Sadikin',
    ARRAY['Budi Gunadi','Menkes Budi','Budi Sadikin',
          'Budi Gunadi Sadikin','Menteri Kesehatan Budi'],
    'minister', 'Independen', 'Menteri Kesehatan'
),
(
    'Yusril Ihza Mahendra',
    ARRAY['Yusril','Pak Yusril','Menko Yusril',
          'Yusril Ihza','Yusril Mahendra'],
    'minister', 'PBB', 'Menko Hukum HAM Imigrasi dan Pemasyarakatan'
),

-- ── KETUA PARTAI ────────────────────────────────────────────────
(
    'Megawati Soekarnoputri',
    ARRAY['Megawati','Bu Mega','Ibu Megawati','Mega',
          'Ketum PDIP','Ketua Umum PDIP Megawati'],
    'party', 'PDI-P', 'Ketua Umum PDI-Perjuangan'
),
(
    'Puan Maharani',
    ARRAY['Puan','Bu Puan','Puan Maharani','Ketua DPR Puan',
          'Mba Puan'],
    'legislator', 'PDI-P', 'Ketua DPR RI'
),

-- ── TOKOH OPOSISI / NON-KABINET ──────────────────────────────
(
    'Anies Baswedan',
    ARRAY['Anies','Pak Anies','Anies Baswedan','Mas Anies',
          'Gubernur Anies','Cagub Anies'],
    'other', 'Nasdem', 'Mantan Gubernur DKI Jakarta'
),
(
    'Ganjar Pranowo',
    ARRAY['Ganjar','Pak Ganjar','Mas Ganjar','Ganjar Pranowo',
          'Capres Ganjar','Gubernur Ganjar'],
    'other', 'PDI-P', 'Mantan Gubernur Jawa Tengah'
),
(
    'Susilo Bambang Yudhoyono',
    ARRAY['SBY','Pak SBY','Susilo Bambang Yudhoyono',
          'Presiden SBY','Ketum Demokrat SBY'],
    'other', 'Demokrat', 'Mantan Presiden RI ke-6 / Ketua Majelis Tinggi Demokrat'
),

-- ── GUBERNUR / KEPALA DAERAH STRATEGIS ────────────────────────
(
    'Ridwan Kamil',
    ARRAY['Ridwan Kamil','Kang Emil','Emil','Gubernur Emil',
          'RK','Cagub RK'],
    'governor', 'Golkar', 'Mantan Gubernur Jawa Barat'
),
(
    'Khofifah Indar Parawansa',
    ARRAY['Khofifah','Bu Khofifah','Gubernur Khofifah',
          'Khofifah Indar'],
    'governor', 'PKB', 'Gubernur Jawa Timur'
),
(
    'Pramono Anung',
    ARRAY['Pramono Anung','Pramono','Gubernur Pramono',
          'Pak Pramono'],
    'governor', 'PDI-P', 'Gubernur DKI Jakarta'
)

ON CONFLICT DO NOTHING;

-- Verifikasi
SELECT canonical_name, entity_type, party_affiliation
FROM political_entities
ORDER BY entity_type, canonical_name;
