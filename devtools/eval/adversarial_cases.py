"""
adversarial_cases.py v3 — Expert-Grade Adversarial Test Suite
=============================================================
Kasus-kasus sulit (edge cases) berdasarkan jurnalisme politik Indonesia.
Menguji ketahanan model terhadap: Name Collision, Sarkasme, Negasi Ganda,
Code-Switching, Atribusi Kutipan, dan Logika Kondisional.

Setiap kasus dirancang untuk mematahkan asumsi model yang dilatih pada
dataset umum (review e-commerce/media sosial).
"""

ADVERSARIAL_CASES = [
    # ─── CATEGORY 1: NAME COLLISION (False Positive Historis) ───
    {
        "id": "NC01",
        "text": "Kapolri Jenderal Listyo Sigit Prabowo menyerahkan 6.000 bansos dalam rangka HUT Bhayangkara ke-80 di Markas Brimob.",
        "context": "Prabowo Subianto",
        "expected_relevant": False, "expected_label": None,
        "category": "name_collision", "difficulty": "Hard",
        "note": "Bug paling persisten: Kapolri (orang BEDA) vs Presiden Prabowo. Model harus nge-blok ini."
    },
    {
        "id": "NC02",
        "text": "Bupati Suryanto resmikan jembatan layang baru sepanjang 2 km di kabupaten Banyumas.",
        "context": "Surya Paloh",
        "expected_relevant": False, "expected_label": None,
        "category": "name_collision", "difficulty": "Medium",
        "note": "Nama depan 'Surya' umum, bisa false match Surya Paloh."
    },
    {
        "id": "NC03",
        "text": "Menteri BUMN Erick Thohir menggelar rapat evaluasi kinerja pertambangan.",
        "context": "Erick Thohir",
        "expected_relevant": True, "expected_label": "neutral",
        "category": "name_collision", "difficulty": "Easy",
        "note": "Control test: Nama unik, harusnya relevan dan netral (fakta kejadian)."
    },

    # ─── CATEGORY 2: SARCASM / IRONY ───
    {
        "id": "SR01",
        "text": "Hebat banget, Prabowo baru aja dilantik udah korupsi APBN triliunan. Memang layak jadi panutan.",
        "context": "Prabowo Subianto",
        "expected_relevant": True, "expected_label": "negative",
        "category": "sarcasm", "difficulty": "Extreme",
        "note": "Sarkasme eksplisit: Kata 'hebat', 'panutan' tapi konteks menyindir. Harus tertangkap negatif."
    },
    {
        "id": "SR02",
        "text": "Mantap, kebijakan baru ini berhasil membuat harga BBM naik berkali-kali. Pintar sekali pemerintah.",
        "context": "Joko Widodo",
        "expected_relevant": True, "expected_label": "negative",
        "category": "sarcasm", "difficulty": "Hard",
        "note": "Ironi tanpa menyebut tokoh langsung, tapi pakai 'pemerintah' (linked to context)."
    },

    # ─── CATEGORY 3: CODE-SWITCHING (Formal + Gaul + Asing) ───
    {
        "id": "CS01",
        "text": "Gibran nggak sopan banget sama senior, parah sih masa wakil presiden gitu kelakuannya.",
        "context": "Gibran Rakabuming Raka",
        "expected_relevant": True, "expected_label": "negative",
        "category": "code_switching", "difficulty": "Hard",
        "note": "Bahasa gaul: 'nggak sopan banget', 'parah sih' = negatif."
    },
    {
        "id": "CS02",
        "text": "Anies perform nya gokil sih, beneran worth it deh dipercaya lead Jakarta lagi.",
        "context": "Anies Baswedan",
        "expected_relevant": True, "expected_label": "positive",
        "category": "code_switching", "difficulty": "Medium",
        "note": "Campur Inggris: 'perform gokil', 'worth it' = positif."
    },

    # ─── CATEGORY 4: QUOTE ATTRIBUTION (Atribusi Kutipan) ───
    {
        "id": "QA01",
        "text": "Megawati dalam orasinya mengkritik keras pemerintahan saat ini. \"Jangan sampai rakyat dikibulin,\" tegasnya.",
        "context": "Megawati Soekarnoputri",
        "expected_relevant": True, "expected_label": "negative",
        "category": "quote_attribution", "difficulty": "Medium",
        "note": "Megawati mengkritik -> sentimen dia terhadap konteks = negatif."
    },
    {
        "id": "QA02",
        "text": "Meskipun Prabowo memuji kinerja Sri Mulyani, para pengamat tetap menilai kebijakan tersebut gagal.",
        "context": "Prabowo Subianto",
        "expected_relevant": True, "expected_label": "positive",
        "category": "quote_attribution", "difficulty": "Hard",
        "note": "Model harus fokus pada tokoh utama (Prabowo memuji), bukan opini pengamat di klausa belakang."
    },

    # ─── CATEGORY 5: MIXED / NUANCED SENTIMENT ───
    {
        "id": "MX01",
        "text": "Kinerja kabinet Prabowo dinilai cukup baik di bidang pertahanan, namun banyak kritik soal lambatnya program makan bergizi.",
        "context": "Prabowo Subianto",
        "expected_relevant": True, "expected_label": "neutral",
        "category": "mixed_sentiment", "difficulty": "Hard",
        "note": "Positif di satu sisi, negatif di sisi lain -> netral/ambigu."
    },
    {
        "id": "MX02",
        "text": "Walaupun Anies dipuji karena keberaniannya, langkahnya dinilai terlalu berisiko oleh mayoritas ekonom.",
        "context": "Anies Baswedan",
        "expected_relevant": True, "expected_label": "neutral",
        "category": "mixed_sentiment", "difficulty": "Hard",
        "note": "Dipuji tapi juga dikritik -> netral."
    },

    # ─── CATEGORY 6: TANGENTIAL MENTION (Bukan Topik Utama) ───
    {
        "id": "TM01",
        "text": "Bursa saham JCI ditutup menguat 1.2% dipicu sektor perbankan. Sebelumnya, Presiden Prabowo meresmikan proyek tol trans-Jawa.",
        "context": "Prabowo Subianto",
        "expected_relevant": True, "expected_label": "neutral",
        "category": "tangential_mention", "difficulty": "Medium",
        "note": "Prabowo disebut tapi artikel murni tentang keuangan/saham -> netral."
    },

    # ─── CATEGORY 7: NEGATION (Negasi Eksplisit & Ganda) ───
    {
        "id": "NG01",
        "text": "Prabowo menyatakan tidak setuju dengan kebijakan tarif baru yang dinilai membebani rakyat.",
        "context": "Prabowo Subianto",
        "expected_relevant": True, "expected_label": "negative",
        "category": "negation", "difficulty": "Medium",
        "note": "Negasi: 'tidak setuju' harusnya tertangkap sebagai negatif."
    },
    {
        "id": "NG02",
        "text": "Pernyataan Prabowo tersebut bukan tanpa dasar, beliau memiliki data lengkap untuk membuktikannya.",
        "context": "Prabowo Subianto",
        "expected_relevant": True, "expected_label": "positive",
        "category": "negation", "difficulty": "Extreme",
        "note": "Double negation: 'bukan tanpa dasar' = beralasan/positif. Model bodoh akan menebak negatif."
    },

    # ─── CATEGORY 8: TEMPORAL / ROLE REFERENCE (Referensi Waktu/Jabatan) ───
    {
        "id": "TR01",
        "text": "Kebijakan presiden sebelumnya dinilai jauh lebih baik dalam menangani krisis dibanding kabinet saat ini.",
        "context": "Joko Widodo",
        "expected_relevant": False, "expected_label": None,
        "category": "temporal_reference", "difficulty": "Hard",
        "note": "Konteks saat ini: Prabowo. 'Presiden sebelumnya' tidak boleh match Jokowi jika NLP fokus pada konteks saat ini."
    },
    {
        "id": "TR02",
        "text": "Mantan Gubernur DKI Jakarta tersebut kembali menjabat sebagai Menteri Investasi.",
        "context": "Anies Baswedan",
        "expected_relevant": False, "expected_label": None,
        "category": "temporal_reference", "difficulty": "Hard",
        "note": "Mantan Gubernur DKI = Anies? Tapi kalimat menyebut Menteri Investasi (Bahlil). Model harus nolak Anies."
    },

    # ─── CATEGORY 9: MULTI-ENTITY (Dua tokoh, sentimen berbeda) ───
    {
        "id": "ME01",
        "text": "Prabowo menyerang keras kebijakan Anies, namun Anies memuji kepemimpinan Prabowo.",
        "context": "Anies Baswedan",
        "expected_relevant": True, "expected_label": "positive",
        "category": "multi_entity", "difficulty": "Extreme",
        "note": "Fokus ke Anies: Anies memuji Prabowo -> positif. Model harus pintar memisahkan aktor."
    },
    {
        "id": "ME02",
        "text": "Megawati menolak tegas kerja sama dengan Prabowo, sementara Prabowo terlihat kecewa dengan sikap tersebut.",
        "context": "Prabowo Subianto",
        "expected_relevant": True, "expected_label": "negative",
        "category": "multi_entity", "difficulty": "Extreme",
        "note": "Fokus ke Prabowo: Prabowo kecewa -> negatif. Megawati menolak (negatif untuk Megawati)."
    },

    # ─── CATEGORY 10: PASSIVE VOICE ───
    {
        "id": "PV01",
        "text": "Kebijakan tersebut dikritik oleh pengamat karena dinilai merugikan tokoh tersebut.",
        "context": "Prabowo Subianto",
        "expected_relevant": True, "expected_label": "negative",
        "category": "passive_voice", "difficulty": "Hard",
        "note": "Passive voice: 'dikritik oleh pengamat' = sentimen negatif terhadap kebijakan tokoh."
    },

    # ─── CATEGORY 11: OBJECTIVE / NUMERICAL REPORTING (Harus Netral) ───
    {
        "id": "OB01",
        "text": "Sri Mulyani mengumumkan bahwa pertumbuhan ekonomi kuartal III mencapai 5.01 persen.",
        "context": "Sri Mulyani",
        "expected_relevant": True, "expected_label": "neutral",
        "category": "objective_reporting", "difficulty": "Medium",
        "note": "Fakta murni angka. Tidak ada kata sifat. Model sering salah menebak positif karena angka naik."
    },
    {
        "id": "OB02",
        "text": "Gibran Rakabuming Raka menghadiri rapat paripurna ke-12 di Gedung DPR/MPR.",
        "context": "Gibran Rakabuming Raka",
        "expected_relevant": True, "expected_label": "neutral",
        "category": "objective_reporting", "difficulty": "Easy",
        "note": "Kehadiran murni fakta, harus netral."
    },

    # ─── CATEGORY 12: CONDITIONAL SENTIMENT (Sentimen Kondisional) ───
    {
        "id": "CD01",
        "text": "Jika Prabowo tidak segera merevisi kebijakan impor ini, petani lokal akan terancam gulung tikar.",
        "context": "Prabowo Subianto",
        "expected_relevant": True, "expected_label": "negative",
        "category": "conditional_sentiment", "difficulty": "Extreme",
        "note": "Kondisional: 'Jika tidak... akan terancam'. Implikasi negatif untuk tokoh."
    },
    {
        "id": "CD02",
        "text": "Selama Anies konsisten dengan visi tersebut, Jakarta akan tetap menjadi kota yang kompetitif.",
        "context": "Anies Baswedan",
        "expected_relevant": True, "expected_label": "positive",
        "category": "conditional_sentiment", "difficulty": "Hard",
        "note": "Kondisional positif: 'Selama konsisten... akan kompetitif'."
    }
]

def get_cases_by_category(category: str = None) -> list:
    """Filter cases berdasarkan kategori."""
    if category is None:
        return ADVERSARIAL_CASES
    return [c for c in ADVERSARIAL_CASES if c["category"] == category]

def get_categories() -> list:
    """List semua kategori adversarial."""
    return sorted(set(c["category"] for c in ADVERSARIAL_CASES))