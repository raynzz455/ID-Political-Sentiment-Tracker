"""
adversarial_cases.py — Test Cases Sulit untuk Validasi Model
=============================================================
Kasus-kasus yang harus ditangani dengan benar oleh model production-grade.
Inspirasi dari bug historis project ini (Prabowo/Listyo Sigit, RK/kriminal, dll).

Tiap case punya:
- text: input artikel
- context: entity yang ditanyakan
- expected_relevant: apakah artikel benar-benar tentang entity itu?
- expected_label: sentimen yang benar (kalau relevant)
- category: jenis adversarial
- note: penjelasan kenapa ini sulit
"""

ADVERSARIAL_CASES = [
    # ─── CATEGORY 1: NAME COLLISION (false positive historis) ───
    {
        "text": "Kapolri Jenderal Listyo Sigit Prabowo menyerahkan 6.000 bansos "
                "dalam rangka HUT Bhayangkara ke-80 di Markas Brimob",
        "context": "Prabowo Subianto",
        "expected_relevant": False,
        "expected_label": None,
        "category": "name_collision",
        "note": "Bug paling persisten: Kapolri (orang BEDA) vs Presiden Prabowo",
    },
    {
        "text": "Polda Jatim berhasil mengungkap sindikat narkoba bermodal miliaran "
                "berkedok kantor pengiriman barang di Surabaya",
        "context": "Ridwan Kamil",
        "expected_relevant": False,
        "expected_label": None,
        "category": "name_collision",
        "note": "Alias 'RK' sering match kriminal/random RK lain",
    },
    {
        "text": "Bupati Suryanto resmikan jembatan layang baru sepanjang 2 km "
                "di kabupaten Banyumas",
        "context": "Surya Paloh",
        "expected_relevant": False,
        "expected_label": None,
        "category": "name_collision",
        "note": "Nama depan 'Surya' umum, bisa false match Surya Paloh",
    },

    # ─── CATEGORY 2: SARCASM / IRONY ───
    {
        "text": "Hebat banget, baru aja dilantik udah korupsi APBN triliunan. "
                "Memang layak jadi panutan bagi generasi muda Indonesia.",
        "context": None,
        "expected_relevant": True,
        "expected_label": "negative",
        "category": "sarcasm",
        "note": "Sarkasme: kata 'hebat', 'panutan' tapi sebenarnya negatif",
    },
    {
        "text": "Luar biasa kebijakannya, harga BBM naik berkali-kali, rakyat "
                "kecil makin sejahtera katanya.",
        "context": None,
        "expected_relevant": True,
        "expected_label": "negative",
        "category": "sarcasm",
        "note": "Ironi: 'luar biasa', 'sejahtera' tapi konteks menyindir",
    },

    # ─── CATEGORY 3: CODE-SWITCHING (formal + gaul) ───
    {
        "text": "Gibran nggak sopan banget sama senior, parah sih masa wakil "
                "presiden gitu kelakuannya ke老年 politisi.",
        "context": "Gibran Rakabuming Raka",
        "expected_relevant": True,
        "expected_label": "negative",
        "category": "code_switching",
        "note": "Campur formal/gaul/Cina: model harus tangkap sentimen negatif",
    },
    {
        "text": "Anies perform nya gokil sih, beneran worth it deh dipercaya "
                "lead Jakarta lagi.",
        "context": "Anies Baswedan",
        "expected_relevant": True,
        "expected_label": "positive",
        "category": "code_switching",
        "note": "Bahasa gaul: 'gokil', 'worth it' = positif",
    },

    # ─── CATEGORY 4: QUOTE ATTRIBUTION ───
    {
        "text": "Megawati dalam orasinya mengkritik keras pemerintahan saat ini. "
                "\"Jangan sampai rakyat terus dikibulin dengan janji manis,\" "
                "tegas ketua umum PDIP tersebut.",
        "context": "Megawati Soekarnoputri",
        "expected_relevant": True,
        "expected_label": "negative",
        "category": "quote_attribution",
        "note": "Megawoto mengkritik → sentimen dia terhadap konteks = negatif",
    },
    {
        "text": "Prabowo memuji kinerja Sri Mulyani dalam mengelola APBN. "
                "\"Beliau sangat kompeten,\" ujarnya.",
        "context": "Prabowo Subianto",
        "expected_relevant": True,
        "expected_label": "positive",
        "category": "quote_attribution",
        "note": "Prabowo memuji → positif terhadap konteksnya",
    },

    # ─── CATEGORY 5: MIXED / NUANCED SENTIMENT ───
    {
        "text": "Kinerja kabinet Prabowo dinilai cukup baik di bidang pertahanan, "
                "namun banyak kritik soal lambatnya program makan bergizi.",
        "context": "Prabowo Subianto",
        "expected_relevant": True,
        "expected_label": "neutral",
        "category": "mixed_sentiment",
        "note": "Positif di satu sisi, negatif di sisi lain → netral/ambigu",
    },
    {
        "text": "Ganjar akhirnya resmi dicalonkan kembali sebagai capreda, "
                "disambut antusias pengurus DPD Jateng.",
        "context": "Ganjar Pranowo",
        "expected_relevant": True,
        "expected_label": "positive",
        "category": "mixed_sentiment",
        "note": "Positif jelas: 'disambut antusias'",
    },

    # ─── CATEGORY 6: ENTITY MENTION BUT NOT MAIN TOPIC ───
    {
        "text": "Bursa saham JCI ditutup menguat 1.2% dipicu sektor perbankan. "
                "Sebelumnya, Presiden Prabowo meresmikan proyek tol trans-Jawa.",
        "context": "Prabowo Subianto",
        "expected_relevant": True,
        "expected_label": "neutral",
        "category": "tangential_mention",
        "note": "Prabowo disebut tapi bukan topik utama artikel keuangan",
    },
]


def get_cases_by_category(category: str = None) -> list:
    """Filter cases berdasarkan kategori."""
    if category is None:
        return ADVERSARIAL_CASES
    return [c for c in ADVERSARIAL_CASES if c["category"] == category]


def get_categories() -> list:
    """List semua kategori adversarial."""
    return sorted(set(c["category"] for c in ADVERSARIAL_CASES))
