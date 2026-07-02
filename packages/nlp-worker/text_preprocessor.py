"""
text_preprocessor.py — ID-Political-Sentiment-Tracker
=======================================================
Preprocessing teks artikel Indonesia sebelum masuk ke IndoBERT.

IndoBERT tokenizer memang handle subword tokenization otomatis,
tapi TIDAK handle:
- Boilerplate jurnalistik Indonesia (header kota, attribution)
- URL, caption foto, tag iklan
- Teks duplikat / redundan dari RSS description
- Encoding noise dari HTML entities yang mungkin masih lolos

Dipanggil oleh drain_queue.py dan cli_test.py sebelum predict_gated().
Import: from text_preprocessor import prepare_text
"""

import re
import unicodedata


# ─────────────────────────────────────────────────────────────
# PATTERN BOILERPLATE JURNALISTIK INDONESIA
# ─────────────────────────────────────────────────────────────

# "JAKARTA, CNN Indonesia --" / "JAKARTA (ANTARA) -"
_DATELINE = re.compile(
    r'^[A-Z][A-Z\s,\.]+(?:,\s*)(?:CNN Indonesia|ANTARA|Kompas\.com|detikcom|Tempo\.co'
    r'|liputan6\.com|Republika\.co\.id|cnnindonesia\.com|[A-Z][a-z]+\.(?:com|co\.id|id))?'
    r'\s*[-–]\s*',
    re.IGNORECASE
)

# "Baca juga: ..." / "Lihat juga: ..." / "Artikel terkait: ..."
_READ_ALSO = re.compile(
    r'(?:Baca|Lihat|Simak|Cek|Artikel)\s+(?:juga|terkait|selengkapnya)\s*:\s*[^\n]{0,200}',
    re.IGNORECASE
)

# "(FOTO: Dok. Kementerian)" / "(Ilustrasi: Reuters)"
_PHOTO_CAP = re.compile(
    r'\((?:FOTO|GRAFIS|VIDEO|ILUSTRASI|Sumber|Dok|Photo)\s*[:\.]?[^)]{0,100}\)',
    re.IGNORECASE
)

# URL
_URL = re.compile(r'https?://\S+|www\.\S+')

# "ADVERTISEMENT" / "IKLAN"
_AD = re.compile(r'\b(?:ADVERTISEMENT|IKLAN|SPONSORED)\b', re.IGNORECASE)

# "Halaman 1 dari 2" / "Halaman berikutnya:"
_PAGINATION = re.compile(
    r'(?:Halaman\s+\d+\s+dari\s+\d+|Halaman berikutnya\s*:|Selanjutnya\s*>>)',
    re.IGNORECASE
)

# Simbol HTML yang masih lolos
_HTML_ENTITY = re.compile(r'&(?:#\d+|[a-zA-Z]+);')
_HTML_TAG    = re.compile(r'<[^>]+>')

# Tanda baca berulang ("!!!", "???", "...")
_PUNCT_REPEAT = re.compile(r'([!?.]){3,}')

# Whitespace berlebih
_WHITESPACE = re.compile(r'\s+')


# ─────────────────────────────────────────────────────────────
# NORMALISASI KARAKTER UNICODE
# ─────────────────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode: NFKC (kompose + kompatibilitas).
    Handles: ＡＢＣ → ABC, ½ → 1/2, fi ligature → fi, dst.
    """
    return unicodedata.normalize("NFKC", text)


# ─────────────────────────────────────────────────────────────
# MAIN CLEANING FUNCTION
# ─────────────────────────────────────────────────────────────

def clean_article(text: str) -> str:
    """
    Bersihkan teks artikel berita Indonesia dari boilerplate dan noise.
    Input: raw text dari raw_texts.text (sudah di-stripHTML sebelumnya oleh Edge Function)
    Output: teks bersih, siap untuk IndoBERT
    """
    if not text:
        return ""

    # 1. Unicode normalization
    text = normalize_unicode(text)

    # 2. HTML artifacts yang mungkin masih lolos
    text = _HTML_TAG.sub(" ", text)
    text = _HTML_ENTITY.sub(" ", text)

    # 3. Hapus URL
    text = _URL.sub("", text)

    # 4. Hapus dateline jurnalistik (paling awal, sebelum konten bermakna)
    text = _DATELINE.sub("", text, count=1)

    # 5. Hapus "Baca juga", caption foto, pagination, iklan
    text = _READ_ALSO.sub(" ", text)
    text = _PHOTO_CAP.sub(" ", text)
    text = _PAGINATION.sub(" ", text)
    text = _AD.sub("", text)

    # 6. Normalisasi tanda baca berulang
    text = _PUNCT_REPEAT.sub(r'\1\1', text)

    # 7. Collapse whitespace
    text = _WHITESPACE.sub(" ", text).strip()

    return text


def prepare_text(title: str | None, body: str | None, max_chars: int = 1500) -> str:
    """
    Gabungkan title + body, bersihkan, truncate untuk IndoBERT.

    Kenapa title didahulukan:
    - Judul artikel mengandung kata kunci utama dan nama tokoh
    - IndoBERT dengan max_length=256 akan truncate dari kanan
    - Pastikan judul selalu masuk dalam window

    max_chars=1500:
    - IndoBERT max 256 token
    - Rata-rata 1 token = 4-6 karakter untuk Indonesian text
    - 1500 chars ≈ 250-375 token → aman, tidak terbuang
    - Lebih dari 1500 karakter = di luar jangkauan model anyway
    """
    title_clean = clean_article(title or "")
    body_clean  = clean_article(body  or "")

    # Gabung: title + separator + body
    if title_clean and body_clean:
        combined = f"{title_clean}. {body_clean}"
    else:
        combined = title_clean or body_clean

    # Truncate di batas kata supaya tidak potong di tengah
    if len(combined) > max_chars:
        combined = combined[:max_chars].rsplit(" ", 1)[0]

    return combined.strip()


# ─────────────────────────────────────────────────────────────
# VALIDASI — apakah teks cukup bermakna untuk inference
# ─────────────────────────────────────────────────────────────

MIN_MEANINGFUL_LENGTH = 15   # karakter minimum setelah cleaning
MIN_WORD_COUNT        = 3    # kata minimum

def is_meaningful(text: str) -> bool:
    """
    Return True kalau teks cukup bermakna untuk diinference.
    Teks yang terlalu pendek setelah cleaning tidak informatif
    dan cenderung menurunkan kualitas distribusi.
    """
    if len(text) < MIN_MEANINGFUL_LENGTH:
        return False
    words = text.split()
    if len(words) < MIN_WORD_COUNT:
        return False
    # Cek apakah mayoritas karakter adalah alfanumerik (bukan symbol/angka saja)
    alnum_count = sum(1 for c in text if c.isalpha())
    if alnum_count < len(text) * 0.4:
        return False
    return True


# ─────────────────────────────────────────────────────────────
# CLI TEST — untuk verifikasi manual
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        {
            "title": "Airlangga Hartarto Tegaskan Pemerintah Dukung Penuh Investasi Asing",
            "body": "JAKARTA, CNN Indonesia -- Menteri Koordinator Bidang Perekonomian "
                    "Airlangga Hartarto menegaskan bahwa pemerintah berkomitmen untuk "
                    "mendukung penuh masuknya investasi asing ke Indonesia. Baca juga: "
                    "Prabowo Terbang ke Gorontalo. (FOTO: Dok. Kementerian Perekonomian) "
                    "https://cdn.cnn.com/img/test.jpg ADVERTISEMENT Selanjutnya >>",
        },
        {
            "title": "PSI: Masa Depan Politik Gibran Tidak Ditentukan Jokowi",
            "body": '<a href="https://news.google.com/rss/articles/CBMi...">PSI: Masa Depan '
                    'Politik Gibran Tidak Ditentukan Jokowi</a> <font color="#6f6f6f">Kompas.com</font>',
        },
        {
            "title": None,
            "body": "x",  # terlalu pendek
        },
    ]

    print("=" * 60)
    print("TEXT PREPROCESSOR TEST")
    print("=" * 60)
    for i, case in enumerate(test_cases, 1):
        result = prepare_text(case["title"], case["body"])
        print(f"\n[{i}] Input title : {(case['title'] or '')[:60]}")
        print(f"    Input body  : {(case['body']  or '')[:60]}...")
        print(f"    → Output    : {result[:120]}")
        print(f"    → Meaningful: {is_meaningful(result)}")
    print("=" * 60)
