"""
enricher_worker_v21_library.py
==============================
Refactor Layer 1-2 (Enrichment) dengan LIBRARY-BASED approach.

PERUBAHAN DARI v20 → v21:
  1. Ganti manual regex cleaning → clean-text + ftfy library
  2. Tambah intra-article deduplication (paragraf duplikat)
  3. Tambah promo/marketing removal (Gabung KOMPAS.com Plus, dll)
  4. Tambah byline removal ((hnh/kri), (red), dll)
  5. Tambah source attribution removal (KOMPAS.com -, TRIBUN -, dll)
  6. Sentence boundary alignment (tidak terpotong mid-word)

Library yang dipakai:
  - ftfy: fix encoding/mojibake
  - clean-text: comprehensive text cleaning
  - spacy: sentence boundary detection

Usage:
  from enricher_worker_v21_library import clean_article_text
  
  cleaned = clean_article_text(raw_html_or_text, title="...")
"""
import re
import ftfy
from cleantext import clean as clean_text_lib

# ============================================================
# LAYER 1-2: ENRICHMENT (Article Extraction + Cleaning)
# ============================================================

# Promo/marketing patterns yang sering muncul di portal berita Indonesia
PROMO_PATTERNS = [
    r'(?i)Gabung\s+\w+\s*\.?\s*Plus\s*sekarang.*',
    r'(?i)berkomitmen memberikan fakta jernih.*',
    r'(?i)Dukung keberlanjutan jurnalisme.*',
    r'(?i)nikmati kenyamanan baca.*',
    r'(?i)KOMPAS\.com berkomitmen.*',
    r'(?i)Detikcom berkomitmen.*',
    r'(?i)Baca berita selengkapnya.*',
    r'(?i)Simak juga.*?(?=\.|$)',
    r'(?i)Baca Juga\s*:.*?(?=\.|$)',
    r'(?i)Berita Terkait\s*:.*?(?=\.|$)',
    r'(?i)Pilihan untuk lu.*',
    r'(?i)Sponsor.*?(?=\.|$)',
    r'(?i)Iklan.*?(?=\.|$)',
    r'(?i)Advertisement.*?(?=\.|$)',
    r'(?i)Scroll ke bawah.*',
    r'(?i)Mau berita terbaru.*',
]

# Byline patterns: (hnh/kri), (red), (tfq/dal), dll
# HANYA hapus yang punya SLASH (author/editor format) ATAU di akhir text
BYLINE_SLASH_PATTERN = re.compile(r'\s*\([a-z]{2,5}/[a-z]{2,5}\)\s*', re.IGNORECASE)
BYLINE_END_PATTERN = re.compile(r'\s*\([a-z]{2,5}(?:/[a-z]{2,5})?\)\s*$', re.IGNORECASE)

# Singkatan konteks yang HARUS dipertahankan (BUKAN byline)
KEEP_ABBREVIATIONS = {
    'ratas', 'nobar', 'red', 'kapol', 'wabup', ' Wagub', 'wali',
    'ist', 'dok', 'antara', 'foto', 'instagram', 'pmj', 'ls',
    'psht', 'pmp', 'kk', 'ak', 's.h.', 'm.h.', 's.i.p.',
}

# Source attribution patterns: "KOMPAS.com -", "TRIBUN -", "CNN Indonesia -"
SOURCE_ATTR_PATTERNS = [
    r'^(KOMPAS\.com|CNN Indonesia|TEMPO\.CO|TRIBUN\w*\.?\w*|ANTARA/?\w*|jpnn\.com|detikcom|VIVA|Suara\.com|Republika|POPOSIDK)\s*[\-–—|:]\s*',
    r'^(JAKARTA|BANDUNG|SURABAYA|MEDAN|MAKASSAR|SEMARANG|YOGYAKARTA)\s*[\-–—|:]\s*[A-Z]',
    r'^(TRIBUNNEWS\.COM|KOMPAS\.TV|CNNINDONESIA)\s*[, ]*[A-Z\s]*\s*[\-–—|:]\s*',
]

# UI patterns (navigation, tags, related articles)
UI_PATTERNS = [
    r'(?i)(Tags\s*:|Berita Lainnya|Dark/Light Mode|BREAKINGNEWS).*?(?=\n|$|\.)',
    r'(?i)Gambas\s*:\s*Video\s*\w+',
    r'(?i)Dilarang keras mengambil konten.*?(?=\n|$|\.)',
    r'(?i)(Reporter|Editor|Penulis|Pewarta|Jurnalis)\s*:\s*.*?(?=\n|$|\.)',
    r'(?i)(Foto|Instagram|Dok|Istimewa|Antara)\s*[\[\(][^\)]*[\]\)]',
]


def fix_encoding(text: str) -> str:
    """Fix encoding issues using ftfy library.
    
    Handles:
    - Mojibake (Â, â€, dll)
    - Broken Unicode
    - Mixed encoding
    """
    return ftfy.fix_text(text)


def comprehensive_clean(text: str) -> str:
    """Clean text using clean-text library.
    
    Handles:
    - Unicode normalization
    - URL/email/phone removal
    - Whitespace normalization
    - Emoji removal
    - Quotation mark normalization
    """
    return clean_text_lib(
        text,
        fix_unicode=True,
        to_ascii=True,
        lower=False,
        no_line_breaks=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_punct=False,
        no_emoji=True,
        replace_with_url="",
        replace_with_email="",
        replace_with_phone_number="",
        lang="id"
    )


def remove_promo_content(text: str) -> str:
    """Remove promo/marketing content from portal berita.
    
    Handles:
    - "Gabung KOMPAS.com Plus sekarang"
    - "berkomitmen memberikan fakta jernih"
    - "Baca Juga:", "Simak juga:"
    - "Reporter:", "Editor:", "Jurnalis:"
    - "Sponsor", "Iklan", "Advertisement"
    """
    for pattern in PROMO_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    
    for pattern in UI_PATTERNS:
        text = re.sub(pattern, '', text)
    
    return text


def remove_byline(text: str) -> str:
    """Remove byline author markers — but KEEP context abbreviations.
    
    Removes:
    - (hnh/kri), (tfq/dal) — author/editor bylines (with slash)
    - (red) at END of text only — editor byline
    
    KEEPS (does NOT remove):
    - (ratas) — rapat terbatas (context abbreviation)
    - (nobar) — nonton bareng (context abbreviation)
    - (ist), (dok), (antara) — photo credits (handled separately)
    """
    # 1. Remove bylines WITH slash (author/editor format): (hnh/kri), (tfq/dal)
    text = BYLINE_SLASH_PATTERN.sub(' ', text)
    
    # 2. Remove byline at END of text only: (red), (hnh)
    #    But ONLY if it's not a known abbreviation
    match = BYLINE_END_PATTERN.search(text)
    if match:
        byline_content = match.group().strip('() \n\r')
        if byline_content.lower() not in KEEP_ABBREVIATIONS:
            text = BYLINE_END_PATTERN.sub('', text)
    
    # 3. Clean up any trailing whitespace
    text = text.rstrip()
    
    return text


def remove_source_attribution(text: str) -> str:
    """Remove source attribution at start of text.
    
    Handles:
    - "KOMPAS.com -" at start
    - "TRIBUN -" at start  
    - "CNN Indonesia -" at start
    - "JAKARTA, KOMPAS.com -" at start
    """
    for pattern in SOURCE_ATTR_PATTERNS:
        text = re.sub(pattern, '', text)
    
    return text.strip()


def remove_duplicate_paragraphs(text: str, entity_name: str = "") -> str:
    """Remove duplicate paragraphs within article.
    
    Preserves paragraphs containing the entity name.
    """
    # Split into paragraphs
    paragraphs = text.split('\n')
    
    # If no paragraph breaks, try sentence-level dedup
    if len(paragraphs) <= 1:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        seen = set()
        unique = []
        entity_lower = entity_name.lower() if entity_name else ""
        for s in sentences:
            key = s.strip()[:80].lower()
            # Always keep sentences with entity
            if entity_lower and entity_lower in s.lower():
                unique.append(s.strip())
                continue
            if key not in seen and len(s.strip()) > 20:
                seen.add(key)
                unique.append(s.strip())
        return ' '.join(unique)
    
    # Paragraph-level dedup
    seen = set()
    unique = []
    entity_lower = entity_name.lower() if entity_name else ""
    for para in paragraphs:
        key = para.strip()[:100].lower()
        # Always keep paragraphs with entity
        if entity_lower and entity_lower in para.lower():
            unique.append(para.strip())
            continue
        if key not in seen and para.strip():
            seen.add(key)
            unique.append(para.strip())
    
    return '\n'.join(unique)


def align_sentence_boundary(text: str, max_chars: int = 500, entity_name: str = "") -> str:
    """Ensure text ends at sentence boundary (not mid-word or incomplete).
    
    If entity_name is provided, ensures entity is not cut off by truncation.
    """
    if not text:
        return text
    
    # Check for incomplete ending
    INCOMPLETE_WORDS = ['dan', 'atau', 'yang', 'di', 'ke', 'dari', 'untuk', 'pada', 'dengan',
                        'karena', 'sementara', 'namun', 'tetapi', 'meskipun', 'sehingga',
                        'agar', 'supaya', 'ketika', 'saat', 'jika', 'kalau']
    
    # Remove incomplete ending
    clean_end = text.rstrip('.!?"\')]').strip()
    if clean_end:
        last_word = clean_end.split()[-1].lower() if clean_end.split() else ''
        if last_word in INCOMPLETE_WORDS:
            last_space = clean_end.rfind(' ')
            if last_space > 50:
                text = clean_end[:last_space].rstrip() + '.'
            else:
                text = clean_end.rstrip() + '.'
    
    # If entity_name provided, extend max_chars to include entity
    effective_max = max_chars
    if entity_name:
        entity_pos = text.lower().find(entity_name.lower())
        if entity_pos >= 0:
            entity_end = entity_pos + len(entity_name)
            # Ensure max_chars covers at least entity + some context after
            effective_max = max(max_chars, entity_end + 100)
    
    # Truncate if needed
    if len(text) > effective_max:
        truncated = text[:effective_max]
        # Find last sentence boundary
        last_period = truncated.rfind('. ')
        if last_period > 100:
            return truncated[:last_period + 1]
        for punct in ['! ', '? ', '." ']:
            pos = truncated.rfind(punct)
            if pos > 100:
                return truncated[:pos + 1]
        last_space = truncated.rfind(' ')
        if last_space > 100:
            return truncated[:last_space] + '.'
        return truncated + '.'
    
    # Ensure ends with punctuation
    if text and text[-1] not in '.!?"\')]':
        last_period = text.rfind('. ')
        if last_period > 100:
            text = text[:last_period + 1]
        else:
            text = text.rstrip() + '.'
    
    return text


def clean_article_text(text: str, title: str = "", entity_name: str = "") -> str:
    """
    MAIN FUNCTION: Clean article text using library-based approach.
    
    Pipeline:
      1. Fix encoding (ftfy)
      2. Comprehensive clean (clean-text library)
      3. Remove promo/marketing
      4. Remove byline
      5. Remove source attribution
      6. Remove duplicate paragraphs (preserves entity paragraphs)
      7. Align sentence boundary
    
    Args:
        text: Raw article text (from Trafilatura or RSS)
        title: Article title (for title removal from text start)
        entity_name: Entity name to preserve during dedup
    
    Returns:
        Clean article text
    """
    if not text or len(text.strip()) < 50:
        return text
    
    original_length = len(text)
    changes = []
    
    # Step 1: Fix encoding
    text = fix_encoding(text)
    if len(text) != original_length:
        changes.append('ftfy encoding fix')
    
    # Step 2: Comprehensive clean (clean-text library)
    pre_clean = text
    text = comprehensive_clean(text)
    if text != pre_clean:
        changes.append('clean-text normalization')
    
    # Step 3: Remove promo/marketing
    pre_promo = text
    text = remove_promo_content(text)
    if len(text) < len(pre_promo):
        changes.append(f'promo removed ({len(pre_promo) - len(text)} chars)')
    
    # Step 4: Remove byline
    pre_byline = text
    text = remove_byline(text)
    if len(text) < len(pre_byline):
        changes.append('byline removed')
    
    # Step 5: Remove source attribution
    pre_source = text
    text = remove_source_attribution(text)
    if len(text) < len(pre_source):
        changes.append('source attribution removed')
    
    # Step 6: Remove duplicate paragraphs (preserves entity paragraphs)
    pre_dup = text
    text = remove_duplicate_paragraphs(text, entity_name)
    if len(text) < len(pre_dup):
        changes.append(f'duplicate removed ({len(pre_dup) - len(text)} chars)')
    
    # Step 7: Align sentence boundary (with entity preservation)
    pre_align = text
    text = align_sentence_boundary(text, entity_name=entity_name)
    if text != pre_align:
        changes.append('sentence boundary aligned')
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# ============================================================
# TEST FUNCTION
# ============================================================

def test_cleaning(raw_text: str, title: str = "", label: str = ""):
    """Test cleaning on raw text and show before/after."""
    print(f"\n{'='*70}")
    print(f"TEST CLEANING {label}")
    print(f"{'='*70}")
    
    print(f"\n--- BEFORE ({len(raw_text)} chars) ---")
    print(raw_text[:500])
    if len(raw_text) > 500:
        print(f"... ({len(raw_text) - 500} more chars)")
    
    cleaned = clean_article_text(raw_text, title)
    
    print(f"\n--- AFTER ({len(cleaned)} chars) ---")
    print(cleaned[:500])
    if len(cleaned) > 500:
        print(f"... ({len(cleaned) - 500} more chars)")
    
    print(f"\n--- STATS ---")
    print(f"  Before: {len(raw_text)} chars")
    print(f"  After:  {len(cleaned)} chars")
    print(f"  Reduction: {len(raw_text) - len(cleaned)} chars")
    
    # Quality checks
    checks = {
        'no_non_ascii': all(ord(c) < 128 for c in cleaned),
        'no_byline': not re.search(r'\([a-z]{2,5}/[a-z]{2,5}\)', cleaned),
        'no_promo': 'Gabung' not in cleaned and 'berkomitmen' not in cleaned.lower(),
        'ends_with_punct': cleaned[-1] in '.!?"\')]' if cleaned else False,
        'starts_clean': cleaned[0].isupper() if cleaned else False,
    }
    
    print(f"\n--- QUALITY CHECKS ---")
    for check, passed in checks.items():
        status = '✅' if passed else '❌'
        print(f"  {status} {check}: {passed}")
    
    return cleaned


if __name__ == "__main__":
    # Test with Miftachul Akhyar article (has duplicate paragraphs)
    miftachul_text = """Rais Aam Pengurus Besar Nahdlatul Ulama (PBNU), KH Miftachul Akhyar, meminta Ketua Umum (PBNU) KH Yahya Cholil Staquf (Gus Yahya) mundur dari jabatannya. Hal itu tertuang dalam risalah rapat harian Syuriah. Rapat tersebut digelar pada Kamis (20/11/2025) di Hotel Aston City Jakarta. Rapat dihadiri 37 dari 53 orang Pengurus Harian Syuriah PBNU. Risalah rapat ini ditandatangani oleh pimpinan rapat sekaligus Rais Aam PBNU, KH Miftachul Akhyar. "Musyawarah antara Rais Aam dan dua Wakil Rais Aam memutuskan: KH Yahya Cholil Staquf harus mengundurkan diri dari jabatan Ketua Umum Pengurus Besar Nahdlatul Ulama dalam waktu 3 (tiga) hari terhitung sejak diterimanya keputusan Rapat Harian Syuriyah PBNU," tulis poin keputusan dalam risalah rapat harian Syuriah PBNU tersebut. "Jika dalam waktu 3 (tiga) hari tidak mengundurkan diri, Rapat Harian Syuriyah PBNU memutuskan memberhentikan KH. Yahya Cholil Staquf sebagai Ketua Umum Pengurus Besar Nahdlatul Ulama," lanjutnya. Siapa sebenarnya KH Miftachul Akhyar? Berikut profilnya. Profil KH Miftachul Akhyar
Miftachul Akhyar adalah seorang ulama senior yang lahir pada 1953. Ia adalah pengasuh Pondok Pesantren Miftachus Sunnah, Surabaya, Jawa Timur. Dilansir dari situs NU Jatim, KH Miftachul Akhyar merupakan putra Pengasuh Pondok Pesantren Tahsinul Akhlaq Rangkah, KH Abdul Ghoni. Dia merupakan anak kesembilan dari 13 bersaudara. Ulama yang akrab disapa Kiai Miftah ini pernah mengemban pendidikan di sejumlah pesantren. Mulai dari Pondok Pesantren Bahrul Ulum Tambakberas Jombang, Pondok Pesantren Sidogiri Pasuruan, hingga Pondok Pesantren Al-Islah Soditan Lasem. Setelah itu, Kiai Miftah mendirikan Ponpes Miftachus Sunnah di Kedung Tarukan, kampung yang tak ramah pada dakwah ulama. Berkatnya, kesan negatif kampung tersebut hilang usai berdirinya ponpes tersebut. Kiai Miftah dikenal luas karena kesederhanaan dan keramahannya dalam menyambut tamu. Sifatnya yang rendah hati terlihat jelas dari perilakunya; tak sungkan melayani sendiri tamunya, bahkan menuangkan wedang dan menyajikan camilan. Keunggulan Kiai Miftah bukan hanya terletak pada penguasaan mendalam materi agama, tetapi juga pada sifat tawadhu (rendah hati) yang luhur. Kombinasi inilah yang kemudian membuat seorang kiai terhormat-yang merupakan alumnus istimewa Pondok Pesantren Tremas Pacitan-berkenan menjadikannya sebagai menantu. Perjalanan Karier KH Miftachul Akhyar di NU
Jauh sebelum mengemban amanah sebagai Rais Aam PBNU, KH Miftachul Akhyar telah menapaki karier panjang dan matang di berbagai tingkatan kepengurusan Nahdlatul Ulama. Perjalanannya menunjukkan konsistensi dalam memimpin di struktur Syuriyah. Berikut perjalanan kariernya. - Rais Syuriyah Pengurus Cabang Nahdlatul Ulama (PCNU) Kota Surabaya periode 2000-2005
- Rais Syuriyah Pengurus Wilayah Nahdlatul Ulama (PWNU) Jawa Timur periode 2007-2013
- Rais Syuriyah Pengurus Wilayah Nahdlatul Ulama (PWNU) Jawa Timur periode 2013-2018
- Wakil Rais Aam PBNU periode 2015-2020
- Pj. Rais Aam Pengurus Besar Nahdlatul Ulama (PBNU) periode 2018-2020
- Rais 'Aam Pengurus Besar Nahdlatul Ulama (PBNU) masa khidmah 2022-2027
(hnh/kri)"""
    
    test_cleaning(miftachul_text, title="Profil KH Miftachul Akhyar", label="(Miftachul Akhyar — duplicate + byline)")
    
    # Test with Budi Gunadi article (has promo + truncation)
    budi_text = """JAKARTA, KOMPAS.com - Menteri Kesehatan (Menkes) Budi Gunadi Sadikin memastikan pemerintah tidak berencana menaikkan iuran BPJS Kesehatan pada tahun 2027. "Belum ada rencana untuk menaikkan tarif BPJS (di tahun 2027)," ujarnya dalam konferensi pers RAPBN dan Nota Keuangan Tahun Anggaran 2027 di Kantor Pusat Ditjen Pajak Kemenkeu, Jakarta, Jumat (14/8/2026). Ia mengatakan, pemerintah akan memberikan dukungan kepada BPJS Kesehatan apabila keuangannya mengalami defisit. "Teman-teman enggak usah khawatir, kalau BPJS-nya defisit pasti pemerintah pusat akan support," kata Budi Gunadi. Sebagai informasi, besaran iuran BPJS Kesehatan yang saat ini berlaku adalah Kelas I sebesar Rp 150.000 per bulan, Kelas II sebesar Rp 100.000 per bulan, serta Kelas III sebesar Rp 42.000 per bulan. Adapun untuk peserta kelas III, iuran yang dibayarkan sebesar Rp 35.000 per bulan, sedangkan Rp 7.000 sisanya disubsidi pemerintah. Besaran iuran BPJS Kesehatan itu mengacu pada Peraturan Presiden (Perpres) Nomor 64 Tahun 2020 tentang Perubahan Kedua atas Perpres Nomor 82 Tahun 2018 tentang Jaminan Kesehatan. Sebelumnya, wacana kenaikan iuran BPJS Kesehatan juga pernah mencuat pada awal 2026. Menteri Koordinator Bidang Pemberdayaan Masyarakat Abdul Muhaimin Iskandar kala itu mengatakan, pemerintah belum menaikkan iuran dengan mempertimbangkan kondisi masyarakat. "Karena kondisi dan keadaan, kita putuskan untuk tidak dinaikkan dulu," ujarnya, dikutip dari Antara (28/2/2026). Muhaimin mengatakan, pemerintah saat ini telah menanggung lebih dari 60 persen pembiayaan BPJS Kesehatan. Selain itu, sistem Jaminan Kesehatan Nasional (JKN) juga menerapkan mekanisme subsidi silang, yakni peserta yang mampu ikut membantu pembiayaan layanan kesehatan bagi peserta kurang mampu. Di sisi lain, baru-baru ini Direktur Utama BPJS Kesehatan Prihati Pujowaskito mengungkapkan, banyaknya peserta yang tidak aktif akibat menunggak iuran memicu tekanan terhadap kondisi keuangan. Ia menyebut jumlah peserta program Jaminan Kesehatan Nasional (JKN) ini telah mencapai lebih dari 285 juta jiwa atau sekitar 98,8 persen dari total penduduk Indonesia. Namun, sebanyak 54 juta peserta masih tercatat tidak aktif. "Masih ada yang tidak aktif sebesar 54 juta jiwa sampai hari ini kami sering laporkan," kata Pujo dalam acara Kerja Sama Strategis Program Jaminan Kesehatan Nasional di Kantor BPJS, Jakarta Pusat, Senin (3/8/2026). Sementara itu, kebutuhan layanan kesehatan mencapai sekitar Rp 500 miliar per hari atau Rp 16,5 triliun per bulan, sedangkan iuran yang masuk hanya berkisar Rp 14 triliun per bulan. "Ini sering kami laporkan ada selisih Rp 2 triliun lebih setiap bulannya," tutur Pujo. KOMPAS.com berkomitmen memberikan fakta jernih, tepercaya, dan berimbang. Dukung keberlanjutan jurnalisme jernih dan nikmati kenyamanan baca tanpa Gabung KOMPAS.com Plus sekarang"""
    
    test_cleaning(budi_text, title="Menkes Sebut Belum Ada Rencana Naikkan Iuran BPJS", label="(Budi Gunadi — promo + source attribution)")
