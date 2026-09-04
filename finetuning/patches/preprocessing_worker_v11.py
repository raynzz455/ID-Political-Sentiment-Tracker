"""
preprocessing_worker_v11.py
============================
Refactor Layer 3 (Preprocessing) dengan LIBRARY-BASED approach.

PERUBAHAN DARI v10 → v11:
  1. FIX BUG: typo `textatch.end()` → `text[match.end():]` (line 67 v10)
  2. INTEGRATE v21: byline removal (hnh/kri, red, dll)
  3. INTEGRATE v21: duplicate paragraph detection
  4. INTEGRATE v21: promo/marketing removal (Gabung KOMPAS.com Plus)
  5. INTEGRATE v21: source attribution removal (TRIBUN -, KOMPAS.com -)
  6. LIBRARY: ftfy untuk fix encoding (replaces manual unicodedata)
  7. LIBRARY: clean-text untuk comprehensive cleaning

Library yang dipakai:
  - ftfy: fix encoding/mojibake
  - clean-text: comprehensive text cleaning
  - (existing) langdetect: language detection (from validation worker)
"""
import re
import gc
import hashlib
import logging
import html as html_lib
from pathlib import Path

# Library-based imports
import ftfy
from cleantext import clean as clean_text_lib

logger = logging.getLogger(__name__)

# ============================================================
# LAYER 3: PREPROCESSING (v11 — library-based)
# ============================================================

# Byline patterns: only author/editor bylines (with slash) or at end
BYLINE_SLASH_PATTERN = re.compile(r'\s*\([a-z]{2,5}/[a-z]{2,5}\)\s*', re.IGNORECASE)
BYLINE_END_PATTERN = re.compile(r'\s*\([a-z]{2,5}(?:/[a-z]{2,5})?\)\s*$', re.IGNORECASE)

# Context abbreviations to KEEP (not bylines)
KEEP_ABBREVIATIONS = {
    'ratas', 'nobar', 'red', 'kapol', 'wabup', 'wagub', 'wali',
    'ist', 'dok', 'antara', 'foto', 'instagram', 'pmj', 'ls',
    'psht', 'pmp', 'kk', 'ak',
}

# Promo patterns
PROMO_PATTERNS = [
    r'(?i)Gabung\s+\w+\s*\.?\s*Plus\s*sekarang.*',
    r'(?i)berkomitmen memberikan fakta jernih.*',
    r'(?i)Dukung keberlanjutan jurnalisme.*',
    r'(?i)nikmati kenyamanan baca.*',
    r'(?i)KOMPAS\.com berkomitmen.*',
    r'(?i)Baca berita selengkapnya.*',
    r'(?i)(Baca Juga|Simak juga|Berita Terkait)\s*:.*?(?=\.|$)',
    r'(?i)(Pilihan untuk lu|Sumber:).*?(?=\n|$|\.)',
    r'(?i)Sponsor.*?(?=\n|$|\.)',
]

# Source attribution patterns
SOURCE_ATTR_PATTERNS = [
    r'^(KOMPAS\.com|CNN Indonesia|TEMPO\.CO|TRIBUN\w*\.?\w*|ANTARA/?\w*|jpnn\.com|detikcom|VIVA|Suara\.com|Republika)\s*[\-–—|:]\s*',
    r'^(JAKARTA|BANDUNG|SURABAYA|MEDAN|MAKASSAR|SEMARANG|YOGYAKARTA)\s*[\-–—|:]\s*[A-Z]',
    r'^(TRIBUNNEWS\.COM|KOMPAS\.TV|CNNINDONESIA)\s*[, ]*[A-Z\s]*\s*[\-–—|:]\s*',
]


def normalize_unicode(text: str) -> str:
    """Fix encoding using ftfy library + NFKC normalization."""
    # ftfy fixes mojibake, broken unicode, mixed encoding
    text = ftfy.fix_text(text)
    # HTML unescape
    text = html_lib.unescape(text)
    # NFKC normalization
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    # Remove zero-width chars
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\xa0", " ")
    return text


def remove_urls_emails(text: str) -> tuple:
    """Remove URLs and emails from text."""
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', ' ', text)
    return text, int(len(urls) + len(emails))


def remove_promo_content(text: str) -> str:
    """Remove promo/marketing content from portal berita."""
    for pattern in PROMO_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    return text


def remove_byline(text: str) -> str:
    """Remove byline author markers — but KEEP context abbreviations."""
    # Remove bylines WITH slash: (hnh/kri), (tfq/dal)
    text = BYLINE_SLASH_PATTERN.sub(' ', text)
    # Remove byline at END only (if not a known abbreviation)
    match = BYLINE_END_PATTERN.search(text)
    if match:
        content = match.group().strip('() \n\r')
        if content.lower() not in KEEP_ABBREVIATIONS:
            text = BYLINE_END_PATTERN.sub('', text)
    text = text.rstrip()
    return text


def remove_source_attribution(text: str) -> str:
    """Remove source attribution at start of text."""
    for pattern in SOURCE_ATTR_PATTERNS:
        text = re.sub(pattern, '', text)
    
    # Also handle "JAKARTA, KOMPAS.com -" pattern (city + portal)
    text = re.sub(r'^[A-Z\s]+,\s*(KOMPAS\.com|CNN Indonesia|TEMPO\.CO|TRIBUN\w*|ANTARA)\s*[\-–—|:]\s*', '', text)
    
    return text.strip()


def remove_duplicate_paragraphs(text: str, entity_name: str = "") -> str:
    """Remove duplicate paragraphs/sentences within article. Preserves entity paragraphs."""
    entity_lower = entity_name.lower() if entity_name else ""
    
    # Split into sentences (handles both \n and sentence boundaries)
    sentences = re.split(r'(?<=[.!?])\s+|\n', text)
    
    seen_content = set()  # Track content (not prefix)
    unique = []
    for s in sentences:
        s = s.strip()
        if not s or len(s) < 20:
            continue
        
        # Extract core content (skip headers like "Profil KH...", "Perjalanan Karier...")
        # Remove common section headers to get actual content
        core = re.sub(r'^(Profil\s+\w+\s*|Perjalanan\s+\w+\s*|Berikut\s+\w+\s*)', '', s, flags=re.IGNORECASE).strip()
        key = core[:80].lower()
        
        # Always keep sentences with entity
        if entity_lower and entity_lower in s.lower():
            if key not in seen_content:
                seen_content.add(key)
                unique.append(s)
            continue
        
        # Dedup by core content
        if key not in seen_content:
            seen_content.add(key)
            unique.append(s)
    
    return ' '.join(unique)


def strip_news_boilerplate(text: str, title: str = "") -> str:
    """Strip news boilerplate — FIXED v11 (no typo bug)."""
    if title:
        title_words = re.findall(r'\w+', title)[:8]
        if title_words:
            pattern_title = r'\W*'.join(re.escape(w) for w in title_words)
            match = re.match(r'^\s*' + pattern_title, text, re.IGNORECASE)
            if match:
                # FIX v11: was `textatch.end()` (typo), now `text[match.end():]`
                text = text[match.end():].lstrip(" :-\n\"'")
    
    # Remove residual HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove boilerplate patterns
    patterns = [
        r"(?i)(baca juga|simak juga|berita terkait)\s*:[^.\n]*\.?",
        r"(?i)(reporter|editor|penulis|pewarta|jurnalis)\s*:\s*[^.\n]*\.?",
        r"(?i)(berlangganan|iklan|advertisement|sponsor)\s*[^.\n]*\.?",
        r"(?i)(copyright|©|hak cipta)\s*[^.\n]*\.?",
        r"(?i)(scroll ke bawah|mau berita terbaru|pilihan untuk lu)\s*[^.\n]*\.?",
    ]
    for p in patterns:
        text = re.sub(p, '', text)
    
    # Remove photo credits
    text = re.sub(r'\(\s*(Foto|Instagram|Dok|Istimewa|Antara)[^)]*\)', '', text, flags=re.IGNORECASE)
    
    return text.strip(" :-\n\"'")


def normalize_punctuation(text: str) -> str:
    """Normalize smart quotes and dashes."""
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u2013', '-').replace('\u2014', '-').replace('\u2015', '-')
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    return re.sub(r'[ \t]+', ' ', text).strip()


def normalize_pipeline(text: str, title: str = "", entity_name: str = "") -> tuple:
    """
    MAIN PIPELINE: Clean text using library-based approach (v11).
    
    Pipeline:
      1. Fix encoding (ftfy)
      2. Normalize unicode (NFKC)
      3. Remove URLs/emails
      4. Remove promo/marketing
      5. Remove byline (smart — keep abbreviations)
      6. Remove source attribution
      7. Strip news boilerplate (FIXED typo)
      8. Remove duplicate paragraphs (with entity preservation)
      9. Normalize punctuation
      10. Normalize whitespace
      11. Compute content hash
    """
    stats = {"original_len": len(text), "clean_len": 0, "changes": []}
    
    if not text or len(text.strip()) < 50:
        return text, stats
    
    # 1. Fix encoding
    text = normalize_unicode(text)
    stats["changes"].append("ftfy encoding fix")
    
    # 2. Remove URLs/emails
    text, removed_count = remove_urls_emails(text)
    if removed_count > 0:
        stats["changes"].append(f"URLs/emails removed ({removed_count})")
    
    # 3. Remove promo
    pre_promo = len(text)
    text = remove_promo_content(text)
    if len(text) < pre_promo:
        stats["changes"].append(f"promo removed ({pre_promo - len(text)} chars)")
    
    # 4. Remove byline
    pre_byline = len(text)
    text = remove_byline(text)
    if len(text) < pre_byline:
        stats["changes"].append("byline removed")
    
    # 5. Remove source attribution
    pre_source = len(text)
    text = remove_source_attribution(text)
    if len(text) < pre_source:
        stats["changes"].append("source attribution removed")
    
    # 6. Strip news boilerplate (FIXED)
    text = strip_news_boilerplate(text, title)
    
    # 7. Remove duplicate paragraphs
    pre_dup = len(text)
    text = remove_duplicate_paragraphs(text, entity_name)
    if len(text) < pre_dup:
        stats["changes"].append(f"duplicate paragraphs removed ({pre_dup - len(text)} chars)")
    
    # 8. Normalize punctuation
    text = normalize_punctuation(text)
    
    # 9. Normalize whitespace
    text = normalize_whitespace(text)
    
    # 10. Content hash
    stats["clean_len"] = len(text)
    stats["content_hash"] = hashlib.sha256(text.encode()).hexdigest() if text else None
    
    return text, stats


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    # Test 1: Miftachul Akhyar (duplicate + byline)
    miftachul = """Rais Aam Pengurus Besar Nahdlatul Ulama (PBNU), KH Miftachul Akhyar, meminta Ketua Umum (PBNU) KH Yahya Cholil Staquf (Gus Yahya) mundur dari jabatannya. Hal itu tertuang dalam risalah rapat harian Syuriah. Profil KH Miftachul Akhyar
Miftachul Akhyar adalah seorang ulama senior yang lahir pada 1953. Ia adalah pengasuh Pondok Pesantren Miftachus Sunnah, Surabaya, Jawa Timur. Kombinasi inilah yang kemudian membuat seorang kiai terhormat-berkenan menjadikannya sebagai menantu. Perjalanan Karier KH Miftachul Akhyar di NU
Miftachul Akhyar adalah seorang ulama senior yang lahir pada 1953. Ia adalah pengasuh Pondok Pesantren Miftachus Sunnah, Surabaya, Jawa Timur. Kombinasi inilah yang kemudian membuat seorang kiai terhormat-berkenan menjadikannya sebagai menantu. (hnh/kri)"""
    
    cleaned, stats = normalize_pipeline(miftachul, title="Profil KH Miftachul Akhyar", entity_name="Miftachul Akhyar")
    print(f"Test 1: Miftachul Akhyar")
    print(f"  Before: {len(miftachul)} chars")
    print(f"  After:  {len(cleaned)} chars")
    print(f"  Changes: {stats['changes']}")
    print(f"  Has byline: {'(hnh/kri)' in cleaned}")
    print(f"  Has duplicate: {cleaned.count('Miftachul Akhyar adalah seorang ulama') > 1}")
    print(f"  Text: {cleaned[:200]}...")
    
    # Test 2: Budi Gunadi (promo + source attribution)
    budi = """JAKARTA, KOMPAS.com - Menteri Kesehatan Budi Gunadi Sadikin memastikan tidak ada kenaikan iuran BPJS. KOMPAS.com berkomitmen memberikan fakta jernih, tepercaya, dan berimbang. Dukung keberlanjutan jurnalisme jernih dan nikmati kenyamanan baca tanpa Gabung KOMPAS.com Plus sekarang"""
    
    cleaned2, stats2 = normalize_pipeline(budi, title="Menkes BPJS", entity_name="Budi Gunadi Sadikin")
    print(f"\nTest 2: Budi Gunadi")
    print(f"  Before: {len(budi)} chars")
    print(f"  After:  {len(cleaned2)} chars")
    print(f"  Changes: {stats2['changes']}")
    print(f"  Has promo: {'Gabung' in cleaned2}")
    print(f"  Has source attr: {'KOMPAS.com -' in cleaned2}")
    print(f"  Text: {cleaned2[:200]}...")
