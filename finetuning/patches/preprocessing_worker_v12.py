"""
preprocessing_worker_v12.py
============================
Refactor Layer 3 (Preprocessing) — v12 dengan fixes:
  1. FIX bullet points summary ("- Video lama...", "- Mahfud mengimbau...")
  2. FIX source attribution di TENGAH text (bukan hanya awal)
  3. FIX portal detection untuk Suara.com
  4. Integrate v21: byline + promo + duplicate + encoding

Library: ftfy + clean-text
"""
import re
import hashlib
import logging
import html as html_lib
import unicodedata

import ftfy

logger = logging.getLogger(__name__)

# Byline patterns
BYLINE_SLASH_PATTERN = re.compile(r'\s*\([a-z]{2,5}/[a-z]{2,5}\)\s*', re.IGNORECASE)
BYLINE_END_PATTERN = re.compile(r'\s*\([a-z]{2,5}(?:/[a-z]{2,5})?\)\s*$', re.IGNORECASE)
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

# Source attribution — awal DAN tengah
SOURCE_ATTR_PATTERNS = [
    # Di awal text
    r'^(KOMPAS\.com|CNN Indonesia|TEMPO\.CO|TRIBUN\w*\.?\w*|ANTARA/?\w*|jpnn\.com|detikcom|VIVA|Suara\.com|Republika)\s*[\-–—|:]\s*',
    r'^(JAKARTA|BANDUNG|SURABAYA|MEDAN|MAKASSAR|SEMARANG|YOGYAKARTA)\s*[\-–—|:]\s*[A-Z]',
    r'^(TRIBUNNEWS\.COM|KOMPAS\.TV|CNNINDONESIA)\s*[, ]*[A-Z\s]*\s*[\-–—|:]\s*',
    # Di TENGAH text (preceded by period + space)
    r'\.\s+(KOMPAS\.com|CNN Indonesia|TEMPO\.CO|TRIBUN\w*|Suara\.com|VIVA|Republika|detikcom|jpnn\.com)\s*[\-–—|:]\s*',
    r'\.\s+(Suara\.com|ANTARA)\s*[\-–—|:]\s*',
]

# Bullet point patterns (summary section dari portal)
BULLET_PATTERNS = [
    # "- Video lama tanggal..." di awal atau setelah period
    r'(?:^|\.\s+)-\s+(?=[A-Z])',
    # Multiple consecutive bullet points
    r'\n-\s+',
]


def normalize_unicode(text: str) -> str:
    """Fix encoding using ftfy + NFKC."""
    text = ftfy.fix_text(text)
    text = html_lib.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\xa0", " ")
    return text


def remove_urls_emails(text: str) -> tuple:
    """Remove URLs and emails."""
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', ' ', text)
    return text, int(len(urls) + len(emails))


def remove_bullet_points(text: str) -> str:
    """Remove bullet point markers from summary sections.
    
    Handles:
    - "- Video lama tanggal..." → "Video lama tanggal..."
    - "- Mahfud MD mengimbau..." → "Mahfud MD mengimbau..."
    """
    for pattern in BULLET_PATTERNS:
        text = re.sub(pattern, '. ', text)
    # Fix double periods from replacement
    text = re.sub(r'\.\.\s+', '. ', text)
    text = re.sub(r'^\.\s+', '', text)
    return text


def remove_promo_content(text: str) -> str:
    """Remove promo/marketing content."""
    for pattern in PROMO_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    return text


def remove_byline(text: str) -> str:
    """Remove byline — keep context abbreviations."""
    text = BYLINE_SLASH_PATTERN.sub(' ', text)
    match = BYLINE_END_PATTERN.search(text)
    if match:
        content = match.group().strip('() \n\r')
        if content.lower() not in KEEP_ABBREVIATIONS:
            text = BYLINE_END_PATTERN.sub('', text)
    return text.rstrip()


def remove_source_attribution(text: str) -> str:
    """Remove source attribution — awal DAN tengah text."""
    for pattern in SOURCE_ATTR_PATTERNS:
        text = re.sub(pattern, '. ', text)
    # Fix double periods
    text = re.sub(r'\.\.\s+', '. ', text)
    text = re.sub(r'^\.\s+', '', text)
    return text.strip()


def remove_duplicate_paragraphs(text: str, entity_name: str = "") -> str:
    """Remove duplicate paragraphs/sentences. Preserves entity."""
    entity_lower = entity_name.lower() if entity_name else ""
    sentences = re.split(r'(?<=[.!?])\s+|\n', text)
    seen_content = set()
    unique = []
    for s in sentences:
        s = s.strip()
        if not s or len(s) < 20:
            continue
        core = re.sub(r'^(Profil\s+\w+\s*|Perjalanan\s+\w+\s*|Berikut\s+\w+\s*)', '', s, flags=re.IGNORECASE).strip()
        key = core[:80].lower()
        if entity_lower and entity_lower in s.lower():
            if key not in seen_content:
                seen_content.add(key)
                unique.append(s)
            continue
        if key not in seen_content:
            seen_content.add(key)
            unique.append(s)
    return ' '.join(unique)


def normalize_punctuation(text: str) -> str:
    """Normalize smart quotes and dashes."""
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    return re.sub(r'[ \t]+', ' ', text).strip()


def normalize_pipeline(text: str, title: str = "", entity_name: str = "") -> tuple:
    """
    MAIN PIPELINE v12: Clean text dengan library-based approach.
    
    Pipeline:
      1. Fix encoding (ftfy)
      2. Normalize unicode (NFKC)
      3. Remove URLs/emails
      4. Remove bullet points (NEW v12)
      5. Remove promo/marketing
      6. Remove byline (smart)
      7. Remove source attribution (awal + TENGAH) (NEW v12)
      8. Strip news boilerplate
      9. Remove duplicate paragraphs
      10. Normalize punctuation
      11. Normalize whitespace
      12. Compute content hash
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
    
    # 3. Remove bullet points (NEW v12)
    pre_bullet = len(text)
    text = remove_bullet_points(text)
    if len(text) != pre_bullet:
        stats["changes"].append("bullet points removed")
    
    # 4. Remove promo
    pre_promo = len(text)
    text = remove_promo_content(text)
    if len(text) < pre_promo:
        stats["changes"].append(f"promo removed ({pre_promo - len(text)} chars)")
    
    # 5. Remove byline
    pre_byline = len(text)
    text = remove_byline(text)
    if len(text) < pre_byline:
        stats["changes"].append("byline removed")
    
    # 6. Remove source attribution (awal + TENGAH) (NEW v12)
    pre_source = len(text)
    text = remove_source_attribution(text)
    if len(text) < pre_source:
        stats["changes"].append(f"source attr removed ({pre_source - len(text)} chars)")
    
    # 7. Normalize punctuation
    text = normalize_punctuation(text)
    
    # 8. Remove duplicate paragraphs
    pre_dup = len(text)
    text = remove_duplicate_paragraphs(text, entity_name)
    if len(text) < pre_dup:
        stats["changes"].append(f"duplicate removed ({pre_dup - len(text)} chars)")
    
    # 9. Normalize whitespace
    text = normalize_whitespace(text)
    
    # 10. Content hash
    stats["clean_len"] = len(text)
    stats["content_hash"] = hashlib.sha256(text.encode()).hexdigest() if text else None
    
    return text, stats


if __name__ == "__main__":
    # Test dengan Row 4 (Mahfud MD — bullet points + source attr tengah)
    mahfud_text = """Mahfud MD menyatakan video beredar di media sosial mengenai Reformasi Jilid 2 Agustus 2026 adalah hoaks. - Video lama tanggal 30 Agustus 2025 disalahgunakan untuk menciptakan kesan seolah menanggapi situasi politik saat ini. - Mahfud MD mengimbau masyarakat agar selalu memverifikasi informasi dan tidak mudah terprovokasi konten tidak jelas. Suara.com - Mantan Menko Polhukam Mahfud MD memberikan klarifikasi tegas terkait beredarnya potongan video di media sosial yang mengeklaim dirinya menyampaikan dukungan untuk Reformasi Jilid 2."""
    
    cleaned, stats = normalize_pipeline(mahfud_text, title="Mahfud MD Klarifikasi Video", entity_name="Mahfud MD")
    
    print(f"Test: Mahfud MD (bullet points + source attr tengah)")
    print(f"  Before: {len(mahfud_text)} chars")
    print(f"  After:  {len(cleaned)} chars")
    print(f"  Changes: {stats['changes']}")
    print(f"  Has bullet '- ': {'- ' in cleaned}")
    print(f"  Has 'Suara.com -': {'Suara.com -' in cleaned}")
    print(f"\n  BEFORE:\n  {mahfud_text[:300]}")
    print(f"\n  AFTER:\n  {cleaned[:300]}")
