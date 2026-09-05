"""
context_extraction_v23.py
==========================
Context Extraction v23 — Token Budget + Quality Filter

UPGRADES dari v22:
  1. QUALITY FILTER: Skip kalimat profil/bio ("merupakan tokoh...", "lahir pada...")
  2. DEDUP: Skip kalimat redundan (similarity check dengan kalimat sebelumnya)
  3. SENTIMENT PRIORITY: Prioritaskan kalimat dengan kata evaluatif
  4. TOKEN BUDGET: Maksimalkan 246 tokens dengan kalimat BERKUALITAS

Filter rules:
  - SKIP: "merupakan tokoh", "lahir pada", "menjabat sebagai", "putra dari"
  - SKIP: kalimat dengan similarity >70% dengan kalimat sebelumnya
  - PRIORITIZE: kalimat dengan kata sentimen (puji, kritik, dukung, tolak, dll)
"""
import re
import logging
from typing import List, Tuple, Set

logger = logging.getLogger(__name__)

MAX_SEQ_LENGTH = 256
PREMISE_TOKENS = 10
AVAILABLE_TOKENS = MAX_SEQ_LENGTH - PREMISE_TOKENS
TOKEN_PER_WORD = 1.3
MIN_CHARS = 100
MAX_CHARS = 1200

# Kalimat yang harus di-skip (profil/bio, tidak ada sentimen)
PROFILE_PATTERNS = [
    r'(?i)merupakan\s+(seorang\s+)?(tokoh|ulama|politisi|ekonom|pengusaha|aktivis|jurnalis|akademisi)',
    r'(?i)lahir\s+(pada|di)\s+\d',
    r'(?i)putra\s+(dari|ke-)',
    r'(?i)menjabat\s+sebagai\s+(Menteri|Gubernur|Walikota|Bupati|Ketua|Direktur)\s+(pada|di|tahun)\s+\d',
    r'(?i)pendidikan\s+(di|dari|formal)',
    r'(?i)karir\s+(dimulai|politik)',
    r'(?i)perjalanan\s+(karier|politik)',
    r'(?i)profil\s+\w+',
    r'(?i)berikut\s+(profil|biografi|perjalanan)',
    r'(?i)\bsumber:\s',
    r'(?i)biodata\s',
    r'(?i)\binformasi\s+pribadi\b',
]

# Kata sentimen — prioritas tinggi untuk context
SENTIMENT_KEYWORDS = {
    # Positive
    'puji', 'dipuji', 'memuji', 'apresiasi', 'mengapresiasi', 'harga',
    'dihormati', 'berhasil', 'menang', 'prestasi', 'penghargaan', 'dianugerahi',
    'mendukung', 'dukungan', 'mendorong', 'optimis', 'positif', 'gemilang',
    'bangga', 'terbaik', 'unggul', 'inovatif', 'tanggap',
    # Negative
    'kritik', 'dikritik', 'mengkritik', 'korupsi', 'tersangka', 'divonis',
    'ditahan', 'dicopot', 'mundur', 'gagal', 'skandal', 'dugaan', 'tuntut',
    'menuntut', 'melaporkan', 'dicela', 'menolak', 'kecewa', 'turun',
    'anjlok', 'negatif', 'kontroversi', 'sanksi', 'hukuman', 'vonis',
    # Neutral indicators (factual statements)
    'menyatakan', 'mengatakan', 'menegaskan', 'mengimbau', 'klarifikasi',
    'menjelaskan', 'mengumumkan', 'melantik', 'bertemu', 'membahas',
}


def estimate_tokens(text: str) -> int:
    words = len(text.split())
    return int(words * TOKEN_PER_WORD)


def find_entity_position(text: str, entity_name: str) -> int:
    entity_lower = entity_name.lower()
    text_lower = text.lower()
    pos = text_lower.find(entity_lower)
    if pos >= 0:
        return pos
    SHORT_FORMS = {
        "joko widodo": ["jokowi"], "prabowo subianto": ["prabowo"],
        "megawati soekarnoputri": ["megawati"], "basuki tjahaja purnama": ["ahok"],
        "abdurrahman wahid": ["gus dur"], "ma'ruf amin": ["ma'ruf"],
        "erick thohir": ["erick"], "sri mulyani indrawati": ["sri mulyani"],
        "anies baswedan": ["anies"], "puan maharani": ["puan"],
        "sufmi dasco ahmad": ["dasco"], "khofifah indar parawansa": ["khofifah"],
        "dedi mulyadi": ["dedi"], "tito karnavian": ["tito"],
    }
    if entity_lower in SHORT_FORMS:
        for sf in SHORT_FORMS[entity_lower]:
            pos = text_lower.find(sf)
            if pos >= 0:
                return pos
    parts = entity_name.split()
    if len(parts) >= 2 and len(parts[-1]) >= 4:
        pos = text_lower.find(parts[-1].lower())
        if pos >= 0:
            return pos
    return -1


def split_sentences(text: str, stanza_nlp=None) -> List[str]:
    if stanza_nlp:
        try:
            doc = stanza_nlp(text)
            return [sent.text for sent in doc.sentences]
        except:
            pass
    sentences = []
    current = []
    for char in text:
        current.append(char)
        if char in '.!?' and len(current) > 1:
            sentences.append(''.join(current).strip())
            current = []
    if current:
        sentences.append(''.join(current).strip())
    return [s for s in sentences if s]


def is_profile_sentence(sentence: str) -> bool:
    """Check if sentence is a profile/bio sentence (skip)."""
    for pattern in PROFILE_PATTERNS:
        if re.search(pattern, sentence):
            return True
    return False


def is_redundant(sentence: str, previous_sentences: List[str], threshold: float = 0.5) -> bool:
    """Check if sentence is redundant with previous sentences.
    
    Uses two methods:
    1. Word overlap (Jaccard similarity) — threshold 0.5
    2. Key phrase match — if >40% words match a previous sentence
    """
    if not previous_sentences:
        return False
    
    def get_words(s):
        return set(w.lower().strip('.,;:!?()"\'[]{}') for w in s.split() if len(w) > 2)
    
    sent_words = get_words(sentence)
    if not sent_words:
        return False
    
    for prev in previous_sentences:
        prev_words = get_words(prev)
        if not prev_words:
            continue
        
        # Method 1: Jaccard similarity
        intersection = len(sent_words & prev_words)
        union = len(sent_words | prev_words)
        if union > 0:
            similarity = intersection / union
            if similarity >= threshold:
                return True
        
        # Method 2: Key phrase match (overlap coefficient)
        # If >50% of sentence words appear in previous sentence
        overlap = intersection / min(len(sent_words), len(prev_words))
        if overlap >= 0.6:
            return True
        
        # Method 3: Check for key phrase duplication
        # "mengimbau masyarakat" in both → redundant
        sent_lower = sentence.lower()
        prev_lower = prev.lower()
        key_phrases = ['mengimbau masyarakat', 'menyatakan bahwa', 'menegaskan bahwa',
                       'mengatakan bahwa', 'menjelaskan bahwa', 'menekankan bahwa']
        for phrase in key_phrases:
            if phrase in sent_lower and phrase in prev_lower:
                return True
    
    return False


def has_sentiment_keyword(sentence: str) -> bool:
    """Check if sentence contains sentiment keywords (priority)."""
    sent_lower = sentence.lower()
    for keyword in SENTIMENT_KEYWORDS:
        if keyword in sent_lower:
            return True
    return False


def extract_context_v23(
    article_text: str,
    entity_name: str,
    stanza_nlp=None,
    max_tokens: int = AVAILABLE_TOKENS,
) -> Tuple[str, int, dict]:
    """
    Extract context with TOKEN BUDGET + QUALITY FILTER.
    
    Pipeline:
      1. Split into sentences
      2. Find anchor sentence (contains entity)
      3. Filter: skip profile, skip redundant
      4. Prioritize: sentiment keywords first
      5. Fill token budget with quality sentences
    
    Returns:
        (context_text, token_estimate, stats)
    """
    stats = {"total_sentences": 0, "skipped_profile": 0, "skipped_redundant": 0,
             "sentiment_priority": 0, "included": 0}
    
    if not article_text or not entity_name:
        return "", 0, stats
    
    entity_pos = find_entity_position(article_text, entity_name)
    if entity_pos < 0:
        fallback = article_text[:500].rsplit(' ', 1)[0]
        if not fallback.endswith('.'):
            fallback += '.'
        return fallback, estimate_tokens(fallback), stats
    
    sentences = split_sentences(article_text, stanza_nlp)
    if not sentences:
        return "", 0, stats
    
    stats["total_sentences"] = len(sentences)
    
    # Find anchor sentence
    anchor_idx = -1
    entity_lower = entity_name.lower()
    for i, sent in enumerate(sentences):
        if entity_lower in sent.lower():
            anchor_idx = i
            break
    
    if anchor_idx < 0:
        SHORT_FORMS = ["jokowi","prabowo","megawati","sby","ahok","gus dur",
                       "erick","bima","sri mulyani","anies","puan","dasco",
                       "khofifah","yusril","dedi","tito"]
        for i, sent in enumerate(sentences):
            for sf in SHORT_FORMS:
                if sf in entity_lower and sf in sent.lower():
                    anchor_idx = i
                    break
            if anchor_idx >= 0:
                break
    
    if anchor_idx < 0:
        char_count = 0
        for i, sent in enumerate(sentences):
            char_count += len(sent)
            if char_count > entity_pos:
                anchor_idx = i
                break
    
    if anchor_idx < 0:
        anchor_idx = 0
    
    # Build context with quality filter
    context_sentences = []
    current_tokens = 0
    included_indices = set()
    
    # Step 1: Add anchor sentence (always include, even if profile)
    anchor_sent = sentences[anchor_idx]
    context_sentences.append(anchor_sent)
    current_tokens += estimate_tokens(anchor_sent)
    included_indices.add(anchor_idx)
    stats["included"] += 1
    
    # Step 2: Build list of candidate sentences (before + after, ordered by distance)
    candidates = []
    for distance in range(1, len(sentences)):
        # After anchor
        idx_after = anchor_idx + distance
        if idx_after < len(sentences) and idx_after not in included_indices:
            candidates.append((idx_after, distance, 'after'))
        # Before anchor
        idx_before = anchor_idx - distance
        if idx_before >= 0 and idx_before not in included_indices:
            candidates.append((idx_before, distance, 'before'))
    
    # Step 3: Sort candidates — sentiment priority first, then by distance
    def candidate_priority(c):
        idx, distance, direction = c
        sent = sentences[idx]
        has_sentiment = has_sentiment_keyword(sent)
        # Sentiment sentences get priority (lower sort key)
        return (0 if has_sentiment else 1, distance)
    
    candidates.sort(key=candidate_priority)
    
    # Step 4: Add candidates with quality filter
    for idx, distance, direction in candidates:
        if current_tokens >= max_tokens:
            break
        
        sent = sentences[idx]
        sent_tokens = estimate_tokens(sent)
        
        if current_tokens + sent_tokens > max_tokens:
            continue
        
        # Quality filter: skip profile
        if is_profile_sentence(sent):
            stats["skipped_profile"] += 1
            continue
        
        # Quality filter: skip redundant
        if is_redundant(sent, context_sentences):
            stats["skipped_redundant"] += 1
            continue
        
        # Add sentence
        if direction == 'before':
            context_sentences.insert(0, sent)
        else:
            context_sentences.append(sent)
        current_tokens += sent_tokens
        included_indices.add(idx)
        stats["included"] += 1
        
        if has_sentiment_keyword(sent):
            stats["sentiment_priority"] += 1
    
    # Join
    context = ' '.join(context_sentences)
    
    # Ensure ends with punctuation
    if context and context[-1] not in '.!?"\')]':
        last_period = context.rfind('. ')
        if last_period > 100:
            context = context[:last_period + 1]
        else:
            context = context.rstrip() + '.'
    
    # Hard cap
    if len(context) > MAX_CHARS:
        truncated = context[:MAX_CHARS]
        last_period = truncated.rfind('. ')
        if last_period > 100:
            context = truncated[:last_period + 1]
        else:
            context = truncated.rsplit(' ', 1)[0] + '.'
    
    return context, current_tokens, stats


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    mahfud_article = """Mahfud MD menyatakan video beredar di media sosial mengenai Reformasi Jilid 2 Agustus 2026 adalah hoaks. Video lama tanggal 30 Agustus 2025 disalahgunakan untuk menciptakan kesan seolah menanggapi situasi politik saat ini. Mahfud MD mengimbau masyarakat agar selalu memverifikasi informasi dan tidak mudah terprovokasi konten tidak jelas. Mantan Menko Polhukam Mahfud MD memberikan klarifikasi tegas terkait beredarnya potongan video di media sosial yang mengeklaim dirinya menyampaikan dukungan untuk Reformasi Jilid 2. Melalui unggahan di akun Instagram resminya, Mahfud MD menjelaskan bahwa video yang beredar merupakan potongan dari sesi diskusi lama. Dalam klarifikasi tersebut, Mahfud menekankan bahwa ia tidak pernah menyampaikan dukungan untuk aksi Reformasi Jilid 2. Ia juga mengimbau masyarakat untuk tidak mudah percaya pada informasi yang belum terverifikasi. Mahfud MD merupakan tokoh hukum dan politik Indonesia yang pernah menjabat sebagai Menko Polhukam."""
    
    print(f"{'='*70}")
    print(f"TEST: Context Extraction v23 — Token Budget + Quality Filter")
    print(f"{'='*70}")
    
    context, tokens, stats = extract_context_v23(mahfud_article, "Mahfud MD")
    
    print(f"\nArticle: {len(mahfud_article)} chars, {stats['total_sentences']} sentences")
    print(f"\nQuality Filter Stats:")
    print(f"  Total sentences:    {stats['total_sentences']}")
    print(f"  Included:           {stats['included']}")
    print(f"  Skipped (profile):  {stats['skipped_profile']}")
    print(f"  Skipped (redundant):{stats['skipped_redundant']}")
    print(f"  Sentiment priority: {stats['sentiment_priority']}")
    
    print(f"\nExtracted Context ({len(context)} chars, ~{tokens} tokens):")
    print(f"---")
    print(context)
    print(f"---")
    print(f"\nToken utilization: {tokens}/{AVAILABLE_TOKENS} ({tokens/AVAILABLE_TOKENS*100:.1f}%)")
    print(f"Entity present: {'mahfud md' in context.lower()}")
    
    # Check if profile sentence is excluded
    has_profile = 'merupakan tokoh' in context.lower()
    has_redundant = context.count('mengimbau masyarakat') > 1
    print(f"\nQuality checks:")
    print(f"  ✅ Profile sentence excluded: {not has_profile}")
    print(f"  ✅ Redundant sentence excluded: {not has_redundant}")
    
    # Compare v22 vs v23
    print(f"\n{'='*70}")
    print(f"COMPARISON: v22 vs v23")
    print(f"{'='*70}")
    
    # v22 (import from previous)
    from context_extraction_v22 import extract_context_optimized
    context_v22, tokens_v22 = extract_context_optimized(mahfud_article, "Mahfud MD")
    
    print(f"\nv22 (token budget only):")
    print(f"  Chars: {len(context_v22)}, Tokens: ~{tokens_v22}")
    print(f"  Has profile: {'merupakan tokoh' in context_v22.lower()}")
    print(f"  Has redundant: {context_v22.count('mengimbau masyarakat') > 1}")
    
    print(f"\nv23 (token budget + quality filter):")
    print(f"  Chars: {len(context)}, Tokens: ~{tokens}")
    print(f"  Has profile: {has_profile}")
    print(f"  Has redundant: {has_redundant}")
    
    print(f"\nImprovement: profile {'REMOVED' if not has_profile else 'STILL EXISTS'}, "
          f"redundant {'REMOVED' if not has_redundant else 'STILL EXISTS'}")
