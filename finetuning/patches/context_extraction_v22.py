"""
context_extraction_v22.py
==========================
Refactor Layer 3.5 (Context Extraction) — v22 dengan token budget optimization.

PERUBAHAN DARI v20 → v22:
  1. TOKEN BUDGET: Memaksimalkan penggunaan token base model (MAX_SEQ=256)
     - Sebelumnya: ambil 3 kalimat (sering <150 tokens, wasteful)
     - Sekarang: ambil kalimat sampai mendekati 256 tokens (optimal)
  
  2. TOKEN ESTIMATION: Estimasi token per kalimat (1 kata ≈ 1.3 tokens)
  
  3. SENTENCE PRIORITY: Prioritaskan kalimat dengan entity + kalimat terkait
     - Kalimat 1: yang mengandung entity (anchor)
     - Kalimat 2-3: sebelum anchor (jika relevan)
     - Kalimat 4-N: setelah anchor (sampai token budget tercapai)

  4. LIBRARY: Stanza sentence segmentation (no manual regex)

Token budget calculation:
  - MAX_SEQ_LENGTH = 256 tokens
  - Premise "Tentang {entity}" = ~5-10 tokens
  - Available for context = ~240-250 tokens
  - Target: 240 tokens ≈ 185 kata ≈ 1000-1200 chars
"""
import re
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Token budget
MAX_SEQ_LENGTH = 256
PREMISE_TOKENS = 10  # "Tentang {entity_name}" ≈ 5-10 tokens
AVAILABLE_TOKENS = MAX_SEQ_LENGTH - PREMISE_TOKENS  # ~246 tokens
TOKEN_PER_WORD = 1.3  # rough estimate for Indonesian
TARGET_CHARS = int(AVAILABLE_TOKENS / TOKEN_PER_WORD * 5)  # ~950 chars (avg 5 chars/word)
MIN_CHARS = 100
MAX_CHARS = 1200  # hard cap


def estimate_tokens(text: str) -> int:
    """Estimate token count for text (1 word ≈ 1.3 tokens for Indonesian)."""
    words = len(text.split())
    return int(words * TOKEN_PER_WORD)


def find_entity_position(text: str, entity_name: str) -> int:
    """Find entity position in text."""
    entity_lower = entity_name.lower()
    text_lower = text.lower()
    pos = text_lower.find(entity_lower)
    if pos >= 0:
        return pos
    # Try short forms
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
    # Try last name
    parts = entity_name.split()
    if len(parts) >= 2 and len(parts[-1]) >= 4:
        pos = text_lower.find(parts[-1].lower())
        if pos >= 0:
            return pos
    return -1


def split_sentences(text: str, stanza_nlp=None) -> List[str]:
    """Split text into sentences using Stanza (or fallback)."""
    if stanza_nlp:
        try:
            doc = stanza_nlp(text)
            return [sent.text for sent in doc.sentences]
        except:
            pass
    
    # Fallback: simple sentence split (not regex pattern, just string ops)
    sentences = []
    current = []
    for char in text:
        current.append(char)
        if char in '.!?' and len(current) > 1:
            # Check next char is whitespace (end of sentence)
            sentences.append(''.join(current).strip())
            current = []
    if current:
        sentences.append(''.join(current).strip())
    return [s for s in sentences if s]


def extract_context_optimized(
    article_text: str,
    entity_name: str,
    stanza_nlp=None,
    max_tokens: int = AVAILABLE_TOKENS,
) -> Tuple[str, int]:
    """
    Extract context with TOKEN BUDGET optimization.
    
    Strategy:
      1. Find entity position → anchor sentence
      2. Expand outward (before + after) until token budget reached
      3. Prioritize sentences near entity
      4. Always end at sentence boundary
    
    Returns:
        (context_text, token_estimate)
    """
    if not article_text or not entity_name:
        return "", 0
    
    # Find entity position
    entity_pos = find_entity_position(article_text, entity_name)
    if entity_pos < 0:
        # Entity not found — fallback to first 500 chars
        fallback = article_text[:500].rsplit(' ', 1)[0]
        if not fallback.endswith('.'):
            fallback += '.'
        return fallback, estimate_tokens(fallback)
    
    # Split into sentences
    sentences = split_sentences(article_text, stanza_nlp)
    if not sentences:
        return "", 0
    
    # Find anchor sentence (containing entity)
    anchor_idx = -1
    entity_lower = entity_name.lower()
    for i, sent in enumerate(sentences):
        if entity_lower in sent.lower():
            anchor_idx = i
            break
    
    if anchor_idx < 0:
        # Try short forms
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
        # Fallback: use sentences around entity_pos
        char_count = 0
        for i, sent in enumerate(sentences):
            char_count += len(sent)
            if char_count > entity_pos:
                anchor_idx = i
                break
    
    if anchor_idx < 0:
        anchor_idx = 0
    
    # Build context with token budget
    context_sentences = [sentences[anchor_idx]]
    current_tokens = estimate_tokens(sentences[anchor_idx])
    
    # Expand: alternate before and after anchor
    before_idx = anchor_idx - 1
    after_idx = anchor_idx + 1
    
    while current_tokens < max_tokens:
        added = False
        
        # Try after first (usually more relevant)
        if after_idx < len(sentences):
            sent = sentences[after_idx]
            sent_tokens = estimate_tokens(sent)
            if current_tokens + sent_tokens <= max_tokens:
                context_sentences.append(sent)
                current_tokens += sent_tokens
                after_idx += 1
                added = True
        
        # Try before
        if before_idx >= 0 and current_tokens < max_tokens:
            sent = sentences[before_idx]
            sent_tokens = estimate_tokens(sent)
            if current_tokens + sent_tokens <= max_tokens:
                context_sentences.insert(0, sent)
                current_tokens += sent_tokens
                before_idx -= 1
                added = True
        
        if not added:
            break
    
    # Join sentences
    context = ' '.join(context_sentences)
    
    # Ensure ends with punctuation
    if context and context[-1] not in '.!?"\')]':
        last_period = context.rfind('. ')
        if last_period > 100:
            context = context[:last_period + 1]
        else:
            context = context.rstrip() + '.'
    
    # Hard cap at MAX_CHARS
    if len(context) > MAX_CHARS:
        truncated = context[:MAX_CHARS]
        last_period = truncated.rfind('. ')
        if last_period > 100:
            context = truncated[:last_period + 1]
        else:
            context = truncated.rsplit(' ', 1)[0] + '.'
    
    return context, current_tokens


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    # Test dengan Mahfud MD article
    mahfud_article = """Mahfud MD menyatakan video beredar di media sosial mengenai Reformasi Jilid 2 Agustus 2026 adalah hoaks. Video lama tanggal 30 Agustus 2025 disalahgunakan untuk menciptakan kesan seolah menanggapi situasi politik saat ini. Mahfud MD mengimbau masyarakat agar selalu memverifikasi informasi dan tidak mudah terprovokasi konten tidak jelas. Mantan Menko Polhukam Mahfud MD memberikan klarifikasi tegas terkait beredarnya potongan video di media sosial yang mengeklaim dirinya menyampaikan dukungan untuk Reformasi Jilid 2. Melalui unggahan di akun Instagram resminya, Mahfud MD menjelaskan bahwa video yang beredar merupakan potongan dari sesi diskusi lama. Dalam klarifikasi tersebut, Mahfud menekankan bahwa ia tidak pernah menyampaikan dukungan untuk aksi Reformasi Jilid 2. Ia juga mengimbau masyarakat untuk tidak mudah percaya pada informasi yang belum terverifikasi. Mahfud MD merupakan tokoh hukum dan politik Indonesia yang pernah menjabat sebagai Menko Polhukam."""
    
    print(f"{'='*70}")
    print(f"TEST: Context Extraction v22 — Token Budget Optimization")
    print(f"{'='*70}")
    print(f"\nArticle length: {len(mahfud_article)} chars")
    print(f"Token budget: {AVAILABLE_TOKENS} tokens (MAX_SEQ=256 - premise=10)")
    print(f"Target chars: ~{TARGET_CHARS}")
    
    context, tokens = extract_context_optimized(mahfud_article, "Mahfud MD")
    
    print(f"\nExtracted Context ({len(context)} chars, ~{tokens} tokens):")
    print(f"---")
    print(context)
    print(f"---")
    print(f"\nToken utilization: {tokens}/{AVAILABLE_TOKENS} ({tokens/AVAILABLE_TOKENS*100:.1f}%)")
    print(f"Char utilization: {len(context)}/{TARGET_CHARS} ({len(context)/TARGET_CHARS*100:.1f}%)")
    print(f"Entity present: {'mahfud md' in context.lower()}")
    
    # Compare with old approach (3 sentences only)
    print(f"\n{'='*70}")
    print(f"COMPARISON: Old approach (3 sentences) vs New (token budget)")
    print(f"{'='*70}")
    
    # Old: 3 sentences
    sentences = split_sentences(mahfud_article)
    old_context = ' '.join(sentences[:3])
    old_tokens = estimate_tokens(old_context)
    
    print(f"\nOLD (3 sentences): {len(old_context)} chars, ~{old_tokens} tokens")
    print(f"  Token utilization: {old_tokens}/{AVAILABLE_TOKENS} ({old_tokens/AVAILABLE_TOKENS*100:.1f}%)")
    print(f"  Context: {old_context[:150]}...")
    
    print(f"\nNEW (token budget): {len(context)} chars, ~{tokens} tokens")
    print(f"  Token utilization: {tokens}/{AVAILABLE_TOKENS} ({tokens/AVAILABLE_TOKENS*100:.1f}%)")
    print(f"  Context: {context[:150]}...")
    
    improvement = tokens / AVAILABLE_TOKENS * 100 - old_tokens / AVAILABLE_TOKENS * 100
    print(f"\nImprovement: +{improvement:.1f}% token utilization")
