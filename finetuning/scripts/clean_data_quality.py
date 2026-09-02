#!/usr/bin/env python3
"""
clean_data_quality.py
=====================
Perbaiki masalah kebersihan data di dataset_gold_standard_final.jsonl.

Masalah yang ditemukan:
  1. 36 rows non-ASCII (Â, •, €, Arabic, Chinese chars, emoji)
  2. 2 rows emoji (📖, Chinese chars in name)
  3. 1 row repeated chars (_____ separator)
  4. 52 rows truncated start (dimulai lowercase, mid-word)
  5. 1 row no terminal punctuation (terpotong mid-word)

Fix strategy:
  1. Hapus noise chars (Â, •, €, emoji, ____)
  2. Normalisasi non-ASCII ke ASCII equivalent (é→e, ö→o, ī→i)
  3. Fix truncated start: cari awal kalimat di article_text asli
  4. Fix no terminal punct: hapus kata terakhir yang tidak lengkap
  5. Hapus rows yang tidak bisa diperbaiki
"""
import json, re, unicodedata
from pathlib import Path
from collections import Counter

DATASET = Path(__file__).resolve().parent.parent / "datasets" / "dataset_gold_standard_final.jsonl"
RAW_DATASET = Path(__file__).resolve().parent.parent / "datasets" / "dataset_v10_final.jsonl"
OUTPUT = Path(__file__).resolve().parent.parent / "datasets" / "dataset_gold_standard_final.jsonl"

# Emoji pattern
EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF\U00002600-\U000026FF\U00002700-\U000027BF]"
)

# Noise chars to remove (Â mojibake, bullet, euro, arrows, etc)
NOISE_CHARS = {
    'Â': '', 'â': '', '•': ' ', '€': ' ', '↑': ' ', '↓': ' ',
    '·': ' ', 'ˈ': '', 'ˌ': '', '̍': '', '̪': '',
}

# Arabic diacritics (harakat) — remove
ARABIC_DIACRITICS = re.compile(r'[\u064B-\u0652\u0670\u0640]')


def normalize_non_ascii(text):
    """Normalize non-ASCII chars to ASCII equivalent or remove noise."""
    # Remove emoji
    text = EMOJI_PATTERN.sub('', text)
    
    # Remove Arabic diacritics
    text = ARABIC_DIACRITICS.sub('', text)
    
    # Replace noise chars
    for char, replacement in NOISE_CHARS.items():
        text = text.replace(char, replacement)
    
    # Normalize Unicode (NFKD = compatibility decomposition)
    # This converts é→e, ö→o, ī→i, etc.
    text = unicodedata.normalize('NFKD', text)
    
    # Remove combining characters (accents that became separate)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    # Remove remaining non-ASCII (Arabic, Chinese, etc) — replace with space
    text = ''.join(c if ord(c) < 128 else ' ' for c in text)
    
    # Clean up multiple spaces
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


def fix_truncated_start(text, article_text, entity_name):
    """Fix text that starts mid-word (lowercase first char).
    
    Strategy: find entity in article_text, re-extract context with proper
    sentence boundary alignment.
    """
    if not text or not text[0].islower():
        return text, False  # not truncated
    
    if not article_text:
        return text, False
    
    # Find entity position in article
    entity_lower = entity_name.lower()
    article_lower = article_text.lower()
    pos = article_lower.find(entity_lower)
    
    if pos < 0:
        # Try short forms
        parts = entity_name.split()
        for p in parts:
            if len(p) >= 4 and p.lower() in article_lower:
                pos = article_lower.find(p.lower())
                break
    
    if pos < 0:
        return text, False  # can't fix
    
    # Find sentence start before entity
    SENTENCE_END = re.compile(r'[.!?]["\')\]]?\s+')
    before_entity = article_text[:pos]
    matches = list(SENTENCE_END.finditer(before_entity))
    
    if matches:
        start = matches[-1].end()
    else:
        start = 0
    
    # Find sentence end after entity (up to 3 sentences)
    end = pos + len(entity_name)
    sent_count = 0
    for match in SENTENCE_END.finditer(article_text[end:]):
        end = end + match.end()
        sent_count += 1
        if sent_count >= 3:
            break
    
    # Extract context
    context = article_text[start:end].strip()
    
    # Clean
    context = normalize_non_ascii(context)
    context = re.sub(r'\s+', ' ', context).strip()
    
    # Verify it starts with uppercase now
    if context and context[0].isupper():
        return context, True
    
    return text, False  # couldn't fix


def fix_no_terminal_punct(text):
    """Fix text that ends mid-word (no terminal punctuation)."""
    if not text or text[-1] in '.!?"\')]':
        return text, False
    
    # Cut to last complete sentence
    SENTENCE_END = re.compile(r'[.!?]["\')\]]?\s+')
    matches = list(SENTENCE_END.finditer(text))
    
    if matches:
        # Cut after last sentence boundary
        cut_pos = matches[-1].end()
        return text[:cut_pos].strip(), True
    
    # If no sentence boundary, cut at last space before end
    last_space = text.rfind(' ', 0, len(text) - 5)
    if last_space > 50:
        return text[:last_space].strip() + '.', True
    
    return text, False


def remove_repeated_chars(text):
    """Remove repeated separator chars (___, ---, ===)."""
    # Replace 4+ repeated chars with single
    text = re.sub(r'_{4,}', ' ', text)
    text = re.sub(r'-{4,}', ' ', text)
    text = re.sub(r'={4,}', ' ', text)
    text = re.sub(r'\*{4,}', ' ', text)
    # Clean up multiple spaces
    text = re.sub(r' +', ' ', text)
    return text.strip()


def main():
    print("=" * 70)
    print("DATA QUALITY CLEANING")
    print("=" * 70)
    
    rows = [json.loads(l) for l in open(DATASET) if l.strip()]
    print(f"Input: {len(rows)} rows")
    
    # Load raw article_text for fixing truncated starts
    raw_articles = {}
    if RAW_DATASET.exists():
        raw_rows = [json.loads(l) for l in open(RAW_DATASET) if l.strip()]
        for r in raw_rows:
            raw_articles[r.get('raw_text_id', '')] = r.get('article_text', '')
        print(f"Loaded {len(raw_articles)} raw articles for context fixing")
    
    stats = {
        'non_ascii_fixed': 0,
        'emoji_removed': 0,
        'truncated_fixed': 0,
        'truncated_failed': 0,
        'no_punct_fixed': 0,
        'repeated_chars_fixed': 0,
        'rows_removed': 0,
    }
    
    cleaned = []
    
    for i, r in enumerate(rows):
        text = r['text']
        original_text = text
        modified = False
        
        # 1. Fix non-ASCII
        if any(ord(c) > 127 for c in text):
            text = normalize_non_ascii(text)
            if text != original_text:
                stats['non_ascii_fixed'] += 1
                modified = True
        
        # 2. Remove emoji (already done in normalize, but check)
        if EMOJI_PATTERN.search(text):
            text = EMOJI_PATTERN.sub('', text)
            stats['emoji_removed'] += 1
            modified = True
        
        # 3. Remove repeated chars
        if re.search(r'[_=\-]{4,}|\*{4,}', text):
            text = remove_repeated_chars(text)
            stats['repeated_chars_fixed'] += 1
            modified = True
        
        # 4. Fix truncated start (lowercase first char)
        if text and text[0].islower() and text[0] not in '"\'':
            raw_id = r.get('raw_text_id', '')
            article = raw_articles.get(raw_id, '')
            new_text, fixed = fix_truncated_start(text, article, r['entity_name'])
            if fixed:
                text = new_text
                stats['truncated_fixed'] += 1
                modified = True
            else:
                stats['truncated_failed'] += 1
        
        # 5. Fix no terminal punctuation
        if text and text[-1] not in '.!?"\')]':
            new_text, fixed = fix_no_terminal_punct(text)
            if fixed:
                text = new_text
                stats['no_punct_fixed'] += 1
                modified = True
        
        # 6. Remove rows that are too short after cleaning or still broken
        if len(text) < 80:
            stats['rows_removed'] += 1
            continue
        
        # 7. Remove rows that still start with lowercase (couldn't fix)
        if text and text[0].islower() and text[0] not in '"\'':
            # Try one more time: capitalize first letter if it's a partial word
            # Find first space, capitalize from there
            space_pos = text.find(' ')
            if space_pos > 0 and space_pos < 10:
                # Skip the partial word at start
                text = text[space_pos+1:].strip()
                if text and text[0].islower():
                    text = text[0].upper() + text[1:]
                if len(text) < 80:
                    stats['rows_removed'] += 1
                    continue
            else:
                # Just capitalize first letter
                text = text[0].upper() + text[1:]
        
        # Update row
        r['text'] = text
        r['context_chars'] = len(text)
        if modified:
            r['data_quality_cleaned'] = True
        
        cleaned.append(r)
    
    # Write output
    with open(OUTPUT, 'w') as f:
        for r in cleaned:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print(f"\n{'='*70}")
    print(f"HASIL CLEANING")
    print(f"{'='*70}")
    print(f"  Input:              {len(rows)} rows")
    print(f"  Output:             {len(cleaned)} rows")
    print(f"  Rows removed:       {stats['rows_removed']}")
    print(f"\n  Fixes applied:")
    print(f"    Non-ASCII fixed:      {stats['non_ascii_fixed']}")
    print(f"    Emoji removed:        {stats['emoji_removed']}")
    print(f"    Truncated start fixed:{stats['truncated_fixed']}")
    print(f"    Truncated failed:     {stats['truncated_failed']}")
    print(f"    No punct fixed:       {stats['no_punct_fixed']}")
    print(f"    Repeated chars fixed: {stats['repeated_chars_fixed']}")
    
    # Verify: re-audit
    print(f"\n{'='*70}")
    print(f"VERIFIKASI: RE-AUDIT SETELAH CLEANING")
    print(f"{'='*70}")
    
    issues_after = {
        'non_ascii': 0,
        'emoji': 0,
        'repeated_chars': 0,
        'truncated_start': 0,
        'no_terminal_punct': 0,
    }
    
    for r in cleaned:
        text = r['text']
        if any(ord(c) > 127 for c in text):
            issues_after['non_ascii'] += 1
        if EMOJI_PATTERN.search(text):
            issues_after['emoji'] += 1
        if re.search(r'[_=\-]{4,}|\*{4,}', text):
            issues_after['repeated_chars'] += 1
        if text and text[0].islower() and text[0] not in '"\'':
            issues_after['truncated_start'] += 1
        if text and text[-1] not in '.!?"\')]':
            issues_after['no_terminal_punct'] += 1
    
    for issue, count in issues_after.items():
        status = "✅" if count == 0 else "❌"
        print(f"  {status} {issue:25s}: {count}")
    
    print(f"\n  Total rows: {len(cleaned)}")


if __name__ == "__main__":
    main()
