#!/usr/bin/env python3
"""
test_enrichment_pipeline.py
===========================
Script penguji yang menampilkan OUTPUT LANGSUNG dari setiap tahap proses
enrichment worker sampai preprocessing.

Untuk setiap row, script menampilkan:
  TAHAP 0: Raw Article (data mentah dari portal berita)
  TAHAP 1: Entity Resolution Output (entity yang ditemukan worker)
  TAHAP 2: Context Extraction Output (context yang di-extract worker)
  TAHAP 3: Preprocessing Output (context setelah dibersihkan)
  TAHAP 4: Label Output (label + confidence dari LLM)
  TAHAP 5: Final Dataset Row (data final yang masuk training)

User bisa melihat output setiap tahap dan menilai apakah tepat atau tidak.

Usage:
  python3 test_enrichment_pipeline.py                    # default 5 rows
  python3 test_enrichment_pipeline.py --n 10              # 10 rows
  python3 test_enrichment_pipeline.py --row 5             # row ke-5 saja
  python3 test_enrichment_pipeline.py --label negative    # hanya label negative
"""
import sys, json, re, argparse, random, textwrap
from pathlib import Path
from collections import Counter

DATASET_FINAL = Path(__file__).resolve().parent.parent / "datasets" / "dataset_gold_standard_final.jsonl"
DATASET_RAW   = Path(__file__).resolve().parent.parent / "datasets" / "dataset_v10_final.jsonl"

SHORT_FORMS = {
    "joko widodo": ["jokowi"], "prabowo subianto": ["prabowo"],
    "megawati soekarnoputri": ["megawati"], "susilo bambang yudhoyono": ["sby"],
    "basuki tjahaja purnama": ["ahok"], "abdurrahman wahid": ["gus dur"],
    "ma'ruf amin": ["ma'ruf", "maruf"], "muhaimin iskandar": ["cak imin"],
    "erick thohir": ["erick"], "bima arya sugiarto": ["bima"],
    "sri mulyani indrawati": ["sri mulyani"], "ridwan kamil": ["rk"],
    "anies baswedan": ["anies"], "pramono anung": ["pram"],
    "puan maharani": ["puan"], "sufmi dasco ahmad": ["dasco"],
    "khofifah indar parawansa": ["khofifah"], "dedi mulyadi": ["dedi"],
    "tito karnavian": ["tito"], "bobby nasution": ["bobby"],
    "bahlil lahadalia": ["bahlil"], "soekarno": ["bung karno"],
    "bacharuddin jusuf habibie": ["habibie"], "bj habibie": ["habibie"],
}


def load_data():
    """Load final dataset + raw articles for comparison."""
    final_rows = [json.loads(l) for l in open(DATASET_FINAL) if l.strip()]
    
    # Load raw articles (with article_text)
    raw_map = {}
    if DATASET_RAW.exists():
        raw_rows = [json.loads(l) for l in open(DATASET_RAW) if l.strip()]
        for r in raw_rows:
            raw_map[r.get('raw_text_id', '')] = r
    
    return final_rows, raw_map


def find_entity_in_text(entity, text):
    """Check if entity or alias is in text."""
    entity_lower = entity.lower()
    text_lower = text.lower()
    
    if entity_lower in text_lower:
        return True, 'exact'
    
    if entity_lower in SHORT_FORMS:
        for sf in SHORT_FORMS[entity_lower]:
            if sf in text_lower:
                return True, f'alias:{sf}'
    
    parts = entity.split()
    if len(parts) >= 2:
        if len(parts[-1]) >= 4 and parts[-1].lower() in text_lower:
            return True, 'last_name'
        if len(parts[0]) >= 4 and parts[0].lower() in text_lower:
            return True, 'first_name'
    
    return False, 'NOT_FOUND'


def detect_entities_in_text(text, entity_db_names=None):
    """Simulate entity resolution — find all potential entities in text."""
    # Find capitalized words/phrases (potential entity names)
    # Look for sequences of capitalized words
    words = text.split()
    entities_found = []
    current_seq = []
    
    for word in words:
        clean = word.strip('.,;:!?()"\'[]{}—–-')
        if clean and clean[0].isupper() and len(clean) >= 3:
            current_seq.append(clean)
        else:
            if current_seq:
                entities_found.append(' '.join(current_seq))
                current_seq = []
    if current_seq:
        entities_found.append(' '.join(current_seq))
    
    # Deduplicate
    seen = set()
    unique = []
    for e in entities_found:
        if e.lower() not in seen and len(e) >= 4:
            seen.add(e.lower())
            unique.append(e)
    
    return unique[:10]  # limit to 10


def simulate_context_extraction(article_text, entity_name):
    """Simulate context extraction — find entity position and extract context."""
    entity_lower = entity_name.lower()
    article_lower = article_text.lower()
    
    # Find entity position
    pos = article_lower.find(entity_lower)
    match_type = 'full_match'
    
    if pos < 0:
        # Try short forms
        if entity_lower in SHORT_FORMS:
            for sf in SHORT_FORMS[entity_lower]:
                pos = article_lower.find(sf)
                if pos >= 0:
                    match_type = f'alias:{sf}'
                    break
    
    if pos < 0:
        # Try last name
        parts = entity_name.split()
        if len(parts) >= 2 and len(parts[-1]) >= 4:
            pos = article_lower.find(parts[-1].lower())
            if pos >= 0:
                match_type = 'last_name'
    
    if pos < 0:
        return None, 'entity_not_found', 0
    
    # Find sentence boundaries
    SENTENCE_END = re.compile(r'[.!?]["\')\]]?\s+')
    
    # Walk backwards to find sentence start
    before = article_text[:pos]
    matches = list(SENTENCE_END.finditer(before))
    start = matches[-1].end() if matches else 0
    
    # Walk forwards to find sentence end (up to 3 sentences)
    end = pos + len(entity_name)
    sent_count = 0
    for match in SENTENCE_END.finditer(article_text[end:]):
        end = end + match.end()
        sent_count += 1
        if sent_count >= 3:
            break
    
    context = article_text[start:end]
    return context, match_type, pos


def simulate_preprocessing(text):
    """Simulate preprocessing — show what cleaning was applied."""
    original = text
    changes = []
    
    # Check non-ASCII
    non_ascii = [c for c in text if ord(c) > 127]
    if non_ascii:
        import unicodedata
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        text = ''.join(c if ord(c) < 128 else ' ' for c in text)
        changes.append(f'Removed {len(non_ascii)} non-ASCII chars')
    
    # Check HTML entities
    html = re.findall(r'&\w+;|&#\d+;', text)
    if html:
        text = re.sub(r'&\w+;', ' ', text)
        text = re.sub(r'&#\d+;', ' ', text)
        changes.append(f'Removed {len(html)} HTML entities')
    
    # Check citation markers
    citations = re.findall(r'\[\d+\]', text)
    if citations:
        text = re.sub(r'\[\d+\]', '', text)
        changes.append(f'Removed {len(citations)} citation markers')
    
    # Check multiple whitespace
    if re.search(r'  +', text):
        count = len(re.findall(r'  +', text))
        text = re.sub(r' +', ' ', text)
        changes.append(f'Normalized {count} multiple whitespace')
    
    # Check tabs/newlines
    if '\t' in text or '\n' in text:
        text = text.replace('\t', ' ').replace('\n', ' ')
        changes.append('Removed tabs/newlines')
    
    # Strip
    text = text.strip()
    
    if text != original and not changes:
        changes.append('Minor cleanup')
    
    return text, changes


def print_separator(char='═', width=80):
    print(char * width)


def print_tahap(num, title):
    print()
    print_separator()
    print(f"  TAHAP {num}: {title}")
    print_separator()


def print_row_pipeline(row, raw_row, row_num):
    """Print full pipeline output for one row."""
    
    print(f"\n{'#'*80}")
    print(f"  ROW {row_num} — Pipeline Output Trace")
    print(f"{'#'*80}")
    
    # ===== TAHAP 0: RAW ARTICLE =====
    print_tahap(0, "RAW ARTICLE (Data Mentah dari Portal Berita)")
    
    source_url = row.get('source_url', 'N/A')
    print(f"\n  Source URL: {source_url}")
    
    # Extract portal name from URL
    if 'cnnindonesia' in source_url:
        portal = "CNN Indonesia"
    elif 'tempo.co' in source_url:
        portal = "Tempo"
    elif 'kompas.com' in source_url:
        portal = "Kompas"
    elif 'tribunnews' in source_url:
        portal = "Tribun News"
    elif 'jpnn' in source_url:
        portal = "JPNN"
    elif 'detik' in source_url:
        portal = "Detik"
    elif 'antaranews' in source_url:
        portal = "Antara News"
    else:
        portal = "Other"
    print(f"  Portal: {portal}")
    
    raw_text_id = row.get('raw_text_id', 'N/A')
    print(f"  Raw Text ID: {raw_text_id}")
    
    if raw_row:
        article_text = raw_row.get('article_text', '')
        context_text_raw = raw_row.get('context_text', '')
        print(f"\n  Article Text ({len(article_text)} chars):")
        for line in textwrap.wrap(article_text[:600], width=76, initial_indent='    ', subsequent_indent='    '):
            print(line)
        if len(article_text) > 600:
            print(f"    ... ({len(article_text) - 600} more chars)")
        
        print(f"\n  Context (sebelum cleaning, {len(context_text_raw)} chars):")
        for line in textwrap.wrap(context_text_raw[:400], width=76, initial_indent='    ', subsequent_indent='    '):
            print(line)
    else:
        print(f"\n  [Raw article tidak tersedia]")
    
    # ===== TAHAP 1: ENTITY RESOLUTION =====
    print_tahap(1, "ENTITY RESOLUTION WORKER OUTPUT")
    
    entity_name = row.get('entity_name', '')
    match_type = row.get('match_type', '')
    
    print(f"\n  Expected Entity: {entity_name}")
    print(f"  Match Type:      {match_type}")
    
    if raw_row:
        article_text = raw_row.get('article_text', '')
        detected_entities = detect_entities_in_text(article_text)
        print(f"\n  All entities detected in article ({len(detected_entities)}):")
        for e in detected_entities:
            marker = ' ◀ TARGET' if e.lower() == entity_name.lower() else ''
            print(f"    • {e}{marker}")
    
    # Verify entity in final text
    found, match_detail = find_entity_in_text(entity_name, row['text'])
    print(f"\n  Entity in final text: {'✅ YES' if found else '❌ NO'} ({match_detail})")
    
    # ===== TAHAP 2: CONTEXT EXTRACTION =====
    print_tahap(2, "CONTEXT EXTRACTION WORKER OUTPUT")
    
    if raw_row:
        article_text = raw_row.get('article_text', '')
        extracted_context, ext_match_type, entity_pos = simulate_context_extraction(article_text, entity_name)
        
        if extracted_context:
            print(f"\n  Entity position in article: char {entity_pos}")
            print(f"  Match type: {ext_match_type}")
            print(f"\n  Extracted Context ({len(extracted_context)} chars):")
            for line in textwrap.wrap(extracted_context[:500], width=76, initial_indent='    ', subsequent_indent='    '):
                print(line)
        else:
            print(f"\n  ❌ Entity not found in article — context extraction failed")
    else:
        print(f"\n  [Raw article tidak tersedia untuk simulasi]")
    
    # ===== TAHAP 3: PREPROCESSING =====
    print_tahap(3, "PREPROCESSING OUTPUT")
    
    final_text = row['text']
    print(f"\n  Final Text after preprocessing ({len(final_text)} chars):")
    for line in textwrap.wrap(final_text, width=76, initial_indent='    ', subsequent_indent='    '):
        print(line)
    
    # Show what was cleaned
    if raw_row:
        raw_context = raw_row.get('context_text', '')
        cleaned_text, changes = simulate_preprocessing(raw_context)
        if changes:
            print(f"\n  Preprocessing changes applied:")
            for change in changes:
                print(f"    • {change}")
        else:
            print(f"\n  No preprocessing changes needed (already clean)")
    
    # Quality checks
    quality_checks = []
    quality_checks.append(('Starts with uppercase', final_text[0].isupper() or final_text[0] in '"\'('))
    quality_checks.append(('Ends with punctuation', final_text[-1] in '.!?"\')]'))
    quality_checks.append(('Length 80-500 chars', 80 <= len(final_text) <= 500))
    quality_checks.append(('Entity in text', found))
    
    print(f"\n  Quality checks:")
    for check_name, passed in quality_checks:
        status = '✅' if passed else '❌'
        print(f"    {status} {check_name}")
    
    # ===== TAHAP 4: LABEL OUTPUT =====
    print_tahap(4, "LABEL OUTPUT (LLM Verification)")
    
    label = row.get('label', '')
    confidence = row.get('label_confidence', 0)
    label_source = row.get('label_source', '')
    reasoning = row.get('verification_reasoning', '')
    gold_relevancy = row.get('gold_relevancy', '')
    
    print(f"\n  Label:           {label}")
    print(f"  Confidence:      {confidence*100:.1f}%")
    print(f"  Label Source:    {label_source}")
    print(f"  Gold Relevancy:  {gold_relevancy}")
    
    if reasoning:
        print(f"\n  LLM Reasoning:")
        for line in textwrap.wrap(reasoning[:300], width=76, initial_indent='    ', subsequent_indent='    '):
            print(line)
    
    # ===== TAHAP 5: FINAL DATASET ROW =====
    print_tahap(5, "FINAL DATASET ROW (Yang Masuk Training)")
    
    print(f"\n  Input untuk model:")
    print(f"    Premise (entity):    Tentang {entity_name}")
    print(f"    Hypothesis (context): {final_text[:100]}...")
    print(f"    Label:               {label}")
    print(f"    Confidence:          {confidence*100:.1f}%")
    
    # Assessment
    print(f"\n  {'─'*60}")
    issues = []
    if not found:
        issues.append('Entity tidak ada di text')
    if not (final_text[0].isupper() or final_text[0] in '"\'('):
        issues.append('Text dimulai lowercase')
    if final_text[-1] not in '.!?"\')]':
        issues.append('Text tidak diakhiri punctuation')
    if not (80 <= len(final_text) <= 500):
        issues.append(f'Panjang text tidak optimal ({len(final_text)} chars)')
    if confidence < 0.90:
        issues.append(f'Confidence rendah ({confidence*100:.1f}%)')
    
    if issues:
        print(f"  ⚠ ISSUES ({len(issues)}):")
        for issue in issues:
            print(f"    • {issue}")
    else:
        print(f"  ✅ ROW QUALITY: SANGAT BAIK — semua checks passed")


def main():
    ap = argparse.ArgumentParser(description="Test Enrichment Pipeline — Output per Tahap")
    ap.add_argument('--n', type=int, default=5, help='Jumlah row (default: 5)')
    ap.add_argument('--row', type=int, default=None, help='Row ke-N saja (0-indexed)')
    ap.add_argument('--label', choices=['positive', 'neutral', 'negative'], default=None,
                    help='Filter by label')
    ap.add_argument('--seed', type=int, default=2024, help='Random seed')
    args = ap.parse_args()
    
    print("=" * 80)
    print("ENRICHMENT PIPELINE — OUTPUT TRACE PER TAHAP")
    print("=" * 80)
    print(f"\nScript ini menampilkan output dari SETIAP TAHAP proses enrichment:")
    print(f"  Tahap 0: Raw Article (data mentah)")
    print(f"  Tahap 1: Entity Resolution (entity detection)")
    print(f"  Tahap 2: Context Extraction (context di sekitar entity)")
    print(f"  Tahap 3: Preprocessing (cleaning)")
    print(f"  Tahap 4: Label (LLM verification)")
    print(f"  Tahap 5: Final Dataset Row (untuk training)")
    
    final_rows, raw_map = load_data()
    print(f"\nDataset: {len(final_rows)} rows, {len(raw_map)} raw articles")
    
    # Filter or sample
    if args.label:
        final_rows = [r for r in final_rows if r['label'] == args.label]
        print(f"Filtered by label '{args.label}': {len(final_rows)} rows")
    
    if args.row is not None:
        if args.row < len(final_rows):
            final_rows = [final_rows[args.row]]
        else:
            print(f"Row {args.row} tidak ada (max: {len(final_rows)-1})")
            return
    else:
        random.seed(args.seed)
        final_rows = random.sample(final_rows, min(args.n, len(final_rows)))
    
    print(f"\nMenampilkan {len(final_rows)} rows...\n")
    
    # Process each row
    for i, row in enumerate(final_rows):
        raw_row = raw_map.get(row.get('raw_text_id', ''), None)
        print_row_pipeline(row, raw_row, i + 1)
        
        if i < len(final_rows) - 1:
            pass  # auto-continue
    
    print(f"\n{'#'*80}")
    print(f"  SELESAI — {len(final_rows)} rows ditampilkan")
    print(f"{'#'*80}")


if __name__ == "__main__":
    main()
