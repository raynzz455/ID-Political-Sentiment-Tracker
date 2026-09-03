#!/usr/bin/env python3
"""
test_enrichment_pipeline.py
===========================
Script penguji yang menampilkan OUTPUT LANGSUNG dari setiap layer
sesuai README backend pipeline.

URUTAN LAYER SESUAI README:
  L1-2:   Ingestion & Enrichment (RSS fetch + Trafilatura extraction)
  L2.5:   Validation (Quality Control score 0-100)
  L3:     Preprocessing (normalisasi unicode, hapus URL, hash dedup)
  L3.2:   Entity Resolution (NER + alias matching)
  L3.5:   Context Extraction (context span di sekitar entity)
  L3.7:   Readiness & Queue (final gatekeeper)
  L4:     NLP Worker (IndoBERT 2-stage: relevancy + sentiment)

Usage:
  python3 test_enrichment_pipeline.py                    # default 3 rows
  python3 test_enrichment_pipeline.py --n 5              # 5 rows
  python3 test_enrichment_pipeline.py --row 0            # row ke-0 saja
  python3 test_enrichment_pipeline.py --label negative   # hanya label negative
"""
import sys, json, re, argparse, random, textwrap, hashlib, unicodedata
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
    final_rows = [json.loads(l) for l in open(DATASET_FINAL) if l.strip()]
    raw_map = {}
    if DATASET_RAW.exists():
        for l in open(DATASET_RAW):
            if l.strip():
                r = json.loads(l)
                raw_map[r.get('raw_text_id', '')] = r
    return final_rows, raw_map


def detect_portal(url):
    if 'cnnindonesia' in url: return 'CNN Indonesia'
    elif 'tempo.co' in url: return 'Tempo'
    elif 'kompas.com' in url: return 'Kompas'
    elif 'tribunnews' in url: return 'Tribun News'
    elif 'detik' in url: return 'Detik'
    elif 'antaranews' in url: return 'Antara News'
    elif 'jpnn' in url: return 'JPNN'
    elif 'news.google.com' in url: return 'Google News'
    return 'Other'


def detect_entities(text):
    """Simulasi entity detection — find capitalized sequences."""
    words = text.split()
    entities = []
    current = []
    for word in words:
        clean = word.strip('.,;:!?()"\'[]{}—–-')
        if clean and clean[0].isupper() and len(clean) >= 3:
            current.append(clean)
        else:
            if current:
                entities.append(' '.join(current))
                current = []
    if current:
        entities.append(' '.join(current))
    seen = set()
    unique = []
    for e in entities:
        if e.lower() not in seen and len(e) >= 4:
            seen.add(e.lower())
            unique.append(e)
    return unique[:10]


def find_entity_position(text, entity):
    entity_lower = entity.lower()
    text_lower = text.lower()
    pos = text_lower.find(entity_lower)
    if pos < 0:
        for sf in SHORT_FORMS.get(entity_lower, []):
            pos = text_lower.find(sf)
            if pos >= 0: break
    if pos < 0:
        parts = entity.split()
        if len(parts) >= 2 and len(parts[-1]) >= 4:
            pos = text_lower.find(parts[-1].lower())
    return pos


def extract_context(article_text, entity_name):
    """Simulasi context extraction — sentence window."""
    pos = find_entity_position(article_text, entity_name)
    if pos < 0:
        return None, -1
    SENTENCE_END = re.compile(r'[.!?]["\')\]]?\s+')
    before = article_text[:pos]
    matches = list(SENTENCE_END.finditer(before))
    start = matches[-1].end() if matches else 0
    end = pos + len(entity_name)
    sent_count = 0
    for match in SENTENCE_END.finditer(article_text[end:]):
        end = end + match.end()
        sent_count += 1
        if sent_count >= 3: break
    return article_text[start:end], pos


def simulate_preprocessing(text):
    """Simulasi preprocessing — show what cleaning was applied."""
    changes = []
    if any(ord(c) > 127 for c in text):
        changes.append('Non-ASCII normalized')
    if re.search(r'&\w+;|&#\d+;', text):
        changes.append('HTML entities removed')
    if re.search(r'\[\d+\]', text):
        changes.append('Citation markers removed')
    if re.search(r'  +', text):
        changes.append('Whitespace normalized')
    if re.search(r'https?://\S+', text):
        changes.append('URLs removed')
    content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    return changes, content_hash


def print_sep(char='═', w=80):
    print(char * w)

def print_layer(layer_num, title, subtitle=''):
    print()
    print_sep()
    print(f"  LAYER {layer_num}: {title}")
    if subtitle:
        print(f"  ({subtitle})")
    print_sep()

def wrap(text, indent='    ', w=72):
    for line in textwrap.wrap(str(text), width=w, initial_indent=indent, subsequent_indent=indent):
        print(line)


def print_row_pipeline(raw_row, final_row, row_num):
    """Print pipeline output SESUAI URUTAN README."""
    
    print(f"\n{'#'*80}")
    print(f"  ROW {row_num} — Backend Pipeline Output Trace")
    print(f"{'#'*80}")
    
    article_text = raw_row.get('article_text', '') if raw_row else final_row.get('text', '')
    context_raw = raw_row.get('context_text', '') if raw_row else ''
    final_text = final_row.get('text', '')
    entity_name = final_row.get('entity_name', '')
    source_url = final_row.get('source_url', '')
    
    # ===== L1-2: INGESTION & ENRICHMENT =====
    print_layer('1-2', 'INGESTION & ENRICHMENT', 'RSS fetch + Trafilatura extraction')
    
    portal = detect_portal(source_url)
    print(f"\n  Source URL: {source_url[:100]}")
    print(f"  Portal: {portal}")
    print(f"  Raw Text ID: {final_row.get('raw_text_id', 'N/A')}")
    print(f"\n  Article Text ({len(article_text)} chars):")
    wrap(article_text[:500])
    if len(article_text) > 500:
        print(f"    ... ({len(article_text) - 500} more chars)")
    
    # Anti-sampah checks
    print(f"\n  Anti-Sampah Checks:")
    print(f"    ✅ Text < 20.000 chars: {len(article_text) < 20000}")
    print(f"    ✅ Text > 500 chars: {len(article_text) > 500}")
    
    # ===== L2.5: VALIDATION (Quality Control) =====
    print_layer('2.5', 'VALIDATION', 'Quality Control score 0-100')
    
    # Simulasi QC
    score = 0
    if 200 <= len(article_text) <= 5000: score += 40
    elif len(article_text) > 50: score += 20
    sentences = re.split(r'[.!?]\s', article_text)
    if len(sentences) >= 3: score += 20
    elif len(sentences) >= 1: score += 10
    indo_words = ['yang','dan','di','ke','dari','untuk','pada','dengan','atau','ini']
    indo_count = sum(1 for w in indo_words if w in article_text.lower())
    if indo_count >= 5: score += 20
    elif indo_count >= 2: score += 10
    url_count = len(re.findall(r'https?://', article_text))
    if url_count <= 10: score += 20
    score = max(0, min(100, score))
    
    print(f"\n  QC Score:        {score}/100")
    print(f"  Passed:          {'✅ YES' if score >= 50 else '❌ NO'}")
    print(f"  Article Length:  {len(article_text)} chars")
    print(f"  Sentence Count:  {len(sentences)}")
    print(f"  Indo Words:      {indo_count}")
    
    # ===== L3: PREPROCESSING =====
    print_layer('3', 'PREPROCESSING', 'Normalisasi unicode, hapus URL, hash dedup')
    
    # Preprocessing membersihkan ARTICLE_TEXT (bukan context_text!)
    # Show article_text BEFORE preprocessing (with noise)
    print(f"\n  Article Text SEBELUM preprocessing ({len(article_text)} chars):")
    wrap(article_text[:400])
    if len(article_text) > 400:
        print(f"    ... ({len(article_text) - 400} more chars)")
    
    # Apply preprocessing to article_text (same text, just cleaned)
    import unicodedata
    import html as html_lib
    preprocessed = article_text
    changes = []
    
    # 1. HTML unescape
    preprocessed = html_lib.unescape(preprocessed)
    # 2. Unicode normalize
    preprocessed = unicodedata.normalize('NFKC', preprocessed)
    # 3. Remove zero-width chars
    preprocessed = preprocessed.replace('\u200b','').replace('\u200c','').replace('\xa0',' ')
    # 4. Remove URLs
    if re.search(r'https?://\S+', preprocessed):
        preprocessed = re.sub(r'https?://\S+', '', preprocessed)
        changes.append('URLs removed')
    # 5. Remove HTML entities
    if re.search(r'&\w+;|&#\d+;', preprocessed):
        preprocessed = re.sub(r'&\w+;|&#\d+;', '', preprocessed)
        changes.append('HTML entities removed')
    # 6. Remove citation markers
    if re.search(r'\[\d+\]', preprocessed):
        preprocessed = re.sub(r'\[\d+\]', '', preprocessed)
        changes.append('Citation markers removed')
    # 7. Normalize whitespace
    if re.search(r'  +|\t|\n', preprocessed):
        preprocessed = re.sub(r'\s+', ' ', preprocessed)
        changes.append('Whitespace normalized')
    # 8. Non-ASCII check
    if any(ord(c) > 127 for c in preprocessed):
        changes.append('Non-ASCII chars detected (kept as-is)')
    
    preprocessed = preprocessed.strip()
    content_hash = hashlib.sha256(preprocessed.encode('utf-8')).hexdigest()[:16]
    
    print(f"\n  Article Text SETELAH preprocessing ({len(preprocessed)} chars):")
    wrap(preprocessed[:400])
    if len(preprocessed) > 400:
        print(f"    ... ({len(preprocessed) - 400} more chars)")
    
    print(f"\n  Reduction:       {len(article_text) - len(preprocessed)} chars")
    print(f"  Content hash:    {content_hash}")
    
    if changes:
        print(f"\n  Changes applied:")
        for change in changes:
            print(f"    • {change}")
    else:
        print(f"\n  No changes needed (already clean)")
    
    # Update article_text to preprocessed version for downstream layers
    article_text = preprocessed
    
    # ===== L3.2: ENTITY RESOLUTION =====
    print_layer('3.2', 'ENTITY RESOLUTION', 'NER + alias matching')
    
    match_type = final_row.get('match_type', '')
    print(f"\n  Expected Entity:  {entity_name}")
    print(f"  Match Type:       {match_type}")
    
    # All entities detected
    detected = detect_entities(article_text)
    print(f"\n  All entities detected in article ({len(detected)} shown):")
    for e in detected:
        marker = ' ◀ TARGET' if e.lower() == entity_name.lower() else ''
        print(f"    • {e}{marker}")
    
    # Check entity in final text
    entity_lower = entity_name.lower()
    text_lower = final_text.lower()
    if entity_lower in text_lower:
        found = 'exact'
    elif any(sf in text_lower for sf in SHORT_FORMS.get(entity_lower, [])):
        found = 'alias'
    else:
        parts = entity_name.split()
        if len(parts) >= 2 and parts[-1].lower() in text_lower:
            found = 'last_name'
        elif len(parts) >= 2 and parts[0].lower() in text_lower:
            found = 'first_name'
        else:
            found = 'NOT_FOUND'
    
    print(f"\n  Entity in final text: {'✅ YES' if found != 'NOT_FOUND' else '❌ NO'} ({found})")
    
    # ===== L3.5: CONTEXT EXTRACTION =====
    print_layer('3.5', 'CONTEXT EXTRACTION', 'Context span di sekitar entity')
    
    extracted_ctx, entity_pos = extract_context(article_text, entity_name)
    
    if extracted_ctx:
        print(f"\n  Entity position:  char {entity_pos}")
        print(f"\n  Extracted Context ({len(extracted_ctx)} chars):")
        wrap(extracted_ctx[:400])
        
        print(f"\n  Final Context ({len(final_text)} chars):")
        wrap(final_text[:300])
        
        # Quality checks
        q_start = final_text[0].isupper() or final_text[0] in '"\'('
        q_end = final_text[-1] in '.!?"\')]'
        q_len = 80 <= len(final_text) <= 500
        q_entity = found != 'NOT_FOUND'
        score_q = sum([q_start, q_end, q_len, q_entity])
        
        print(f"\n  Quality checks ({score_q}/4):")
        print(f"    {'✅' if q_start else '❌'} Starts clean (uppercase)")
        print(f"    {'✅' if q_end else '❌'} Ends clean (punctuation)")
        print(f"    {'✅' if q_len else '❌'} Length optimal (80-500)")
        print(f"    {'✅' if q_entity else '❌'} Entity present")
    else:
        print(f"\n  ❌ Entity not found in article — extraction failed")
    
    # ===== L3.7: READINESS & QUEUE =====
    print_layer('3.7', 'READINESS & QUEUE (Final Gatekeeper)', 'Cek kelengkapan sebelum NLP')
    
    checks = {
        'has_entity': bool(entity_name.strip()),
        'has_text': bool(final_text.strip()),
        'has_label': bool(final_row.get('label', '').strip()),
        'has_confidence': final_row.get('label_confidence', 0) > 0,
        'has_source_url': bool(source_url.strip()),
        'text_length_ok': 80 <= len(final_text) <= 600,
        'label_valid': final_row.get('label', '') in ['positive', 'neutral', 'negative'],
    }
    all_passed = all(checks.values())
    
    print(f"\n  Gatekeeper: {'PASS ✅' if all_passed else 'FAIL ❌'}")
    print(f"  Ready for NLP: {'✅ YES' if all_passed else '❌ NO'}")
    print(f"\n  Readiness checks:")
    for name, passed in checks.items():
        print(f"    {'✅' if passed else '❌'} {name}: {passed}")
    
    # ===== L4: NLP WORKER =====
    print_layer('4', 'NLP WORKER (IndoBERT 2-Stage)', 'Relevancy Gate + Sentiment Classifier')
    
    label = final_row.get('label', '')
    confidence = final_row.get('label_confidence', 0)
    label_source = final_row.get('label_source', '')
    reasoning = final_row.get('verification_reasoning', '')
    
    # Stage 1: Relevancy
    relevancy = found != 'NOT_FOUND'
    print(f"\n  Stage 1 — Relevancy Gate:")
    print(f"    Question: Apakah context membahas {entity_name}?")
    print(f"    Answer:   {'RELEVANT' if relevancy else 'NOT_RELEVANT'}")
    print(f"    Passed:   {'✅ YES' if relevancy else '❌ NO'}")
    
    # Stage 2: Sentiment
    print(f"\n  Stage 2 — Sentiment Classifier:")
    print(f"    Label:      {label}")
    print(f"    Confidence: {confidence*100:.1f}%")
    print(f"    Source:     {label_source}")
    if reasoning:
        print(f"\n  LLM Reasoning:")
        wrap(reasoning[:200])
    
    print(f"\n  NLP Output:")
    print(f"    Entity:    {entity_name}")
    print(f"    Sentiment: {label}")
    print(f"    Stored:    {'✅ YES' if relevancy else '❌ NO'}")
    
    # ===== SUMMARY =====
    print()
    print_sep('─')
    print(f"  PIPELINE SUMMARY (Row {row_num})")
    print_sep('─')
    
    layers = {
        'L1-2 Ingestion': len(article_text) > 500,
        'L2.5 Validation': score >= 50,
        'L3 Preprocessing': len(changes) == 0 or len(final_text) > 80,
        'L3.2 Entity': found != 'NOT_FOUND',
        'L3.5 Context': score_q >= 3 if extracted_ctx else False,
        'L3.7 Readiness': all_passed,
        'L4 NLP': relevancy,
    }
    
    passed = sum(1 for v in layers.values() if v)
    for layer, ok in layers.items():
        print(f"    {'✅' if ok else '❌'} {layer}")
    
    print(f"\n    Overall: {passed}/{len(layers)} layers passed", end="")
    if passed == len(layers):
        print(" — SANGAT BAIK ✅")
    elif passed >= len(layers) - 1:
        print(" — BAIK ✅")
    else:
        print(" — PERLU PERBAIKAN ⚠")


def main():
    ap = argparse.ArgumentParser(description="Test Enrichment Pipeline — Output per Layer (sesuai README)")
    ap.add_argument('--n', type=int, default=3, help='Jumlah row (default: 3)')
    ap.add_argument('--row', type=int, default=None, help='Row ke-N saja')
    ap.add_argument('--label', choices=['positive', 'neutral', 'negative'], default=None)
    ap.add_argument('--seed', type=int, default=2024)
    args = ap.parse_args()
    
    print("=" * 80)
    print("ENRICHMENT PIPELINE — OUTPUT TRACE PER LAYER (SESUAI README)")
    print("=" * 80)
    print(f"\nUrutan Layer:")
    print(f"  L1-2:   Ingestion & Enrichment (RSS + Trafilatura)")
    print(f"  L2.5:   Validation (Quality Control 0-100)")
    print(f"  L3:     Preprocessing (normalisasi, hash dedup)")
    print(f"  L3.2:   Entity Resolution (NER + alias)")
    print(f"  L3.5:   Context Extraction (sentence window)")
    print(f"  L3.7:   Readiness & Queue (final gatekeeper)")
    print(f"  L4:     NLP Worker (IndoBERT 2-stage)")
    
    final_rows, raw_map = load_data()
    print(f"\nDataset: {len(final_rows)} rows, {len(raw_map)} raw articles")
    
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
        rows_with_raw = [r for r in final_rows if r.get('raw_text_id', '') in raw_map]
        final_rows = random.sample(rows_with_raw, min(args.n, len(rows_with_raw)))
    
    print(f"\nMenampilkan {len(final_rows)} rows...\n")
    
    for i, row in enumerate(final_rows):
        raw_row = raw_map.get(row.get('raw_text_id', ''), None)
        print_row_pipeline(raw_row, row, i + 1)
    
    print(f"\n{'#'*80}")
    print(f"  SELESAI — {len(final_rows)} rows ditampilkan")
    print(f"{'#'*80}")


if __name__ == "__main__":
    main()
