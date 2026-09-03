#!/usr/bin/env python3
"""
test_backend_pipeline.py
=========================
Script penguji OUTPUT dari SETIAP LAYER backend pipeline sesuai README.

Workflow backend lengkap (7 layer):
  L1-2:   Ingestion & Enrichment (RSS fetch + Trafilatura extraction)
  L2.5:   Validation (Quality Control 0-100)
  L3:     Preprocessing (normalisasi unicode, hapus URL, hash dedup)
  L3.2:   Entity Resolution (NER + alias matching)
  L3.5:   Context Extraction (context span di sekitar entity)
  L3.7:   Readiness & Queue (final gatekeeper, pgmq)
  L4:     NLP Worker (IndoBERT 2-stage: relevancy + sentiment)
  L5:     Database (Supabase storage)
  L6:     Dashboard (Next.js frontend)

Script ini menampilkan output dari setiap layer untuk verifikasi manual.
User bisa melihat output setiap tahap dan menilai apakah tepat atau tidak.

Usage:
  python3 test_backend_pipeline.py                    # default 3 rows
  python3 test_backend_pipeline.py --n 5              # 5 rows
  python3 test_backend_pipeline.py --row 0            # row ke-0 saja
  python3 test_backend_pipeline.py --layer 3          # hanya layer 3 (preprocessing)

NOTE: Layer 1 (RSS fetch) dan Layer 4 (NLP inference) memerlukan:
  - Supabase connection (untuk ambil raw data)
  - IndoBERT model (untuk sentiment prediction)
  Script ini fokus pada Layer 2-3.7 yang bisa di-test dari dataset lokal.
"""
import sys, os, json, re, argparse, random, textwrap, hashlib, unicodedata
from pathlib import Path
from collections import Counter

# Paths
DATASET_FINAL = Path(__file__).resolve().parent.parent / "datasets" / "dataset_gold_standard_final.jsonl"
DATASET_RAW   = Path(__file__).resolve().parent.parent / "datasets" / "dataset_v10_final.jsonl"
DATASET_MERGED = Path(__file__).resolve().parent.parent / "datasets" / "dataset_merged_final.jsonl"

# Short forms for entity matching
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

EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]"
)


def load_all_data():
    """Load semua dataset untuk perbandingan antar layer."""
    data = {
        'final': [],
        'raw': {},
        'merged': {},
    }
    
    # Final dataset (hasil akhir)
    if DATASET_FINAL.exists():
        data['final'] = [json.loads(l) for l in open(DATASET_FINAL) if l.strip()]
    
    # Raw dataset (dengan article_text)
    if DATASET_RAW.exists():
        for l in open(DATASET_RAW):
            if l.strip():
                r = json.loads(l)
                data['raw'][r.get('raw_text_id', '')] = r
    
    # Merged dataset (dataset awal sebelum LLM verify)
    if DATASET_MERGED.exists():
        for l in open(DATASET_MERGED):
            if l.strip():
                r = json.loads(l)
                data['merged'][r.get('raw_text_id', '')] = r
    
    return data


# ============================================================
# LAYER 1-2: INGESTION & ENRICHMENT
# ============================================================

def test_layer_1_2_ingestion_enrichment(raw_row):
    """Layer 1-2: Ingestion (RSS fetch) & Enrichment (Trafilatura extraction).
    
    Output yang diperiksa:
    - Source URL (dari RSS feed)
    - Article text (hasil Trafilatura extraction)
    - Title (headline)
    - Anti-sampah checks (>20k chars, title mismatch)
    """
    result = {
        'source_url': raw_row.get('source_url', ''),
        'article_text': raw_row.get('article_text', ''),
        'title': raw_row.get('title', raw_row.get('headline', '')),
        'raw_text_id': raw_row.get('raw_text_id', ''),
    }
    
    # Anti-sampah checks (Layer 2)
    article = result['article_text']
    result['checks'] = {
        'text_length_ok': len(article) < 20000,  # anti section leakage
        'text_not_empty': len(article.strip()) > 50,
        'has_content': len(article.strip()) > 100,
    }
    
    # Detect portal from URL
    url = result['source_url']
    if 'cnnindonesia' in url: result['portal'] = 'CNN Indonesia'
    elif 'tempo.co' in url: result['portal'] = 'Tempo'
    elif 'kompas.com' in url: result['portal'] = 'Kompas'
    elif 'tribunnews' in url: result['portal'] = 'Tribun News'
    elif 'detik' in url: result['portal'] = 'Detik'
    elif 'antaranews' in url: result['portal'] = 'Antara News'
    elif 'jpnn' in url: result['portal'] = 'JPNN'
    elif 'news.google.com' in url: result['portal'] = 'Google News (aggregator)'
    else: result['portal'] = 'Other'
    
    return result


# ============================================================
# LAYER 2.5: VALIDATION (Quality Control)
# ============================================================

def test_layer_2_5_validation(raw_row):
    """Layer 2.5: Validation — Quality Control score (0-100).
    
    Simulasi QC: menilai kualitas teks berdasarkan:
    - Length (terlalu pendek = reject)
    - Coherence (apakah teks nyambung?)
    - Language (apakah Bahasa Indonesia?)
    - Spam detection
    """
    article = raw_row.get('article_text', '')
    
    score = 0
    issues = []
    
    # Check 1: Length (20-100 = poor, 100-500 = ok, 500-5000 = good, >5000 = too long)
    if len(article) < 50:
        score += 0
        issues.append(f'too_short ({len(article)} chars)')
    elif len(article) < 200:
        score += 30
        issues.append('short_article')
    elif len(article) <= 5000:
        score += 40
    else:
        score += 20
        issues.append(f'very_long ({len(article)} chars)')
    
    # Check 2: Has sentences (coherence)
    sentences = re.split(r'[.!?]\s', article)
    if len(sentences) >= 3:
        score += 20
    elif len(sentences) >= 1:
        score += 10
    else:
        score += 0
        issues.append('no_sentences')
    
    # Check 3: Has Indonesian words (language detection simple)
    indo_words = ['yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'pada', 'dengan',
                  'atau', 'ini', 'itu', 'tidak', 'akan', 'sudah', 'juga']
    indo_count = sum(1 for w in indo_words if w in article.lower())
    if indo_count >= 5:
        score += 20
    elif indo_count >= 2:
        score += 10
    else:
        score += 0
        issues.append('not_indonesian')
    
    # Check 4: Not spam (no excessive URLs, no repeated content)
    url_count = len(re.findall(r'https?://', article))
    if url_count > 10:
        score -= 20
        issues.append(f'spam_urls ({url_count})')
    else:
        score += 20
    
    # Clamp 0-100
    score = max(0, min(100, score))
    
    return {
        'qc_score': score,
        'passed': score >= 50,
        'issues': issues,
        'article_length': len(article),
        'sentence_count': len(sentences),
        'indo_word_count': indo_count,
    }


# ============================================================
# LAYER 3: PREPROCESSING
# ============================================================

def test_layer_3_preprocessing(raw_row, final_row):
    """Layer 3: Preprocessing — normalisasi unicode, hapus URL, hash dedup.
    
    Output yang diperiksa:
    - Unicode normalization (NFKD)
    - URL removal
    - HTML entity removal
    - Citation marker removal
    - Whitespace normalization
    - Content hash (untuk dedup)
    """
    raw_text = raw_row.get('context_text', '') or raw_row.get('article_text', '')
    final_text = final_row.get('text', '')
    
    changes = []
    
    # Check what was cleaned
    # 1. Non-ASCII
    non_ascii_raw = [c for c in raw_text if ord(c) > 127]
    non_ascii_final = [c for c in final_text if ord(c) > 127]
    if non_ascii_raw and not non_ascii_final:
        changes.append(f'Non-ASCII removed: {len(non_ascii_raw)} chars')
    
    # 2. HTML entities
    html_raw = re.findall(r'&\w+;|&#\d+;', raw_text)
    if html_raw:
        changes.append(f'HTML entities removed: {len(html_raw)}')
    
    # 3. Citation markers
    citations_raw = re.findall(r'\[\d+\]', raw_text)
    if citations_raw:
        changes.append(f'Citation markers removed: {len(citations_raw)}')
    
    # 4. URLs
    urls_raw = re.findall(r'https?://\S+', raw_text)
    if urls_raw:
        changes.append(f'URLs removed: {len(urls_raw)}')
    
    # 5. Multiple whitespace
    if re.search(r'  +', raw_text):
        changes.append('Multiple whitespace normalized')
    
    # 6. Tabs/newlines
    if '\t' in raw_text or '\n' in raw_text:
        changes.append('Tabs/newlines removed')
    
    # Content hash (for dedup)
    content_hash = hashlib.sha256(final_text.encode('utf-8')).hexdigest()[:16]
    
    return {
        'raw_length': len(raw_text),
        'final_length': len(final_text),
        'length_reduction': len(raw_text) - len(final_text),
        'changes_applied': changes,
        'content_hash': content_hash,
        'is_clean': len(non_ascii_final) == 0 and not re.search(r'&\w+;|\[\d+\]|  +', final_text),
    }


# ============================================================
# LAYER 3.2: ENTITY RESOLUTION
# ============================================================

def test_layer_3_2_entity_resolution(raw_row, final_row):
    """Layer 3.2: Entity Resolution — NER + alias matching.
    
    Output yang diperiksa:
    - Entity yang ditemukan (target)
    - Match type (full, alias, first_name, last_name)
    - All entities detected in article
    - Entity confidence
    """
    entity_name = final_row.get('entity_name', '')
    match_type = final_row.get('match_type', '')
    article = raw_row.get('article_text', '')
    
    # Find all capitalized sequences (potential entities)
    words = article.split()
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
    unique_entities = []
    for e in entities_found:
        if e.lower() not in seen and len(e) >= 4:
            seen.add(e.lower())
            unique_entities.append(e)
    
    # Check entity in final text
    entity_lower = entity_name.lower()
    text_lower = final_row['text'].lower()
    
    if entity_lower in text_lower:
        found_in_final = 'exact'
    elif any(sf in text_lower for sf in SHORT_FORMS.get(entity_lower, [])):
        found_in_final = 'alias'
    else:
        parts = entity_name.split()
        if len(parts) >= 2 and parts[-1].lower() in text_lower:
            found_in_final = 'last_name'
        elif len(parts) >= 2 and parts[0].lower() in text_lower:
            found_in_final = 'first_name'
        else:
            found_in_final = 'NOT_FOUND'
    
    return {
        'expected_entity': entity_name,
        'match_type': match_type,
        'all_entities_detected': unique_entities[:10],
        'total_entities_detected': len(unique_entities),
        'entity_in_final_text': found_in_final,
        'is_target_found': found_in_final != 'NOT_FOUND',
    }


# ============================================================
# LAYER 3.5: CONTEXT EXTRACTION
# ============================================================

def test_layer_3_5_context_extraction(raw_row, final_row):
    """Layer 3.5: Context Extraction — context span di sekitar entity.
    
    Output yang diperiksa:
    - Entity position in article
    - Extracted context (sentence window)
    - Context quality (clean start, clean end, length, entity present)
    """
    article = raw_row.get('article_text', '')
    entity = final_row.get('entity_name', '')
    final_text = final_row.get('text', '')
    
    entity_lower = entity.lower()
    article_lower = article.lower()
    
    # Find entity position
    pos = article_lower.find(entity_lower)
    if pos < 0:
        # Try short forms
        for sf in SHORT_FORMS.get(entity_lower, []):
            pos = article_lower.find(sf)
            if pos >= 0: break
    
    if pos < 0:
        # Try last name
        parts = entity.split()
        if len(parts) >= 2 and len(parts[-1]) >= 4:
            pos = article_lower.find(parts[-1].lower())
    
    if pos < 0:
        return {
            'entity_position': -1,
            'extraction_status': 'FAILED — entity not found in article',
            'extracted_context': '',
            'final_context': final_text,
        }
    
    # Simulate sentence window extraction
    SENTENCE_END = re.compile(r'[.!?]["\')\]]?\s+')
    before = article[:pos]
    matches = list(SENTENCE_END.finditer(before))
    start = matches[-1].end() if matches else 0
    
    end = pos + len(entity)
    sent_count = 0
    for match in SENTENCE_END.finditer(article[end:]):
        end = end + match.end()
        sent_count += 1
        if sent_count >= 3: break
    
    extracted = article[start:end]
    
    # Quality checks on final text
    quality = {
        'starts_clean': final_text[0].isupper() or final_text[0] in '"\'(',
        'ends_clean': final_text[-1] in '.!?"\')]',
        'length_ok': 80 <= len(final_text) <= 500,
        'entity_present': entity_lower in final_text.lower() or 
                          any(sf in final_text.lower() for sf in SHORT_FORMS.get(entity_lower, [])),
    }
    quality_score = sum(quality.values())
    
    return {
        'entity_position': pos,
        'extraction_status': 'SUCCESS',
        'extracted_context': extracted,
        'extracted_length': len(extracted),
        'final_context': final_text,
        'final_length': len(final_text),
        'quality_checks': quality,
        'quality_score': f'{quality_score}/4',
    }


# ============================================================
# LAYER 3.7: READINESS & QUEUE (Final Gatekeeper)
# ============================================================

def test_layer_3_7_readiness(final_row):
    """Layer 3.7: Readiness — final gatekeeper before queue.
    
    Checks:
    - Has entity
    - Has context
    - Has label
    - Has confidence
    - Has source URL
    - Ready for NLP processing
    """
    checks = {
        'has_entity': bool(final_row.get('entity_name', '').strip()),
        'has_text': bool(final_row.get('text', '').strip()),
        'has_label': bool(final_row.get('label', '').strip()),
        'has_confidence': final_row.get('label_confidence', 0) > 0,
        'has_source_url': bool(final_row.get('source_url', '').strip()),
        'has_reasoning': bool(final_row.get('verification_reasoning', '').strip()),
        'text_length_ok': 80 <= len(final_row.get('text', '')) <= 600,
        'label_valid': final_row.get('label', '') in ['positive', 'neutral', 'negative'],
    }
    
    all_passed = all(checks.values())
    
    return {
        'checks': checks,
        'all_passed': all_passed,
        'ready_for_nlp': all_passed,
        'gatekeeper_status': 'PASS ✅' if all_passed else 'FAIL ❌',
    }


# ============================================================
# LAYER 4: NLP WORKER (Simulasi — model belum di-load)
# ============================================================

def test_layer_4_nlp_worker(final_row):
    """Layer 4: NLP Worker — IndoBERT 2-stage pipeline.
    
    Stage 1: Relevancy Gate — "Apakah context benar-benar membahas entity?"
    Stage 2: Sentiment Classifier — positive/neutral/negative
    
    NOTE: Ini simulasi — model IndoBERT belum di-load di sandbox.
    Output menunjukkan apa yang NLP worker akan terima dan hasilkan.
    """
    entity = final_row.get('entity_name', '')
    text = final_row.get('text', '')
    label = final_row.get('label', '')
    confidence = final_row.get('label_confidence', 0)
    
    # Simulasi Stage 1: Relevancy Gate
    entity_lower = entity.lower()
    text_lower = text.lower()
    
    relevancy_passed = False
    if entity_lower in text_lower:
        relevancy_passed = True
    elif any(sf in text_lower for sf in SHORT_FORMS.get(entity_lower, [])):
        relevancy_passed = True
    
    # Simulasi Stage 2: Sentiment
    # (Model akan output ini, sekarang kita pakai label dari dataset)
    
    return {
        'stage_1_relevancy': {
            'question': f'Apakah context membahas {entity}?',
            'answer': 'RELEVANT' if relevancy_passed else 'NOT_RELEVANT',
            'passed': relevancy_passed,
        },
        'stage_2_sentiment': {
            'input': f'Premise: "Tentang {entity}" | Hypothesis: "{text[:100]}..."',
            'predicted_label': label,  # dari dataset (model akan predict ini)
            'confidence': f'{confidence*100:.1f}%',
            'note': 'Label dari dataset gold standard. Model akan predict saat training selesai.',
        },
        'nlp_output': {
            'entity': entity,
            'sentiment': label,
            'confidence': confidence,
            'would_be_stored': relevancy_passed,
        },
    }


# ============================================================
# PRINT FUNCTIONS
# ============================================================

def print_separator(char='═', width=80):
    print(char * width)

def print_layer(num, title, subtitle=''):
    print()
    print_separator()
    print(f"  LAYER {num}: {title}")
    if subtitle:
        print(f"  {subtitle}")
    print_separator()

def print_wrapped(text, indent='    ', width=72):
    for line in textwrap.wrap(str(text), width=width, initial_indent=indent, subsequent_indent=indent):
        print(line)

def print_checks(checks, indent='    '):
    for name, passed in checks.items():
        status = '✅' if passed else '❌'
        print(f"{indent}{status} {name}: {passed}")


def print_row_pipeline(raw_row, final_row, row_num):
    """Print full backend pipeline output for one row."""
    
    print(f"\n{'#'*80}")
    print(f"  ROW {row_num} — Backend Pipeline Output Trace")
    print(f"{'#'*80}")
    
    # ===== LAYER 1-2: INGESTION & ENRICHMENT =====
    print_layer('1-2', 'INGESTION & ENRICHMENT', '(RSS fetch + Trafilatura extraction)')
    
    l1_result = test_layer_1_2_ingestion_enrichment(raw_row)
    print(f"\n  Source URL: {l1_result['source_url'][:100]}")
    print(f"  Portal: {l1_result['portal']}")
    print(f"  Raw Text ID: {l1_result['raw_text_id']}")
    print(f"\n  Article Text ({len(l1_result['article_text'])} chars):")
    print_wrapped(l1_result['article_text'][:500])
    if len(l1_result['article_text']) > 500:
        print(f"    ... ({len(l1_result['article_text']) - 500} more chars)")
    
    print(f"\n  Anti-Sampah Checks (Layer 2):")
    print_checks(l1_result['checks'])
    
    # ===== LAYER 2.5: VALIDATION =====
    print_layer('2.5', 'VALIDATION (Quality Control)', '(Skor kualitas teks 0-100)')
    
    l25_result = test_layer_2_5_validation(raw_row)
    print(f"\n  QC Score:        {l25_result['qc_score']}/100")
    print(f"  Passed:          {'✅ YES' if l25_result['passed'] else '❌ NO'}")
    print(f"  Article Length:  {l25_result['article_length']} chars")
    print(f"  Sentence Count:  {l25_result['sentence_count']}")
    print(f"  Indo Words:      {l25_result['indo_word_count']}")
    if l25_result['issues']:
        print(f"\n  Issues:")
        for issue in l25_result['issues']:
            print(f"    ⚠ {issue}")
    
    # ===== LAYER 3: PREPROCESSING =====
    print_layer('3', 'PREPROCESSING', '(Normalisasi unicode, hapus URL, hash dedup)')
    
    l3_result = test_layer_3_preprocessing(raw_row, final_row)
    print(f"\n  Raw length:      {l3_result['raw_length']} chars")
    print(f"  Final length:    {l3_result['final_length']} chars")
    print(f"  Reduction:       {l3_result['length_reduction']} chars")
    print(f"  Content hash:    {l3_result['content_hash']}")
    print(f"  Is clean:        {'✅ YES' if l3_result['is_clean'] else '❌ NO'}")
    
    if l3_result['changes_applied']:
        print(f"\n  Changes applied:")
        for change in l3_result['changes_applied']:
            print(f"    • {change}")
    else:
        print(f"\n  No changes needed (already clean)")
    
    print(f"\n  Final Text:")
    print_wrapped(final_row['text'][:300])
    
    # ===== LAYER 3.2: ENTITY RESOLUTION =====
    print_layer('3.2', 'ENTITY RESOLUTION', '(NER + alias matching)')
    
    l32_result = test_layer_3_2_entity_resolution(raw_row, final_row)
    print(f"\n  Expected Entity:  {l32_result['expected_entity']}")
    print(f"  Match Type:       {l32_result['match_type']}")
    print(f"  Entity in text:   {l32_result['entity_in_final_text']}")
    print(f"  Target found:     {'✅ YES' if l32_result['is_target_found'] else '❌ NO'}")
    print(f"\n  All entities detected ({l32_result['total_entities_detected']} total):")
    for e in l32_result['all_entities_detected']:
        marker = ' ◀ TARGET' if e.lower() == l32_result['expected_entity'].lower() else ''
        print(f"    • {e}{marker}")
    
    # ===== LAYER 3.5: CONTEXT EXTRACTION =====
    print_layer('3.5', 'CONTEXT EXTRACTION', '(Context span di sekitar entity)')
    
    l35_result = test_layer_3_5_context_extraction(raw_row, final_row)
    print(f"\n  Entity position:  char {l35_result['entity_position']}")
    print(f"  Extraction:       {l35_result['extraction_status']}")
    
    if l35_result['extracted_context']:
        print(f"\n  Extracted Context ({l35_result['extracted_length']} chars):")
        print_wrapped(l35_result['extracted_context'][:400])
    
    print(f"\n  Final Context ({l35_result['final_length']} chars):")
    print_wrapped(l35_result['final_context'][:300])
    
    print(f"\n  Quality checks ({l35_result['quality_score']}):")
    print_checks(l35_result['quality_checks'])
    
    # ===== LAYER 3.7: READINESS & QUEUE =====
    print_layer('3.7', 'READINESS & QUEUE (Final Gatekeeper)', '(Cek kelengkapan sebelum NLP)')
    
    l37_result = test_layer_3_7_readiness(final_row)
    print(f"\n  Gatekeeper Status: {l37_result['gatekeeper_status']}")
    print(f"  Ready for NLP:     {'✅ YES' if l37_result['ready_for_nlp'] else '❌ NO'}")
    print(f"\n  Readiness checks:")
    print_checks(l37_result['checks'])
    
    # ===== LAYER 4: NLP WORKER =====
    print_layer('4', 'NLP WORKER (IndoBERT 2-Stage)', '(Relevancy Gate + Sentiment Classifier)')
    
    l4_result = test_layer_4_nlp_worker(final_row)
    print(f"\n  Stage 1 — Relevancy Gate:")
    print(f"    Question: {l4_result['stage_1_relevancy']['question']}")
    print(f"    Answer:   {l4_result['stage_1_relevancy']['answer']}")
    print(f"    Passed:   {'✅ YES' if l4_result['stage_1_relevancy']['passed'] else '❌ NO'}")
    
    print(f"\n  Stage 2 — Sentiment Classifier:")
    print(f"    Input:    {l4_result['stage_2_sentiment']['input'][:100]}")
    print(f"    Label:    {l4_result['stage_2_sentiment']['predicted_label']}")
    print(f"    Confidence: {l4_result['stage_2_sentiment']['confidence']}")
    print(f"    Note:     {l4_result['stage_2_sentiment']['note']}")
    
    print(f"\n  NLP Output:")
    print(f"    Entity:    {l4_result['nlp_output']['entity']}")
    print(f"    Sentiment: {l4_result['nlp_output']['sentiment']}")
    print(f"    Confidence: {l4_result['nlp_output']['confidence']*100:.1f}%")
    print(f"    Stored:    {'✅ YES' if l4_result['nlp_output']['would_be_stored'] else '❌ NO'}")
    
    # ===== SUMMARY =====
    print()
    print_separator('─')
    print(f"  PIPELINE SUMMARY (Row {row_num})")
    print_separator('─')
    
    all_layers = {
        'L1-2 Ingestion': all(l1_result['checks'].values()),
        'L2.5 Validation': l25_result['passed'],
        'L3 Preprocessing': l3_result['is_clean'],
        'L3.2 Entity': l32_result['is_target_found'],
        'L3.5 Context': l35_result['quality_checks']['starts_clean'] and l35_result['quality_checks']['ends_clean'],
        'L3.7 Readiness': l37_result['all_passed'],
        'L4 NLP': l4_result['stage_1_relevancy']['passed'],
    }
    
    passed = sum(1 for v in all_layers.values() if v)
    total = len(all_layers)
    
    for layer, ok in all_layers.items():
        status = '✅' if ok else '❌'
        print(f"    {status} {layer}")
    
    print(f"\n    Overall: {passed}/{total} layers passed", end="")
    if passed == total:
        print(" — SANGAT BAIK ✅")
    elif passed >= total - 1:
        print(" — BAIK ✅")
    elif passed >= total - 2:
        print(" — CUKUP ⚠")
    else:
        print(" — BURUK ❌")


def main():
    ap = argparse.ArgumentParser(description="Test Backend Pipeline — Output per Layer")
    ap.add_argument('--n', type=int, default=3, help='Jumlah row (default: 3)')
    ap.add_argument('--row', type=int, default=None, help='Row ke-N saja')
    ap.add_argument('--seed', type=int, default=2024, help='Random seed')
    args = ap.parse_args()
    
    print("=" * 80)
    print("BACKEND PIPELINE — OUTPUT TRACE PER LAYER")
    print("=" * 80)
    print(f"\nWorkflow (sesuai README):")
    print(f"  L1-2:   Ingestion & Enrichment (RSS + Trafilatura)")
    print(f"  L2.5:   Validation (Quality Control 0-100)")
    print(f"  L3:     Preprocessing (normalisasi, hash dedup)")
    print(f"  L3.2:   Entity Resolution (NER + alias)")
    print(f"  L3.5:   Context Extraction (sentence window)")
    print(f"  L3.7:   Readiness & Queue (final gatekeeper)")
    print(f"  L4:     NLP Worker (IndoBERT 2-stage)")
    
    data = load_all_data()
    print(f"\nDataset: {len(data['final'])} final rows, {len(data['raw'])} raw articles")
    
    if not data['final']:
        print("ERROR: Final dataset tidak ditemukan!")
        return
    
    # Select rows
    rows = data['final']
    if args.row is not None:
        if args.row < len(rows):
            rows = [rows[args.row]]
        else:
            print(f"Row {args.row} tidak ada (max: {len(rows)-1})")
            return
    else:
        random.seed(args.seed)
        # Only select rows that have raw articles
        rows_with_raw = [r for r in rows if r.get('raw_text_id', '') in data['raw']]
        rows = random.sample(rows_with_raw, min(args.n, len(rows_with_raw)))
    
    print(f"\nMenampilkan {len(rows)} rows...\n")
    
    for i, row in enumerate(rows):
        raw_row = data['raw'].get(row.get('raw_text_id', ''), row)
        print_row_pipeline(raw_row, row, i + 1)
    
    print(f"\n{'#'*80}")
    print(f"  SELESAI — {len(rows)} rows ditampilkan")
    print(f"{'#'*80}")


if __name__ == "__main__":
    main()
