#!/usr/bin/env python3
"""
test_enrichment_quality.py
==========================
Script penguji formal untuk menilai output enrichment worker sampai preprocessing.

Menguji 5 aspek kualitas data:
  1. Entity Resolution — apakah entity cocok dengan text?
  2. Context Quality — apakah context bersih dan lengkap?
  3. Preprocessing — apakah data bersih (no noise, no artifacts)?
  4. Label Quality — apakah label dan confidence valid?
  5. Data Integrity — apakah tidak ada duplicates/empty fields?

Output:
  - Console: detailed report dengan grade per komponen
  - JSON: structured report untuk audit
  - Penilaian akhir: SANGAT BAIK / BAIK / CUKUP / BURUK

Usage:
  python3 test_enrichment_quality.py                      # full dataset
  python3 test_enrichment_quality.py --n 100              # sample 100 rows
  python3 test_enrichment_quality.py --sample-only        # sample 30 rows only
  python3 test_enrichment_quality.py --fix-issues         # auto-fix masalah
"""
import sys, os, json, re, argparse, random, statistics, unicodedata
from pathlib import Path
from collections import Counter

DATASET = Path(__file__).resolve().parent.parent / "datasets" / "dataset_gold_standard_final.jsonl"
REPORT  = Path(__file__).resolve().parent / "enrichment_quality_report.json"

# Short forms for entity matching
SHORT_FORMS = {
    "joko widodo": ["jokowi"], "prabowo subianto": ["prabowo"],
    "megawati soekarnoputri": ["megawati"], "susilo bambang yudhoyono": ["sby"],
    "basuki tjahaja purnama": ["ahok"], "abdurrahman wahid": ["gus dur"],
    "ma'ruf amin": ["ma'ruf", "maruf"], "muhaimin iskandar": ["cak imin"],
    "erick thohir": ["erick"], "bima arya sugiarto": ["bima"],
    "sri mulyani indrawati": ["sri mulyani"], "ridwan kamil": ["rk", "kang emil"],
    "anies baswedan": ["anies"], "pramono anung": ["pram"],
    "puan maharani": ["puan"], "agus harimurti yudhoyono": ["ahy"],
    "sandiaga uno": ["sandi"], "sufmi dasco ahmad": ["dasco"],
    "khofifah indar parawansa": ["khofifah"], "yusril ihza mahendra": ["yusril"],
    "dedi mulyadi": ["dedi"], "tito karnavian": ["tito"],
    "bobby nasution": ["bobby"], "bahlil lahadalia": ["bahlil"],
    "soekarno": ["bung karno", "soekarno"], "soeharto": ["pak harto"],
    "bacharuddin jusuf habibie": ["habibie"], "bj habibie": ["habibie"],
}

# Emoji pattern
EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]"
)


def load_data(n=None, seed=2024):
    """Load dataset, optionally sample N rows."""
    rows = [json.loads(l) for l in open(DATASET) if l.strip()]
    if n and n < len(rows):
        random.seed(seed)
        rows = random.sample(rows, n)
    return rows


def check_entity_in_text(entity, text):
    """Check if entity (or alias) appears in text. Returns (found, match_type)."""
    entity_lower = entity.lower()
    text_lower = text.lower()
    
    # 1. Exact match
    if entity_lower in text_lower:
        return True, 'exact'
    
    # 2. Short form match
    if entity_lower in SHORT_FORMS:
        for sf in SHORT_FORMS[entity_lower]:
            if sf in text_lower:
                return True, f'alias:{sf}'
    
    # 3. Last name match
    parts = entity.split()
    if len(parts) >= 2:
        if len(parts[-1]) >= 4 and parts[-1].lower() in text_lower:
            return True, 'last_name'
        if len(parts[0]) >= 4 and parts[0].lower() in text_lower:
            return True, 'first_name'
    
    return False, 'mismatch'


def test_entity_resolution(rows):
    """Test 1: Entity Resolution Quality."""
    results = {
        'exact_match': 0,
        'partial_match': 0,
        'mismatch': 0,
        'mismatches_detail': [],
    }
    
    for i, r in enumerate(rows):
        found, match_type = check_entity_in_text(r['entity_name'], r['text'])
        if found:
            if match_type == 'exact':
                results['exact_match'] += 1
            else:
                results['partial_match'] += 1
        else:
            results['mismatch'] += 1
            results['mismatches_detail'].append({
                'row': i, 'entity': r['entity_name'],
                'label': r['label'], 'text_preview': r['text'][:100]
            })
    
    total = len(rows)
    results['total'] = total
    results['accuracy'] = round((results['exact_match'] + results['partial_match']) / total * 100, 2)
    results['mismatch_rate'] = round(results['mismatch'] / total * 100, 2)
    
    # Grade
    if results['mismatch_rate'] < 1: results['grade'] = 'SANGAT BAIK'
    elif results['mismatch_rate'] < 5: results['grade'] = 'BAIK'
    elif results['mismatch_rate'] < 10: results['grade'] = 'CUKUP'
    else: results['grade'] = 'BURUK'
    
    return results


def test_context_quality(rows):
    """Test 2: Context Extraction Quality."""
    results = {
        'quality_4': 0, 'quality_3': 0, 'quality_2': 0, 'quality_1': 0, 'quality_0': 0,
        'issues': {
            'truncated_start': 0,
            'no_terminal_punct': 0,
            'too_short': 0,
            'too_long': 0,
            'entity_not_in_text': 0,
        },
        'details': [],
    }
    
    for i, r in enumerate(rows):
        text = r['text']
        score = 0
        issues = []
        
        # Check 1: Clean start (uppercase or quote)
        if text and (text[0].isupper() or text[0] in '"\'('):
            score += 1
        else:
            results['issues']['truncated_start'] += 1
            issues.append('truncated_start')
        
        # Check 2: Clean end (terminal punctuation)
        if text and text[-1] in '.!?"\')]':
            score += 1
        else:
            results['issues']['no_terminal_punct'] += 1
            issues.append('no_terminal_punct')
        
        # Check 3: Optimal length (80-500 chars)
        if 80 <= len(text) <= 500:
            score += 1
        elif len(text) < 80:
            results['issues']['too_short'] += 1
            issues.append(f'too_short({len(text)})')
        else:
            results['issues']['too_long'] += 1
            issues.append(f'too_long({len(text)})')
        
        # Check 4: Entity in text
        found, _ = check_entity_in_text(r['entity_name'], text)
        if found:
            score += 1
        else:
            results['issues']['entity_not_in_text'] += 1
            issues.append('entity_not_in_text')
        
        # Record quality
        results[f'quality_{score}'] += 1
        
        if score < 3:
            results['details'].append({
                'row': i, 'entity': r['entity_name'], 'score': score,
                'issues': issues, 'text_preview': text[:100]
            })
    
    total = len(rows)
    results['total'] = total
    results['perfect_rate'] = round(results['quality_4'] / total * 100, 2)
    results['good_rate'] = round((results['quality_4'] + results['quality_3']) / total * 100, 2)
    results['poor_rate'] = round((results['quality_0'] + results['quality_1'] + results['quality_2']) / total * 100, 2)
    results['avg_score'] = round(
        (4 * results['quality_4'] + 3 * results['quality_3'] + 
         2 * results['quality_2'] + 1 * results['quality_1']) / total, 2
    )
    
    # Grade
    if results['perfect_rate'] >= 90: results['grade'] = 'SANGAT BAIK'
    elif results['good_rate'] >= 90: results['grade'] = 'BAIK'
    elif results['good_rate'] >= 70: results['grade'] = 'CUKUP'
    else: results['grade'] = 'BURUK'
    
    return results


def test_preprocessing(rows):
    """Test 3: Preprocessing Quality (kebersihan data)."""
    results = {
        'issues': {
            'non_ascii': 0,
            'html_entities': 0,
            'mojibake': 0,
            'multi_whitespace': 0,
            'tabs': 0,
            'citations': 0,
            'urls_in_text': 0,
            'emoji': 0,
            'repeated_chars': 0,
            'zero_width': 0,
            'control_chars': 0,
        },
        'details': [],
    }
    
    for i, r in enumerate(rows):
        text = r['text']
        row_issues = []
        
        # Non-ASCII
        if any(ord(c) > 127 for c in text):
            results['issues']['non_ascii'] += 1
            row_issues.append('non_ascii')
        
        # HTML entities
        if re.search(r'&\w+;|&#\d+;', text):
            results['issues']['html_entities'] += 1
            row_issues.append('html_entities')
        
        # Mojibake
        if re.search(r'Ã[\x80-\xBF]|â€[\x80-\x9F]|Â§|Â°', text):
            results['issues']['mojibake'] += 1
            row_issues.append('mojibake')
        
        # Multiple whitespace
        if re.search(r'  +', text):
            results['issues']['multi_whitespace'] += 1
            row_issues.append('multi_whitespace')
        
        # Tabs
        if '\t' in text:
            results['issues']['tabs'] += 1
            row_issues.append('tabs')
        
        # Citation markers
        if re.search(r'\[\d+\]', text):
            results['issues']['citations'] += 1
            row_issues.append('citations')
        
        # URLs in text
        if re.search(r'https?://\S+', text):
            results['issues']['urls_in_text'] += 1
            row_issues.append('urls_in_text')
        
        # Emoji
        if EMOJI_PATTERN.search(text):
            results['issues']['emoji'] += 1
            row_issues.append('emoji')
        
        # Repeated chars
        if re.search(r'[_=\-]{4,}|\*{4,}', text):
            results['issues']['repeated_chars'] += 1
            row_issues.append('repeated_chars')
        
        # Zero-width chars
        if any(ord(c) in [0x200B, 0x200C, 0x200D, 0xFEFF] for c in text):
            results['issues']['zero_width'] += 1
            row_issues.append('zero_width')
        
        # Control chars
        if any(ord(c) < 32 and c not in '\n\r\t' for c in text):
            results['issues']['control_chars'] += 1
            row_issues.append('control_chars')
        
        if row_issues:
            results['details'].append({
                'row': i, 'entity': r['entity_name'], 'issues': row_issues
            })
    
    total = len(rows)
    results['total'] = total
    total_issues = sum(results['issues'].values())
    results['total_issues'] = total_issues
    results['clean_rate'] = round((1 - total_issues / (total * 11)) * 100, 2)
    
    # Grade
    if total_issues == 0: results['grade'] = 'SANGAT BAIK'
    elif total_issues < 10: results['grade'] = 'BAIK'
    elif total_issues < 50: results['grade'] = 'CUKUP'
    else: results['grade'] = 'BURUK'
    
    return results


def test_label_quality(rows):
    """Test 4: Label Quality (confidence + source)."""
    confs = [r.get('label_confidence', 0.5) for r in rows]
    labels = Counter(r['label'] for r in rows)
    sources = Counter(r.get('label_source', '') for r in rows)
    
    results = {
        'mean_confidence': round(statistics.mean(confs) * 100, 2),
        'median_confidence': round(statistics.median(confs) * 100, 2),
        'min_confidence': round(min(confs) * 100, 2),
        'high_confidence_count': sum(1 for c in confs if c >= 0.95),
        'high_confidence_pct': round(sum(1 for c in confs if c >= 0.95) / len(rows) * 100, 2),
        'low_confidence_count': sum(1 for c in confs if c < 0.90),
        'low_confidence_pct': round(sum(1 for c in confs if c < 0.90) / len(rows) * 100, 2),
        'llm_verified_count': sum(1 for r in rows if 'llm' in r.get('label_source', '')),
        'llm_verified_pct': round(sum(1 for r in rows if 'llm' in r.get('label_source', '')) / len(rows) * 100, 2),
        'label_distribution': dict(labels),
        'label_sources': dict(sources),
    }
    
    results['total'] = len(rows)
    
    # Grade
    if results['mean_confidence'] >= 95: results['grade'] = 'SANGAT BAIK'
    elif results['mean_confidence'] >= 90: results['grade'] = 'BAIK'
    elif results['mean_confidence'] >= 80: results['grade'] = 'CUKUP'
    else: results['grade'] = 'BURUK'
    
    return results


def test_data_integrity(rows):
    """Test 5: Data Integrity (duplicates, empty fields)."""
    text_counts = Counter(r['text'] for r in rows)
    dupes = sum(1 for t, c in text_counts.items() if c > 1)
    
    results = {
        'total_rows': len(rows),
        'duplicates': dupes,
        'empty_texts': sum(1 for r in rows if not r['text'].strip()),
        'empty_entities': sum(1 for r in rows if not r['entity_name'].strip()),
        'empty_labels': sum(1 for r in rows if not r.get('label', '').strip()),
        'empty_confidence': sum(1 for r in rows if r.get('label_confidence', 0) == 0),
        'empty_reasoning': sum(1 for r in rows if not r.get('verification_reasoning', '').strip()),
    }
    
    results['integrity_rate'] = round(
        (1 - (dupes + results['empty_texts'] + results['empty_entities'] + 
              results['empty_labels']) / len(rows)) * 100, 2
    )
    
    # Grade
    if dupes == 0 and results['empty_texts'] == 0 and results['empty_entities'] == 0:
        results['grade'] = 'SANGAT BAIK'
    elif dupes < 5 and results['empty_texts'] == 0:
        results['grade'] = 'BAIK'
    elif dupes < 20:
        results['grade'] = 'CUKUP'
    else:
        results['grade'] = 'BURUK'
    
    return results


def print_section(title, results, score_key=None):
    """Print a test section."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    if title.startswith('1. ENTITY'):
        print(f"  Exact match:      {results['exact_match']}/{results['total']} ({results['exact_match']/results['total']*100:.1f}%)")
        print(f"  Partial (alias):  {results['partial_match']}/{results['total']} ({results['partial_match']/results['total']*100:.1f}%)")
        print(f"  Mismatch:         {results['mismatch']}/{results['total']} ({results['mismatch_rate']}%)")
        print(f"  Accuracy:         {results['accuracy']}%")
        print(f"  Grade: {results['grade']}")
        if results['mismatches_detail']:
            print(f"\n  Mismatches ({len(results['mismatches_detail'])}):")
            for m in results['mismatches_detail'][:5]:
                print(f"    Row {m['row']} [{m['entity']}]: {m['text_preview'][:80]}...")
    
    elif title.startswith('2. CONTEXT'):
        print(f"  Quality 4/4 (perfect): {results['quality_4']}/{results['total']} ({results['quality_4']/results['total']*100:.1f}%)")
        print(f"  Quality 3/4 (good):    {results['quality_3']}/{results['total']} ({results['quality_3']/results['total']*100:.1f}%)")
        print(f"  Quality 2/4 (medium):  {results['quality_2']}/{results['total']} ({results['quality_2']/results['total']*100:.1f}%)")
        print(f"  Quality 0-1 (poor):    {results['quality_0']+results['quality_1']}/{results['total']}")
        print(f"  Perfect rate:    {results['perfect_rate']}%")
        print(f"  Good rate:       {results['good_rate']}%")
        print(f"  Avg score:       {results['avg_score']}/4.00")
        print(f"  Grade: {results['grade']}")
        print(f"\n  Issues:")
        for issue, count in results['issues'].items():
            if count > 0:
                print(f"    {issue}: {count}")
    
    elif title.startswith('3. PREPROCESSING'):
        print(f"  Total issues:    {results['total_issues']}")
        print(f"  Clean rate:      {results['clean_rate']}%")
        print(f"  Grade: {results['grade']}")
        if results['total_issues'] > 0:
            print(f"\n  Issues breakdown:")
            for issue, count in results['issues'].items():
                if count > 0:
                    print(f"    {issue}: {count}")
    
    elif title.startswith('4. LABEL'):
        print(f"  Mean confidence:   {results['mean_confidence']}%")
        print(f"  Median:            {results['median_confidence']}%")
        print(f"  High conf (≥0.95): {results['high_confidence_count']} ({results['high_confidence_pct']}%)")
        print(f"  Low conf (<0.90):  {results['low_confidence_count']} ({results['low_confidence_pct']}%)")
        print(f"  LLM-verified:      {results['llm_verified_count']} ({results['llm_verified_pct']}%)")
        print(f"  Grade: {results['grade']}")
        print(f"\n  Label distribution:")
        for k, v in sorted(results['label_distribution'].items(), key=lambda x: -x[1]):
            print(f"    {k:10s}: {v:5d} ({v/results['total']*100:.1f}%)")
    
    elif title.startswith('5. DATA INTEGRITY'):
        print(f"  Total rows:        {results['total_rows']}")
        print(f"  Duplicates:        {results['duplicates']}")
        print(f"  Empty texts:       {results['empty_texts']}")
        print(f"  Empty entities:    {results['empty_entities']}")
        print(f"  Empty labels:      {results['empty_labels']}")
        print(f"  Empty reasoning:   {results['empty_reasoning']}")
        print(f"  Integrity rate:    {results['integrity_rate']}%")
        print(f"  Grade: {results['grade']}")


def main():
    ap = argparse.ArgumentParser(description="Test Enrichment Worker + Preprocessing Quality")
    ap.add_argument('--n', type=int, default=None, help='Sample N rows (default: all)')
    ap.add_argument('--seed', type=int, default=2024, help='Random seed for sampling')
    ap.add_argument('--sample-only', action='store_true', help='Sample 30 rows only (quick test)')
    args = ap.parse_args()
    
    if args.sample_only:
        args.n = 30
    
    print("=" * 70)
    print("PENGUJI KUALITAS: ENRICHMENT WORKER + PREPROCESSING")
    print(f"{'='*70}")
    
    rows = load_data(args.n, args.seed)
    print(f"Testing {len(rows)} rows\n")
    
    # Run all tests
    entity_results = test_entity_resolution(rows)
    context_results = test_context_quality(rows)
    preprocessing_results = test_preprocessing(rows)
    label_results = test_label_quality(rows)
    integrity_results = test_data_integrity(rows)
    
    # Print reports
    print_section("1. ENTITY RESOLUTION QUALITY", entity_results)
    print_section("2. CONTEXT EXTRACTION QUALITY", context_results)
    print_section("3. PREPROCESSING QUALITY (KEBERSIHAN)", preprocessing_results)
    print_section("4. LABEL QUALITY", label_results)
    print_section("5. DATA INTEGRITY", integrity_results)
    
    # Overall assessment
    print(f"\n{'='*70}")
    print(f"PENILAIAN AKHIR KESELURUHAN")
    print(f"{'='*70}")
    
    grade_scores = {'SANGAT BAIK': 100, 'BAIK': 85, 'CUKUP': 70, 'BURUK': 50}
    scores = {
        'Entity Resolution': grade_scores[entity_results['grade']],
        'Context Quality': grade_scores[context_results['grade']],
        'Preprocessing': grade_scores[preprocessing_results['grade']],
        'Label Quality': grade_scores[label_results['grade']],
        'Data Integrity': grade_scores[integrity_results['grade']],
    }
    
    print(f"\n  {'Komponen':<25} {'Score':>8} {'Grade':>15}")
    print(f"  {'-'*50}")
    for name, score in scores.items():
        grade = [k for k, v in grade_scores.items() if v == score][0]
        marker = '✅' if 'BAIK' in grade else '⚠' if grade == 'CUKUP' else '❌'
        print(f"  {name:<25} {score:>7.0f}% {marker} {grade:>12}")
    
    avg = statistics.mean(scores.values())
    print(f"\n  {'RATA-RATA':<25} {avg:>7.1f}%")
    
    if avg >= 95:
        overall = "SANGAT BAIK ✅"
    elif avg >= 85:
        overall = "BAIK ✅"
    elif avg >= 70:
        overall = "CUKUP ⚠"
    else:
        overall = "BURUK ❌"
    
    print(f"\n  KESIMPULAN: Output enrichment + preprocessing = {overall}")
    
    # Save report
    report = {
        'test_config': {
            'n_rows': len(rows),
            'seed': args.seed,
            'timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
        },
        'results': {
            'entity_resolution': entity_results,
            'context_quality': context_results,
            'preprocessing': preprocessing_results,
            'label_quality': label_results,
            'data_integrity': integrity_results,
        },
        'overall_score': round(avg, 2),
        'overall_grade': overall,
    }
    with open(REPORT, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {REPORT}")


if __name__ == "__main__":
    main()
