#!/usr/bin/env python3
"""
test_context_extraction.py — Script Penguji Context Extraction Worker
=====================================================================
Menguji output context extraction worker terhadap dataset gold standard.
Krusial untuk NLP: context yang salah = model belajar dari teks yang salah.

Quality checks:
  1. has_entity: entity name (atau alias) ada di context
  2. length_ok: 80-600 chars (optimal untuk IndoBERT MAX_SEQ=256)
  3. starts_clean: dimulai huruf besar/quote (bukan mid-word)
  4. ends_clean: diakhiri punctuation (.!?")

Usage:
  python3 test_context_extraction.py                    # default 20 rows
  python3 test_context_extraction.py --n 50
  python3 test_context_extraction.py --expert stanza
  python3 test_context_extraction.py --expert paragraph
  python3 test_context_extraction.py --expert all
"""
import sys, json, time, argparse, random, statistics
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "patches"))

DATASET = Path(__file__).resolve().parent.parent / "datasets" / "dataset_gold_standard_final.jsonl"
REPORT  = Path(__file__).resolve().parent / "context_extraction_report.json"

MIN_CHARS, MAX_CHARS = 80, 600

SHORT_FORMS = ["jokowi","prabowo","megawati","sby","ahok","gus dur","erick",
    "bima","sri mulyani","anies","puan","dasco","khofifah","yusril","dedi","tito"]


def load_sample(n, seed=2024):
    rows = [json.loads(l) for l in open(DATASET) if l.strip()]
    random.seed(seed)
    sample, seen = [], set()
    shuffled = rows.copy(); random.shuffle(shuffled)
    for r in shuffled:
        if r['entity_name'] not in seen and len(sample) < n:
            sample.append(r); seen.add(r['entity_name'])
    return sample


def find_pos(text, entity):
    text_lower = text.lower(); entity_lower = entity.lower()
    pos = text_lower.find(entity_lower)
    if pos >= 0: return pos
    parts = entity.split()
    for p in parts:
        if len(p) >= 4 and p.lower() in text_lower:
            return text_lower.find(p.lower())
    return -1


def assess_quality(ctx, entity):
    has_entity = entity.lower() in ctx.lower()
    if not has_entity:
        entity_lower = entity.lower()
        for sf in SHORT_FORMS:
            if sf in entity_lower and sf in ctx.lower():
                has_entity = True; break
    length_ok = MIN_CHARS <= len(ctx) <= MAX_CHARS
    starts_clean = ctx[0].isupper() or ctx[0] in '"\'('
    ends_clean = ctx[-1] in '.!?"\')]'
    score = sum([has_entity, length_ok, starts_clean, ends_clean])
    return {'has_entity': has_entity, 'length_ok': length_ok,
            'starts_clean': starts_clean, 'ends_clean': ends_clean, 'score': score}


def test_stanza_window(sample, nlp):
    from context_extraction_moe import SentenceWindowExtractor
    ext = SentenceWindowExtractor(nlp_pipeline=nlp)
    results = []
    for i, r in enumerate(sample):
        text, entity = r['text'], r['entity_name']
        pos = find_pos(text, entity)
        if pos < 0:
            results.append({'row': i+1, 'entity': entity, 'extracted': False,
                            'reason': 'entity not found', 'quality_score': 0})
            continue
        spans = ext.extract(text, entity, pos)
        if not spans:
            results.append({'row': i+1, 'entity': entity, 'extracted': False,
                            'reason': 'empty output', 'quality_score': 0})
            continue
        ctx = spans[0]
        q = assess_quality(ctx.text, entity)
        results.append({'row': i+1, 'entity': entity, 'extracted': True,
                        'context_len': len(ctx.text), 'quality_score': q['score'],
                        'has_entity': q['has_entity'], 'length_ok': q['length_ok'],
                        'starts_clean': q['starts_clean'], 'ends_clean': q['ends_clean'],
                        'preview': ctx.text[:150]})
    return results


def test_paragraph(sample):
    from context_extraction_moe import ParagraphExtractor
    ext = ParagraphExtractor()
    results = []
    for i, r in enumerate(sample):
        text, entity = r['text'], r['entity_name']
        pos = find_pos(text, entity)
        if pos < 0:
            results.append({'row': i+1, 'entity': entity, 'extracted': False,
                            'reason': 'entity not found', 'quality_score': 0})
            continue
        spans = ext.extract(text, entity, pos)
        if not spans:
            results.append({'row': i+1, 'entity': entity, 'extracted': False,
                            'reason': 'empty output', 'quality_score': 0})
            continue
        ctx = spans[0]
        q = assess_quality(ctx.text, entity)
        results.append({'row': i+1, 'entity': entity, 'extracted': True,
                        'context_len': len(ctx.text), 'quality_score': q['score'],
                        'has_entity': q['has_entity'], 'length_ok': q['length_ok'],
                        'starts_clean': q['starts_clean'], 'ends_clean': q['ends_clean'],
                        'preview': ctx.text[:150]})
    return results


def calc_metrics(results):
    total = len(results)
    if not total: return {}
    extracted = [r for r in results if r.get('extracted')]
    ext_count = len(extracted)
    rate = ext_count / total
    q_dist = Counter(r.get('quality_score', 0) for r in extracted)
    high = q_dist.get(4, 0) + q_dist.get(3, 0)
    med = q_dist.get(2, 0)
    low = q_dist.get(1, 0) + q_dist.get(0, 0)
    lengths = [r['context_len'] for r in extracted if r.get('context_len')]
    avg_len = statistics.mean(lengths) if lengths else 0
    has_ent = sum(1 for r in extracted if r.get('has_entity'))
    return {'extraction_rate': round(rate, 4), 'extracted': ext_count, 'total': total,
            'high_quality': high, 'medium_quality': med, 'low_quality': low,
            'high_pct': round(high/ext_count*100, 1) if ext_count else 0,
            'avg_length': round(avg_len, 1),
            'entity_coverage': round(has_ent/ext_count*100, 1) if ext_count else 0,
            'quality_dist': dict(q_dist)}


def print_report(name, results, metrics):
    print(f"\n{'='*60}\nEXPERT: {name}\n{'='*60}")
    print(f"  Extraction rate: {metrics['extraction_rate']*100:.1f}% ({metrics['extracted']}/{metrics['total']})")
    print(f"  High quality:    {metrics['high_quality']} ({metrics['high_pct']}%)")
    print(f"  Medium quality:  {metrics['medium_quality']}")
    print(f"  Low quality:     {metrics['low_quality']}")
    print(f"  Avg length:      {metrics['avg_length']} chars")
    print(f"  Entity coverage: {metrics['entity_coverage']}%")
    if 'avg_ms' in metrics: print(f"  Time/row:        {metrics['avg_ms']}ms")
    fails = [r for r in results if not r.get('extracted')]
    if fails:
        print(f"\n  Failures ({len(fails)}):")
        for f in fails[:3]:
            print(f"    Row {f['row']}: {f['entity']} — {f.get('reason','?')}")


def main():
    ap = argparse.ArgumentParser(description="Test Context Extraction Worker")
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--expert', choices=['all','stanza','paragraph'], default='all')
    ap.add_argument('--seed', type=int, default=2024)
    args = ap.parse_args()

    print(f"{'='*60}\nCONTEXT EXTRACTION WORKER TEST\nRows: {args.n} | Expert: {args.expert}\n{'='*60}")

    sample = load_sample(args.n, args.seed)
    print(f"Loaded {len(sample)} rows ({len(set(r['entity_name'] for r in sample))} entities)")

    all_results = {}

    if args.expert in ['all', 'stanza']:
        print("\n--- Stanza Sentence Window ---")
        try:
            import stanza
            nlp = stanza.Pipeline('id', processors='tokenize,pos,lemma,depparse',
                                  use_gpu=False, verbose=False, logging_level='ERROR')
            t0 = time.time()
            results = test_stanza_window(sample, nlp)
            elapsed = time.time() - t0
            metrics = calc_metrics(results)
            metrics['avg_ms'] = round(elapsed / len(sample) * 1000, 1)
            print_report("Stanza Sentence Window", results, metrics)
            all_results['stanza_window'] = {'metrics': metrics, 'details': results}
        except Exception as e:
            print(f"  [SKIP] Stanza not available: {e}")

    if args.expert in ['all', 'paragraph']:
        print("\n--- Paragraph Extractor ---")
        t0 = time.time()
        results = test_paragraph(sample)
        elapsed = time.time() - t0
        metrics = calc_metrics(results)
        metrics['avg_ms'] = round(elapsed / len(sample) * 1000, 1)
        print_report("Paragraph Extractor", results, metrics)
        all_results['paragraph'] = {'metrics': metrics, 'details': results}

    if len(all_results) > 1:
        print(f"\n{'='*60}\nCOMPARISON\n{'='*60}")
        print(f"  {'Expert':<20} {'Ext%':>8} {'HighQ%':>8} {'AvgLen':>8} {'EntCov':>8} {'ms':>6}")
        for name, data in all_results.items():
            m = data['metrics']
            print(f"  {name:<20} {m['extraction_rate']*100:>7.1f}% {m['high_pct']:>7.1f}% {m['avg_length']:>7.1f} {m['entity_coverage']:>7.1f}% {m.get('avg_ms',0):>5.1f}")

    report = {'config': {'n': args.n, 'expert': args.expert, 'seed': args.seed,
                         'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')},
              'results': all_results}
    with open(REPORT, 'w') as f: json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport: {REPORT}")


if __name__ == "__main__":
    main()
