#!/usr/bin/env python3
"""
test_entity_resolution.py — Script Penguji Entity Resolution Worker
===================================================================
Menguja output entity resolution worker (Stanza + RapidFuzz) terhadap
dataset gold standard. Krusial untuk NLP: jika entity salah detect,
context extraction juga salah, dan model belajar dari data yang salah.

Metrics: accuracy, precision, recall, F1, false positives/negatives

Usage:
  python3 test_entity_resolution.py                    # default 20 rows
  python3 test_entity_resolution.py --n 50              # test 50 rows
  python3 test_entity_resolution.py --expert stanza     # Stanza only
  python3 test_entity_resolution.py --expert rapidfuzz  # RapidFuzz only
  python3 test_entity_resolution.py --expert all        # all experts
"""
import sys, json, time, argparse, random, statistics
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "patches"))

DATASET = Path(__file__).resolve().parent.parent / "datasets" / "dataset_gold_standard_final.jsonl"
REPORT  = Path(__file__).resolve().parent / "entity_resolution_report.json"

SHORT_FORMS = {
    "joko widodo": ["jokowi"], "prabowo subianto": ["prabowo"],
    "megawati soekarnoputri": ["megawati"], "susilo bambang yudhoyono": ["sby"],
    "basuki tjahaja purnama": ["ahok"], "abdurrahman wahid": ["gus dur"],
    "ma'ruf amin": ["ma'ruf"], "muhaimin iskandar": ["cak imin"],
    "erick thohir": ["erick"], "bima arya sugiarto": ["bima"],
    "sri mulyani indrawati": ["sri mulyani"], "ridwan kamil": ["rk"],
    "anies baswedan": ["anies"], "pramono anung": ["pram"],
    "puan maharani": ["puan"], "agus harimurti yudhoyono": ["ahy"],
    "sandiaga uno": ["sandi"], "sufmi dasco ahmad": ["dasco"],
    "khofifah indar parawansa": ["khofifah"], "yusril ihza mahendra": ["yusril"],
    "dedi mulyadi": ["dedi"], "tito karnavian": ["tito"],
    "bobby nasution": ["bobby"], "bahlil lahadalia": ["bahlil"],
}


def load_sample(n, seed=2024):
    rows = [json.loads(l) for l in open(DATASET) if l.strip()]
    random.seed(seed)
    sample, seen = [], set()
    shuffled = rows.copy(); random.shuffle(shuffled)
    for r in shuffled:
        if r['entity_name'] not in seen and len(sample) < n:
            sample.append(r); seen.add(r['entity_name'])
    return sample


def build_db():
    rows = [json.loads(l) for l in open(DATASET) if l.strip()]
    entity_db, alias_map, id_to_name, entity_names = {}, {}, {}, []
    for r in rows:
        name = r['entity_name']; name_lower = name.lower()
        if name_lower not in entity_db:
            ent_id = f"ent-{len(entity_db)+1}"
            entity_db[name_lower] = ent_id
            id_to_name[ent_id] = name
            entity_names.append(name_lower)
            if name_lower in SHORT_FORMS:
                for sf in SHORT_FORMS[name_lower]:
                    alias_map[sf] = name; entity_names.append(sf)
            parts = name.split()
            if len(parts) >= 2 and len(parts[-1]) >= 4:
                alias_map[parts[-1].lower()] = name; entity_names.append(parts[-1].lower())
            if len(parts) >= 2 and len(parts[0]) >= 4:
                alias_map[parts[0].lower()] = name; entity_names.append(parts[0].lower())
    return entity_db, alias_map, id_to_name, entity_names


def calc_metrics(results):
    total = len(results)
    if not total: return {}
    tp = sum(1 for r in results if r['found_expected'])
    fn = sum(1 for r in results if not r['found_expected'])
    fp = sum(len(r['false_positives']) for r in results)
    acc = tp / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {'accuracy': round(acc,4), 'precision': round(prec,4),
            'recall': round(rec,4), 'f1': round(f1,4),
            'tp': tp, 'fn': fn, 'fp': fp, 'total': total}


def test_stanza(sample, nlp, entity_db, alias_map, id_to_name):
    from entity_resolution_moe import StanzaNERMatcher
    matcher = StanzaNERMatcher(nlp, entity_db, alias_map, id_to_name)
    results = []
    for i, r in enumerate(sample):
        expected = r['entity_name']
        mentions = matcher.find(r['text'])
        found = list(set(m.entity_name for m in mentions))
        found_exp = any(expected.lower() in f.lower() or f.lower() in expected.lower() for f in found)
        false_pos = [f for f in found if expected.lower() not in f.lower() and f.lower() not in expected.lower()]
        results.append({'row': i+1, 'expected': expected, 'found': found,
                        'found_expected': found_exp, 'false_positives': false_pos,
                        'num_mentions': len(mentions)})
    return results


def test_rapidfuzz(sample, entity_db, alias_map, id_to_name, entity_names):
    from entity_resolution_moe import RapidFuzzMatcher
    matcher = RapidFuzzMatcher(entity_db, alias_map, id_to_name, entity_names)
    results = []
    for i, r in enumerate(sample):
        expected = r['entity_name']
        mentions = matcher.find(r['text'])
        found = list(set(m.entity_name for m in mentions))
        found_exp = any(expected.lower() in f.lower() or f.lower() in expected.lower() for f in found)
        false_pos = [f for f in found if expected.lower() not in f.lower() and f.lower() not in expected.lower()]
        results.append({'row': i+1, 'expected': expected, 'found': found,
                        'found_expected': found_exp, 'false_positives': false_pos,
                        'num_mentions': len(mentions)})
    return results


def print_report(name, results, metrics):
    print(f"\n{'='*60}\nEXPERT: {name}\n{'='*60}")
    print(f"  Accuracy:  {metrics['accuracy']*100:.1f}% ({metrics['tp']}/{metrics['total']})")
    print(f"  Precision: {metrics['precision']*100:.1f}%")
    print(f"  Recall:    {metrics['recall']*100:.1f}%")
    print(f"  F1 Score:  {metrics['f1']*100:.1f}%")
    print(f"  False neg: {metrics['fn']}, False pos: {metrics['fp']}")
    if 'avg_ms' in metrics: print(f"  Time/row:  {metrics['avg_ms']}ms")
    fails = [r for r in results if not r['found_expected']]
    if fails:
        print(f"\n  Failures ({len(fails)}):")
        for f in fails[:5]:
            print(f"    Row {f['row']}: expected='{f['expected']}' → found={f['found'][:3]}")


def main():
    ap = argparse.ArgumentParser(description="Test Entity Resolution Worker")
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--expert', choices=['all','stanza','rapidfuzz'], default='all')
    ap.add_argument('--seed', type=int, default=2024)
    args = ap.parse_args()

    print(f"{'='*60}\nENTITY RESOLUTION WORKER TEST\nRows: {args.n} | Expert: {args.expert}\n{'='*60}")

    sample = load_sample(args.n, args.seed)
    print(f"Loaded {len(sample)} rows ({len(set(r['entity_name'] for r in sample))} entities)")

    entity_db, alias_map, id_to_name, entity_names = build_db()
    print(f"Entity DB: {len(entity_db)} entities, {len(entity_names)} names")

    all_results = {}

    if args.expert in ['all', 'stanza']:
        print("\n--- Stanza PROPN ---")
        try:
            import stanza
            nlp = stanza.Pipeline('id', processors='tokenize,pos,lemma,depparse',
                                  use_gpu=False, verbose=False, logging_level='ERROR')
            t0 = time.time()
            results = test_stanza(sample, nlp, entity_db, alias_map, id_to_name)
            elapsed = time.time() - t0
            metrics = calc_metrics(results)
            metrics['avg_ms'] = round(elapsed / len(sample) * 1000, 1)
            print_report("Stanza PROPN", results, metrics)
            all_results['stanza'] = {'metrics': metrics, 'details': results}
        except Exception as e:
            print(f"  [SKIP] Stanza not available: {e}")

    if args.expert in ['all', 'rapidfuzz']:
        print("\n--- RapidFuzz ---")
        t0 = time.time()
        results = test_rapidfuzz(sample, entity_db, alias_map, id_to_name, entity_names)
        elapsed = time.time() - t0
        metrics = calc_metrics(results)
        metrics['avg_ms'] = round(elapsed / len(sample) * 1000, 1)
        print_report("RapidFuzz", results, metrics)
        all_results['rapidfuzz'] = {'metrics': metrics, 'details': results}

    if len(all_results) > 1:
        print(f"\n{'='*60}\nCOMPARISON\n{'='*60}")
        print(f"  {'Expert':<15} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'ms/row':>8}")
        for name, data in all_results.items():
            m = data['metrics']
            print(f"  {name:<15} {m['accuracy']*100:>7.1f}% {m['precision']*100:>7.1f}% {m['recall']*100:>7.1f}% {m['f1']*100:>7.1f}% {m.get('avg_ms',0):>7.1f}")

    report = {'config': {'n': args.n, 'expert': args.expert, 'seed': args.seed,
                         'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')},
              'results': all_results}
    with open(REPORT, 'w') as f: json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport: {REPORT}")


if __name__ == "__main__":
    main()
