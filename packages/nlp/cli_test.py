"""
cli_test.py — Dev Tool untuk Inspeksi & Testing NLP
=====================================================
Tujuan: Peek antrian, cek statistik DB, dan test teks manual.
Untuk memproses antrian secara massal, gunakan: python -m packages.nlp.nlp_worker
"""
import os
import sys
import argparse
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

try:
    from supabase import create_client, Client
except ImportError:
    print("[ERROR] pip install supabase"); sys.exit(1)

from packages.shared.db_client import get_client
from packages.nlp.sentiment_model import get_pipeline

def cmd_inspect(sb: Client, args):
    res = sb.rpc("dequeue_nlp_batch", {"p_vt": 60, "p_qty": 5}).execute()
    items = res.data or []

    print(f"\n{'='*60}\nPEEK QUEUE (peek-only)\n{'='*60}\nItems returned: {len(items)}\n")

    for i, item in enumerate(items, 1):
        text = (item.get("text") or "")[:120]
        title = (item.get("title") or "(no title)")[:80]
        print(f"[{i}] source: {item.get('source', '?')}")
        print(f"    title:  {title}")
        print(f"    text:   {text}{'...' if len(item.get('text','')) > 120 else ''}")
        print()

def cmd_single(sb: Client, args):
    text = args.text
    pipeline = get_pipeline()

    print(f"\n{'='*60}\nSINGLE TEST (IndoBERT Fallback)\n{'='*60}\nText: {text}\n")

    result = pipeline.predict_gated(text=text, context=None)

    print(f"Label: {result.label}")
    print(f"Confidence: {result.sentiment_confidence:.3f}")
    print(f"Scores: neg={result.scores[0]:.3f}, neu={result.scores[1]:.3f}, pos={result.scores[2]:.3f}")
    print(f"{'='*60}\n")

def cmd_stats(sb: Client, args):
    print(f"\n{'='*60}\nDB STATS\n{'='*60}")

    res = sb.table("raw_texts").select("status").execute()
    status_counts = Counter(r["status"] for r in res.data)
    print("\nraw_texts by status:")
    for status, c in status_counts.most_common():
        print(f"  {status:15s} {c:5d}")
    print(f"  {'TOTAL':15s} {len(res.data):5d}")

    res2 = sb.table("sentiment_scores").select("id", count="exact").execute()
    print(f"\nsentiment_scores total: {len(res2.data)}")

    print("\nTop tokoh di sentiment_scores (kalau ada):")
    res3 = sb.table("sentiment_scores") \
             .select("entity_id, political_entities(canonical_name)") \
             .limit(500) \
             .execute()
    entity_counter = Counter()
    for r in res3.data:
        pe = r.get("political_entities") or {}
        name = pe.get("canonical_name", "?")        
        entity_counter[name] += 1
    for name, c in entity_counter.most_common(10):
        print(f"  {name:30s} {c:5d}")

    print(f"\n{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description="ID-Sentiment Dev CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="Lihat isi queue tanpa proses")
    p_inspect.set_defaults(func=cmd_inspect)

    p_single = sub.add_parser("single", help="Test 1 teks manual dengan model fallback")
    p_single.add_argument("text", type=str, help="teks untuk dianalisis")
    p_single.set_defaults(func=cmd_single)

    p_stats = sub.add_parser("stats", help="Lihat statistik DB")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    sb = get_client()
    args.func(sb, args)

if __name__ == "__main__":
    main()