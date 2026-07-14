"""
main.py — ID-Political-Sentiment-Tracker Orchestrator
=========================================================
Entry point tunggal untuk menjalankan seluruh atau sebagian pipeline.

Usage:
  1. Jalankan semua layer preprocessing (Layer 1 - 3.7):
     python main.py run-prep --limit 100            (Proses 100 artikel per batch, unlimited total)
     python main.py run-prep --max-total 500        (Proses maksimal 500 artikel total)

  2. Jalankan NLP Worker (Layer 4):
     python main.py run-nlp --target 500
     python main.py run-nlp --all                   (Drain sampai antrian habis)

  3. Jalankan worker spesifik saja:
     python main.py run-worker validation --limit 50
     python main.py run-worker entity --max-total 1000
"""

import argparse
import time
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

# Import Workers
try:
    from packages.enrichment import enricher_worker
    from packages.recovery import gnews_resolver_worker
    from packages.validation import validation_worker, preprocessing_worker
    from packages.entity import entity_resolution_worker
    from packages.context import context_worker, nlp_readiness_worker
    from packages.nlp import nlp_worker
    from devtools.sql_tools.check_db_stats import main as check_status
except ImportError as e:
    print(f"[ERROR] Gagal meload module worker: {e}")
    print("Pastikan struktur direktori 'packages/' sudah benar dan ada file __init__.py di dalamnya.")
    sys.exit(1)


def run_prep_pipeline(limit: int, max_total: int):
    """Menjalankan Layer 1 hingga 3.7 secara berurutan."""
    print(f"\n{'='*70}")
    print(f"🚀 STARTING PREPROCESSING PIPELINE (Batch: {limit} | Max Total: {'Unlimited' if max_total == 0 else max_total})")
    print(f"{'='*70}")
    
    start_time = time.time()

    print("\n--- [1/6] Running Enricher Worker ---")
    enricher_worker.main(limit=limit, max_total=max_total)
    
    print("\n--- [2/6] Running Validation Worker ---")
    validation_worker.main(limit=limit, max_total=max_total)
    
    print("\n--- [3/6] Running Preprocessing Worker ---")
    preprocessing_worker.main(limit=limit, max_total=max_total)
    
    print("\n--- [4/6] Running Entity Resolution Worker ---")
    entity_resolution_worker.main(limit=limit, max_total=max_total)
    
    print("\n--- [5/6] Running Context Worker ---")
    context_worker.main(limit=limit, max_total=max_total)
    
    print("\n--- [6/6] Running NLP Readiness Worker ---")
    nlp_readiness_worker.main(limit=limit, max_total=max_total)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✅ PREPROCESSING PIPELINE FINISHED in {elapsed:.2f} seconds.")
    print(f"{'='*70}\n")


def run_nlp_worker(target: int, batch_size: int, run_all: bool):
    """Menjalankan Layer 4 (NLP Inference)."""
    print(f"\n{'='*70}")
    print(f"🧠 STARTING NLP WORKER (Target: {'ALL' if run_all else target})")
    print(f"{'='*70}")
    
    args_to_pass = []
    if run_all:
        args_to_pass.append("--all")
    else:
        args_to_pass.extend(["--target", str(target)])
        
    args_to_pass.extend(["--batch-size", str(batch_size)])
    
    original_sys_argv = sys.argv
    sys.argv = ["nlp_worker.py"] + args_to_pass
    
    try:
        nlp_worker.main()
    except Exception as e:
        print(f"[ERROR] NLP Worker crashed: {e}")
    finally:
        sys.argv = original_sys_argv


def run_specific_worker(worker_name: str, limit: int, max_total: int):
    """Menjalankan satu worker spesifik berdasarkan nama."""
    print(f"\n{'='*70}")
    print(f"⚙️ RUNNING SPECIFIC WORKER: {worker_name.upper()} (Batch: {limit} | Max Total: {'Unlimited' if max_total == 0 else max_total})")
    print(f"{'='*70}")
    
    workers = {
        "enricher": enricher_worker,
        "gnews_resolver": gnews_resolver_worker,
        "validation": validation_worker,
        "preprocessing": preprocessing_worker,
        "entity": entity_resolution_worker,
        "context": context_worker,
        "readiness": nlp_readiness_worker
    }
    
    if worker_name not in workers:
        print(f"[ERROR] Worker '{worker_name}' tidak ditemukan.")
        print(f"Pilih salah satu: {', '.join(workers.keys())}")
        return
        
    worker_module = workers[worker_name]
    
    try:
        worker_module.main(limit=limit, max_total=max_total)
    except TypeError:
        worker_module.main()
    except Exception as e:
        print(f"[ERROR] Worker {worker_name} crashed: {e}")


def main():
    parser = argparse.ArgumentParser(description="ID-Political-Sentiment-Tracker Orchestrator")
    
    subparsers = parser.add_subparsers(dest="command", help="Perintah yang tersedia")

    # 1. Command: run-prep
    parser_prep = subparsers.add_parser("run-prep", help="Jalankan Layer 1 hingga 3.7 (Preprocessing)")
    parser_prep.add_argument("--limit", type=int, default=100, help="Jumlah row per batch per worker (default 100)")
    parser_prep.add_argument("--max-total", type=int, default=0, help="Batas total proses (0 = unlimited, default 0)")

    # 2. Command: run-nlp
    parser_nlp = subparsers.add_parser("run-nlp", help="Jalankan Layer 4 (NLP Inference)")
    parser_nlp.add_argument("--target", type=int, default=300, help="Jumlah artikel yang diproses")
    parser_nlp.add_argument("--batch-size", type=int, default=30, help="Ukuran batch dequeue")
    parser_nlp.add_argument("--all", action="store_true", help="Drain sampai antrian habis (unlimited)")

    # 3. Command: run-worker
    parser_worker = subparsers.add_parser("run-worker", help="Jalankan worker spesifik")
    parser_worker.add_argument("name", type=str, help="Nama worker (enricher, gnews_resolver, validation, preprocessing, entity, context, readiness)")
    parser_worker.add_argument("--limit", type=int, default=100, help="Jumlah row per batch (default 100)")
    parser_worker.add_argument("--max-total", type=int, default=0, help="Batas total proses (0 = unlimited, default 0)")

    # 4. Command: status
    parser_status = subparsers.add_parser("status", help="Cek status & kesehatan pipeline di database")
    parser_status.set_defaults(func=lambda args: check_status())
    
    args = parser.parse_args()

    if args.command == "run-prep":
        run_prep_pipeline(limit=args.limit, max_total=args.max_total)
    elif args.command == "run-nlp":
        run_nlp_worker(target=args.target, batch_size=args.batch_size, run_all=args.all)
    elif args.command == "run-worker":
        run_specific_worker(worker_name=args.name, limit=args.limit, max_total=args.max_total)
    elif args.command == "status":
        check_status()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()