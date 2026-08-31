"""
colab_setup_v4.py — Self-contained setup for Colab
==================================================
Run this in Colab first to set up everything.

Usage in Colab cell:
  !pip install -q transformers peft scikit-learn accelerate sentencepiece
  # Upload this file (colab_setup_v4.py) and dataset_gold_standard_final.jsonl
  # Then run:
  !python colab_setup_v4.py setup
  !python colab_setup_v4.py finetune --task sentiment --kfold 5
  !python colab_setup_v4.py evaluate --task sentiment
"""
import os, sys, json, argparse, subprocess, shutil
from pathlib import Path

REPO_URL = "https://github.com/raynzz455/ID-Political-Sentiment-Tracker.git"
REPO_DIR = "/content/ID-Political-Sentiment-Tracker"
DATASET = "dataset_gold_standard_final.jsonl"

def run(cmd, check=True):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr, text=True)
    if check and result.returncode != 0:
        print(f"FAILED: code {result.returncode}"); sys.exit(result.returncode)
    return result

def cmd_setup():
    """Clone repo + copy uploaded files into it."""
    print("=== SETUP ===")
    # Clone the base repo (has packages/, devtools/, etc.)
    if not Path(REPO_DIR).exists():
        run(f"git clone {REPO_URL} {REPO_DIR}")
    else:
        print(f"Repo exists at {REPO_DIR}")

    # Copy uploaded files from /content/ into repo
    finetune_dir = Path(REPO_DIR) / "finetuning"
    finetune_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        ("colab_setup_v4.py", "colab_setup_v4.py"),
        ("finetune_v4.py", "finetune_v4.py"),
        ("evaluate_v4.py", "evaluate_v4.py"),
        ("hyperparams_v4.py", "configs/hyperparams_v4.py"),
        ("sentiment_model_v6.py", "patches/sentiment_model_v6.py"),
        (DATASET, f"datasets/{DATASET}"),
    ]
    for src, dst in files_to_copy:
        src_path = Path("/content") / src
        dst_path = finetune_dir / dst
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"  Copied: {src} -> finetuning/{dst}")
        else:
            print(f"  MISSING: {src_path} (upload it first!)")

    # Verify dataset
    ds = finetune_dir / "datasets" / DATASET
    if ds.exists():
        count = sum(1 for _ in open(ds))
        print(f"\nDataset OK: {count} rows")
    else:
        print(f"\nERROR: Dataset not found at {ds}")
        print("Upload dataset_gold_standard_final.jsonl to /content/ first!")

def cmd_finetune(args):
    """Run fine-tuning."""
    task = args.task
    kfold = args.kfold
    print(f"=== FINETUNE: task={task}, kfold={kfold} ===")
    run(f"cd {REPO_DIR}/finetuning && python finetune_v4.py --task {task} --kfold {kfold}")

def cmd_evaluate(args):
    """Run evaluation."""
    task = args.task
    print(f"=== EVALUATE: task={task} ===")
    kfold_file = Path(REPO_DIR) / "finetuning" / "runs" / f"{task}_v4" / "kfold_results.json"
    if kfold_file.exists():
        run(f"cd {REPO_DIR}/finetuning && python evaluate_v4.py --task {task} --kfold-results {kfold_file}")
    else:
        print(f"K-fold results not found: {kfold_file}")
        print("Run finetune first.")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command")
    sub.add_parser("setup")
    p_ft = sub.add_parser("finetune")
    p_ft.add_argument("--task", choices=["sentiment","relevancy"], default="sentiment")
    p_ft.add_argument("--kfold", type=int, default=5)
    p_ev = sub.add_parser("evaluate")
    p_ev.add_argument("--task", choices=["sentiment","relevancy"], default="sentiment")
    args = ap.parse_args()
    if args.command == "setup": cmd_setup()
    elif args.command == "finetune": cmd_finetune(args)
    elif args.command == "evaluate": cmd_evaluate(args)
    else: ap.print_help()

if __name__ == "__main__":
    main()
