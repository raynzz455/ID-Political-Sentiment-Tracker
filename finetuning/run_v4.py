"""
run_v4.py — Python runner for the v4 fine-tuning + evaluation pipeline
======================================================================
Equivalent of run_v4.sh but cross-platform (works on Windows too).

WHAT THIS DOES:
  1. Verifies Python + required packages (torch, transformers, peft, ...)
  2. Verifies the gold-standard dataset exists
  3. Runs K-fold fine-tuning for sentiment and/or relevancy
  4. Prints K-fold summaries for each task
  5. Lists the output tree

OUTPUTS (all under finetuning/runs/):
  runs/sentiment_v4/
    ├── fold_1/{lora/, tokenizer/, metrics.json}
    ├── ...
    ├── fold_5/...
    └── kfold_results.json

  runs/relevancy_v4/
    ├── fold_1/...fold_5/...
    └── kfold_results.json

USAGE:
  cd /path/to/ID-Political-Sentiment-Tracker
  python finetuning/run_v4.py                       # both tasks, kfold=5
  python finetuning/run_v4.py --task sentiment      # only sentiment
  python finetuning/run_v4.py --kfold 3             # 3-fold instead of 5
  python finetuning/run_v4.py --eval-only           # skip training, just summarize
  python finetuning/run_v4.py --python /path/to/python  # use specific interpreter

NOTE: For Colab, use colab_complete_pipeline_v4.py instead (handles cloning
      + install + upload to HuggingFace). This script is for LOCAL execution.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def banner(task: str, kfold: int, eval_only: bool, python: str):
    print("=" * 64)
    print("  v4 Fine-tuning Pipeline (LOCAL Python runner)")
    print("=" * 64)
    print(f"  Script dir : {SCRIPT_DIR}")
    py_ver = subprocess.run([python, "--version"], capture_output=True, text=True).stdout.strip()
    print(f"  Python     : {py_ver} ({python})")
    print(f"  Task       : {task}")
    print(f"  K-fold     : {kfold}")
    print(f"  Eval-only  : {eval_only}")
    print("=" * 64)
    print()


def step_verify_env(python: str) -> bool:
    """Step 1: verify required packages are installed."""
    print(">>> [1/5] Verifying environment...")
    required = ["torch", "transformers", "peft", "sklearn", "numpy"]
    missing = []
    for pkg in required:
        r = subprocess.run([python, "-c", f"import {pkg}"], capture_output=True)
        if r.returncode != 0:
            missing.append(pkg)
    if missing:
        print(f"    ERROR: missing packages: {', '.join(missing)}")
        print(f"    Run: pip install -r finetuning/requirements_finetune.txt")
        return False

    # CUDA check
    r = subprocess.run(
        [python, "-c", "import torch; print('yes' if torch.cuda.is_available() else 'no')"],
        capture_output=True, text=True,
    )
    cuda = r.stdout.strip()
    print(f"    torch CUDA available: {cuda}")
    if cuda == "no":
        print("    WARNING: No CUDA — training will be extremely slow on CPU.")
        print("             Consider running on Colab via colab_complete_pipeline_v4.py")
        try:
            yn = input("    Continue anyway? [y/N] ").strip()
        except EOFError:
            yn = ""
        if yn.lower() not in ("y", "yes"):
            return False
    print("    All required packages present.")
    return True


def step_verify_dataset() -> Path | None:
    """Step 2: verify the gold-standard dataset exists."""
    print("\n>>> [2/5] Verifying dataset...")
    dataset = SCRIPT_DIR / "datasets" / "dataset_gold_standard_final.jsonl"
    if not dataset.exists():
        print(f"    ERROR: Dataset not found at:\n      {dataset}")
        print(f"\n    Place dataset_gold_standard_final.jsonl in:\n      {SCRIPT_DIR / 'datasets'}")
        return None
    with open(dataset) as f:
        nrows = sum(1 for _ in f)
    print(f"    Dataset: {dataset}")
    print(f"    Rows   : {nrows}")
    if nrows < 100:
        print(f"    WARNING: Very small dataset ({nrows} rows). Results will be unreliable.")
    return dataset


def run_finetune_task(python: str, task: str, kfold: int, eval_only: bool):
    """Step 3/4: run K-fold fine-tuning for a task."""
    print("\n" + "=" * 64)
    print(f">>> Fine-tuning: {task} (kfold={kfold})")
    print("=" * 64)
    if eval_only:
        print("    (skipped — --eval-only)")
        return 0
    cmd = [python, "finetune_v4.py", "--task", task, "--kfold", str(kfold)]
    print(f"    $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(SCRIPT_DIR)).returncode


def summarize_task(python: str, task: str):
    """Step 5: print K-fold summary for a task."""
    results_file = SCRIPT_DIR / "runs" / f"{task}_v4" / "kfold_results.json"
    print("\n" + "-" * 64)
    print(f"  K-fold summary: {task}")
    print("-" * 64)
    if not results_file.exists():
        print(f"  (no kfold_results.json found at {results_file})")
        return
    cmd = [python, "evaluate_v4.py", "--task", task, "--kfold-results", str(results_file)]
    subprocess.run(cmd, cwd=str(SCRIPT_DIR))


def show_output_tree():
    """Step 6: show the output directory structure."""
    print("\n>>> [5/5] Output structure:")
    print()
    runs_dir = SCRIPT_DIR / "runs"
    if not runs_dir.exists():
        print("    (no runs/ directory yet)")
        return
    # Walk the tree, max depth 3
    for root, dirs, files in os.walk(runs_dir):
        depth = Path(root).relative_to(runs_dir).parts
        if len(depth) > 3:
            continue
        indent = "    " + "  " * len(depth)
        print(f"{indent}{Path(root).name}/")
        for f in sorted(files):
            print(f"{indent}  {f}")


def main():
    ap = argparse.ArgumentParser(description="v4 local pipeline runner")
    ap.add_argument("--task", choices=["sentiment", "relevancy", "both"], default="both")
    ap.add_argument("--kfold", type=int, default=5, help="0=disabled, 5=recommended")
    ap.add_argument("--eval-only", action="store_true", help="skip training, just summarize")
    ap.add_argument("--python", default=sys.executable, help="Python interpreter to use")
    args = ap.parse_args()

    banner(args.task, args.kfold, args.eval_only, args.python)

    if not step_verify_env(args.python):
        sys.exit(1)
    if not step_verify_dataset():
        sys.exit(1)

    tasks = ["sentiment", "relevancy"] if args.task == "both" else [args.task]
    for task in tasks:
        rc = run_finetune_task(args.python, task, args.kfold, args.eval_only)
        if rc != 0:
            print(f"\nERROR: fine-tuning {task} failed with exit code {rc}")
            sys.exit(rc)

    print("\n>>> [4/5] K-fold summaries...")
    for task in tasks:
        summarize_task(args.python, task)

    show_output_tree()

    print("\n" + "=" * 64)
    print(f"  DONE. Outputs saved under: {SCRIPT_DIR / 'runs'}")
    print("=" * 64)
    print("\nNext steps:")
    print("  - Review kfold_results.json for per-fold metrics")
    print("  - Evaluate single fold with confidence sweep:")
    print(f"      {args.python} evaluate_v4.py --task sentiment --run-dir runs/sentiment_v4/fold_1")
    print("  - Upload best fold to HuggingFace (see colab_complete_pipeline_v4.py)")
    print()


if __name__ == "__main__":
    main()
