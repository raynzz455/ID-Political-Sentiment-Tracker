"""
upload_to_hf.py — Upload trained models to HuggingFace (Python API, no CLI)
==========================================================================
FIX v4.6: huggingface-cli deprecated → use Python API (HfApi.upload_folder)

This script finds the best fold from kfold_results.json and uploads it to HF.

Usage:
  # Set token first
  import os
  os.environ['HF_TOKEN'] = 'hf_xxxxx'

  # Upload both models
  !python upload_to_hf.py

  # Or with explicit token
  !python upload_to_hf.py --token hf_xxxxx

  # Upload specific task only
  !python upload_to_hf.py --task sentiment
  !python upload_to_hf.py --task relevancy
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path

# Model repos (TANPA -v4, sesuai dengan yang user buat di HF)
HF_MODELS = {
    "sentiment": "raynzz455/id-political-sentiment-sentiment",
    "relevancy": "raynzz455/id-political-sentiment-relevancy",
}

# Output dirs
RUN_DIRS = {
    "sentiment": Path(__file__).resolve().parent / "runs" / "sentiment_v4",
    "relevancy": Path(__file__).resolve().parent / "runs" / "relevancy_v4",
}

# Colab path fallback
COLAB_BASE = Path("/content/ID-Political-Sentiment-Tracker/finetuning/runs")


def find_runs_dir(task: str) -> Path | None:
    """Find runs directory — check local first, then Colab path."""
    local = RUN_DIRS[task]
    if local.exists():
        return local

    colab = COLAB_BASE / f"{task}_v4"
    if colab.exists():
        return colab

    return None


def find_best_fold(runs_dir: Path) -> dict | None:
    """Find best fold from kfold_results.json."""
    kf_path = runs_dir / "kfold_results.json"
    if not kf_path.exists():
        print(f"❌ kfold_results.json not found at {kf_path}")
        return None

    kfold = json.load(open(kf_path))
    fold_results = kfold.get("fold_results", kfold.get("folds", []))
    if not fold_results:
        print(f"❌ No fold results in {kf_path}")
        return None

    best = max(fold_results, key=lambda x: x.get("macro_f1", 0))
    best_fold = best.get("fold", 1)
    fold_dir = runs_dir / f"fold_{best_fold}"

    return {
        "fold": best_fold,
        "macro_f1": best.get("macro_f1", 0),
        "accuracy": best.get("accuracy", 0),
        "fold_dir": fold_dir,
        "temperature": best.get("temperature", 1.0),
    }


def upload_model(task: str, token: str) -> bool:
    """Upload best fold for a task to HuggingFace."""
    print(f"\n{'=' * 60}")
    print(f"  Upload: {task}")
    print(f"{'=' * 60}")

    # Find runs directory
    runs_dir = find_runs_dir(task)
    if not runs_dir:
        print(f"❌ Runs directory not found for {task}")
        print(f"   Checked: {RUN_DIRS[task]}")
        print(f"   Checked: {COLAB_BASE / f'{task}_v4'}")
        return False

    print(f"✅ Found runs: {runs_dir}")

    # Find best fold
    info = find_best_fold(runs_dir)
    if not info:
        return False

    if not info["fold_dir"].exists():
        print(f"❌ Fold directory not found: {info['fold_dir']}")
        return False

    print(f"✅ Best fold: {info['fold']} (f1={info['macro_f1']:.4f}, acc={info['accuracy']:.4f})")
    print(f"   Source: {info['fold_dir']}")

    # List files to upload
    files = list(info["fold_dir"].rglob("*"))
    file_count = sum(1 for f in files if f.is_file())
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    print(f"   Files: {file_count} ({total_size / 1e6:.1f} MB)")

    # Upload
    hf_model = HF_MODELS[task]
    print(f"   Target: https://huggingface.co/{hf_model}")

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)

        # Verify token
        whoami = api.whoami()
        print(f"   Account: {whoami.get('name', 'unknown')}")

        # Upload
        print(f"\n   Uploading...")
        api.upload_folder(
            folder_path=str(info["fold_dir"]),
            repo_id=hf_model,
            repo_type="model",
            token=token,
        )
        print(f"\n✅ Upload SUCCESS!")
        print(f"   View at: https://huggingface.co/{hf_model}")
        return True

    except ImportError:
        print("❌ huggingface_hub not installed")
        print("   Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print(f"\n   Manual upload:")
        print(f"   1. Go to: https://huggingface.co/{hf_model}")
        print(f"   2. Click 'Add file' → 'Upload files'")
        print(f"   3. Drag files from: {info['fold_dir']}")
        return False


def main():
    ap = argparse.ArgumentParser(description="Upload trained models to HuggingFace")
    ap.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    ap.add_argument("--task", choices=["sentiment", "relevancy", "both"], default="both",
                    help="Which task to upload (default: both)")
    args = ap.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("❌ No HF_TOKEN provided")
        print("   Usage: python upload_to_hf.py --token hf_xxxxx")
        print("   Or:    export HF_TOKEN=hf_xxxxx && python upload_to_hf.py")
        sys.exit(1)

    print("=" * 60)
    print("  HuggingFace Upload (Python API — no CLI)")
    print("=" * 60)
    print(f"  Token: {token[:10]}...{token[-4:]}")

    tasks = ["sentiment", "relevancy"] if args.task == "both" else [args.task]
    results = {}
    for task in tasks:
        results[task] = upload_model(task, token)

    # Summary
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    for task, success in results.items():
        status = "✅ Uploaded" if success else "❌ Failed"
        url = f"https://huggingface.co/{HF_MODELS[task]}"
        print(f"  {task:12s}: {status}")
        if success:
            print(f"                {url}")

    if all(results.values()):
        print(f"\n🎉 All models uploaded!")
        print(f"\nNext: switch production to fine-tuned models:")
        print(f"  export NLP_RELEVANCY_MODEL={HF_MODELS['relevancy']}")
        print(f"  export NLP_SENTIMENT_MODEL={HF_MODELS['sentiment']}")
    else:
        print(f"\n⚠️  Some uploads failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
