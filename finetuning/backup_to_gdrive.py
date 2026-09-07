"""
backup_to_gdrive.py
===================
Auto-backup fine-tuning outputs to Google Drive + (optional) HuggingFace.

Run AFTER training completes. Prevents loss of model weights when Colab
session expires (Colab ephemeral disk is wiped on disconnect).

WHAT IT BACKS UP:
  - finetuning/runs/sentiment_v4/  (all folds + kfold_results.json)
  - finetuning/runs/relevancy_v4/  (all folds + kfold_results.json)
  - finetuning/configs/hyperparams_v4.py  (for reproducibility)

DESTINATIONS (pick one or both):
  1. Google Drive  → /content/drive/MyDrive/finetuning_runs_v4_YYYYMMDD_HHMMSS/
  2. HuggingFace   → raynzz455/id-political-sentiment-{sentiment,relevancy}-v4

USAGE in Colab:
  # After training finishes:
  !python backup_to_gdrive.py                          # backup to Drive only
  !python backup_to_gdrive.py --upload-hf              # backup to Drive + HF
  !python backup_to_gdrive.py --upload-hf --hf-token hf_xxx  # explicit token
  !python backup_to_gdrive.py --skip-drive             # only HF upload
  !python backup_to_gdrive.py --zip-download           # also download zip to browser
"""
from __future__ import annotations
import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs"
CONFIGS_DIR = SCRIPT_DIR / "configs"
DRIVE_BASE = Path("/content/drive/MyDrive")

HF_SENTIMENT_MODEL = "raynzz455/id-political-sentiment-sentiment"
HF_RELEVANCY_MODEL = "raynzz455/id-political-sentiment-relevancy"


def print_banner(msg):
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}")


def mount_gdrive() -> bool:
    """Mount Google Drive if not already mounted."""
    drive_mount = Path("/content/drive/MyDrive")
    if drive_mount.exists():
        print("✅ Google Drive already mounted")
        return True

    print("Mounting Google Drive...")
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
        if drive_mount.exists():
            print("✅ Google Drive mounted successfully")
            return True
    except ImportError:
        print("⚠️  Not running in Colab — skipping Drive mount")
    except Exception as e:
        print(f"❌ Failed to mount Drive: {e}")
    return False


def backup_to_drive(timestamp: str) -> Path | None:
    """Copy runs/ + configs/ to Google Drive with timestamp."""
    print_banner("BACKUP TO GOOGLE DRIVE")

    if not RUNS_DIR.exists():
        print(f"❌ No runs/ directory found at {RUNS_DIR}")
        print("   Run training first: python finetune_v4.py --task sentiment --kfold 5")
        return None

    if not mount_gdrive():
        print("⚠️  Google Drive not available — skipping Drive backup")
        return None

    # Create timestamped backup folder
    backup_name = f"finetuning_runs_v4_{timestamp}"
    backup_dir = DRIVE_BASE / backup_name
    print(f"Backup destination: {backup_dir}")

    # Remove old backup with same name (unlikely but safe)
    if backup_dir.exists():
        print(f"  Removing existing backup: {backup_dir}")
        shutil.rmtree(backup_dir)

    backup_dir.mkdir(parents=True, exist_ok=True)

    # Copy runs/
    runs_dst = backup_dir / "runs"
    print(f"  Copying runs/ ...")
    shutil.copytree(RUNS_DIR, runs_dst)
    n_fold_dirs = sum(1 for p in runs_dst.rglob("fold_*") if p.is_dir())
    print(f"  ✅ Copied runs/ ({n_fold_dirs} fold directories)")

    # Copy configs/ (for reproducibility)
    if CONFIGS_DIR.exists():
        print(f"  Copying configs/ ...")
        shutil.copytree(CONFIGS_DIR, backup_dir / "configs")
        print(f"  ✅ Copied configs/")

    # Copy hyperparams file
    hp_file = CONFIGS_DIR / "hyperparams_v4.py"
    if hp_file.exists():
        shutil.copy(hp_file, backup_dir / "hyperparams_v4.py")

    # Create README with backup info
    readme = backup_dir / "BACKUP_README.md"
    readme.write_text(f"""# Fine-tuning Backup — {timestamp}

## Contents
- `runs/sentiment_v4/` — 5-fold LoRA adapters + metrics for sentiment task
- `runs/relevancy_v4/` — 5-fold LoRA adapters + metrics for relevancy task
- `configs/hyperparams_v4.py` — hyperparameter config used

## How to Restore
1. Copy `runs/` folder back to `finetuning/runs/` in the repo
2. Or load LoRA adapter directly:
   ```python
   from transformers import AutoTokenizer, AutoModelForSequenceClassification
   from peft import PeftModel
   tok = AutoTokenizer.from_pretrained("runs/sentiment_v4/fold_1/tokenizer")
   model = AutoModelForSequenceClassification.from_pretrained("apriandito/indobert-sentiment-classifier")
   model = PeftModel.from_pretrained(model, "runs/sentiment_v4/fold_1/lora")
   ```

## Backup Info
- Created: {datetime.now().isoformat()}
- Source: {RUNS_DIR}
- Total size: {sum(f.stat().st_size for f in backup_dir.rglob('*') if f.is_file()) / 1e6:.1f} MB
""")

    # Calculate total size
    total_size = sum(f.stat().st_size for f in backup_dir.rglob("*") if f.is_file())
    print(f"\n✅ Backup complete: {backup_dir}")
    print(f"   Total size: {total_size / 1e6:.1f} MB")
    return backup_dir


def find_best_folds() -> dict:
    """Find best fold (by macro_f1) for each task."""
    best = {}
    for task, model_name in [("sentiment", HF_SENTIMENT_MODEL),
                              ("relevancy", HF_RELEVANCY_MODEL)]:
        kf_path = RUNS_DIR / f"{task}_v4" / "kfold_results.json"
        if not kf_path.exists():
            print(f"⚠️  No kfold_results.json for {task}")
            continue
        kfold = json.load(open(kf_path))
        fold_results = kfold.get("fold_results", kfold.get("folds", []))
        if not fold_results:
            print(f"⚠️  No fold results for {task}")
            continue
        best_fold = max(fold_results, key=lambda x: x.get("macro_f1", 0))
        best[task] = {
            "fold": best_fold.get("fold", 1),
            "macro_f1": best_fold.get("macro_f1", 0),
            "accuracy": best_fold.get("accuracy", 0),
            "fold_dir": RUNS_DIR / f"{task}_v4" / f"fold_{best_fold.get('fold', 1)}",
            "hf_model": model_name,
        }
        print(f"  {task}: best fold = {best[task]['fold']} "
              f"(f1={best[task]['macro_f1']:.4f}, acc={best[task]['accuracy']:.4f})")
    return best


def upload_to_huggingface(hf_token: str, best_folds: dict):
    """Upload best fold for each task to HuggingFace Hub."""
    print_banner("UPLOAD TO HUGGINGFACE HUB")

    if not hf_token:
        print("❌ No HF_TOKEN provided")
        print("   Get token at: https://huggingface.co/settings/tokens")
        return

    # Login
    print("Logging in to HuggingFace...")
    r = subprocess.run(
        ["huggingface-cli", "login", "--token", hf_token],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"❌ Login failed: {r.stderr}")
        return
    print("✅ Logged in")

    for task, info in best_folds.items():
        fold_dir = info["fold_dir"]
        hf_model = info["hf_model"]
        if not fold_dir.exists():
            print(f"⚠️  Fold dir not found: {fold_dir}")
            continue

        print(f"\nUploading {task} (fold {info['fold']}, f1={info['macro_f1']:.4f})...")
        print(f"  Source: {fold_dir}")
        print(f"  Target: huggingface.co/{hf_model}")

        cmd = ["huggingface-cli", "upload", hf_model, str(fold_dir),
               "--token", hf_token]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  ✅ Uploaded successfully")
            print(f"     View at: https://huggingface.co/{hf_model}")
        else:
            print(f"  ❌ Upload failed: {r.stderr}")
            print(f"     Try manual: huggingface-cli upload {hf_model} {fold_dir} --token YOUR_TOKEN")


def create_zip_download():
    """Create a zip file and trigger browser download (Colab only)."""
    print_banner("ZIP + DOWNLOAD TO BROWSER")

    if not RUNS_DIR.exists():
        print("❌ No runs/ directory to zip")
        return

    zip_path = Path("/content/v4_runs_backup.zip")
    print(f"Creating zip: {zip_path}")
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=str(RUNS_DIR))
    size_mb = zip_path.stat().st_size / 1e6
    print(f"✅ Zip created: {size_mb:.1f} MB")

    try:
        from google.colab import files  # type: ignore
        print("Triggering browser download...")
        files.download(str(zip_path))
        print("✅ Download started — check your browser downloads")
    except ImportError:
        print(f"⚠️  Not in Colab — zip saved at {zip_path}, download manually")  # FIX FT#7: add f-string
    except Exception as e:
        print(f"⚠️  Download trigger failed: {e}")
        print(f"   Zip is at: {zip_path}")


def print_summary(drive_dir: Path | None, best_folds: dict, hf_uploaded: bool):
    """Print final summary of what was backed up."""
    print_banner("BACKUP SUMMARY")

    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    if drive_dir:
        print(f"✅ Google Drive: {drive_dir}")
    else:
        print(f"❌ Google Drive: not backed up")

    print()
    print("Best folds identified:")
    for task, info in best_folds.items():
        print(f"  {task}: fold {info['fold']} "
              f"(f1={info['macro_f1']:.4f}, acc={info['accuracy']:.4f})")

    print()
    if hf_uploaded:
        print("✅ HuggingFace: uploaded")
        for task, info in best_folds.items():
            print(f"  {task}: https://huggingface.co/{info['hf_model']}")
    else:
        print("❌ HuggingFace: not uploaded (use --upload-hf)")

    print()
    print("=" * 60)
    print("  Your model weights are now SAFE from Colab disconnect!")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="Backup fine-tuning outputs to Drive + HF")
    ap.add_argument("--upload-hf", action="store_true",
                    help="Upload best fold to HuggingFace Hub")
    ap.add_argument("--hf-token", default=None,
                    help="HuggingFace token (or set HF_TOKEN env var)")
    ap.add_argument("--skip-drive", action="store_true",
                    help="Skip Google Drive backup")
    ap.add_argument("--zip-download", action="store_true",
                    help="Also create zip and trigger browser download")
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Find best folds first
    print_banner("FINDING BEST FOLDS")
    best_folds = find_best_folds()
    if not best_folds:
        print("❌ No folds found. Run training first.")
        sys.exit(1)

    # Backup to Google Drive
    drive_dir = None
    if not args.skip_drive:
        drive_dir = backup_to_drive(timestamp)

    # Upload to HuggingFace
    hf_uploaded = False
    if args.upload_hf:
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if not token:
            print("\n❌ No HF_TOKEN. Get one at: https://huggingface.co/settings/tokens")
            print("   Then run: python backup_to_gdrive.py --upload-hf --hf-token hf_xxxxx")
        else:
            upload_to_huggingface(token, best_folds)
            hf_uploaded = True

    # Optional: zip download
    if args.zip_download:
        create_zip_download()

    # Summary
    print_summary(drive_dir, best_folds, hf_uploaded)


if __name__ == "__main__":
    main()
