"""
colab_complete_pipeline_v4.py
=============================
Complete Colab pipeline for v4 fine-tuning.

Steps: install deps → clone repo → fine-tune sentiment → fine-tune relevancy
       → evaluate → upload to HuggingFace → print summary

Usage:
  python colab_complete_pipeline_v4.py --steps all
  python colab_complete_pipeline_v4.py --steps sentiment-only
"""
import os, sys, json, argparse, subprocess
from pathlib import Path

REPO_URL = "https://github.com/raynzz455/ID-Political-Sentiment-Tracker.git"
REPO_DIR = "/content/ID-Political-Sentiment-Tracker"
HF_TOKEN_ENV = "HF_TOKEN"
HF_MODEL_SENTIMENT = "raynzz455/id-political-sentiment-sentiment-v4"
HF_MODEL_RELEVANCY = "raynzz455/id-political-sentiment-relevancy-v4"
DATASET = "dataset_gold_standard.jsonl"
K_FOLD = 5

def run(cmd, cwd=None, check=True):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, stdout=sys.stdout, stderr=sys.stderr, text=True)
    if check and result.returncode != 0:
        print(f"FAILED: code {result.returncode}"); sys.exit(result.returncode)
    return result

def step_install():
    print("\n=== STEP 1: Install ===")
    run("pip install -q transformers peft scikit-learn accelerate sentencepiece")
    run('python -c "import torch; print(f\'CUDA: {torch.cuda.is_available()}\')"')

def step_clone():
    print("\n=== STEP 2: Clone ===")
    if not Path(REPO_DIR).exists():
        run(f"git clone {REPO_URL} {REPO_DIR}")
    else:
        run(f"cd {REPO_DIR} && git pull", check=False)
    ds = Path(REPO_DIR) / "finetuning" / "datasets" / DATASET
    if not ds.exists():
        print(f"ERROR: {ds} not found. Run cleaning pipeline first."); sys.exit(1)
    run(f"wc -l {ds}")

def step_finetune_sentiment():
    print("\n=== STEP 3: Fine-tune Sentiment ===")
    run(f"cd {REPO_DIR}/finetuning && python finetune_v4.py --task sentiment --kfold {K_FOLD}")

def step_finetune_relevancy():
    print("\n=== STEP 4: Fine-tune Relevancy ===")
    run(f"cd {REPO_DIR}/finetuning && python finetune_v4.py --task relevancy --kfold {K_FOLD}")

def step_evaluate():
    print("\n=== STEP 5: Evaluate ===")
    for task, dir_name in [("sentiment","sentiment_v4"), ("relevancy","relevancy_v4")]:
        kf = Path(REPO_DIR)/"finetuning"/"runs"/dir_name/"kfold_results.json"
        if kf.exists():
            print(f"\n--- {task} K-fold ---")
            run(f'cd {REPO_DIR}/finetuning && python evaluate_v4.py --task {task} --kfold-results {kf}')

def step_upload():
    print("\n=== STEP 6: Upload ===")
    token = os.environ.get(HF_TOKEN_ENV)
    if not token:
        print(f"{HF_TOKEN_ENV} not set. Skipping upload."); return
    run(f"huggingface-cli login --token {token}")
    for task, hf_model, dir_name in [
        ("sentiment", HF_MODEL_SENTIMENT, "sentiment_v4"),
        ("relevancy", HF_MODEL_RELEVANCY, "relevancy_v4"),
    ]:
        kf = Path(REPO_DIR)/"finetuning"/"runs"/dir_name/"kfold_results.json"
        if kf.exists():
            kfold = json.load(open(kf))
            best = max(kfold.get("fold_results",[]), key=lambda x: x.get("macro_f1",0))
            fold_dir = Path(REPO_DIR)/"finetuning"/"runs"/dir_name/f"fold_{best.get('fold',1)}"
            if fold_dir.exists():
                print(f"Uploading {task} (fold {best.get('fold')}, f1={best.get('macro_f1',0):.4f})")
                run(f"huggingface-cli upload {hf_model} {fold_dir} --token {token}")

def step_summary():
    print("\n=== COMPLETE ===")
    for task, dir_name in [("sentiment","sentiment_v4"), ("relevancy","relevancy_v4")]:
        kf = Path(REPO_DIR)/"finetuning"/"runs"/dir_name/"kfold_results.json"
        if kf.exists():
            kfold = json.load(open(kf))
            print(f"{task}: acc={kfold.get('mean_accuracy',0):.4f} ± {kfold.get('std_accuracy',0):.4f}, "
                  f"f1={kfold.get('mean_macro_f1',0):.4f} ± {kfold.get('std_macro_f1',0):.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="all")
    args = ap.parse_args()
    steps = args.steps.lower().split(",")
    if "all" in steps: steps = ["install","clone","sentiment","relevancy","evaluate","upload","summary"]
    elif "sentiment-only" in steps: steps = ["install","clone","sentiment","evaluate","summary"]
    step_map = {"install":step_install,"clone":step_clone,"sentiment":step_finetune_sentiment,
                "relevancy":step_finetune_relevancy,"evaluate":step_evaluate,"upload":step_upload,"summary":step_summary}
    for s in steps:
        s = s.strip()
        if s in step_map: step_map[s]()

if __name__ == "__main__":
    main()
