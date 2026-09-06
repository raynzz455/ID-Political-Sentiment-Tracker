#!/usr/bin/env bash
# =============================================================================
# run_v4.sh — Local runner for the v4 fine-tuning + evaluation pipeline
# =============================================================================
#
# WHAT THIS DOES (in order):
#   1. Verifies Python + required packages (torch, transformers, peft, ...)
#   2. Verifies the gold-standard dataset exists
#   3. Runs K-fold fine-tuning for the sentiment task
#   4. Runs K-fold fine-tuning for the relevancy task
#   5. Prints K-fold summaries for both tasks
#   6. (Optional) Evaluates a single fold with the confidence-threshold sweep
#
# OUTPUTS (all under finetuning/runs/):
#   runs/sentiment_v4/
#     ├── fold_1/{lora/, tokenizer/, metrics.json}   <- per-fold LoRA adapter
#     ├── fold_2/...
#     ├── fold_5/...
#     └── kfold_results.json                          <- aggregate metrics
#
#   runs/relevancy_v4/
#     ├── fold_1/...fold_5/...
#     └── kfold_results.json
#
# USAGE:
#   cd /path/to/ID-Political-Sentiment-Tracker
#   bash finetuning/run_v4.sh                  # default: both tasks, kfold=5
#   bash finetuning/run_v4.sh --task sentiment # only sentiment
#   bash finetuning/run_v4.sh --kfold 3        # 3-fold instead of 5
#   bash finetuning/run_v4.sh --eval-only      # skip training, just summarize
#
# PREREQUISITES:
#   - Python 3.10+
#   - pip install -r finetuning/requirements_finetune.txt
#   - GPU with >= 8GB VRAM (adversarial training auto-disables < 12GB)
#   - Dataset at: finetuning/datasets/dataset_gold_standard_final.jsonl
#
# NOTE: For Colab, use colab_complete_pipeline_v4.py instead (it handles
#       cloning + install + upload to HuggingFace). This script is for LOCAL
#       execution on your own machine.
# =============================================================================
set -euo pipefail

# ---- Defaults ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TASK="both"          # sentiment | relevancy | both
KFOLD=5
EVAL_ONLY=false
PYTHON="${PYTHON:-python3}"

# ---- Parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        --kfold) KFOLD="$2"; shift 2 ;;
        --eval-only) EVAL_ONLY=true; shift ;;
        --python) PYTHON="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

cd "$SCRIPT_DIR"

# ---- Banner -----------------------------------------------------------------
echo "================================================================"
echo "  v4 Fine-tuning Pipeline (LOCAL runner)"
echo "================================================================"
echo "  Script dir : $SCRIPT_DIR"
echo "  Python     : $($PYTHON --version 2>&1)"
echo "  Task       : $TASK"
echo "  K-fold     : $KFOLD"
echo "  Eval-only  : $EVAL_ONLY"
echo "================================================================"
echo ""

# ---- Step 1: Verify environment --------------------------------------------
echo ">>> [1/5] Verifying environment..."

if ! $PYTHON -c "import torch" 2>/dev/null; then
    echo "ERROR: torch not installed."
    echo "  Run: pip install -r finetuning/requirements_finetune.txt"
    exit 1
fi

CUDA_OK=$($PYTHON -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null || echo "no")
echo "    torch CUDA available: $CUDA_OK"
if [[ "$CUDA_OK" == "no" ]]; then
    echo "    WARNING: No CUDA — training will be extremely slow on CPU."
    echo "             Consider running on Colab via colab_complete_pipeline_v4.py"
    read -p "    Continue anyway? [y/N] " yn
    [[ "$yn" =~ ^[Yy]$ ]] || exit 0
fi

for pkg in transformers peft sklearn numpy; do
    if ! $PYTHON -c "import $pkg" 2>/dev/null; then
        echo "ERROR: $pkg not installed."
        echo "  Run: pip install -r finetuning/requirements_finetune.txt"
        exit 1
    fi
done
echo "    All required packages present."

# ---- Step 2: Verify dataset ------------------------------------------------
echo ""
echo ">>> [2/5] Verifying dataset..."
DATASET="$SCRIPT_DIR/datasets/dataset_gold_standard_final.jsonl"
if [[ ! -f "$DATASET" ]]; then
    echo "ERROR: Dataset not found at:"
    echo "  $DATASET"
    echo ""
    echo "Place dataset_gold_standard_final.jsonl in:"
    echo "  $SCRIPT_DIR/datasets/"
    exit 1
fi
NROWS=$(wc -l < "$DATASET")
echo "    Dataset: $DATASET"
echo "    Rows   : $NROWS"
if [[ "$NROWS" -lt 100 ]]; then
    echo "    WARNING: Very small dataset ($NROWS rows). Results will be unreliable."
fi

# ---- Helper: run a task ----------------------------------------------------
run_finetune_task() {
    local task="$1"
    echo ""
    echo "================================================================"
    echo ">>> [3/5] Fine-tuning: $task (kfold=$KFOLD)"
    echo "================================================================"
    if $EVAL_ONLY; then
        echo "    (skipped — --eval-only)"
        return
    fi
    $PYTHON finetune_v4.py --task "$task" --kfold "$KFOLD"
}

summarize_task() {
    local task="$1"
    local results_file="$SCRIPT_DIR/runs/${task}_v4/kfold_results.json"
    echo ""
    echo "----------------------------------------------------------------"
    echo "  K-fold summary: $task"
    echo "----------------------------------------------------------------"
    if [[ ! -f "$results_file" ]]; then
        echo "  (no kfold_results.json found at $results_file)"
        return
    fi
    $PYTHON evaluate_v4.py --task "$task" --kfold-results "$results_file"
}

# ---- Step 3+4: Run fine-tuning ---------------------------------------------
if [[ "$TASK" == "sentiment" || "$TASK" == "both" ]]; then
    run_finetune_task "sentiment"
fi
if [[ "$TASK" == "relevancy" || "$TASK" == "both" ]]; then
    run_finetune_task "relevancy"
fi

# ---- Step 5: Summarize -----------------------------------------------------
echo ""
echo ">>> [4/5] K-fold summaries..."
if [[ "$TASK" == "sentiment" || "$TASK" == "both" ]]; then
    summarize_task "sentiment"
fi
if [[ "$TASK" == "relevancy" || "$TASK" == "both" ]]; then
    summarize_task "relevancy"
fi

# ---- Step 6: Show output tree ----------------------------------------------
echo ""
echo ">>> [5/5] Output structure:"
echo ""
# Try tree first, fall back to find
if command -v tree &>/dev/null; then
    tree -L 3 "$SCRIPT_DIR/runs" 2>/dev/null || find "$SCRIPT_DIR/runs" -maxdepth 3 -type d 2>/dev/null | sort
else
    find "$SCRIPT_DIR/runs" -maxdepth 3 2>/dev/null | sort | sed 's/^/    /'
fi

echo ""
echo "================================================================"
echo "  DONE. Outputs saved under: $SCRIPT_DIR/runs/"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  - Review kfold_results.json for per-fold metrics"
echo "  - Evaluate single fold with confidence sweep:"
echo "      $PYTHON evaluate_v4.py --task sentiment --run-dir runs/sentiment_v4/fold_1"
echo "  - Upload best fold to HuggingFace (see colab_complete_pipeline_v4.py)"
echo ""
