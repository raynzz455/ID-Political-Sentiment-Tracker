#!/bin/bash
# cleanup_repo.sh
# ================
# Cleanup obsolete/superseded files from ID-Political-Sentiment-Tracker repo.
#
# Run from repo root: bash finetuning/scripts/cleanup_repo.sh
#
# Categories deleted:
#   1. Superseded datasets (v1=dataset.jsonl, v4, intermediate label files)
#   2. Superseded scripts (build_dataset_v2, build_enhanced, build_gold, relabel, llm_verify_all, llm_verify_sdk)
#   3. One-time test scripts (test_v15_vs_v14_regex, test_overconfidence_simulation)
#   4. Duplicate patches (original v15 patch — keep deployed version)
#   5. Empty/stub dirs (apps/web if only README, supabase/.temp)
#
# Files KEPT (do NOT delete):
#   - dataset_v9.jsonl (current best)
#   - dataset_v3.jsonl, v7.jsonl (reference, used by v9 merge)
#   - dataset_v2.jsonl (superset of v3, historical)
#   - dataset_v5.jsonl, v6.jsonl (used by v9 merge)
#   - dataset_enhanced.jsonl (used by v9 merge)
#   - gold_labels.jsonl (27 human-labeled, keep for reference)
#   - llm_verified_v3.jsonl, v8.jsonl, v9.jsonl (verification results)
#   - All current production scripts (finetune.py, finetune_v3.py, evaluate.py,
#     build_dataset_v3.py, build_dataset_v9.py, verify_dataset_v*.mjs,
#     tahap2_llm_hybrid_pipeline.py, infer_calibrated.py, upload_huggingface.py,
#     hyperparams.py, dataset_schema.py, export_and_label_new_data.py)
#   - hyperparams_v2.py, hyperparams_v3.py, hyperparams_optimized.py
#   - colab_complete_pipeline.py (kept — main Colab script)
#   - patches/entity_resolution_worker_v15_deployed.py (kept — reference)

set -e

REPO="${1:-/tmp/idpst_repo}"
cd "$REPO"

echo "============================================================"
echo "REPO CLEANUP — removing obsolete/superseded files"
echo "============================================================"
echo "Repo: $REPO"
echo ""

# 1. Superseded datasets
echo "[1/5] Removing superseded datasets..."
DELETE_DATASETS=(
  "finetuning/datasets/dataset.jsonl"              # v1, 909 rows, original — superseded
  "finetuning/datasets/dataset_v4.jsonl"           # 1011 rows, pure model labels — superseded
  "finetuning/datasets/llm_labels.jsonl"           # 375 rows, intermediate — merged into v3/v9
  "finetuning/datasets/llm_verified_labels.jsonl"   # 144 rows, intermediate — merged into v3
)
for f in "${DELETE_DATASETS[@]}"; do
  if [ -f "$f" ]; then
    sz=$(du -h "$f" | cut -f1)
    rm "$f"
    echo "  deleted: $f ($sz)"
  fi
done

# 2. Superseded scripts
echo ""
echo "[2/5] Removing superseded scripts..."
DELETE_SCRIPTS=(
  "finetuning/scripts/build_dataset_v2.py"          # superseded by build_dataset_v3 → v9
  "finetuning/scripts/build_enhanced_dataset.py"    # superseded by build_dataset_v9
  "finetuning/scripts/build_gold_labels.py"          # one-time use, gold merged
  "finetuning/scripts/relabel_dataset.py"            # superseded by LLM verification
  "finetuning/scripts/llm_verify_all.py"             # superseded by verify_dataset_v*.mjs
  "finetuning/scripts/llm_verify_sdk.mjs"            # superseded by verify_dataset_v9.mjs
)
for f in "${DELETE_SCRIPTS[@]}"; do
  if [ -f "$f" ]; then
    rm "$f"
    echo "  deleted: $f"
  fi
done

# 3. One-time test scripts
echo ""
echo "[3/5] Removing one-time test scripts..."
DELETE_TESTS=(
  "finetuning/scripts/test_v15_vs_v14_regex.py"
  "finetuning/scripts/test_v15_vs_v14.py"
  "finetuning/scripts/test_overconfidence_simulation.py"
)
for f in "${DELETE_TESTS[@]}"; do
  if [ -f "$f" ]; then
    rm "$f"
    echo "  deleted: $f"
  fi
done

# 4. Duplicate patches (keep deployed version)
echo ""
echo "[4/5] Removing duplicate patches..."
DELETE_PATCHES=(
  "finetuning/patches/entity_resolution_worker_v15.py"  # original non-deployed patch
)
for f in "${DELETE_PATCHES[@]}"; do
  if [ -f "$f" ]; then
    rm "$f"
    echo "  deleted: $f"
  fi
done

# 5. Old colab script (superseded by colab_complete_pipeline.py)
echo ""
echo "[5/5] Removing superseded colab scripts..."
DELETE_COLAB=(
  "finetuning/colab_tahap2.py"  # superseded by colab_complete_pipeline.py
)
for f in "${DELETE_COLAB[@]}"; do
  if [ -f "$f" ]; then
    rm "$f"
    echo "  deleted: $f"
  fi
done

# Check for empty dirs
echo ""
echo "Checking for empty directories..."
find finetuning -type d -empty 2>/dev/null | while read d; do
  echo "  empty dir: $d (rmdir)"
  rmdir "$d" 2>/dev/null || true
done

echo ""
echo "============================================================"
echo "CLEANUP COMPLETE"
echo "============================================================"
echo ""
echo "Remaining finetuning/ structure:"
find finetuning -type f | sort
echo ""
echo "Space saved:"
du -sh finetuning/ 2>/dev/null
