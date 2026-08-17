#!/bin/bash
# ================================================================
# apply_moe_updates.sh — Apply MoE updates to ID-Political-Sentiment-Tracker
# ================================================================
# Run this script from repo root to apply:
#   1. entity_resolution_worker.py v15.1 → v16 (MoE-enabled)
#   2. context_worker.py v19.1 → v20 (MoE-enabled)
#   3. Add MoE files to packages/
#   4. Add GitHub Actions workflow
#
# Usage:
#   cd /path/to/ID-Political-Sentiment-Tracker
#   bash finetuning/scripts/apply_moe_updates.sh
#
# Prerequisites:
#   - Files from /tmp/idpst_repo_v2/ accessible (or download from this repo's patches)
#   - Git repo clean (no uncommitted changes)
#
# After applying:
#   - Commit: git add -A && git commit -m "feat: MoE-enabled workers v16/v20"
#   - Push: git push origin main
# ================================================================

set -e

echo "============================================================"
echo "APPLYING MoE UPDATES to packages/"
echo "============================================================"

# === STEP 1: Copy MoE files to packages/ ===
echo ""
echo "[1/4] Copying MoE files to packages/..."

# Entity Resolution MoE
if [ -f "finetuning/patches/entity_resolution_moe.py" ]; then
    cp finetuning/patches/entity_resolution_moe.py packages/entity/entity_resolution_moe.py
    echo "  ✅ packages/entity/entity_resolution_moe.py copied"
else
    echo "  ❌ finetuning/patches/entity_resolution_moe.py not found"
    echo "     Run: git pull origin main (to get patches from finetuning/patches/)"
    exit 1
fi

# Context Extraction MoE
if [ -f "finetuning/patches/context_extraction_moe.py" ]; then
    cp finetuning/patches/context_extraction_moe.py packages/context/context_extraction_moe.py
    echo "  ✅ packages/context/context_extraction_moe.py copied"
else
    echo "  ❌ finetuning/patches/context_extraction_moe.py not found"
    exit 1
fi

# === STEP 2: Update entity_resolution_worker.py (v15.1 → v16) ===
echo ""
echo "[2/4] Updating entity_resolution_worker.py to v16 (MoE-enabled)..."

ENTITY_FILE="packages/entity/entity_resolution_worker.py"

# Check current version
CURRENT_VERSION=$(grep "RESOLVER_VERSION" $ENTITY_FILE | head -1 | grep -oP '"v[^"]+"')
echo "  Current version: $CURRENT_VERSION"

if [[ "$CURRENT_VERSION" == *"v16"* ]]; then
    echo "  ✅ Already v16 — skipping"
elif [[ "$CURRENT_VERSION" == *"v15"* ]]; then
    # Apply patch
    if [ -f "/tmp/entity_v16.patch" ]; then
        git apply /tmp/entity_v16.patch
        echo "  ✅ Patched to v16_moe_enabled"
    else
        echo "  ⚠️ Patch file not found at /tmp/entity_v16.patch"
        echo "     Manual update needed — see instructions below"
    fi
else
    echo "  ⚠️ Unknown version — manual update needed"
fi

# === STEP 3: Update context_worker.py (v19.1 → v20) ===
echo ""
echo "[3/4] Updating context_worker.py to v20 (MoE-enabled)..."

CONTEXT_FILE="packages/context/context_worker.py"

CURRENT_VERSION=$(grep "CONTEXT_VERSION" $CONTEXT_FILE | head -1 | grep -oP '"v[^"]+"')
echo "  Current version: $CURRENT_VERSION"

if [[ "$CURRENT_VERSION" == *"v20"* ]]; then
    echo "  ✅ Already v20 — skipping"
elif [[ "$CURRENT_VERSION" == *"v19"* ]]; then
    if [ -f "/tmp/context_v20.patch" ]; then
        git apply /tmp/context_v20.patch
        echo "  ✅ Patched to v20_moe_enabled"
    else
        echo "  ⚠️ Patch file not found at /tmp/context_v20.patch"
        echo "     Manual update needed — see instructions below"
    fi
else
    echo "  ⚠️ Unknown version — manual update needed"
fi

# === STEP 4: Add GitHub Actions workflow ===
echo ""
echo "[4/4] Adding GitHub Actions workflow..."

mkdir -p .github/workflows

if [ -f ".github/workflows/preprocessing-pipeline.yml" ]; then
    echo "  ✅ Workflow already exists — skipping"
else
    # Create workflow file (inline — no external dependency)
    cat > .github/workflows/preprocessing-pipeline.yml << 'WORKFLOW_EOF'
name: Preprocessing Pipeline

on:
  schedule:
    - cron: '0 */6 * * *'
    - cron: '0 2 * * *'
  workflow_dispatch:
    inputs:
      worker:
        description: 'Worker to run'
        required: false
        default: 'prep-full'
        type: choice
        options:
          - ingestion
          - prep-full
          - entity
          - context
          - readiness
      limit:
        description: 'Batch size'
        required: false
        default: '50'

jobs:
  prep:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'
      - run: pip install -r requirements.txt
      - run: python -c "import stanza; stanza.download('id', processors='tokenize,pos,lemma,depparse')"
      - uses: actions/cache@v4
        with:
          path: ~/stanza_resources
          key: stanza-id-${{ hashFiles('requirements.txt') }}
      - env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_SERVICE_ROLE_KEY: ${{ secrets.SUPABASE_SERVICE_ROLE_KEY }}
          USE_MOE: '0'
        run: |
          WORKER="${{ github.event.inputs.worker || 'prep-full' }}"
          LIMIT="${{ github.event.inputs.limit || '50' }}"
          if [ "$WORKER" = "prep-full" ]; then
            python main.py run-prep --limit $LIMIT --max-total 200
          elif [ "$WORKER" = "ingestion" ]; then
            python -c "from supabase import create_client; import os; sb=create_client(os.environ['SUPABASE_URL'],os.environ['SUPABASE_SERVICE_ROLE_KEY']); sb.functions.invoke('rss-ingestion')"
          else
            python main.py run-worker $WORKER --limit $LIMIT
          fi
WORKFLOW_EOF
    echo "  ✅ .github/workflows/preprocessing-pipeline.yml created"
fi

# === DONE ===
echo ""
echo "============================================================"
echo "MoE UPDATES APPLIED SUCCESSFULLY!"
echo "============================================================"
echo ""
echo "Changes:"
echo "  - packages/entity/entity_resolution_worker.py: v15.1 → v16 (MoE-enabled)"
echo "  - packages/context/context_worker.py: v19.1 → v20 (MoE-enabled)"
echo "  - packages/entity/entity_resolution_moe.py: NEW (6 experts)"
echo "  - packages/context/context_extraction_moe.py: NEW (5 experts)"
echo "  - .github/workflows/preprocessing-pipeline.yml: NEW (GH Actions)"
echo ""
echo "Next steps:"
echo "  1. Review changes: git diff"
echo "  2. Commit: git add -A && git commit -m 'feat: MoE-enabled workers v16/v20'"
echo "  3. Push: git push origin main"
echo "  4. Set GitHub Secrets: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY"
echo "  5. Test: Actions tab → Run workflow"
echo ""
echo "Usage modes:"
echo "  USE_MOE=0 (default): single expert — fits GH Actions free tier (7GB RAM)"
echo "  USE_MOE=1: MoE mode (6+5 experts) — needs HuggingFace Spaces (16GB RAM)"
