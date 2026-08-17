#!/bin/bash
# ================================================================
# push_updates.sh — Push all MoE updates to GitHub
# ================================================================
# This script pushes all updates from the local clone to GitHub.
#
# Usage:
#   1. Set your GitHub token: export GH_TOKEN=ghp_your_token_here
#   2. Run: bash push_updates.sh
#
# OR if you have SSH key configured:
#   bash push_updates.sh --ssh
# ================================================================

set -e

REPO_URL="https://github.com/raynzz455/ID-Political-Sentiment-Tracker.git"
BUNDLE_FILE="/home/z/my-project/idpst-updates.bundle"

echo "============================================================"
echo "PUSH MoE UPDATES TO GITHUB"
echo "============================================================"

# Method 1: Using bundle (most reliable)
if [ -f "$BUNDLE_FILE" ]; then
    echo ""
    echo "Method: Git bundle"
    echo "Bundle file: $BUNDLE_FILE (4.9MB)"
    echo ""
    echo "Steps:"
    echo "  1. Clone fresh: git clone $REPO_URL my-repo"
    echo "  2. cd my-repo"
    echo "  3. Fetch from bundle: git fetch $BUNDLE_FILE"
    echo "  4. Merge: git merge FETCH_HEAD"
    echo "  5. Push: git push origin main"
    echo ""
    echo "OR (one-liner):"
    echo "  git clone $REPO_URL && cd ID-Political-Sentiment-Tracker && git fetch $BUNDLE_FILE && git merge FETCH_HEAD --allow-unrelated-histories && git push origin main"
fi

# Method 2: Using token
if [ -n "$GH_TOKEN" ]; then
    echo ""
    echo "Method: Token-based push"
    cd /tmp/idpst_final 2>/dev/null || {
        echo "Local repo not found. Using bundle method."
        exit 0
    }
    git remote set-url origin "https://raynzz455:${GH_TOKEN}@github.com/raynzz455/ID-Political-Sentiment-Tracker.git"
    git push origin main
    echo "✅ Pushed successfully!"
fi

# Method 3: Manual instructions
echo ""
echo "============================================================"
echo "MANUAL PUSH (if automated methods fail)"
echo "============================================================"
echo ""
echo "Option A: Apply bundle"
echo "  git clone https://github.com/raynzz455/ID-Political-Sentiment-Tracker.git"
echo "  cd ID-Political-Sentiment-Tracker"
echo "  git fetch /home/z/my-project/idpst-updates.bundle"
echo "  git merge FETCH_HEAD --allow-unrelated-histories"
echo "  git push origin main"
echo ""
echo "Option B: Copy files manually"
echo "  # Copy these files from /tmp/idpst_final/ to your repo:"
echo "  # - packages/entity/entity_resolution_worker.py (v16)"
echo "  # - packages/context/context_worker.py (v20)"
echo "  # - packages/entity/entity_resolution_moe.py (NEW)"
echo "  # - packages/context/context_extraction_moe.py (NEW)"
echo "  # - .github/workflows/preprocessing-pipeline.yml (NEW)"
echo "  # Then: git add -A && git commit -m 'feat: MoE v16/v20' && git push"
