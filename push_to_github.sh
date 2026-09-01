#!/bin/bash
# push_to_github.sh
# =================
# Script untuk push semua commits ke GitHub repo.
#
# Cara pakai:
#   1. Download file ini + all-commits-backup.bundle dari sandbox
#   2. Jalankan di local machine Anda:
#      bash push_to_github.sh
#
# ATAU alternatif dengan Personal Access Token:
#   1. Buat PAT di: https://github.com/settings/tokens (scope: repo)
#   2. Set environment variable: export GH_TOKEN=your_token_here
#   3. Jalankan: bash push_to_github.sh

set -e

REPO_URL="https://github.com/raynzz455/ID-Political-Sentiment-Tracker.git"
BUNDLE_FILE="all-commits-backup.bundle"
TEMP_DIR="/tmp/idpst_push_$$"

echo "=========================================="
echo "PUSH COMMITS TO GITHUB"
echo "=========================================="
echo ""

# Check if bundle exists
if [ ! -f "$BUNDLE_FILE" ]; then
    echo "ERROR: $BUNDLE_FILE not found!"
    echo "Download all-commits-backup.bundle from sandbox first."
    exit 1
fi

# Create temp dir
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

echo "[1/5] Cloning fresh repo from GitHub..."
git clone "$REPO_URL" repo
cd repo

echo ""
echo "[2/5] Fetching from bundle..."
git fetch "$OLDPWD/$BUNDLE_FILE" main:bundle-main

echo ""
echo "[3/5] Merging bundle commits..."
git merge bundle-main --allow-unrelated-histories -m "Merge: all v4 finetuning pipeline + dataset + workers"

echo ""
echo "[4/5] Pushing to GitHub..."
if [ -n "$GH_TOKEN" ]; then
    # Push with token
    git push "https://$GH_TOKEN@github.com/raynzz455/ID-Political-Sentiment-Tracker.git" main
else
    # Push without token (will prompt for credentials)
    git push origin main
fi

echo ""
echo "[5/5] Cleanup..."
cd /
rm -rf "$TEMP_DIR"

echo ""
echo "=========================================="
echo "SUCCESS! All commits pushed to GitHub."
echo "=========================================="
echo ""
echo "Verify at: https://github.com/raynzz455/ID-Political-Sentiment-Tracker"
