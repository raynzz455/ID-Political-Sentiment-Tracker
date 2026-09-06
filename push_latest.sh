#!/bin/bash
# =============================================================================
# push_latest.sh — Push latest commits to GitHub (with token)
# =============================================================================
#
# Skenario: Repo GitHub sudah sinkron, hanya ada beberapa commit baru di
# lokal yang perlu di-push. Tidak ada credential helper di sandbox, jadi
# butuh Personal Access Token (PAT).
#
# CARA PAKAI (2 opsi):
#
# Opsi A — Jalankan langsung di sandbox (jika Anda berikan token ke AI):
#   GH_TOKEN=ghp_xxxxxxxxxxxx bash push_latest.sh
#
# Opsi B — Jalankan di local machine Anda:
#   1. Download repo ini (atau git clone dari sandbox bundle)
#   2. Buat PAT di: https://github.com/settings/tokens (scope: "repo")
#   3. Jalankan:
#        export GH_TOKEN=ghp_your_token_here
#        bash push_latest.sh
#
# KEAMANAN:
#   - Token hanya dipakai di memori (set sebagai remote URL sementara)
#   - Setelah push, token dihapus dari git config (remote URL kembali plain)
#   - Token tidak pernah ditulis ke file permanen
# =============================================================================
set -e

REPO="raynzz455/ID-Political-Sentiment-Tracker"
REPO_URL="https://github.com/${REPO}.git"

# ---- Banner ----
echo "=========================================="
echo "  PUSH LATEST COMMITS TO GITHUB"
echo "=========================================="
echo "  Repo: $REPO"
echo ""

cd "$(dirname "$0")"

# ---- Check token ----
if [ -z "$GH_TOKEN" ]; then
    echo "ERROR: GH_TOKEN environment variable not set."
    echo ""
    echo "Cara dapat token:"
    echo "  1. Buka https://github.com/settings/tokens"
    echo "  2. Generate new token (classic) → scope: 'repo'"
    echo "  3. Set env var:"
    echo "       export GH_TOKEN=ghp_xxxxxxxxxxxx"
    echo "  4. Run lagi: bash push_latest.sh"
    exit 1
fi

# ---- Show what will be pushed ----
echo "[1/4] Commits to push:"
git log origin/main..HEAD --oneline 2>/dev/null || git log --oneline -5
echo ""

echo "[2/4] Files changed:"
git diff origin/main..HEAD --stat 2>/dev/null | tail -20
echo ""

# ---- Set remote URL with token (temporary) ----
echo "[3/4] Pushing..."
git remote set-url origin "https://${GH_TOKEN}@github.com/${REPO}.git"

# ---- Push ----
if git push origin main 2>&1; then
    echo ""
    echo "✅ PUSH SUCCESSFUL!"
else
    # Restore plain URL on failure
    git remote set-url origin "$REPO_URL"
    echo ""
    echo "❌ PUSH FAILED — check error above."
    echo "   Common causes:"
    echo "   - Token expired or revoked"
    echo "   - Token doesn't have 'repo' scope"
    echo "   - Network issue"
    exit 1
fi

# ---- Restore plain URL (security: remove token from git config) ----
git remote set-url origin "$REPO_URL"
echo "🔐 Token removed from git config (security)"

# ---- Verify ----
echo ""
echo "[4/4] Verification:"
git fetch origin --quiet
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)
echo "  Local HEAD : $LOCAL"
echo "  Remote HEAD: $REMOTE"
if [ "$LOCAL" = "$REMOTE" ]; then
    echo ""
    echo "=========================================="
    echo "  ✅ IN SYNC — local and remote match!"
    echo "=========================================="
    echo ""
    echo "  View at: https://github.com/$REPO"
    echo "  Commits pushed: $(git rev-list --count origin/main~3..origin/main 2>/dev/null || echo '3+')"
else
    echo ""
    echo "  ⚠️  Still out of sync — check manually."
fi
