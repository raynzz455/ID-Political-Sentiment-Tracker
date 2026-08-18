#!/bin/bash
# ================================================================
# push_stanza_fix.sh — Push Stanza crash fix to GitHub
# ================================================================
# 2 methods: bundle or patch
# ================================================================

echo "============================================================"
echo "PUSH STANZA CRASH FIX TO GITHUB"
echo "============================================================"
echo ""
echo "Method 1: Git Bundle (recommended)"
echo "  cd /path/to/ID-Political-Sentiment-Tracker"
echo "  git fetch /home/z/my-project/entity-context-fix.bundle"
echo "  git merge FETCH_HEAD"
echo "  git push origin main"
echo ""
echo "Method 2: Git Patch"
echo "  cd /path/to/ID-Political-Sentiment-Tracker"
echo "  git am < /home/z/my-project/stanza-fix.patch"
echo "  git push origin main"
echo ""
echo "Method 3: Manual copy (if above fails)"
echo "  Copy these 2 files from /tmp/idpst_original/:"
echo "    packages/entity/entity_resolution_worker.py"
echo "    packages/context/context_worker.py"
echo "  Then: git add -A && git commit -m 'fix: Stanza crash' && git push"
echo ""
echo "Files ready:"
ls -lh /home/z/my-project/entity-context-fix.bundle /home/z/my-project/stanza-fix.patch 2>&1
