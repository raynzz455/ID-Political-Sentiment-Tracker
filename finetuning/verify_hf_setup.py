"""
verify_hf_setup.py
==================
Verify HuggingFace setup: check token, repos, and permissions.

Usage:
  python verify_hf_setup.py --token hf_xxxxx
  python verify_hf_setup.py  # will prompt for token
"""
from __future__ import annotations
import argparse
import sys
import os


def verify_token(token: str) -> bool:
    """Check if token is valid (can authenticate)."""
    print("=== Step 1: Verify Token ===")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        user_info = api.whoami()
        username = user_info.get("name", "")
        print(f"  ✅ Token valid")
        print(f"  ✅ Authenticated as: {username}")
        if username != "raynzz455":
            print(f"  ⚠️  WARNING: Username is '{username}', but code expects 'raynzz455'")
            print(f"     You'll need to update HF_ORG in:")
            print(f"     - finetuning/configs/hyperparams_v4.py")
            print(f"     - finetuning/v4_all_in_one.py")
            print(f"     - finetuning/backup_to_gdrive.py")
            print(f"     - finetuning/colab_complete_pipeline_v4.py")
            return False
        return True
    except ImportError:
        print("  ❌ huggingface_hub not installed")
        print("     Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"  ❌ Token invalid: {e}")
        return False


def verify_repo(token: str, repo_id: str) -> bool:
    """Check if model repo exists and is accessible."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        info = api.model_info(repo_id)
        print(f"  ✅ Repo exists: {repo_id}")
        print(f"     Created: {info.created_at}")
        print(f"     Downloads: {info.downloads}")
        return True
    except Exception as e:
        error_str = str(e).lower()
        if "404" in error_str or "not found" in error_str:
            print(f"  ❌ Repo NOT FOUND: {repo_id}")
            print(f"     Create at: https://huggingface.co/new")
            print(f"     Owner: raynzz455")
            print(f"     Name: {repo_id.split('/')[-1]}")
        elif "401" in error_str or "403" in error_str:
            print(f"  ❌ No write access to: {repo_id}")
            print(f"     Token doesn't have Write scope, or repo owned by another user")
        else:
            print(f"  ❌ Error checking {repo_id}: {e}")
        return False


def verify_write_permission(token: str, repo_id: str) -> bool:
    """Check if token has write permission to repo."""
    print(f"\n=== Step 3: Verify Write Permission ===")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        # Try to upload a tiny test file
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("HF setup verification test file - safe to delete")
            test_path = f.name
        api.upload_file(
            path_or_fileobj=test_path,
            path_in_repo="_verify_setup.txt",
            repo_id=repo_id,
            token=token,
        )
        os.unlink(test_path)
        # Delete the test file
        api.delete_file(
            path_in_repo="_verify_setup.txt",
            repo_id=repo_id,
            token=token,
        )
        print(f"  ✅ Write permission confirmed for {repo_id}")
        return True
    except Exception as e:
        print(f"  ❌ Write test failed for {repo_id}: {e}")
        print(f"     Check: token has 'Write' scope, repo owned by 'raynzz455'")
        return False


def main():
    ap = argparse.ArgumentParser(description="Verify HuggingFace setup")
    ap.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    args = ap.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("❌ No token provided.")
        print("   Usage: python verify_hf_setup.py --token hf_xxxxx")
        print("   Or:    export HF_TOKEN=hf_xxxxx && python verify_hf_setup.py")
        sys.exit(1)

    print("=" * 60)
    print("  HuggingFace Setup Verification")
    print("=" * 60)

    # Step 1: Verify token
    if not verify_token(token):
        print("\n❌ Setup INCOMPLETE — fix token issues first")
        sys.exit(1)

    # Step 2: Verify repos exist
    print("\n=== Step 2: Verify Model Repos ===")
    repos = [
        "raynzz455/id-political-sentiment-sentiment",
        "raynzz455/id-political-sentiment-relevancy",
    ]
    repos_ok = True
    for repo_id in repos:
        if not verify_repo(token, repo_id):
            repos_ok = False

    if not repos_ok:
        print("\n❌ Setup INCOMPLETE — create missing repos first")
        print("   Go to: https://huggingface.co/new")
        sys.exit(1)

    # Step 3: Verify write permission (test upload to first repo)
    if not verify_write_permission(token, repos[0]):
        print("\n❌ Setup INCOMPLETE — fix write permissions")
        sys.exit(1)

    # Summary
    print("\n" + "=" * 60)
    print("  ✅ SETUP COMPLETE — Ready for finetuning!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. In Colab, set token before running pipeline:")
    print("     import os")
    print("     os.environ['HF_TOKEN'] = '" + token[:10] + "...'")
    print()
    print("  2. Run pipeline:")
    print("     !python colab_complete_pipeline_v4.py")
    print()
    print("  3. Models will auto-upload to:")
    for r in repos:
        print(f"     https://huggingface.co/{r}")
    print()
    print("  4. After upload, switch production to v4 models:")
    print("     export NLP_RELEVANCY_MODEL=raynzz455/id-political-sentiment-relevancy")
    print("     export NLP_SENTIMENT_MODEL=raynzz455/id-political-sentiment-sentiment")


if __name__ == "__main__":
    main()
