"""
merge_and_upload_lora.py — Merge LoRA adapter → Full model + re-upload ke HuggingFace
====================================================================================
MASALAH YANG DISELESAIKAN:
  Model di HuggingFace (Raynzz455/id-political-sentiment-{relevancy,sentiment})
  di-upload sebagai LoRA adapter (lora/adapter_config.json + adapter_model.safetensors).
  Tapi code production (packages/nlp/sentiment_model.py) pakai:
      AutoModelForSequenceClassification.from_pretrained(model_id)
  yang mengharapkan FULL model (config.json + model.safetensors di root).

  Akibatnya: kalau set NLP_RELEVANCY_MODEL=Raynzz455/... → CRASH (no config.json).

SOLUSI:
  Script ini:
    1. Download base model (apriandito/indobert-*)
    2. Download LoRA adapter dari repo Anda
    3. Merge LoRA weights ke base model (merge_and_unload)
    4. Save full model + tokenizer ke root directory
    5. Re-upload ke HuggingFace repo yang sama (overwrite)

  Setelah merge, repo akan punya:
    - config.json          ← FULL model config (bisa di-load standard)
    - model.safetensors    ← FULL merged weights
    - tokenizer files
    - lora/                ← (opsional, tetap disimpan untuk audit)
    - metrics.json, evaluation.json (tetap ada)

CARA PAKAI (di Google Colab):
  1. Upload file ini ke Colab, atau paste isi nya ke cell
  2. Run:
     !pip install transformers peft huggingface_hub torch
     HF_TOKEN = "hf_xxx"  # token write Anda
     merge_and_upload("relevancy", HF_TOKEN)
     merge_and_upload("sentiment", HF_TOKEN)
  3. Setelah selesai, test load dengan code production:
     from transformers import AutoModelForSequenceClassification
     m = AutoModelForSequenceClassification.from_pretrained(
         "Raynzz455/id-political-sentiment-relevancy")
     print(m.config.id2label)  # harus {0: 'not_relevant', 1: 'relevant'}

PRASYARAT:
  - HF_TOKEN dengan write scope
  - Repo sudah ada (model sudah di-upload sebelumnya)
  - Base model accessible (apriandito/indobert-* — public)
"""
import os
import json
import shutil
from pathlib import Path


def merge_and_upload(task: str, hf_token: str, repo_owner: str = "Raynzz455"):
    """Merge LoRA adapter ke base model, lalu upload full model.

    Args:
        task: "relevancy" atau "sentiment"
        hf_token: HuggingFace token (write scope)
        repo_owner: username HF (default Raynzz455)
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
    from peft import PeftModel
    from huggingface_hub import HfApi

    task = task.lower().strip()
    if task == "relevancy":
        base_model_id = "apriandito/indobert-relevancy-classifier"
        repo_id = f"{repo_owner}/id-political-sentiment-relevancy"
    elif task == "sentiment":
        base_model_id = "apriandito/indobert-sentiment-classifier"
        repo_id = f"{repo_owner}/id-political-sentiment-sentiment"
    else:
        raise ValueError(f"task harus 'relevancy' atau 'sentiment', dapat: {task}")

    print("=" * 65)
    print(f"MERGE TASK: {task.upper()}")
    print(f"  Base model : {base_model_id}")
    print(f"  Repo target: {repo_id}")
    print(f"  Adapter    : {repo_id}/lora/adapter_config.json")
    print("=" * 65)

    # ── 1. Load base model ──────────────────────────────────────
    print("\n[1/5] Loading base model...")
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    print(f"  ✓ Base loaded. Labels: {base_model.config.id2label}")

    # ── 2. Load LoRA adapter on top ─────────────────────────────
    print("\n[2/5] Loading LoRA adapter...")
    # PeftModel.from_pretrained butuh repo_id + subfolder
    peft_model = PeftModel.from_pretrained(
        base_model,
        repo_id,
        subfolder="lora",
        token=hf_token,
    )
    print(f"  ✓ Adapter loaded. PEFT type: {peft_model.peft_config}")

    # ── 3. Merge LoRA → base (merge_and_unload) ─────────────────
    print("\n[3/5] Merging LoRA weights into base model...")
    merged_model = peft_model.merge_and_unload()
    print(f"  ✓ Merged. Type: {type(merged_model).__name__}")
    print(f"  ✓ Labels preserved: {merged_model.config.id2label}")

    # ── 4. Save ke temporary directory ──────────────────────────
    print("\n[4/5] Saving merged full model...")
    out_dir = Path(f"./merged_{task}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    merged_model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))

    # Verify files
    files = sorted(f.name for f in out_dir.iterdir())
    print(f"  ✓ Saved to {out_dir}/")
    for f in files:
        size = (out_dir / f).stat().st_size / (1024 * 1024)
        print(f"     {f} ({size:.1f} MB)")

    # Sanity: pastikan config.json ada (ini yang production butuh)
    config_path = out_dir / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"config.json tidak ada di {out_dir} — merge gagal!")

    # ── 5. Upload ke HuggingFace (overwrite root) ───────────────
    print("\n[5/5] Uploading merged model ke HuggingFace...")
    api = HfApi(token=hf_token)

    # Upload semua file dari out_dir ke ROOT repo
    # (bukan subfolder lora/, tapi root — biar from_pretrained(model_id) jalan)
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"merge: LoRA adapter merged into full model for production use",
    )

    print(f"\n✅ DONE! Repo {repo_id} sekarang berisi full model.")
    print(f"   Production code bisa langsung load:")
    print(f"     AutoModelForSequenceClassification.from_pretrained('{repo_id}')")

    # Cleanup
    shutil.rmtree(out_dir)
    print(f"   (temp dir {out_dir} dibersihkan)")

    # ── Verifikasi ──────────────────────────────────────────────
    print("\n[VERIFIKASI] Test load dengan standard API...")
    test_model = AutoModelForSequenceClassification.from_pretrained(
        repo_id, token=hf_token
    )
    print(f"  ✓ Load berhasil! id2label = {test_model.config.id2label}")
    del test_model

    return True


def main():
    """CLI entry untuk merge kedua model."""
    print("╔" + "═" * 63 + "╗")
    print("║  LoRA Merge & Upload Tool — untuk production deployment       ║")
    print("╚" + "═" * 63 + "╝")
    print()

    # Ambil token dari env atau input
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        hf_token = input("Masukkan HF_TOKEN (write scope): ").strip()
    if not hf_token:
        print("❌ HF_TOKEN required. Set env var atau paste saat prompt.")
        return

    # Login
    from huggingface_hub import login
    login(token=hf_token)

    # Merge kedua model
    for task in ["relevancy", "sentiment"]:
        try:
            merge_and_upload(task, hf_token)
        except Exception as e:
            print(f"\n❌ Gagal merge {task}: {e}")
            import traceback
            traceback.print_exc()
            print(f"   Lanjut ke task berikutnya...")

    print("\n" + "=" * 65)
    print("SELESAI. Setelah merge berhasil, set GitHub Variables:")
    print("  NLP_RELEVANCY_MODEL = Raynzz455/id-political-sentiment-relevancy")
    print("  NLP_SENTIMENT_MODEL = Raynzz455/id-political-sentiment-sentiment")
    print("=" * 65)


if __name__ == "__main__":
    main()
