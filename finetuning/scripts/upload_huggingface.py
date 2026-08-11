"""
upload_huggingface.py
=====================
Upload finetuned LoRA adapters + merged model to HuggingFace Hub.

Usage:
    python upload_huggingface.py --task sentiment --hf-token YOUR_HF_TOKEN
    python upload_huggingface.py --task relevancy --hf-token YOUR_HF_TOKEN

Or set HF_TOKEN env var:
    export HF_TOKEN=your_token_here
    python upload_huggingface.py --task sentiment
"""
import argparse, json, os, sys
from pathlib import Path

def upload_model(task, run_dir, hf_token):
    from huggingface_hub import HfApi, create_repo
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel
    import torch

    sys.path.insert(0, str(Path(__file__).parent.parent / "configs"))
    import hyperparams_optimized as H

    if task == "sentiment":
        base_model = H.SENTIMENT_BASE
        hf_repo = H.HF_SENTIMENT_MODEL
        labels = H.SENTIMENT_LABELS
    else:
        base_model = H.RELEVANCY_BASE
        hf_repo = H.HF_RELEVANCY_MODEL
        labels = H.RELEVANCY_LABELS

    run = Path(run_dir)
    print(f"\nUploading {task} model to: {hf_repo}")

    metrics = {}
    metrics_path = run / "metrics.json"
    if metrics_path.exists():
        metrics = json.load(open(metrics_path))

    print("Loading base + LoRA...")
    tokenizer = AutoTokenizer.from_pretrained(run / "tokenizer")
    base = AutoModelForSequenceClassification.from_pretrained(base_model)
    model = PeftModel.from_pretrained(base, run / "lora")
    model = model.merge_and_unload()

    T = metrics.get("temperature", 1.0)
    model.config.temperature = T

    local_save = run / "merged_model"
    local_save.mkdir(exist_ok=True)
    model.save_pretrained(local_save)
    tokenizer.save_pretrained(local_save)

    test_m = metrics.get("test_metrics", {})
    card = f"""---
language: id
tags:
  - indonesian
  - sentiment-analysis
  - political-sentiment
  - indobert
  - lora
  - finetuned
license: mit
base_model: {base_model}
---

# {hf_repo}

IndoBERT finetuned with LoRA for Indonesian political sentiment analysis.

## Performance
- Accuracy: {test_m.get('accuracy', 'N/A')}
- macro-F1: {test_m.get('macro_f1', 'N/A')}
- Temperature: {T}

## Usage
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("{hf_repo}")
model = AutoModelForSequenceClassification.from_pretrained("{hf_repo}")

entity = "Prabowo Subianto"
context = "Presiden Prabowo menegaskan program ekonomi."
inputs = tok(entity, context, truncation=True, max_length=256, return_tensors="pt")
probs = torch.softmax(model(**inputs).logits / {T}, dim=-1)
label = {labels}[probs.argmax()]
```

## Labels: {labels}
"""
    (local_save / "README.md").write_text(card)

    api = HfApi(token=hf_token)
    create_repo(hf_repo, exist_ok=True, token=hf_token)
    api.upload_folder(folder_path=str(local_save), repo_id=hf_repo, repo_type="model", token=hf_token)
    print(f"Uploaded: https://huggingface.co/{hf_repo}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["sentiment","relevancy"], required=True)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN",""))
    a = p.parse_args()
    if not a.hf_token:
        print("ERROR: set HF_TOKEN env or --hf-token"); sys.exit(1)
    upload_model(a.task, a.run_dir or f"./runs/{a.task}", a.hf_token)
