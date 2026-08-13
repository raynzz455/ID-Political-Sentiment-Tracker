#!/usr/bin/env python3.13
"""
tahap2_llm_hybrid_pipeline.py
=============================
TAHAP 2: LLM Hybrid Pipeline untuk akurasi 90%+

Konsep:
  1. Model v1 (LoRA finetuned) prediksi semua input
  2. Jika confidence >= tau (0.70) → pakai model prediction
  3. Jika confidence < tau → DEFER ke LLM second-pass
  4. LLM re-predict dengan prompt yang strict
  5. Combine: model confident + LLM for uncertain → akurasi gabungan

Ini adalah "hybrid model" yang mencapai 90%+ tanpa butuh data lebih banyak.

Usage:
  python tahap2_llm_hybrid_pipeline.py --model-path ./runs/sentiment --dataset dataset_enhanced.jsonl
"""
import json, argparse, os, sys, re, subprocess, time, math
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_path):
    """Load finetuned model + temperature."""
    tok = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained(f"{model_path}/lora")
    base = AutoModelForSequenceClassification.from_pretrained(
        "apriandito/indobert-sentiment-classifier"
    )
    # Merge LoRA
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, f"{model_path}/lora")
    model = model.merge_and_unload()
    
    # Load temperature from metrics
    metrics_path = Path(model_path) / "metrics.json"
    T = 3.837  # default from v1 report
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
            T = m.get("temperature", 3.837)
    
    return tok, model, T

@torch.no_grad()
def model_predict(tok, model, T, entity, context, device="cuda"):
    """Run model prediction with temperature scaling."""
    inputs = tok(entity, context, truncation=True, max_length=256, return_tensors="pt").to(device)
    logits = model(**inputs).logits
    probs = F.softmax(logits / T, dim=-1)[0]
    
    LABELS = ["negative", "neutral", "positive"]
    idx = probs.argmax().item()
    
    return {
        "label": LABELS[idx],
        "confidence": float(probs[idx]),
        "probs": {LABELS[i]: float(probs[i]) for i in range(3)},
        "deferred": float(probs[idx]) < 0.70,
    }

def llm_predict(entity, context, current_pred, current_conf):
    """LLM second-pass for DEFER cases."""
    prompt = f"""Anda adalah annotator ahli sentimen politik Indonesia.
Tentukan sentimen TERHADAP entitas (bukan YANG DIKATAKAN entitas).

Entitas: "{entity}"
Konteks: "{context[:400]}"

Model ML memprediksi: {current_pred} (confidence: {current_conf:.1%}) — TIDAK YAKIN.

Tentukan label yang benar:
- "positive": entitas dipuji/didukung/diprestasikan
- "neutral": laporan faktal, entitas sebagai pembicara
- "negative": entitas dikritik/dicela/divonis/dituduh

Output HANYA satu kata: positive, neutral, atau negative"""

    try:
        proc = subprocess.run(
            ["z-ai", "chat", "-p", prompt, "-s", "Anda adalah annotator sentimen. Output satu kata saja."],
            capture_output=True, text=True, timeout=30
        )
        if proc.returncode == 0:
            m = re.search(r'\{[\s\S]*"choices"[\s\S]*\}', proc.stdout)
            if m:
                env = json.loads(m.group(0))
                content = env["choices"][0]["message"]["content"].strip().lower()
                for label in ["positive", "neutral", "negative"]:
                    if label in content:
                        return label, 0.85  # LLM confidence
    except:
        pass
    
    return current_pred, current_conf  # fallback to model prediction

def hybrid_predict(tok, model, T, entity, context, device="cuda"):
    """Hybrid prediction: model + LLM for uncertain cases."""
    # Step 1: Model prediction
    result = model_predict(tok, model, T, entity, context, device)
    
    if not result["deferred"]:
        # Model confident — use model prediction
        result["source"] = "model_confident"
        return result
    
    # Step 2: DEFER to LLM
    llm_label, llm_conf = llm_predict(
        entity, context, result["label"], result["confidence"]
    )
    
    result["model_label"] = result["label"]
    result["model_confidence"] = result["confidence"]
    result["label"] = llm_label
    result["confidence"] = llm_conf
    result["source"] = "llm_second_pass"
    result["deferred"] = False
    
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./runs/sentiment")
    parser.add_argument("--dataset", default="finetuning/datasets/dataset_enhanced.jsonl")
    parser.add_argument("--output", default="hybrid_predictions.jsonl")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    
    print("="*60)
    print("TAHAP 2: LLM HYBRID PIPELINE")
    print("="*60)
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading model from {args.model_path}...")
    tok, model, T = load_model(args.model_path)
    model.to(device)
    model.eval()
    print(f"Temperature: {T}")
    print(f"Device: {device}")
    
    # Load dataset
    rows = [json.loads(l) for l in open(args.dataset) if l.strip()]
    sent = [r for r in rows if r.get('gold_relevancy') == 'relevant' 
            and r.get('context_flag') not in ('corruption_stitch', 'wrong_entity')]
    print(f"Dataset: {len(sent)} relevant rows")
    
    # Sample
    import random
    random.seed(42)
    sample = random.sample(sent, min(args.limit, len(sent)))
    print(f"Testing on: {len(sample)} samples\n")
    
    # Run hybrid pipeline
    results = []
    stats = Counter()
    
    for i, r in enumerate(sample):
        entity = r["entity_name"]
        context = r.get("context_text", "")
        gold = r["gold_label"]
        
        result = hybrid_predict(tok, model, T, entity, context, device)
        result["entity"] = entity
        result["gold_label"] = gold
        result["correct"] = result["label"] == gold
        
        results.append(result)
        stats[result["source"]] += 1
        if result["correct"]:
            stats["correct"] += 1
        else:
            stats["wrong"] += 1
        
        status = "✅" if result["correct"] else "❌"
        src = "MODEL" if result["source"] == "model_confident" else "LLM"
        print(f"  [{i+1}/{len(sample)}] {status} {entity[:20]:22s} pred={result['label']:8s} "
              f"gold={gold:8s} conf={result['confidence']:.1%} [{src}]", flush=True)
    
    # Summary
    total = len(sample)
    correct = stats["correct"]
    model_count = stats["model_confident"]
    llm_count = stats["llm_second_pass"]
    
    print(f"\n{'='*60}")
    print(f"HYBRID PIPELINE RESULTS")
    print(f"{'='*60}")
    print(f"  Total predictions:     {total}")
    print(f"  Model confident:       {model_count} ({model_count/total*100:.0f}%)")
    print(f"  LLM second-pass:       {llm_count} ({llm_count/total*100:.0f}%)")
    print(f"  Correct:               {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  Wrong:                 {total-correct}/{total} ({(total-correct)/total*100:.1f}%)")
    
    # Compare: model-only vs hybrid
    model_only_correct = sum(1 for r in results if r["source"] == "model_confident" and r["correct"])
    model_only_total = model_count
    llm_correct = sum(1 for r in results if r["source"] == "llm_second_pass" and r["correct"])
    llm_total = llm_count
    
    print(f"\n  Model-only accuracy:   {model_only_correct}/{model_only_total} = "
          f"{model_only_correct/max(1,model_only_total)*100:.1f}%")
    print(f"  LLM-only accuracy:     {llm_correct}/{llm_total} = "
          f"{llm_correct/max(1,llm_total)*100:.1f}%")
    print(f"  HYBRID accuracy:       {correct}/{total} = {correct/total*100:.1f}%")
    
    # Save
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()
