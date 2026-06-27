# Layer 4 — NLP Worker (IndoBERT inference)

> 🚧 **Belum diimplementasi.** Folder ini adalah placeholder.

## Tujuan

Dequeue batch dari pgmq queue, jalankan IndoBERT sentiment inference (3-class: negative/neutral/positive), lalu:
1. Resolve `entity_id` berdasarkan matching `aliases` di tabel `political_entities`
2. Insert hasil via RPC `insert_sentiment_score(...)`

## Tech Stack

- **Runtime:** Python 3.10+ (Hugging Face Spaces, free CPU tier)
- **Framework:** FastAPI + Uvicorn
- **Inference:** `onnxruntime` (CPU) + model IndoBERT fine-tuned, **INT8 quantized** (hemat memori)
- **Queue:** Supabase pgmq (via `pgmq` SQL API atau HTTP)

## Data flow

```
pgmq queue (nlp_processing_queue)
  → dequeue batch (16-32 items)
  → tokenize + ONNX forward pass → [neg, neu, pos] scores
  → argmax → label + confidence
  → match text vs political_entities.aliases (case-insensitive, array overlap)
  → RPC: insert_sentiment_score(raw_text_id, entity_id, label, ...)
  → update raw_texts.status = 'processed'
```

## Rekomendasi model

- Base: `indobenchmark/indobert-base-p1`
- Fine-tuned untuk sentiment: cari di HuggingFace dengan keyword `indobert-sentiment`
- Wajib **export ke ONNX + quantize INT8** agar muat di free CPU tier (~100MB model).

Contoh konversi:
```python
# convert.py (jalankan LOKAL sekali, commit hanya file .onnx)
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model_id = "your-finetuned-indobert"
model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
model.save_pretrained("./models/indobert-sentiment")

# Quantize INT8 untuk memperkecil:
# quantize_dynamic("./models/indobert-sentiment/model.onnx",
#                  "./models/indobert-sentiment-int8.onnx", weight_type=quantize_dynamic.QUInt8)
```

## Skeleton code (untuk mulai)

```python
# app.py
import os, asyncio
from fastapi import FastAPI
from supabase import create_client
import onnxruntime as ort
import numpy as np

app = FastAPI()
session = ort.InferenceSession(os.environ["ONNX_MODEL_PATH"])
supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"]
)

def predict(text: str):
    # TODO: tokenize dengan tokenizer yang sama saat training
    # inputs = tokenizer(text, ...)
    out = session.run(None, {"input_ids": ..., "attention_mask": ...})
    scores = softmax(out[0])           # [neg, neu, pos]
    label = ["negative","neutral","positive"][np.argmax(scores)]
    return label, float(max(scores)), scores

@app.post("/process-batch")
async def process_batch():
    # 1. Dequeue dari pgmq
    msgs = supabase.rpc("pgmq.read", {"queue_name": "nlp_processing_queue",
                                       "vt": 30, "qty": 16}).execute().data
    for m in msgs:
        text = ...  # ambil raw_text berdasarkan msg.message
        label, conf, scores = predict(text)

        # 2. Match entity via aliases
        entities = match_entities(text)  # SELECT dari political_entities

        # 3. Insert score
        for entity_id in entities:
            supabase.rpc("insert_sentiment_score", {
                "p_raw_text_id": raw_text_id, "p_entity_id": entity_id,
                "p_label": label, "p_neg": float(scores[0]),
                "p_neu": float(scores[1]), "p_pos": float(scores[2]),
                "p_confidence": conf,
            })

        # 4. Update status & ack message
        supabase.table("raw_texts").update({"status":"processed"}).eq("id", raw_text_id).execute()
    return {"processed": len(msgs)}
```

## Entity matching (penting)

Matching dilakukan di worker, BUKAN di DB, karena butuh fuzzy/case-insensitive. Strategi sederhana:

```python
def match_entities(text: str) -> list[str]:
    # Cache semua political_entities + aliases saat startup
    text_lower = text.lower()
    return [e["id"] for e in ENTITY_CACHE
            if any(a.lower() in text_lower for a in e["aliases"])]
```

## Aturan

- ⚠️ Pakai **service_role key** — perlu INSERT ke `sentiment_scores` & UPDATE `raw_texts.status`.
- ⚠️ Jangan kirim `scored_month` — diisi trigger otomatis.
- ⚠️ File `.onnx` TIDAK di-commit (sudah di-`.gitignore`). Upload manual ke HF Spaces.
- ⚠️ Pastikan skor memenuhi CHECK constraint: tiap score `BETWEEN 0 AND 1`.

## Referensi

- RPC insert: skema blok #10 (`insert_sentiment_score`)
- Skema entity matching: kolom `political_entities.aliases TEXT[]`
- Arsitektur: [`../docs/architecture.md`](../docs/architecture.md) — Layer 4 & 5
