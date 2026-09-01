"""
recover_dataset.py — Rich ML Dataset Exporter (Refactored v2)
================================================================
Mengekspor artikel FULLTEXT yang sudah validated + Context + Entity ke JSON.
Berguna untuk Ground Truth Evaluation atau Fine-Tuning model.
"""
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from devtools.common import get_supabase, setup_argparse

def main(limit: int = 1000) -> None:
    sb = get_supabase()
    print(f"[EXPORT] Mengambil {limit} artikel validated + context untuk ML Dataset...")
    
    # Pakai nested select Supabase untuk join raw_texts, entity_contexts, dan political_entities
    res = sb.table("raw_texts") \
            .select("id, title, text, source_url, source, published_at, resolved_domain, entity_contexts(context_text, entity_id, political_entities(canonical_name))") \
            .eq("status", "validated") \
            .eq("content_type", "FULLTEXT") \
            .limit(limit) \
            .execute()
            
    articles = res.data or []
    if not articles:
        print("[EXPORT] Tidak ada data validated untuk diekspor.")
        return
        
    json_file = "dataset_ml_training.json"
    
    # Flatten data untuk ML Training
    ml_data = []
    for art in articles:
        contexts = art.get("entity_contexts") or []
        for ctx in contexts:
            entity = ctx.get("political_entities") or {}
            ml_data.append({
                "raw_text_id": art["id"],
                "title": art["title"],
                "text": art["text"],
                "source_url": art["source_url"],
                "source": art["source"],
                "domain": art.get("resolved_domain"),
                "published_at": art.get("published_at"),
                "entity_id": ctx.get("entity_id"),
                "entity_name": entity.get("canonical_name"),
                "context_text": ctx.get("context_text")
            })
            
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(ml_data, f, ensure_ascii=False, indent=2)
        
    print(f"[EXPORT] ✅ Berhasil mengekspor {len(ml_data)} baris data ke {json_file}")

if __name__ == "__main__":
    parser = setup_argparse("Dataset Exporter for ML Training")
    args = parser.parse_args()
    main(limit=args.limit)