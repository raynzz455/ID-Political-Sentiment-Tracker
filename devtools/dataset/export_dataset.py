"""
export_dataset.py — Rich ML Dataset Exporter
"""
import os, sys, json
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

try:
    from supabase import create_client
except ImportError as e:
    print(f"[ERROR] {e}"); sys.exit(1)

def main(limit: int = 1000):
    sb = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_ROLE_KEY"))
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
        print("[EXPORT] Tidak ada data.")
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    main(limit=args.limit)