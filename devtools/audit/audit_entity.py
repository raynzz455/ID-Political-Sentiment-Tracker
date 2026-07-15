"""
audit_entity.py — Post-Pipeline Quality Audit for Layer 3
"""
import os, sys
from pathlib import Path
from dotenv import load_dotenv
from collections import Counter

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

try:
    from supabase import create_client
except ImportError:
    print("[ERROR] pip install supabase"); sys.exit(1)

def main():
    sb = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_ROLE_KEY"))
    print("\n" + "="*50)
    print("🧠 AUDIT ENTITY RESOLUTION")
    print("="*50)
    
    # 1. Top 10 Entitas yang paling banyak disebut
    res = sb.table("entity_mentions").select("entity_id, political_entities(canonical_name)").limit(1000).execute()
    entities = Counter()
    for r in res.data:
        pe = r.get("political_entities") or {}
        name = pe.get("canonical_name", "Unknown")
        entities[name] += 1
        
    print("\n[TOP 10 ENTITAS PERTAMA (Dari 1000 baris terakhir)]")
    for e, c in entities.most_common(10):
        print(f"  {e:30s}: {c} mentions")
        
    print("="*50 + "\n")

if __name__ == "__main__":
    main()