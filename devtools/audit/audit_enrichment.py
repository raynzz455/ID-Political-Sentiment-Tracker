"""
audit_enrichment.py — Post-Pipeline Quality Audit for Layer 2.5
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
    print("📊 AUDIT ENRICHMENT WORKER")
    print("="*50)
    
    # 1. Distribusi Content Type & Status
    res = sb.table("raw_texts").select("content_type, status").execute()
    stats = Counter(f"{r['content_type']}_{r['status']}" for r in res.data)
    
    print("\n[VOLUME & STATUS]")
    for k, v in stats.most_common():
        print(f"  {k:30s}: {v}")
        
    # 2. Top 10 Domain yang Berhasil Di-enrich
    res2 = sb.table("raw_texts").select("resolved_domain").not_.is_("resolved_domain", "null").execute()
    domains = Counter(r["resolved_domain"] for r in res2.data)
    
    print("\n[TOP 10 DOMAIN FULLTEXT]")
    for d, c in domains.most_common(10):
        print(f"  {d:30s}: {c}")
        
    print("="*50 + "\n")

if __name__ == "__main__":
    main()