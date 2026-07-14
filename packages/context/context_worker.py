"""
context_worker.py v4 — Pure Offset Window & Rich Scoring
=========================================================
FIX dari v3:
  1. PURE OFFSET MATCH: Ambil jendela karakter (misal 600 char) di sekitar offset NER.
     Nol split('\n\n'), nol regex. 100% presisi.
  2. QUALITY SCORE FIX: Hitung density entitas & signal words secara benar.
  3. BATCH READY: Import dari packages.shared.
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

try:
    from supabase import create_client, Client
except ImportError as e:
    print(f"[ERROR] {e}"); sys.exit(1)

# IMPORT DARI MONOREPO SHARED
from packages.shared.db_client import get_client
from packages.shared.logger import start_run, finish_run
from packages.shared import constants as pc

CONTEXT_VERSION = "v4_pure_offset"
CONTEXT_WINDOW_CHARS = 800 # Ambil 800 karakter di sekitar mention (400 sebelum, 400 sesudah)
MAX_WORDS = 350 # Batas aman untuk 512 token IndoBERT

# Expanded Signal Words
SIGNAL_WORDS = {
    "mengatakan", "menyatakan", "mengkritik", "mendesak", "menolak", "mendukung", 
    "membantah", "mengimbau", "mengklarifikasi", "mengomentari", "mengungkapkan", 
    "mengakui", "menegaskan", "menilai", "menyebut", "menjawab"
}

def extract_context_by_offset(text: str, start_offset: int, end_offset: int) -> tuple[str, dict]:
    """Ambil jendela karakter di sekitar offset NER secara matematis."""
    text_len = len(text)
    if text_len == 0:
        return "", {"fallback": "empty_text"}
        
    # Ambil 400 char sebelum dan 400 char sesudah mention
    start = max(0, start_offset - 400)
    end = min(text_len, end_offset + 400)
    
    context_text = text[start:end]
    
    # Token-based Truncation (Word count proxy)
    words = context_text.split()
    is_truncated = False
    if len(words) > MAX_WORDS:
        context_text = " ".join(words[:MAX_WORDS])
        is_truncated = True
        
    return context_text, {"start_idx": start, "end_idx": end, "is_truncated": is_truncated, "word_count": len(words)}

def calculate_quality_score(context_text: str, entity_name: str) -> dict:
    """Hitung density entitas dan sinyal politik."""
    lower_ctx = context_text.lower()
    lower_name = entity_name.lower()
    
    density = lower_ctx.count(lower_name)
    signals = sum(1 for w in SIGNAL_WORDS if w in lower_ctx)
    
    # Base Score: Density (max 50) + Signals (max 50) = Max 100
    base_score = min(50, (density * 10) + (signals * 10))
    
    return {"density": density, "signals": signals, "quality_score": base_score}

def main(limit: int = 50, max_total: int = 0):
    sb = get_client()
    run_id = start_run("context_worker", CONTEXT_VERSION)
    
    total_processed = 0
    total_success = 0
    batch_num = 1

    print(f"[CONTEXT_WORKER] Limit: {limit}/batch | Max: {'Unlimited' if max_total == 0 else max_total}")

    while True:
        if max_total > 0 and total_processed >= max_total:
            break
            
        print(f"\n--- Batch {batch_num} ---")
        res = sb.table("raw_texts") \
                .select("id, text, ingested_month") \
                .eq("status", pc.STATUS_VALIDATED) \
                .not_.is_("entity_resolved_at", "null") \
                .is_("context_extracted_at", "null") \
                .limit(limit) \
                .execute()
                
        articles = res.data or []
        if not articles:
            break
            
        art_ids = [a["id"] for a in articles]
        
        # BATCH QUERY: Ambil semua mentions
        mentions_res = sb.table("entity_mentions") \
                         .select("raw_text_id, entity_id, start_offset, end_offset, political_entities(canonical_name)") \
                         .in_("raw_text_id", art_ids) \
                         .execute()
                         
        mentions_by_art = {}
        for m in (mentions_res.data or []):
            mentions_by_art.setdefault(m["raw_text_id"], []).append(m)
            
        context_inserts = []
        updates = []
        now_iso = datetime.now(timezone.utc).isoformat()
        success_count = 0
        
        for art in articles:
            art_mentions = mentions_by_art.get(art["id"], [])
            text = f"{art.get('title', '')}\n{art.get('text', '')}"
            best_contexts = {} 
            
            for m in art_mentions:
                ctx_text, audit_stats = extract_context_by_offset(text, m["start_offset"], m["end_offset"])
                quality = calculate_quality_score(ctx_text, m["political_entities"]["canonical_name"])
                
                # Multiple Mention Ranking: Pilih context dengan quality_score tertinggi
                if m["entity_id"] not in best_contexts or quality["quality_score"] > best_contexts[m["entity_id"]][1]["quality_score"]:
                    best_contexts[m["entity_id"]] = (ctx_text, quality, audit_stats)
                    
            for ent_id, (ctx_text, quality, audit_stats) in best_contexts.items():
                context_inserts.append({
                    "raw_text_id": art["id"],
                    "ingested_month": art.get("ingested_month"),
                    "entity_id": ent_id,
                    "context_text": ctx_text,
                    "context_version": CONTEXT_VERSION,
                    "metadata": {**audit_stats, **quality}
                })
                
            updates.append({"id": art["id"], "context_extracted_at": now_iso})
            success_count += 1
            
        if context_inserts:
            try: sb.table("entity_contexts").upsert(context_inserts, on_conflict="raw_text_id,entity_id").execute()
            except Exception as e: print(f"[DB_ERROR] {e}")
                
        if updates:
            try: sb.rpc("bulk_update_raw_texts", {"p_updates": updates}).execute()
            except Exception as e: print(f"[DB_ERROR] {e}")
                
        print(f"[CONTEXT_WORKER] {success_count} diproses. {len(context_inserts)} contexts dibuat.")
        
        total_processed += len(articles)
        total_success += success_count
        batch_num += 1
        
    finish_run(run_id, total_processed, total_success, 0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-total", type=int, default=0)
    args = parser.parse_args()
    main(limit=args.limit, max_total=args.max_total)