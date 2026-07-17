"""
entity_resolution_worker.py v5 — Zero-Spacy / Regex Matcher (Optimized)
=========================================================================
FIX v5:
  1. TIME FILTER & ANTI-CRASH: Filter 30 hari terakhir agar tidak timeout.
  2. SCHEMA FIX: Nama tabel diperbaiki dari 'discovery_candidates' ke 'entity_candidates'.
  3. CHUNKED UPSERT: Membagi insert mappings & mentions agar tidak kena payload limit.
  4. CLEAN LOGGING: Menggunakan modul logging terstruktur.
"""

import re
import sys
import time
import logging
import argparse
from datetime import datetime, timezone, timedelta  
from pathlib import Path
from dotenv import load_dotenv
from rapidfuzz import process, fuzz
from collections import Counter

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

try:
    from supabase import Client  
except ImportError as e:
    print(f"[ERROR] {e}"); sys.exit(1)

from packages.shared.db_client import get_client
from packages.shared.logger import start_run, finish_run
from packages.shared import constants as pc

# Setup Clean Logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

RESOLVER_VERSION = "v5_regex_opt"
TITLES_RE = re.compile(r'\b(Dr|Prof|H|Hj|Ir|Jenderal|Mayor|Bapak|Ibu|Pak|Bu|Sri|H\.|Ir\.)\b\.?', re.IGNORECASE)

def normalize_name(name: str) -> str:
    name = TITLES_RE.sub('', name).strip()
    return re.sub(r'\s+', ' ', name)

def load_caches(sb: Client):
    logger.info("Loading caches ke memori...")
    
    pe_res = sb.table("political_entities").select("id, canonical_name, aliases").execute()
    entity_db_map = {} 
    alias_map = {}     
    regex_patterns = [] 
    
    for r in (pe_res.data or []):
        canon_lower = r["canonical_name"].lower()
        entity_db_map[canon_lower] = r["id"]
        
        try:
            regex_patterns.append((re.compile(r'\b' + re.escape(r["canonical_name"]) + r'\b', re.IGNORECASE), canon_lower))
        except re.error:
            pass
            
        for alias in (r.get("aliases") or []):
            if len(alias) < 4: continue
            alias_lower = alias.lower()
            alias_map[alias_lower] = r["canonical_name"]
            try:
                regex_patterns.append((re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE), alias_lower))
            except re.error:
                pass
                
    return alias_map, entity_db_map, regex_patterns

def process_articles_batch(articles: list, alias_map: dict, entity_db_map: dict, regex_patterns: list) -> list:
    """Memproses batch artikel menggunakan Regex Matcher (Sangat Cepat)."""
    results = []
    
    for art in articles:
        text = f"{art.get('title', '')}\n{art.get('text', '')}"
        title_lower = (art.get('title') or "").lower()
        metadata = art.get("metadata") or {}
        ingested_month = art.get("ingested_month")
        
        # Pre-Attribution Check (GNews/DDG)
        if metadata.get("configured_entity_id"):
            ent_id = metadata["configured_entity_id"]
            ent_name = next((k for k, v in entity_db_map.items() if v == ent_id), None)
            
            first_offset = 0
            end_offset = 0
            if ent_name:
                match = re.search(re.escape(ent_name), text, re.IGNORECASE)
                if match:
                    first_offset = match.start()
                    end_offset = match.end()
            
            results.append({
                "raw_text_id": art["id"],
                "ingested_month": ingested_month,
                "mappings": [{"entity_id": ent_id, "is_main_entity": True, "confidence": 1.0, "resolver_source": "pre_attributed"}],
                "mentions": [{"entity_id": ent_id, "text": ent_name or "Unknown", "count": 1, "start": first_offset, "end": end_offset}],
                "unknowns": {}
            })
            continue
            
        entity_data = {} 
        unknown_entities = Counter()
        found_matches = [] 
        
        for pattern, key in regex_patterns:
            for match in pattern.finditer(text):
                found_matches.append((match.start(), match.end(), match.group(), key))
                
        found_matches.sort(key=lambda x: x[0])
        
        last_end = -1
        for start, end, matched_text, key in found_matches:
            if start < last_end:
                continue
                
            resolved_name = None
            resolver_source = "regex_exact"
            confidence = 1.0
            
            if key in alias_map:
                resolved_name = alias_map[key]
            elif key in entity_db_map:
                resolved_name = key
            else:
                norm_name = normalize_name(matched_text)
                match_fuzz = process.extractOne(norm_name, list(entity_db_map.keys()), scorer=fuzz.WRatio, score_cutoff=90)
                if match_fuzz:
                    resolved_name = match_fuzz[0]
                    resolver_source = "fuzzy_match"
                    confidence = match_fuzz[1] / 100.0
                else:
                    unknown_entities[norm_name] += 1
                    last_end = end
                    continue
                    
            if resolved_name and resolved_name.lower() in entity_db_map:
                ent_id = entity_db_map[resolved_name.lower()]
                
                if ent_id not in entity_data:
                    entity_data[ent_id] = {
                        "count": 0, "in_title": resolved_name.lower() in title_lower,
                        "src": resolver_source, "conf": confidence, 
                        "first_offset": start, "last_offset": end,
                        "sample_mention": matched_text
                    }
                
                entity_data[ent_id]["count"] += 1
                entity_data[ent_id]["last_offset"] = end
                entity_data[ent_id]["conf"] = max(entity_data[ent_id]["conf"], confidence)
                
            last_end = end
                    
        ranked_entities = sorted(entity_data.items(), key=lambda item: (item[1]["in_title"], item[1]["count"]), reverse=True)
        
        mappings = []
        mentions = []
        
        for idx, (ent_id, data) in enumerate(ranked_entities):
            is_main = (idx == 0)
            mappings.append({
                "entity_id": ent_id, 
                "is_main_entity": is_main, 
                "confidence": data["conf"], 
                "resolver_source": data["src"]
            })
            
            mentions.append({
                "entity_id": ent_id, "text": data["sample_mention"],
                "count": data["count"], "start": data["first_offset"], "end": data["last_offset"]
            })
        
        unknowns_meta = {}
        for unk_name, unk_count in unknown_entities.items():
            unk_match = re.search(re.escape(unk_name), text, re.IGNORECASE)
            unk_sent = text[max(0, unk_match.start()-50):unk_match.end()+50] if unk_match else ""
            unknowns_meta[unk_name] = {"count": unk_count, "context": unk_sent.strip()[:200]}
            
        results.append({
            "raw_text_id": art["id"],
            "ingested_month": ingested_month,
            "mappings": mappings,
            "mentions": mentions,
            "unknowns": unknowns_meta
        })
        
    return results

def chunked_upsert(sb, table_name: str, data: list, chunk_size: int = 50):
    """Helper untuk upsert data dalam chunk agar tidak kena payload limit."""
    if not data: return
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        try:
            sb.table(table_name).upsert(chunk, on_conflict="raw_text_id,entity_id").execute()
        except Exception as e:
            logger.error(f"Upsert Error ({table_name}): {e}")

def main(limit: int = 50, max_total: int = 0):
    sb = get_client()
    run_id = start_run("entity_resolution_worker", RESOLVER_VERSION)
    
    alias_map, entity_db_map, regex_patterns = load_caches(sb)
    logger.info(f"Loaded {len(regex_patterns)} regex patterns ke memori.")
    
    total_processed = 0
    total_success = 0
    batch_num = 1

    logger.info(f"[ENTITY_RESOLVER v5] Limit: {limit}/batch | Max: {'Unlimited' if max_total == 0 else max_total}")

    while True:
        # 1. STOP JIKA SUDAH MENCAPAI MAX TOTAL
        if max_total > 0 and total_processed >= max_total:
            logger.info(f"Max total ({max_total}) tercapai. Berhenti.")
            break
            
        logger.info(f"--- Batch {batch_num} ---")
        
        # 2. HITUNG LIMIT UNTUK BATCH INI
        current_limit = limit
        if max_total > 0:
            current_limit = min(limit, max_total - total_processed)
        
        # Filter 30 hari terakhir & try-except agar tidak crash
        try:
            # PERBAIKAN PENGGUNAAN DATETIME
            time_filter = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            res = sb.table("raw_texts") \
                    .select("id, title, text, metadata, ingested_month") \
                    .eq("status", pc.STATUS_VALIDATED) \
                    .not_.is_("preprocessed_at", "null") \
                    .is_("entity_resolved_at", "null") \
                    .gte("ingested_at", time_filter) \
                    .limit(current_limit) \
                    .execute()
        except Exception as e:
            logger.warning(f"DB Query Timeout/Error: {e}. Menunggu 10 detik sebelum retry...")
            time.sleep(10)
            continue

        articles = res.data or []
        if not articles:
            logger.info("Tidak ada artikel untuk di-resolve.")
            break
            
        logger.info(f"Memproses {len(articles)} artikel dengan Regex Matcher...")
        batch_results = process_articles_batch(articles, alias_map, entity_db_map, regex_patterns)
        
        all_mappings = []
        all_mentions = []
        all_unknowns = {}
        resolved_updates = []
        now_iso = datetime.now(timezone.utc).isoformat()
        success_count = 0
        
        for result in batch_results:
            resolved_updates.append({
                "id": result["raw_text_id"],
                "entity_resolved_at": now_iso,  
                "resolver_version": RESOLVER_VERSION
            })
            
            if result["mappings"]:
                all_mappings.extend([{**m, "raw_text_id": result["raw_text_id"], "ingested_month": result["ingested_month"]} for m in result["mappings"]])
                all_mentions.extend([{**m, "raw_text_id": result["raw_text_id"], "ingested_month": result["ingested_month"]} for m in result["mentions"]])
                success_count += 1
                
            all_unknowns.update(result["unknowns"])
        
        try:
            if resolved_updates:
                for i in range(0, len(resolved_updates), 25):
                    sb.rpc("bulk_update_raw_texts", {"p_updates": resolved_updates[i:i+25]}).execute()
                
            chunked_upsert(sb, "article_entity_map", all_mappings, chunk_size=50)
                
            if all_mentions:
                db_mentions = [{
                    "raw_text_id": m["raw_text_id"], 
                    "ingested_month": m["ingested_month"],
                    "entity_id": m["entity_id"], 
                    "mention_text": m["text"],
                    "start_offset": m["start"], 
                    "end_offset": m["end"]
                } for m in all_mentions]
                chunked_upsert(sb, "entity_mentions", db_mentions, chunk_size=50)
                
            if all_unknowns:
                unknown_payload = [{
                    "detected_name": name, 
                    "status": "pending",
                    "mention_count": data["count"],
                    "sample_titles": [data["context"]] 
                } for name, data in all_unknowns.items()]
                chunked_upsert(sb, "entity_candidates", unknown_payload, chunk_size=50)
                
        except Exception as e:
            logger.error(f"DB Error: {e}")
            
        logger.info(f"{success_count} artikel berhasil di-resolve. Mappings: {len(all_mappings)} | Mentions: {len(all_mentions)} | Unknowns: {len(all_unknowns)}")
        
        total_processed += len(articles)
        total_success += success_count
        batch_num += 1
        
    finish_run(run_id, total_processed, total_success, 0)
    logger.info("Eksekusi Entity Resolver Selesai.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-total", type=int, default=0)
    args = parser.parse_args()
    main(limit=args.limit, max_total=args.max_total)