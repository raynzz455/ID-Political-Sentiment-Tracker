"""
entity_resolution_worker.py v12 — Co-occurrence Fix & Word-Boundary Matching
=========================================================================
FIX v12 (lihat diagnosis di percakapan sebelumnya):
  1. CO-OCCURRENCE REGRESSION FIX: v11 (jalur configured_entity_id) langsung
     `continue` begitu entity dari scraping_config ketemu -- TIDAK menjalankan
     regex matching umum, jadi entitas co-mention lain (mis. "Prabowo bertemu
     Jokowi" dari config khusus Prabowo) TIDAK PERNAH ter-map. Ini regresi
     dari fix yang sama yang pernah dibuat di versi sebelum Stanza rewrite ini.
     Sekarang: configured_entity_id tetap DIJAMIN masuk sbg is_main_entity,
     TAPI regex matching umum tetap jalan penuh utk menangkap entitas lain.
  2. is_false_positive FIX: v11 pakai substring check (`matched_lower in
     person_lower`) -- "budi" adalah substring "budiman", tapi itu 2 nama
     depan yang beda. Sekarang exact match kata-per-kata.
  3. CONFIGURABLE TIME WINDOW: --days-back (default tetap 30).
  4. entity_mentions sekarang UPSERT dgn on_conflict eksplisit (butuh UNIQUE
     (raw_text_id, entity_id, start_offset) -- lihat migration), bukan
     delete-lalu-insert polos. article_entity_map: DELETE manual yg
     mendahului upsert DIHAPUS (redundan -- upsert dgn on_conflict sudah
     menangani replace, delete tambahan cuma menambah window race condition).
  5. entity_resolved_at HANYA ditandai utk artikel yang db write-nya BENERAN
     sukses (v11 menandai semua artikel di batch tanpa syarat, walau upsert
     gagal utk sebagian -- exception cuma di-log).

FIX v11 (riwayat, tetap berlaku): STANZA NLP INTEGRATION, CANONICAL
VERIFICATION, SALIENCE GATE, MULTI-MENTION STORAGE -- lihat kode.
"""

import re
import time
import random
import logging
import argparse
import stanza
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

from packages.shared.db_client import get_client
from packages.shared.logger import start_run, finish_run
from packages.shared import constants as pc

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("stanza").setLevel(logging.WARNING) # Reduksi log stanza

RESOLVER_VERSION = "v12_cooccurrence_fix"
DEFAULT_DAYS_BACK = 30

# Load Stanza Pipeline SEKALI di awal
logger.info("Memuat Stanza POS Tagger (Bahasa Indonesia)...")
try:
    NLP = stanza.Pipeline('id', processors='tokenize,pos', verbose=False, use_gpu=True)
except Exception as e:
    logger.warning(f"Gagal load GPU Stanza, fallback ke CPU: {e}")
    NLP = stanza.Pipeline('id', processors='tokenize,pos', verbose=False, use_gpu=False)

def normalize_name(name: str) -> str:
    return re.sub(r'\s+', ' ', name).strip()

def load_caches(sb):
    logger.info("Loading caches ke memori...")
    
    pe_res = sb.table("political_entities").select("id, canonical_name, aliases").execute()
    entity_db_map = {} 
    alias_map = {}     
    id_to_name = {}    
    regex_patterns = [] 
    
    for r in (pe_res.data or []):
        canon_lower = r["canonical_name"].lower()
        entity_db_map[canon_lower] = r["id"]
        id_to_name[r["id"]] = r["canonical_name"]
        
        try:
            regex_patterns.append((re.compile(r'\b' + re.escape(r["canonical_name"]) + r'\b', re.IGNORECASE), canon_lower))
        except re.error:
            pass
            
        for alias in (r.get("aliases") or []):
            if len(alias) < 2: continue
            alias_lower = alias.lower()
            alias_map[alias_lower] = r["canonical_name"]
            try:
                regex_patterns.append((re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE), alias_lower))
            except re.error:
                pass
                
    return alias_map, entity_db_map, id_to_name, regex_patterns

def extract_full_persons(text: str) -> list:
    """Cheat Code: Gruping Proper Noun (PROPN) yang berurutan menjadi satu nama utuh."""
    doc = NLP(text)
    persons = []
    current_person = []
    
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos == 'PROPN':
                current_person.append(word.text)
            else:
                if current_person:
                    persons.append(" ".join(current_person))
                    current_person = []
        if current_person:
            persons.append(" ".join(current_person))
            
    return persons

def is_false_positive(matched_text: str, canonical_name: str, full_persons: list) -> bool:
    """Cek apakah alias yang ketemu ternyata bagian dari nama orang lain.
    Exact match KATA-PER-KATA (bukan substring) -- v11 pakai `matched_lower
    in person_lower` yang bisa salah cocok token pendek yang kebetulan jadi
    prefix nama lain (mis. "budi" adalah substring "budiman", tapi Budi
    Gunawan != Budiman Sudjatmiko)."""
    matched_lower = matched_text.lower()
    canonical_lower = canonical_name.lower()

    for person in full_persons:
        person_lower = person.lower()
        person_words = person_lower.split()
        if matched_lower in person_words and len(person_lower) > len(matched_lower):
            # Alias ada di dalam nama utuh (sbg KATA, bukan cuma substring).
            # Apakah nama utuh ini milik tokoh kita?
            canonical_parts = [p for p in canonical_lower.split() if p != matched_lower]

            # Jika ada bagian canonical yang cocok (sbg kata), ini bukan false positive.
            # Contoh: matched="bobby", canonical="bobby nasution", person="bobby afif nasution" -> "nasution" cocok! -> Bukan FP.
            if any(part in person_words for part in canonical_parts):
                return False
            else:
                # Contoh: matched="bobby", canonical="bobby nasution", person="bobby danuardi" -> FP!
                return True
    return False

def process_articles_batch(articles: list, alias_map: dict, entity_db_map: dict, id_to_name: dict, regex_patterns: list) -> list:
    results = []
    
    for art in articles:
        text = f"{art.get('title', '')}\n{art.get('text', '')}"
        title_lower = (art.get('title') or "").lower()
        metadata = art.get("metadata") or {}
        ingested_month = art.get("ingested_month")
        
        # 1. Ekstrak semua nama orang utuh via Stanza (Safety Net)
        full_persons = extract_full_persons(text)

        # === CO-OCCURRENCE FIX (v12) ===
        # v11 di titik ini langsung `continue` begitu configured_entity_id
        # ketemu -- SELURUH regex matching umum di bawah TIDAK PERNAH jalan,
        # jadi entitas co-mention lain (mis. "Prabowo bertemu Jokowi" dari
        # scraping_config Prabowo) tidak pernah ter-map. Sekarang: catat
        # entity_id yang dikonfigurasi (dijamin masuk sbg main entity), tapi
        # TETAP lanjut ke regex matching umum di bawah, bukan skip.
        configured_entity_id = metadata.get("configured_entity_id")
        configured_entity_name = id_to_name.get(configured_entity_id, "") if configured_entity_id else ""

        entity_data = {} 
        found_matches = [] 
        
        # 2. Jalankan Regex Exact Match
        for pattern, key in regex_patterns:
            for match in pattern.finditer(text):
                found_matches.append((match.start(), match.end(), match.group(), key))
                
        found_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        
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
                
            if resolved_name and resolved_name.lower() in entity_db_map:
                ent_id = entity_db_map[resolved_name.lower()]
                
                # === 3. HYBRID VALIDATION (Cek Bobby Danuardi vs Bobby Nasution) ===
                if is_false_positive(matched_text, resolved_name, full_persons):
                    logger.debug(f"  [REJECT] '{matched_text}' diabaikan, terdeteksi sebagai nama orang lain.")
                    last_end = end
                    continue
                
                if ent_id not in entity_data:
                    entity_data[ent_id] = {
                        "count": 0, "in_title": resolved_name.lower() in title_lower,
                        "src": resolver_source, "conf": confidence, "offsets": []
                    }
                
                entity_data[ent_id]["count"] += 1
                entity_data[ent_id]["conf"] = max(entity_data[ent_id]["conf"], confidence)
                entity_data[ent_id]["offsets"].append({"start": start, "end": end, "text": matched_text})
                
            last_end = end

        # Kalau configured_entity_id tidak ketemu sendiri lewat regex (mis.
        # cara penyebutannya tidak match pattern manapun), tetap paksa masuk
        # dgn offset kosong -- supaya main entity dari scraping_config TETAP
        # terjamin ter-map walau NER/regex tidak "melihatnya" secara eksplisit.
        if configured_entity_id and configured_entity_id not in entity_data:
            entity_data[configured_entity_id] = {
                "count": 0, "in_title": configured_entity_name.lower() in title_lower if configured_entity_name else False,
                "src": "pre_attributed", "conf": 1.0, "offsets": []
            }

        ranked_entities = sorted(entity_data.items(), key=lambda item: (item[1]["in_title"], item[1]["count"]), reverse=True)
        
        # === 4. SALIENCE GATE (Filter Tokoh Figuran) ===
        valid_entities = []
        for ent_id, data in ranked_entities:
            # Syarat 1: Tokoh ada di judul berita
            # Syarat 2: Tokoh disebut minimal 2 kali di body text
            # Syarat 3: ATAU memang entity yang dikonfigurasi scraping_config
            #           (sinyal editorial eksplisit, tetap lolos walau sepi mention)
            if data["in_title"] or data["count"] > 1 or ent_id == configured_entity_id:
                valid_entities.append((ent_id, data))
                
        # Safety Net: Jika tidak ada yang lolos (misal semua cuma disebut 1x), 
        # ambil 1 tokoh peringkat pertama agar artikel tidak kosong.
        if not valid_entities and ranked_entities:
            valid_entities.append(ranked_entities[0])
            
        ranked_entities = valid_entities
        # ============================================
        
        mappings = []
        mentions = []
        
        for idx, (ent_id, data) in enumerate(ranked_entities):
            # is_main_entity: entity dari scraping_config SELALU main (sinyal
            # editorial eksplisit menang atas urutan/frekuensi), else entitas
            # peringkat pertama hasil ranking (in_title / count terbanyak).
            is_main = (ent_id == configured_entity_id) if configured_entity_id else (idx == 0)
            mappings.append({
                "entity_id": ent_id, "is_main_entity": is_main, 
                "confidence": data["conf"], "resolver_source": data["src"]
            })
            
            # Buat 1 row untuk setiap kemunculan tokoh (Multi-Mention)
            for offset in data["offsets"]:
                mentions.append({
                    "entity_id": ent_id, "text": offset["text"],
                    "count": data["count"], "start": offset["start"], "end": offset["end"]
                })
            
        results.append({
            "raw_text_id": art["id"],
            "ingested_month": ingested_month,
            "mappings": mappings,
            "mentions": mentions
        })
        
    return results

def chunked_upsert_tracked(sb, table_name: str, data: list, on_conflict: str, chunk_size: int = 50) -> set:
    """Upsert per-chunk, MENGEMBALIKAN set raw_text_id yang GAGAL ditulis
    (chunk yg exception). Dipakai main() utk memutuskan artikel mana yang
    boleh ditandai entity_resolved_at -- fix v12 (lihat changelog: v11
    menandai SEMUA artikel tanpa syarat walau sebagian upsert gagal)."""
    failed_ids = set()
    if not data:
        return failed_ids
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        try:
            sb.table(table_name).upsert(chunk, on_conflict=on_conflict).execute()
        except Exception as e:
            logger.error(f"Upsert Error ({table_name}): {e}")
            failed_ids.update(c["raw_text_id"] for c in chunk)
    return failed_ids

def main(limit: int = 50, max_total: int = 0, days_back: int = DEFAULT_DAYS_BACK):
    sb = get_client()
    run_id = start_run("entity_resolution_worker", RESOLVER_VERSION)
    
    alias_map, entity_db_map, id_to_name, regex_patterns = load_caches(sb)
    logger.info(f"Loaded {len(regex_patterns)} regex patterns ke memori.")
    
    total_processed = 0
    total_success = 0
    batch_num = 1

    logger.info(f"[ENTITY_RESOLVER v12] Co-occurrence Fix | Limit: {limit}/batch | Days back: {days_back}")

    while True:
        if max_total > 0 and total_processed >= max_total:
            break
            
        current_limit = min(limit, max_total - total_processed) if max_total > 0 else limit
        
        try:
            time_filter = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
            res = sb.table("raw_texts") \
                    .select("id, title, text, metadata, ingested_month") \
                    .eq("status", pc.STATUS_VALIDATED) \
                    .not_.is_("preprocessed_at", "null") \
                    .is_("entity_resolved_at", "null") \
                    .gte("ingested_at", time_filter) \
                    .limit(current_limit) \
                    .execute()
        except Exception as e:
            logger.warning(f"DB Query Timeout/Error: {e}. Menunggu 10 detik...")
            time.sleep(10)
            continue

        articles = res.data or []
        if not articles:
            break
            
        logger.info(f"Memproses {len(articles)} artikel dengan Hybrid NLP + Salience Gate...")
        batch_results = process_articles_batch(articles, alias_map, entity_db_map, id_to_name, regex_patterns)
        
        all_mappings = []
        all_mentions = []
        now_iso = datetime.now(timezone.utc).isoformat()
        # succeeded_ids: mulai dari asumsi SEMUA sukses, turunkan kalau ada
        # chunk upsert yg gagal (fix v12 -- v11 menandai semua tanpa syarat)
        succeeded_ids = {result["raw_text_id"] for result in batch_results}

        for result in batch_results:
            if result["mappings"]:
                all_mappings.extend([{**m, "raw_text_id": result["raw_text_id"], "ingested_month": result["ingested_month"]} for m in result["mappings"]])
                all_mentions.extend([{**m, "raw_text_id": result["raw_text_id"], "ingested_month": result["ingested_month"]} for m in result["mentions"]])

        # === UPSERT, BUKAN DELETE+INSERT (fix v12) ===
        # article_entity_map: DELETE manual yg tadinya mendahului upsert
        # DIHAPUS -- chunked_upsert dgn on_conflict="raw_text_id,entity_id"
        # SUDAH menangani replace-if-exists, delete tambahan cuma menambah
        # window race condition kalau ada run lain yg overlap (lihat temuan
        # concurrency GH Actions sebelumnya) tanpa manfaat tambahan.
        mapping_fail_ids = chunked_upsert_tracked(sb, "article_entity_map", all_mappings, on_conflict="raw_text_id,entity_id")
        succeeded_ids -= mapping_fail_ids

        # entity_mentions: upsert dgn on_conflict eksplisit (butuh UNIQUE
        # (raw_text_id, entity_id, start_offset) -- lihat migration 012).
        # v11 pakai on_conflict=None (insert polos) + delete manual sebelumnya;
        # sekarang idempotent tanpa perlu delete sama sekali.
        db_mentions = [{
            "raw_text_id": m["raw_text_id"],
            "ingested_month": m["ingested_month"],
            "entity_id": m["entity_id"],
            "mention_text": m["text"],
            "start_offset": m["start"],
            "end_offset": m["end"]
        } for m in all_mentions]
        mention_fail_ids = chunked_upsert_tracked(sb, "entity_mentions", db_mentions, on_conflict="raw_text_id,entity_id,start_offset")
        succeeded_ids -= mention_fail_ids

        # entity_resolved_at HANYA utk artikel yang mapping & mention-nya
        # (kalau ada) benar2 tertulis -- sisanya otomatis dicoba lagi run
        # berikutnya (masih entity_resolved_at IS NULL).
        resolved_updates = [
            {"id": rid, "entity_resolved_at": now_iso, "resolver_version": RESOLVER_VERSION}
            for rid in succeeded_ids
        ]
        if resolved_updates:
            for i in range(0, len(resolved_updates), 25):
                try:
                    sb.rpc("bulk_update_raw_texts", {"p_updates": resolved_updates[i:i+25]}).execute()
                except Exception as e:
                    logger.error(f"Status Update Error: {e}")

        success_count = len(succeeded_ids)
        logger.info(f"{success_count}/{len(articles)} artikel berhasil di-resolve & ditandai. Mappings: {len(all_mappings)} | Mentions: {len(all_mentions)}")
        
        total_processed += len(articles)
        total_success += success_count
        batch_num += 1
        
        sleep_time = random.uniform(2, 5)
        logger.info(f"Menunggu {sleep_time:.1f}s sebelum batch berikutnya...")
        time.sleep(sleep_time)
        
    finish_run(run_id, total_processed, total_success, 0)
    logger.info("Eksekusi Entity Resolver (v12 Co-occurrence Fix) Selesai.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-total", type=int, default=0)
    parser.add_argument("--days-back", type=int, default=DEFAULT_DAYS_BACK, help="Jangkauan hari ke belakang utk ingested_at (default 30). Backfill bisa pakai angka besar.")
    args = parser.parse_args()
    main(limit=args.limit, max_total=args.max_total, days_back=args.days_back)