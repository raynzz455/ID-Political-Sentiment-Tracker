"""
context_worker.py v14 — Target-Centric Subgraph & Token Cap
=====================================================================
FIX v14:
  1. SUBJECT/OBJECT VERIFICATION: Menggunakan Dependency Parsing untuk memastikan
     tokoh adalah Subjek (nsubj) atau Objek (obj) dari kata kerja utama (root).
     Jika tokoh cuma posesif (nmod), sistem mencari kalimat lain.
  2. QUOTE BACKTRACK: Jika tokoh berada di akhir kalimat (didahului "kata/ujar"),
     sistem menarik 2 kalimat sebelumnya untuk menangkap kutipan utuh.
  3. TOKEN CAP (MAX 180 WORDS): Membatasi total kata agar tidak melebihi limit
     token IndoBERT (256 tokens). Jika kepanjangan, kalimat konteks dibuang.
"""

import re
import time
import logging
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv

import stanza

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

from packages.shared.db_client import get_client
from packages.shared.logger import start_run, finish_run
from packages.shared import constants as pc

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("stanza").setLevel(logging.WARNING)

CONTEXT_VERSION = "v14_target_subgraph"

logger.info("Memuat Stanza Pipeline (tokenize, pos, lemma, depparse)...")
try:
    NLP = stanza.Pipeline('id', processors='tokenize,pos,lemma,depparse', verbose=False, use_gpu=True)
except Exception as e:
    logger.warning(f"Gagal load GPU Stanza, fallback ke CPU: {e}")
    NLP = stanza.Pipeline('id', processors='tokenize,pos,lemma,depparse', verbose=False, use_gpu=False)

ACTIVE_MARKERS = {"mengkritik", "menyindir", "menolak", "mengecam", "menegaskan", "menyatakan", "mengatakan", "menuding", "menyerang", "membela", "menilai", "mengaku", "mengklaim", "mengimbau", "mengingatkan", "menyampaikan", "menjelaskan", "menambahkan"}
PASSIVE_MARKERS = {"dikecam", "dikritik", "dipuji", "ditahan", "dipecat", "dituding", "dituduh", "dilaporkan", "dicekal", "disindir"}
PRONOUNS = {"dia", "ia", "beliau", "mereka", "nya"}
QUOTE_CHARS = set('“"”‘’')
ATTRIBUTION_WORDS = {"kata", "ujar", "tegas", "tutur", "sebut", "ungkap", "papar", "jelaskan", "tambahkan", "nyatakan"}

MAX_CONTEXT_WORDS = 180 # Batas aman untuk IndoBERT 256 tokens

def get_paragraph_index(text: str, offset: int) -> int:
    return text[:offset].count('\n\n')

def is_core_argument(sent, entity_name: str) -> bool:
    """Cek apakah tokoh adalah Subjek/Objek utama, bukan cuma posesif."""
    entity_lower = entity_name.lower()
    for word in sent.words:
        if entity_lower in word.text.lower() or entity_lower in word.lemma.lower():
            # Jika dia subjek (nsubj) atau objek (obj/obj) dari root, itu aktor utama
            if word.deprel in ['nsubj', 'nsubj:pass', 'obj', 'iobj', 'csubj']:
                return True
            # Jika dia modifier posesif (nmod/poss), dia bukan aktor utama
            if word.deprel in ['nmod', 'nmod:poss', 'amod', 'appos']:
                return False
    return True # Default True jika Stanza ragu

def process_articles_batch(articles: list, mentions_by_art: dict) -> list:
    results = []
    
    for art in articles:
        art_id = art["id"]
        title = (art.get("title") or "").strip()
        body = (art.get("text") or "").strip()
        
        if title and body and body.startswith(title):
            body = body[len(title):].lstrip(" :-\n")
            
        clean_text = f"{title}\n\n{body}" if title and body else (body or title)
        if not clean_text: continue
        
        doc = NLP(clean_text)
        sentences = []
        for sent in doc.sentences:
            if len(sent.text.strip()) > 10:
                sentences.append({
                    "text": sent.text,
                    "start": sent.tokens[0].start_char,
                    "end": sent.tokens[-1].end_char,
                    "parsed": sent
                })
                
        if not sentences: continue
        
        art_mentions = mentions_by_art.get(art_id, [])
        best_contexts = {} 
        
        for m in art_mentions:
            entity_id = m["entity_id"]
            entity_name = m["political_entities"]["canonical_name"]
            start_offset = m.get("start_offset", -1)
            
            if start_offset < 0: continue
            
            anchor_idx = -1
            for i, s in enumerate(sentences):
                if s["start"] <= start_offset < s["end"]:
                    anchor_idx = i
                    break
            
            if anchor_idx == -1: continue
            
            anchor_sent = sentences[anchor_idx]
            context_parts = []
            
            # === 1. TARGET-CENTRIC VERIFICATION ===
            is_main_actor = is_core_argument(anchor_sent["parsed"], entity_name)
            
            # === 2. ANALISIS KALIMAT ANCHOR ===
            root_word = ""
            has_action = False
            is_attribution_end = False # Cek kalimat: "...," kata Prabowo.
            
            for word in anchor_sent["parsed"].words:
                if word.deprel == 'root':
                    root_word = (word.lemma or word.text).lower()
                    if root_word in ACTIVE_MARKERS or root_word in PASSIVE_MARKERS:
                        has_action = True
                    if root_word in ATTRIBUTION_WORDS:
                        is_attribution_end = True

            # === 3. QUOTE BACKTRACK (Jika tokoh di akhir kalimat kutipan) ===
            if is_attribution_end and anchor_idx > 0:
                context_parts.append(sentences[anchor_idx - 1]["text"])
                if anchor_idx > 1 and any(qc in sentences[anchor_idx - 1]["text"] for qc in QUOTE_CHARS):
                    # Kalimat sebelumnya ada kutip, tarik 1 kalimat lagi ke belakang
                    context_parts.insert(0, sentences[anchor_idx - 2]["text"])
            
            context_parts.append(anchor_sent["text"])
            
            # === 4. SMART LOOK-AHEAD (Hanya jika tokoh adalah aktor utama) ===
            if is_main_actor and has_action and anchor_idx + 1 < len(sentences):
                next_sent = sentences[anchor_idx + 1]
                first_word = next_sent["parsed"].words[0].text.lower()
                if first_word in PRONOUNS or any(qc in next_sent["text"][:5] for qc in QUOTE_CHARS):
                    context_parts.append(next_sent["text"])
            
            elif not has_action and anchor_idx + 1 < len(sentences):
                # Jika netral, ambil 1 kalimat setelahnya
                context_parts.append(sentences[anchor_idx + 1]["text"])
            
            ctx_text = " ".join(context_parts)
            
            # === 5. TOKEN CAP MANAGEMENT (Batasasi 180 kata) ===
            words_list = ctx_text.split()
            if len(words_list) > MAX_CONTEXT_WORDS:
                # Jika kepanjangan, potong dari belakang (prioritaskan kalimat anchor)
                # Tapi pastikan anchor_sent tidak terpotong.
                # Cara aman: ambil kalimat anchor + sisa kata setelahnya
                anchor_text = anchor_sent["text"]
                anchor_len = len(anchor_text.split())
                
                if anchor_len >= MAX_CONTEXT_WORDS:
                    ctx_text = " ".join(anchor_text.split()[:MAX_CONTEXT_WORDS]) # Paksa potong anchor
                else:
                    # Ambil anchor, lalu isi sisa dengan kalimat sebelum/sesudah
                    remaining_space = MAX_CONTEXT_WORDS - anchor_len
                    other_text = " ".join([c for c in context_parts if c != anchor_text])
                    other_text = " ".join(other_text.split()[:remaining_space])
                    ctx_text = other_text + " " + anchor_text if context_parts[0] != anchor_text else anchor_text + " " + other_text
            
            para_idx = get_paragraph_index(clean_text, start_offset)
            quality = {
                "quality_score": 90 if (has_action and is_main_actor) else 50,
                "attr_score": 40 if has_action else 10,
                "pos_score": 20 if para_idx == 0 else 10,
                "has_quote": any(qc in ctx_text for qc in QUOTE_CHARS),
                "is_main_actor": is_main_actor,
                "paragraph_idx": para_idx,
                "winner_window": "stanza_v14_subgraph"
            }
            
            if entity_id not in best_contexts or quality["quality_score"] > best_contexts[entity_id][1]["quality_score"]:
                best_contexts[entity_id] = (ctx_text, quality)
                
        for ent_id, (ctx_text, quality) in best_contexts.items():
            results.append({
                "raw_text_id": art_id,
                "ingested_month": art.get("ingested_month"),
                "entity_id": ent_id,
                "context_text": ctx_text,
                "context_version": CONTEXT_VERSION,
                "metadata": quality
            })
            
    return results

def main(limit: int = 50, max_total: int = 0):
    sb = get_client()
    run_id = start_run("context_worker", CONTEXT_VERSION)
    
    total_processed = 0
    total_success = 0
    batch_num = 1

    logger.info(f"[CONTEXT_WORKER v14] Target Subgraph Mode | Limit: {limit}/batch")

    while True:
        if max_total > 0 and total_processed >= max_total:
            break
            
        current_limit = min(limit, max_total - total_processed) if max_total > 0 else limit
        
        try:
            time_filter = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            res = sb.table("raw_texts") \
                    .select("id, title, text, ingested_month") \
                    .eq("status", pc.STATUS_VALIDATED) \
                    .not_.is_("entity_resolved_at", "null") \
                    .is_("context_extracted_at", "null") \
                    .gte("ingested_at", time_filter) \
                    .limit(current_limit) \
                    .execute()
        except Exception as e:
            logger.warning(f"DB Query Timeout/Error: {e}. Menunggu 10 detik...")
            time.sleep(10)
            continue
                
        articles = res.data or []
        if not articles: break
            
        art_ids = [a["id"] for a in articles]
        
        try:
            mentions_res = sb.table("entity_mentions") \
                             .select("raw_text_id, entity_id, start_offset, end_offset, political_entities(canonical_name)") \
                             .in_("raw_text_id", art_ids) \
                             .execute()
        except Exception as e:
            time.sleep(5)
            continue
                         
        mentions_by_art = {}
        for m in (mentions_res.data or []):
            mentions_by_art.setdefault(m["raw_text_id"], []).append(m)
            
        context_inserts = process_articles_batch(articles, mentions_by_art)
        updates = [{"id": a["id"], "context_extracted_at": datetime.now(timezone.utc).isoformat()} for a in articles]
        
        try:
            if art_ids:
                for i in range(0, len(art_ids), 25):
                    chunk_ids = art_ids[i:i+25]
                    sb.table("entity_contexts").delete().in_("raw_text_id", chunk_ids).execute()

            if context_inserts:
                for i in range(0, len(context_inserts), 25):
                    chunk = context_inserts[i:i + 25]
                    try: sb.table("entity_contexts").insert(chunk).execute()
                    except Exception as e: logger.error(f"Insert Error: {e}")
                
            if updates:
                for i in range(0, len(updates), 25):
                    chunk = updates[i:i + 25]
                    try: sb.rpc("bulk_update_raw_texts", {"p_updates": chunk}).execute()
                    except Exception as e: logger.error(f"RPC Error: {e}")
                
        except Exception as e:
            logger.error(f"DB Error: {e}")
            
        logger.info(f"{len(articles)} diproses. {len(context_inserts)} contexts dibuat.")
        total_processed += len(articles)
        total_success += len(context_inserts)
        batch_num += 1
        
    finish_run(run_id, total_processed, total_success, 0)
    logger.info("Eksekusi Context Worker (v14 Target Subgraph) Selesai.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-total", type=int, default=0)
    args = parser.parse_args()
    main(limit=args.limit, max_total=args.max_total)