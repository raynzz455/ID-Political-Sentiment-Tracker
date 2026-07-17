"""
context_worker.py v8 — Semantic Candidate & Attribution Scoring
=====================================================================
PERUBAAHAN v8:
  1. MULTI-CANDIDATE: 1 mention menghasilkan 3 kandidat (Narrow, Adaptive, Paragraph).
  2. ATTRIBUTION HUNTING: Boost skor drastis jika context mengandung kutipan langsung.
  3. ADAPTIVE WINDOW: Ukuran jendela menyesuaikan target 250-450 kata, bukan kalimat tetap.
  4. CLEAN BEFORE SCORE: Boilerplate dibuang sebelum teks dipecah dan dinilai.
  5. PARAGRAPH IMPORTANCE: Skor posisi berbasis struktur (Lead/Body/Closing), bukan offset.
  6. RICH METADATA: Menyimpan metrik kandidat untuk keperluan riset/training.
"""
import re
import time
import logging
import argparse
from datetime import datetime, timezone, timedelta

from packages.shared.db_client import get_client
from packages.shared.logger import start_run, finish_run
from packages.shared import constants as pc

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

CONTEXT_VERSION = "v8_semantic_ir"

# 1. Action Verbs (Memicu opini)
ACTION_VERBS = {"mengkritik", "mendesak", "menolak", "mendukung", "membantah", "mengecam", "menyerang", "mengancam", "menepis", "menuding"}

# 2. Sentiment Hints (Kata sifat politik)
SENTIMENT_HINTS = {"berhasil", "maju", "sejahtera", "solusi", "apresiasi", "gagal", "korupsi", "oligarki", "merugikan", "konflik", "tersangka"}

# 3. Attribution Markers (Penanda kutipan langsung)
ATTRIBUTION_MARKERS = ["kata", "ujar", "tegas", "tutur", "sebut", "ungkap", "katakan", "papar"]

# 4. Coreferences (Diperluas ke jabatan)
COREFERENCES = {"beliau", "ia", "dia", "presiden", "menteri", "wakil presiden", "ketua", "ketum", "bupati", "gubernur", "capres", "cawapres", "politikus", "mantan"}

BOILERPLATE_RE = re.compile(r'(Baca Juga|Simak Juga|Berita Terkait|Advertisement|Ikuti Kami|Copyright|©|Reportase:|Jurnalis:|Editor:).*?(?=\n|$)', re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z“"])')

def deep_clean_text(text: str) -> str:
    """Bersihkan boilerplate sebelum di-split."""
    text = BOILERPLATE_RE.sub('', text)
    return text.strip()

def split_sentences(text: str) -> list:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if len(s.strip()) > 10]

def get_paragraph_index(text: str, offset: int) -> int:
    """Hitung index paragraf (0=Lead) berdasarkan offset."""
    return text[:offset].count('\n\n')

def generate_candidates(sentences: list, mention_idx: int) -> dict:
    """Hasilkan 3 kandidat konteks dari index kalimat mention."""
    candidates = {}
    
    # C1: Narrow Window (1 sebelum, 1 mention, 1 sesudah)
    start = max(0, mention_idx - 1)
    end = min(len(sentences), mention_idx + 2)
    c1 = " ".join(sentences[start:end])
    candidates["narrow"] = c1
    
    # C2: Adaptive Window (Target 250-450 kata, ekspansi dua arah)
    target_words = 300
    start = mention_idx
    end = mention_idx
    curr_words = len(sentences[mention_idx].split())
    
    while curr_words < target_words:
        expanded = False
        if start > 0:
            start -= 1
            curr_words += len(sentences[start].split())
            expanded = True
        if end < len(sentences) - 1 and curr_words < target_words + 50:
            end += 1
            curr_words += len(sentences[end].split())
            expanded = True
        if not expanded: break
    c2 = " ".join(sentences[start:end+1])
    candidates["adaptive"] = c2
    
    # C3: Full Paragraph (Cari batas \n\n terdekat)
    # Pseudo-paragraph: gabungan adaptive jika tidak ada \n\n
    candidates["paragraph"] = c2 # Fallback, akan dioptimasi jika ada \n\n
    
    return candidates

def calculate_semantic_score(context_text: str, entity_name: str, paragraph_idx: int) -> dict:
    """Skor berbasis Opini, Kutipan, dan Posisi Struktural."""
    lower_ctx = context_text.lower()
    lower_name = entity_name.lower()
    words = context_text.split()
    
    # 1. Attribution & Quote Score (Maks 40) - Filsafat v8: Kutipan adalah raja sentimen
    attr_hits = sum(1 for m in ATTRIBUTION_MARKERS if f"{m} " in lower_ctx or f"{m}lah" in lower_ctx)
    has_quote = '"' in context_text or '“' in context_text or '”' in context_text
    attr_score = min(40, (attr_hits * 20) + (20 if has_quote else 0))
    
    # 2. Sentiment Richness (Maks 30) - Mencari konteks yang punya "warna" opini
    actions = sum(1 for v in ACTION_VERBS if v in lower_ctx)
    hints = sum(1 for h in SENTIMENT_HINTS if h in lower_ctx)
    rich_score = min(30, (actions * 15) + (hints * 10))
    
    # 3. Paragraph Importance (Maks 20) - Berbasis struktur, bukan offset
    if paragraph_idx == 0:
        pos_score = 20  # Lead Paragraph
    elif paragraph_idx <= 2:
        pos_score = 15  # Early Body
    elif paragraph_idx <= 5:
        pos_score = 10  # Mid Body
    else:
        pos_score = 5   # Closing/Footer
    
    # 4. Coreference Score (Maks 10)
    coref_hits = sum(1 for c in COREFERENCES if c in lower_ctx)
    coref_score = min(10, coref_hits * 3)
    
    # 5. Density (Maks 10)
    density = lower_ctx.count(lower_name)
    density_score = min(10, density * 3)
    
    total_score = attr_score + rich_score + pos_score + coref_score + density_score
    confidence = min(100, total_score)
    
    return {
        "attr_score": attr_score,
        "rich_score": rich_score,
        "pos_score": pos_score,
        "quality_score": total_score,
        "context_confidence": confidence,
        "word_count": len(words),
        "has_quote": has_quote
    }

def process_articles_batch(articles: list, mentions_by_art: dict) -> list:
    results = []
    
    for art in articles:
        art_id = art["id"]
        title = art.get("title") or ""
        raw_text = f"{title}\n{art.get('text', '')}"
        
        # CLEAN BEFORE SCORE
        clean_text = deep_clean_text(raw_text)
        sentences = split_sentences(clean_text)
        text_length = len(clean_text)
        
        art_mentions = mentions_by_art.get(art_id, [])
        best_contexts = {} 
        
        for m in art_mentions:
            entity_id = m["entity_id"]
            entity_name = m["political_entities"]["canonical_name"]
            start_off = m["start_offset"]
            
            # Cari index kalimat di mana mention ini berada
            mention_idx = 0
            char_count = 0
            for i, sent in enumerate(sentences):
                if start_off < char_count + len(sent):
                    mention_idx = i
                    break
                char_count += len(sent) + 1
            
            # GENERATE MULTI-CANDIDATES
            candidates = generate_candidates(sentences, mention_idx)
            para_idx = get_paragraph_index(clean_text, start_off)
            
            # RANKING KANDIDAT
            best_candidate = None
            best_score = -1
            
            for c_type, c_text in candidates.items():
                metrics = calculate_semantic_score(c_text, entity_name, para_idx)
                if metrics["quality_score"] > best_score:
                    best_score = metrics["quality_score"]
                    best_candidate = (c_text, metrics, c_type)
            
            if best_candidate:
                ctx_text, quality, win_type = best_candidate
                
                # Tambahkan Rich Metadata untuk Riset/Training
                quality["candidate_count"] = len(candidates)
                quality["winner_window"] = win_type
                quality["mention_order"] = mention_idx
                quality["paragraph_idx"] = para_idx
                
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

    logger.info(f"[CONTEXT_WORKER v8] Semantic IR Mode | Limit: {limit}/batch")

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
        if not articles:
            break
            
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
        
        if context_inserts:
            for i in range(0, len(context_inserts), 25):
                chunk = context_inserts[i:i + 25]
                try: sb.table("entity_contexts").upsert(chunk, on_conflict="raw_text_id,entity_id").execute()
                except Exception as e: logger.error(f"Upsert Error: {e}")
                
        if updates:
            for i in range(0, len(updates), 25):
                chunk = updates[i:i + 25]
                try: sb.rpc("bulk_update_raw_texts", {"p_updates": chunk}).execute()
                except Exception as e: logger.error(f"RPC Error: {e}")
                
        logger.info(f"{len(articles)} diproses. {len(context_inserts)} contexts dibuat.")
        total_processed += len(articles)
        total_success += len(context_inserts)
        batch_num += 1
        
    finish_run(run_id, total_processed, total_success, 0)
    logger.info("Eksekusi Context Worker (Semantic IR) Selesai.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-total", type=int, default=0)
    args = parser.parse_args()
    main(limit=args.limit, max_total=args.max_total)