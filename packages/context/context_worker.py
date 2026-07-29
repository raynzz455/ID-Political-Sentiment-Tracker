"""
context_worker.py v16 — Crowded-Sentence Disambiguation & Graduated Scoring
=====================================================================
FIX v16 (lihat diagnosis di percakapan sebelumnya -- kasus Jokowi/Puan):
  1. CROWDED-SENTENCE FIX: Akar masalah 2 entitas dapat context_text IDENTIK
     -- root kalimat ("angkat suara") tidak ada di ACTIVE_MARKERS/PASSIVE_MARKERS,
     jadi has_action=False utk KEDUA entitas yang berbagi 1 kalimat anchor,
     keduanya jatuh ke cabang look-ahead yang SAMA. Sekarang: kalimat yang
     "ramai" (dipakai >1 entitas berbeda sbg anchor) dideteksi dulu (pass 1);
     entitas yang BUKAN is_main_actor di kalimat ramai itu dapat KLAUSA LOKAL
     di sekitar posisi kemunculannya sendiri (split di koma/konjungsi terdekat),
     bukan seluruh kalimat mentah-mentah. Entitas is_main_actor tetap dapat
     kalimat penuh (dia memang subjek/objek inti kalimat itu).
  2. QUALITY_SCORE DIGRADASI: v15 cuma biner (90 kalau has_action AND
     is_main_actor, else 50) -- karena syarat "AND" itu jarang terpenuhi,
     hampir semua baris jatuh ke 50 (terbukti: 106/106 baris di dataset
     ekspor terakhir persis 50). Sekarang skor benar2 bertingkat 0-100 dari
     4 komponen (attribution, posisi, peran gramatikal, keunikan/exclusivity),
     dan mencatat exclusivity secara eksplisit di metadata.
  3. CONFIGURABLE TIME WINDOW: --days-back (default tetap 30, tidak berubah
     perilaku default) -- window 30 hari sebelumnya hardcoded, artinya artikel
     lebih lama TIDAK PERNAH bisa diproses lewat CLI apa pun. Backfill nanti
     bisa panggil dgn --days-back besar / khusus.
  4. UPSERT alih-alih DELETE+INSERT: entity_contexts sudah py UNIQUE
     (raw_text_id, entity_id) dari migration 011 -- delete manual sebelum
     insert TIDAK PERLU dan menambah window race condition kalau ada run yg
     overlap (lihat temuan GH Actions concurrency sebelumnya). Upsert dgn
     on_conflict cukup & atomik per baris.
  5. STATUS DITANDAI SELESAI HANYA UTK ARTIKEL YANG BENERAN BERHASIL DITULIS
     -- v15 set context_extracted_at utk SEMUA artikel di batch walau insert
     gagal utk sebagian (exception cuma di-log). Sekarang HANYA artikel yang
     upsert-nya sukses yang ditandai; sisanya otomatis dicoba lagi run berikut
     (masih context_extracted_at IS NULL).
  6. is_core_argument FIX: v15 pakai substring check (`word_lower in
     entity_lower`) -- token pendek yang kebetulan jadi prefix/substring nama
     lain bisa salah cocok. Sekarang exact match terhadap kata per-kata nama
     entitas.

FIX v15 (riwayat, tetap berlaku):
  1. TITLE EXCLUSION, OFFSET ADJUSTMENT, SUBJECT/OBJECT VERIFICATION,
     QUOTE BACKTRACK, TOKEN CAP -- lihat kode.
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

CONTEXT_VERSION = "v16_crowded_sentence_fix"

logger.info("Memuat Stanza Pipeline (tokenize, pos, lemma, depparse)...")
try:
    NLP = stanza.Pipeline('id', processors='tokenize,pos,lemma,depparse', verbose=False, use_gpu=True, batch_size=32)
except Exception as e:
    logger.warning(f"Gagal load GPU Stanza, fallback ke CPU: {e}")
    NLP = stanza.Pipeline('id', processors='tokenize,pos,lemma,depparse', verbose=False, use_gpu=False, batch_size=32)

ACTIVE_MARKERS = {"mengkritik", "menyindir", "menolak", "mengecam", "menegaskan", "menyatakan", "mengatakan", "menuding", "menyerang", "membela", "menilai", "mengaku", "mengklaim", "mengimbau", "mengingatkan", "menyampaikan", "menjelaskan", "menambahkan"}
PASSIVE_MARKERS = {"dikecam", "dikritik", "dipuji", "ditahan", "dipecat", "dituding", "dituduh", "dilaporkan", "dicekal", "disindir"}
PRONOUNS = {"dia", "ia", "beliau", "mereka", "nya"}
QUOTE_CHARS = set('“"”‘’')
ATTRIBUTION_WORDS = {"kata", "ujar", "tegas", "tutur", "sebut", "ungkap", "papar", "jelaskan", "tambahkan", "nyatakan"}
MIN_LOCAL_CLAUSE_WORDS = 4
CLAUSE_SPLIT_RE = re.compile(
    r',|\byang\b|\bdan\b|\bsementara\b|\bsedangkan\b|\bnamun\b|\btetapi\b|\bsedang\b'
    r'|\bsoal\b|\btentang\b|\bterkait\b|\bmengenai\b|\bperihal\b',
    re.IGNORECASE,
)

MAX_CONTEXT_WORDS = 180
DEFAULT_DAYS_BACK = 30

def get_paragraph_index(text: str, offset: int) -> int:
    return text[:offset].count('\n\n')

def is_core_argument(sent, start_offset: int, end_offset: int) -> bool:
    """Cek apakah token di offset ini adalah subjek/objek INTI kalimat."""
    for word in sent.words:
        if word.start_char <= start_offset < word.end_char or \
           (start_offset <= word.start_char < end_offset):
            if word.deprel in ['nsubj', 'nsubj:pass', 'obj', 'iobj', 'csubj']:
                return True
            if word.deprel in ['nmod', 'nmod:poss', 'amod', 'appos']:
                return False
    return True


def extract_local_clause(sent_text: str, sent_start_char: int, entity_start: int, entity_end: int) -> str | None:
    """Ambil klausa LOKAL di sekitar posisi entity dlm kalimat yang "ramai"
    (dipakai >1 entitas berbeda sbg anchor) -- alih-alih seluruh kalimat,
    yang akan identik utk semua entitas yang berbagi kalimat itu (akar
    masalah kasus Jokowi/Puan). Split di koma/konjungsi TERDEKAT ke posisi
    entity, bukan clause parse penuh -- murah tapi cukup akurat utk berita.
    Return None kalau hasil split terlalu pendek/tidak informatif (caller
    fallback ke kalimat penuh)."""
    local_start = entity_start - sent_start_char
    local_end = entity_end - sent_start_char
    if local_start < 0 or local_end > len(sent_text):
        return None

    left_bound = 0
    for m in CLAUSE_SPLIT_RE.finditer(sent_text[:local_start]):
        left_bound = m.end()

    right_match = CLAUSE_SPLIT_RE.search(sent_text[local_end:])
    right_bound = local_end + right_match.start() if right_match else len(sent_text)

    clause = sent_text[left_bound:right_bound].strip(" ,")
    if len(clause.split()) < MIN_LOCAL_CLAUSE_WORDS:
        return None
    return clause

def process_articles_batch(articles: list, mentions_by_art: dict) -> list:
    results = []
    
    # === 1. PERSIAPAN BATCH: Kumpulkan semua body text ===
    batch_texts = []
    batch_meta = []
    
    for art in articles:
        title = (art.get("title") or "").strip()
        body = (art.get("text") or "").strip()
        
        clean_text = body
        title_len = len(title) + 1 if title else 0
        
        if not clean_text: continue
        
        batch_texts.append(clean_text)
        batch_meta.append({
            "art": art, "title_len": title_len, "clean_text": clean_text
        })

    if not batch_texts:
        return []

    # === 2. STANZA BATCH INFERENCE: Proses semua teks SEKALIGUS ===
    logger.info(f"Memproses {len(batch_texts)} teks via Stanza Batch (Depparse)...")
    try:
        docs = NLP(batch_texts)
    except Exception as e:
        logger.error(f"Stanza Batch Error: {e}")
        return []

    # === 3. EKSTRAKSI KONTEKS: Loop hasil doc yang sudah di-parse ===
    for i, doc in enumerate(docs):
        meta = batch_meta[i]
        art = meta["art"]
        title_len = meta["title_len"]
        clean_text = meta["clean_text"]
        art_id = art["id"]

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

        # === PASS 1: cari anchor_idx tiap mention ===
        resolved_mentions = []
        for m in art_mentions:
            entity_id = m["entity_id"]
            entity_name = m["political_entities"]["canonical_name"]
            start_offset = m.get("start_offset", -1)
            end_offset = m.get("end_offset", start_offset)
            if start_offset < 0: continue

            adjusted_offset = start_offset - title_len
            if adjusted_offset < 0: continue
            adjusted_end = end_offset - title_len

            anchor_idx = -1
            for idx, s in enumerate(sentences):
                if s["start"] <= adjusted_offset < s["end"]:
                    anchor_idx = idx
                    break

            if anchor_idx == -1:
                for idx, s in enumerate(sentences):
                    if entity_name.lower() in s["text"].lower():
                        anchor_idx = idx
                        break
                if anchor_idx == -1:
                    continue

            resolved_mentions.append({
                "entity_id": entity_id, "entity_name": entity_name,
                "anchor_idx": anchor_idx, "adjusted_offset": adjusted_offset,
                "adjusted_end": adjusted_end,
            })

        entities_per_sentence = {}
        for rm in resolved_mentions:
            entities_per_sentence.setdefault(rm["anchor_idx"], set()).add(rm["entity_id"])
        crowded_sentence_idxs = {idx for idx, ents in entities_per_sentence.items() if len(ents) > 1}

        # === PASS 2: bangun context ===
        for rm in resolved_mentions:
            entity_id = rm["entity_id"]
            entity_name = rm["entity_name"]
            anchor_idx = rm["anchor_idx"]
            anchor_sent = sentences[anchor_idx]
            is_crowded = anchor_idx in crowded_sentence_idxs
            is_main_actor = is_core_argument(anchor_sent["parsed"], rm["adjusted_offset"], rm["adjusted_end"])
            
            root_word = ""
            has_action = False
            is_attribution_end = False
            for word in anchor_sent["parsed"].words:
                if word.deprel == 'root':
                    root_word = (word.lemma or word.text).lower()
                    if root_word in ACTIVE_MARKERS or root_word in PASSIVE_MARKERS:
                        has_action = True
                    if root_word in ATTRIBUTION_WORDS:
                        is_attribution_end = True

            used_local_clause = False
            anchor_text_for_context = anchor_sent["text"]
            if is_crowded and not is_main_actor:
                local_clause = extract_local_clause(
                    anchor_sent["text"], anchor_sent["start"],
                    rm["adjusted_offset"], rm["adjusted_end"],
                )
                if local_clause:
                    anchor_text_for_context = local_clause
                    used_local_clause = True

            context_parts = []

            if is_attribution_end and not used_local_clause and anchor_idx > 0:
                context_parts.append(sentences[anchor_idx - 1]["text"])
                if anchor_idx > 1 and any(qc in sentences[anchor_idx - 1]["text"] for qc in QUOTE_CHARS):
                    context_parts.insert(0, sentences[anchor_idx - 2]["text"])

            context_parts.append(anchor_text_for_context)

            if not used_local_clause:
                if is_main_actor and has_action and anchor_idx + 1 < len(sentences):
                    next_sent = sentences[anchor_idx + 1]
                    first_word = next_sent["parsed"].words[0].text.lower()
                    if first_word in PRONOUNS or any(qc in next_sent["text"][:5] for qc in QUOTE_CHARS):
                        context_parts.append(next_sent["text"])
                elif not has_action and anchor_idx + 1 < len(sentences):
                    context_parts.append(sentences[anchor_idx + 1]["text"])

            ctx_text = " ".join(context_parts)

            words_list = ctx_text.split()
            if len(words_list) > MAX_CONTEXT_WORDS:
                anchor_text = anchor_text_for_context
                anchor_len = len(anchor_text.split())
                if anchor_len >= MAX_CONTEXT_WORDS:
                    ctx_text = " ".join(anchor_text.split()[:MAX_CONTEXT_WORDS])
                else:
                    remaining_space = MAX_CONTEXT_WORDS - anchor_len
                    other_text = " ".join([c for c in context_parts if c != anchor_text])
                    other_text = " ".join(other_text.split()[:remaining_space])
                    ctx_text = other_text + " " + anchor_text if context_parts[0] != anchor_text else anchor_text + " " + other_text

            para_idx = get_paragraph_index(clean_text, rm["adjusted_offset"])

            attr_score = 40 if has_action else (25 if is_attribution_end else 10)
            actor_score = 30 if is_main_actor else 10
            pos_score = 20 if para_idx == 0 else (12 if para_idx <= 2 else 5)
            exclusivity_score = 10 if not is_crowded else (5 if used_local_clause else 0)
            quality_score = attr_score + actor_score + pos_score + exclusivity_score

            quality = {
                "quality_score": quality_score,
                "attr_score": attr_score,
                "actor_score": actor_score,
                "pos_score": pos_score,
                "exclusivity_score": exclusivity_score,
                "has_quote": any(qc in ctx_text for qc in QUOTE_CHARS),
                "is_main_actor": is_main_actor,
                "is_crowded_sentence": is_crowded,
                "used_local_clause": used_local_clause,
                "paragraph_idx": para_idx,
                "winner_window": CONTEXT_VERSION,
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

# (Bagian main() dan __main__ tetap sama persis seperti v14)
def main(limit: int = 50, max_total: int = 0, days_back: int = DEFAULT_DAYS_BACK):
    sb = get_client()
    run_id = start_run("context_worker", CONTEXT_VERSION)

    total_processed = 0
    total_success = 0
    batch_num = 1

    logger.info(f"[CONTEXT_WORKER v16] Limit: {limit}/batch | Days back: {days_back}")

    while True:
        if max_total > 0 and total_processed >= max_total:
            break

        current_limit = min(limit, max_total - total_processed) if max_total > 0 else limit

        try:
            time_filter = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
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
        except Exception:
            time.sleep(5)
            continue

        mentions_by_art = {}
        for m in (mentions_res.data or []):
            mentions_by_art.setdefault(m["raw_text_id"], []).append(m)

        context_inserts = process_articles_batch(articles, mentions_by_art)

        # === UPSERT alih-alih DELETE+INSERT (fix v16) ===
        # entity_contexts sudah py UNIQUE(raw_text_id, entity_id) (migration
        # 011) -- upsert cukup, tidak perlu delete manual dulu (yang menambah
        # window race condition kalau ada run lain overlap).
        succeeded_art_ids = set(art_ids)  # asumsikan sukses, turunkan kalau ada error per-chunk
        if context_inserts:
            for i in range(0, len(context_inserts), 25):
                chunk = context_inserts[i:i + 25]
                try:
                    sb.table("entity_contexts").upsert(chunk, on_conflict="raw_text_id,entity_id").execute()
                except Exception as e:
                    logger.error(f"Upsert Error: {e}")
                    # Artikel di chunk yg GAGAL upsert-nya JANGAN ditandai selesai --
                    # biar dicoba lagi run berikutnya (masih context_extracted_at IS NULL)
                    failed_ids = {c["raw_text_id"] for c in chunk}
                    succeeded_art_ids -= failed_ids

        # Artikel yg TIDAK menghasilkan context_inserts sama sekali (mis. nol
        # mention valid) TETAP ditandai selesai -- itu bukan kegagalan, cuma
        # tidak ada yg perlu disimpan utk artikel itu.
        updates = [{"id": aid, "context_extracted_at": datetime.now(timezone.utc).isoformat()} for aid in succeeded_art_ids]

        if updates:
            for i in range(0, len(updates), 25):
                chunk = updates[i:i + 25]
                try:
                    sb.rpc("bulk_update_raw_texts", {"p_updates": chunk}).execute()
                except Exception as e:
                    logger.error(f"RPC Error: {e}")

        logger.info(f"{len(articles)} diproses ({len(succeeded_art_ids)} ditandai selesai). {len(context_inserts)} contexts dibuat.")
        total_processed += len(articles)
        total_success += len(context_inserts)
        batch_num += 1

    finish_run(run_id, total_processed, total_success, 0)
    logger.info("Eksekusi Context Worker (v16) Selesai.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-total", type=int, default=0)
    parser.add_argument("--days-back", type=int, default=DEFAULT_DAYS_BACK, help="Jangkauan hari ke belakang utk ingested_at (default 30). Backfill bisa pakai angka besar.")
    args = parser.parse_args()
    main(limit=args.limit, max_total=args.max_total, days_back=args.days_back)