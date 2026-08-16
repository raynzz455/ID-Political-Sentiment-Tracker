"""
context_worker.py v18 — Precision Context Extraction
=====================================================
CRITICAL FIXES over v17:
  1. QUALITY_SCORE FIX: split ATTRIBUTION_WORDS from sentiment predicates.
     Attribution verbs (mengatakan/menegaskan) NO LONGER get attr_score=40.
     Only sentiment predicates (mengkritik/mengecam/dipuji) get high score.
     This kills the speaker_not_target bias (33.7% of v17 contexts).
  2. MULTI-MENTION RETENTION: keep ALL context spans per entity (not just best).
     Stored as list in entity_contexts.metadata.spans. Downstream
     nlp_worker v15 aggregates via predict_gated_multi().
  3. RELEVANCY PRE-FILTER: run the relevancy model on each span BEFORE
     storing. Spans with relevancy < 0.5 are stored but flagged low_relevancy.
     nlp_worker skips low_relevancy spans (token savings + precision boost).
  4. TITLE EXCLUSION preserved (v17 fix kept): context from body only.
  5. CROWDED-SENTENCE FIX preserved: local clause extraction for multi-entity
     sentences.

GITHUB ACTIONS COMPATIBILITY:
  - Stanza 'tokenize,pos,lemma,depparse' (same as v17). No new deps.
  - Relevancy model load adds ~5s startup + ~400MB RAM (IndoBERT-base).
    Total RAM ~1.2GB, well within 7GB limit.
  - Per-article: ~6s (Stanza) + ~0.5s (relevancy on 2-3 spans) = ~6.5s.
    200 articles / 4 threads = ~325s = 5.4 min. Within 45-min timeout.
  - Idempotent upsert preserved.

ACCURACY IMPACT (projected):
  - context precision: 55% -> ~85% (relevancy pre-filter removes bg spans)
  - speaker_not_target: 33.7% -> ~15% (quality_score fix)
  - background_only: 39.9% -> ~20% (multi-mention + relevancy filter)
  - Sentiment training signal: 211 -> ~500 clean rows (2.4x boost)
"""
import re
import time
import logging
import torch
import argparse
import json
import stanza
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

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

CONTEXT_VERSION = "v19.1_max_token"
MAX_NLP_WORKERS = 4 if torch.cuda.is_available() else 2

logger.info("Memuat Stanza Pipeline (tokenize,pos,lemma,depparse)...")
try:
    NLP = stanza.Pipeline('id', processors='tokenize,pos,lemma,depparse', verbose=False, use_gpu=True, batch_size=32)
except Exception as e:
    logger.warning(f"Gagal load GPU Stanza, fallback ke CPU: {e}")
    NLP = stanza.Pipeline('id', processors='tokenize,pos,lemma,depparse', verbose=False, use_gpu=False, batch_size=32)

# v18: Load relevancy model for pre-filtering
RELEVANCY_MODEL_ID = "apriandito/indobert-relevancy-classifier"
RELEVANCY_THRESHOLD = 0.5
_relevancy_pipeline = None

def get_relevancy_pipeline():
    global _relevancy_pipeline
    if _relevancy_pipeline is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            logger.info(f"Loading relevancy model: {RELEVANCY_MODEL_ID}")
            tok = AutoTokenizer.from_pretrained(RELEVANCY_MODEL_ID)
            model = AutoModelForSequenceClassification.from_pretrained(RELEVANCY_MODEL_ID)
            model.to("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()
            _relevancy_pipeline = (tok, model)
        except Exception as e:
            logger.warning(f"Relevancy model load failed (pre-filter disabled): {e}")
            _relevancy_pipeline = None
    return _relevancy_pipeline

@torch.no_grad()
def check_relevancy(entity_name: str, context_text: str) -> float:
    """Run relevancy model on (entity, context) pair. Returns prob[relevant]."""
    pipe = get_relevancy_pipeline()
    if pipe is None:
        return 1.0  # if model unavailable, don't filter (fail-open)
    tok, model = pipe
    device = next(model.parameters()).device
    enc = tok(entity_name, context_text, truncation=True, max_length=256, return_tensors="pt").to(device)
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    # find "relevant" label index
    id2label = model.config.id2label
    rel_idx = None
    for idx, label in id2label.items():
        if label.lower().strip() in {"relevan", "relevant", "yes", "ya", "1", "true"}:
            rel_idx = idx
            break
    if rel_idx is None:
        rel_idx = 1
    return float(probs[rel_idx])

# v18.1: EXPANDED verb sets (v14.2 lemma forms, 70.7% coverage).
# IMPORTANT: Stanza returns ROOT lemmas (dikritik→kritik, mengecam→kecam, memuji→puji).
# Verb sets MUST use lemma forms. Passive detected via deprel=nsubj:pass (same lemma).
SENTIMENT_PREDICATES_ACTIVE = {
    # Negative evaluation (entity criticized/accused/sanctioned)
    "kritik","kecam","sindir","serang","hina","cela","ejek","tuding",
    "tuduh","lapor","cekal","tahan","vonis","tangkap","pidana","anggap",
    "nilai","sorot","gugur","bongkar","pecat","mundur","undur","berhenti",
    "ganti","razia","sita","denda","hukum","ganjar",
    # v18.2: EXPANDED negative framing verbs (from dynamic test findings)
    "duga","dugaan","diduga","terduga","tersangkut","terlibat","didakwa",
    "tuduh","menuduh","tuding","menuding","curiga","dicurigai",
    "skandal","kontroversi","viral",
    "korupsi","suap","pungli","gratifikasi","penyelewengan",
    "pelanggar","melanggar","menyimpang","penyimpangan",
    "salah","salahgunakan","penyalahgunaan",
    "beban","merugikan","rugi","kerugian",
    "bukti","terbukti","buktikan","membuktikan",
    # Positive evaluation (entity praised/supported/endorsed)
    "puji","dukung","apresiasi","restui","sahkan","setuju","kukuhkan",
    "akui","legitimasi",
    # Active opposition/support (entity takes stance)
    "bela","tolak","keberatan","menentang",
    # Judgment/evaluation verbs
    "pandang","sikapi","persepsi",
    # Revelation/exposure (negative framing)
    "ungkap",
}
ATTRIBUTION_WORDS = {
    # Core speaking verbs (entity is SPEAKER — neutral, not target)
    "kata","nyata","tegas","jelaskan","tambah","imbau","ingat","sampai",
    "aku","klaim","nilai","ungkap","jawab","ujar","tutur","sebut","papar",
    "ucap","sampaikan","katakan","ungkapkan","nyatakan","tegaskan",
    "tambahkan","imbaukan","ingatkan","balas","tanggapi",
    # Suggestion/request verbs (entity proposes)
    "saran","menyaran","rekomendasi","usul","ajak","mengajak",
    "pinta","minta","meminta","perintah","wantiwanti",
    # Emphasis verbs (entity highlights)
    "tekan","tekankan","menekankan","sorot","soroti","tandai","tanda",
    # Appointment/indication (entity designates)
    "tunjuk","menunjuk",
}
# v18.3: NOUN-based negative framing detection
# These are nouns (not verbs) that indicate entity is TARGET of negative framing.
# Stanza lemmatizes "dugaan" -> "dugaan" (noun), "korupsi" -> "korupsi" (noun).
# Without this, "dugaan keterlibatan AHY" won't trigger sentiment detection.
NEGATIVE_FRAMING_NOUNS = {
    "dugaan", "terduga", "tersangka", "tersangkut",
    "korupsi", "suap", "pungli", "gratifikasi", "penyelewengan",
    "skandal", "kontroversi", "polemik",
    "kasus", "perkara", "tuntutan", "tuntutan",
    "pelanggaran", "penyimpangan", "penyalahgunaan",
    "rugi", "kerugian", "beban",
    "vonis", "hukuman", "pidana", "dakwaan",
    "bukti", "ketahuan", "terbukti",
}
# v18.3: POSITIVE framing nouns (entity praised)
POSITIVE_FRAMING_NOUNS = {
    "pujian", "apresiasi", "dukungan", "restu", "persetujuan",
    "prestasi", "pencapaian", "kesuksesan", "sukses",
    "penghargaan", "pengakuan", "legitimasi",
}

PRONOUNS = {"dia", "ia", "beliau", "mereka", "nya"}
QUOTE_CHARS = set('“"”‘’')
MIN_LOCAL_CLAUSE_WORDS = 4
CLAUSE_SPLIT_RE = re.compile(
    r',|\byang\b|\bdan\b|\bsementara\b|\bsedangkan\b|\bnamun\b|\btetapi\b|\bsedang\b'
    r'|\bsoal\b|\btentang\b|\bterkait\b|\bmengenai\b|\bperihal\b',
    re.IGNORECASE,
)

# v19: TOKEN-OPTIMIZED — target 90% utilization (230 tokens of 256)
# Indonesian: 1 token ≈ 3.5 chars. 230 tokens ≈ 800 chars ≈ 160 words.
# Old v18.3: MAX_CONTEXT_WORDS=180 but actual output only 50 words (32% utilization).
# v19: increase surrounding sentences + target 160 words per context.
MAX_CONTEXT_WORDS = 160
# v19: TARGET_CHARS for quality control (800 chars = ~230 tokens)
MAX_CONTEXT_CHARS = 850
# v19: how many surrounding sentences to include (was effectively 1-2)
CONTEXT_WINDOW_SENTENCES = 3  # anchor ± 1-2 surrounding sentences
DEFAULT_DAYS_BACK = 30

def get_paragraph_index(text: str, offset: int) -> int:
    return text[:offset].count('\n\n')

def is_core_argument(sent, start_offset: int, end_offset: int) -> bool:
    for word in sent.words:
        if word.start_char <= start_offset < word.end_char or \
           (start_offset <= word.start_char < end_offset):
            if word.deprel in ['nsubj', 'nsubj:pass', 'obj', 'iobj', 'csubj']:
                return True
            if word.deprel in ['nmod', 'nmod:poss', 'amod', 'appos']:
                return False
    return True

def extract_local_clause(sent_text: str, sent_start_char: int, entity_start: int, entity_end: int) -> str | None:
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


def process_single_article_context(art: dict, mentions_by_art: dict) -> list:
    """v18: process 1 article — multi-mention + relevancy pre-filter."""
    art_id = art["id"]
    title = (art.get("title") or "").strip()
    body = (art.get("text") or "").strip()
    clean_text = body  # title excluded (v17 fix preserved)
    if not clean_text: return []

    try:
        doc = NLP(clean_text)
    except Exception as e:
        logger.error(f"ID: {art_id[:8]} | Stanza Error: {e}")
        return []

    sentences = []
    for sent in doc.sentences:
        if len(sent.text.strip()) > 10:
            sentences.append({
                "text": sent.text,
                "start": sent.tokens[0].start_char,
                "end": sent.tokens[-1].end_char,
                "parsed": sent
            })
    if not sentences: return []

    art_mentions = mentions_by_art.get(art_id, [])
    # v18: collect ALL spans per entity (not just best)
    all_spans = {}  # entity_id -> list of (ctx_text, quality)

    # PASS 1: find anchor_idx per mention
    resolved_mentions = []
    for m in art_mentions:
        entity_id = m["entity_id"]
        entity_name = m["political_entities"]["canonical_name"]
        start_offset = m.get("start_offset", -1)
        end_offset = m.get("end_offset", start_offset)
        if start_offset < 0: continue
        # v18: offsets are now body-only (entity_worker v14 sends body offsets)
        adjusted_offset = start_offset
        if adjusted_offset < 0: continue
        adjusted_end = end_offset
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

    # PASS 2: build context for EACH mention (v18: not just best)
    for rm in resolved_mentions:
        entity_id = rm["entity_id"]
        entity_name = rm["entity_name"]
        anchor_idx = rm["anchor_idx"]
        anchor_sent = sentences[anchor_idx]
        is_crowded = anchor_idx in crowded_sentence_idxs
        is_main_actor = is_core_argument(anchor_sent["parsed"], rm["adjusted_offset"], rm["adjusted_end"])

        root_word = ""
        has_sentiment_predicate = False
        has_attribution = False
        has_negative_noun = False
        has_positive_noun = False
        for word in anchor_sent["parsed"].words:
            lemma = (word.lemma or word.text).lower()
            if word.deprel == 'root':
                root_word = lemma
                if root_word in SENTIMENT_PREDICATES_ACTIVE:
                    has_sentiment_predicate = True
                if root_word in ATTRIBUTION_WORDS:
                    has_attribution = True
            # v18.3: check for negative/positive framing NOUNS anywhere in sentence
            if word.upos in ('NOUN', 'PROPN'):
                if lemma in NEGATIVE_FRAMING_NOUNS:
                    has_negative_noun = True
                elif lemma in POSITIVE_FRAMING_NOUNS:
                    has_positive_noun = True
        # v18.3: nouns can also trigger sentiment predicate (for framing detection)
        if has_negative_noun or has_positive_noun:
            has_sentiment_predicate = True

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

        # v19: TOKEN-OPTIMIZED context extraction
        # Goal: fill context to ~800 chars (230 tokens) for max model signal
        context_parts = []
        # Always include anchor sentence
        context_parts.append(anchor_text_for_context)
        # v19: add surrounding sentences (prev first, then next) to fill token budget
        prev_idx = anchor_idx - 1
        next_idx = anchor_idx + 1
        prev_added = 0
        next_added = 0
        max_each_side = 4  # v19.1: take up to 4 prev + 4 next = 9 sentences max

        # First: if attribution, prioritize prev sentence (quote context)
        if has_attribution and not used_local_clause and prev_idx >= 0:
            context_parts.insert(0, sentences[prev_idx]["text"])
            prev_added += 1
            prev_idx -= 1
            # Check if prev prev also has quote chars (continued quote)
            if prev_idx >= 0 and any(qc in sentences[prev_idx + 1]["text"] for qc in QUOTE_CHARS):
                context_parts.insert(0, sentences[prev_idx]["text"])
                prev_added += 1
                prev_idx -= 1

        # Then: add more surrounding sentences to fill token budget
        while (prev_added + next_added) < (max_each_side * 2):
            current_chars = len(" ".join(context_parts))
            if current_chars >= MAX_CONTEXT_CHARS:
                break

            # Alternate: add next sentence if available and adds value
            added_this_round = False
            if next_idx < len(sentences) and next_added < max_each_side:
                next_sent = sentences[next_idx]
                # Skip if very short or unrelated (e.g., different paragraph jump)
                if len(next_sent["text"]) > 20:
                    context_parts.append(next_sent["text"])
                    next_added += 1
                    next_idx += 1
                    added_this_round = True

            if prev_idx >= 0 and prev_added < max_each_side:
                prev_sent = sentences[prev_idx]
                if len(prev_sent["text"]) > 20:
                    context_parts.insert(0, prev_sent["text"])
                    prev_added += 1
                    prev_idx -= 1
                    added_this_round = True

            if not added_this_round:
                break

        ctx_text = " ".join(context_parts)
        # v19: truncate by CHARS (not words) for precise token control
        if len(ctx_text) > MAX_CONTEXT_CHARS:
            # Keep anchor in middle, truncate surrounding
            anchor_text = anchor_text_for_context
            anchor_start = ctx_text.find(anchor_text)
            if anchor_start >= 0:
                # Keep anchor + balanced surrounding
                anchor_end = anchor_start + len(anchor_text)
                remaining = MAX_CONTEXT_CHARS - len(anchor_text)
                left_budget = remaining // 2
                right_budget = remaining - left_budget
                left_part = ctx_text[:anchor_start][-left_budget:].strip()
                right_part = ctx_text[anchor_end:][:right_budget].strip()
                ctx_text = (left_part + " " + anchor_text + " " + right_part).strip()
            else:
                ctx_text = ctx_text[:MAX_CONTEXT_CHARS]

        para_idx = get_paragraph_index(clean_text, rm["adjusted_offset"])

        # v18: QUALITY_SCORE — sentiment predicate gets 40, attribution gets 10 (was 25)
        attr_score = 40 if has_sentiment_predicate else (10 if has_attribution else 10)
        actor_score = 30 if is_main_actor else 10
        pos_score = 20 if para_idx == 0 else (12 if para_idx <= 2 else 5)
        exclusivity_score = 10 if not is_crowded else (5 if used_local_clause else 0)
        quality_score = attr_score + actor_score + pos_score + exclusivity_score

        # v18: relevancy pre-filter
        relevancy_score = check_relevancy(entity_name, ctx_text)

        quality = {
            "quality_score": quality_score,
            "attr_score": attr_score,
            "actor_score": actor_score,
            "pos_score": pos_score,
            "exclusivity_score": exclusivity_score,
            "has_sentiment_predicate": has_sentiment_predicate,
            "has_attribution": has_attribution,
            "has_negative_noun": has_negative_noun,
            "has_positive_noun": has_positive_noun,
            "is_main_actor": is_main_actor,
            "used_local_clause": used_local_clause,
            "para_idx": para_idx,
            "relevancy_score": round(relevancy_score, 3),
            "is_relevant": relevancy_score >= RELEVANCY_THRESHOLD,
        }

        all_spans.setdefault(entity_id, []).append((ctx_text, quality))

    # v18: store ALL spans per entity (not just best)
    results = []
    for ent_id, spans in all_spans.items():
        # sort by quality_score desc, but keep all
        spans.sort(key=lambda x: x[1]["quality_score"], reverse=True)
        best_ctx, best_quality = spans[0]
        # store all spans in metadata for nlp_worker v15 to aggregate
        all_span_texts = [s[0] for s in spans]
        best_quality["all_spans"] = all_span_texts[:5]  # cap at 5 spans
        best_quality["span_count"] = len(spans)
        results.append({
            "raw_text_id": art_id,
            "ingested_month": art.get("ingested_month"),
            "entity_id": ent_id,
            "context_text": best_ctx,
            "context_version": CONTEXT_VERSION,
            "metadata": best_quality,
        })

    if results:
        logger.info(f"ID: {art_id[:8]} | Contexts: {len(results)} entities | "
                    f"Total spans: {sum(len(s) for s in all_spans.values())} | "
                    f"Relevant: {sum(1 for r in results if r['metadata']['is_relevant'])}")
    else:
        logger.info(f"ID: {art_id[:8]} | Contexts: 0 (Skipped)")
    return results


def process_articles_batch(articles: list, mentions_by_art: dict) -> list:
    results = []
    with ThreadPoolExecutor(max_workers=MAX_NLP_WORKERS) as pool:
        futures = {pool.submit(process_single_article_context, art, mentions_by_art): art for art in articles}
        for future in as_completed(futures):
            try:
                res = future.result()
                if res: results.extend(res)
            except Exception as e:
                logger.error(f"Context worker thread crashed: {e}")
    return results


def main(limit: int = 50, max_total: int = 0, days_back: int = DEFAULT_DAYS_BACK):
    sb = get_client()
    run_id = start_run("context_worker", CONTEXT_VERSION)
    total_processed = 0
    total_success = 0
    batch_num = 1
    logger.info(f"[CONTEXT_WORKER v18] Precision Multi | Limit: {limit}/batch | Days back: {days_back} | Threads: {MAX_NLP_WORKERS}")
    while True:
        if max_total > 0 and total_processed >= max_total: break
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
        succeeded_art_ids = set(art_ids)
        if context_inserts:
            for i in range(0, len(context_inserts), 25):
                chunk = context_inserts[i:i + 25]
                try:
                    sb.table("entity_contexts").upsert(chunk, on_conflict="raw_text_id,entity_id").execute()
                except Exception as e:
                    logger.error(f"Upsert Error: {e}")
                    failed_ids = {c["raw_text_id"] for c in chunk}
                    succeeded_art_ids -= failed_ids
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
    logger.info("Eksekusi Context Worker (v18 Precision Multi) Selesai.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-total", type=int, default=0)
    parser.add_argument("--days-back", type=int, default=DEFAULT_DAYS_BACK)
    args = parser.parse_args()
    main(limit=args.limit, max_total=args.max_total, days_back=args.days_back)
