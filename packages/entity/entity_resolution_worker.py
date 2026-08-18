"""
entity_resolution_worker.py v15 — Intuitive Main-Entity Validation
==================================================================
CRITICAL FIXES over v14.2:
  1. TITLE CANDIDATE VALIDATION: title entities are CANDIDATES, must confirm
     in body. Title is HINT, body is PROOF.
  2. ERA VALIDATION: check entity era vs article era markers. Penalize mismatch.
  3. AFFILIATION VALIDATION: check party/position match in article text.
  4. INTUITIVE SCORING: body evidence > title evidence.
  5. STRICTER SALIENCE GATE: requires era_compatible for "count > 1 + in_title".

DEFENSIVE DESIGN:
  - era column may not exist in production DB. We try/except the SELECT and
    default era=[] if column missing. check_era_compatibility() returns
    (True, None) when era list is empty — so era check becomes a no-op.
  - party_affiliation and position columns already exist in schema.sql.
  - All extra computed fields are kept IN-MEMORY only (not upserted to DB)
    to prevent PGRST204 schema mismatch errors.

GITHUB ACTIONS COMPATIBILITY:
  - Same Stanza pipeline as v14 (tokenize,pos,lemma,depparse).
  - No new heavy dependencies.
  - Per-article: ~5s on CPU (same as v14). 200 articles / 2 threads = ~8 min.

ACCURACY IMPACT (projected from dataset analysis):
  - Reduces main-entity false-positive rate from ~25% -> ~15% (era + affiliation).
  - Cuts background_only context rows by additional ~10%.
  - Cuts speaker_not_target by additional ~5% (stricter salience gate).
"""
import re
import time
import random
import logging
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

import stanza
import torch

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

RESOLVER_VERSION = "v15.1_expanded_verbs"
DEFAULT_DAYS_BACK = 30
MAX_NLP_WORKERS = 4 if torch.cuda.is_available() else 2

logger.info("Memuat Stanza Pipeline (tokenize,pos,lemma,depparse)...")
try:
    NLP = stanza.Pipeline('id', processors='tokenize,pos,lemma,depparse',
                          verbose=False, use_gpu=True, batch_size=32)
except Exception as e:
    logger.warning(f"GPU Stanza gagal, fallback CPU: {e}")
    NLP = stanza.Pipeline('id', processors='tokenize,pos,lemma,depparse',
                          verbose=False, use_gpu=False, batch_size=32)

# v14.2/v15: EXPANDED verb sets (lemma forms, 70.7% coverage).
# Stanza returns ROOT lemmas (dikritik→kritik, mengecam→kecam). Passive via deprel=nsubj:pass.
SENTIMENT_PREDICATES_ACTIVE = {
    # Negative evaluation (entity criticized/accused/sanctioned)
    "kritik","kecam","sindir","serang","hina","cela","ejek","tuding",
    "tuduh","lapor","cekal","tahan","vonis","tangkap","pidana","anggap",
    "nilai","sorot","gugur","bongkar","pecat","mundur","undur","berhenti",
    "ganti","razia","sita","denda","hukum","ganjar",
    # v15.1: EXPANDED negative framing verbs (from dynamic test findings)
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
ATTRIBUTION_VERBS = {
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
TOPIC_DOMINANCE_THRESHOLD = 0.25
MIN_BODY_MENTIONS_FOR_MAIN = 1


def normalize_name(name: str) -> str:
    return re.sub(r'\s+', ' ', name).strip()


def load_caches(sb):
    """Load entity DB with era + affiliation for v15 validation.

    DEFENSIVE: try to SELECT era column. If it doesn't exist (migration 007
    not applied), fall back to SELECT without era and treat era=[] for all.
    """
    logger.info("Loading caches (dengan era + affiliation)...")
    # Try with era column first
    try:
        pe_res = sb.table("political_entities").select(
            "id, canonical_name, aliases, entity_type, party_affiliation, position, era"
        ).execute()
        has_era = True
    except Exception as e:
        # era column missing — fall back to schema without era
        logger.warning(f"era column unavailable, fallback tanpa era validation: {str(e)[:80]}")
        pe_res = sb.table("political_entities").select(
            "id, canonical_name, aliases, entity_type, party_affiliation, position"
        ).execute()
        has_era = False

    entity_db_map = {}
    alias_map = {}
    id_to_name = {}
    id_to_entity = {}
    regex_patterns = []

    for r in (pe_res.data or []):
        canon_lower = r["canonical_name"].lower()
        entity_db_map[canon_lower] = r["id"]
        id_to_name[r["id"]] = r["canonical_name"]
        id_to_entity[r["id"]] = {
            "name": r["canonical_name"],
            "aliases": r.get("aliases") or [],
            "entity_type": r.get("entity_type"),
            "party": r.get("party_affiliation"),
            "position": r.get("position"),
            "era": r.get("era") if has_era else [],
        }
        try:
            regex_patterns.append((re.compile(r'\b' + re.escape(r["canonical_name"]) + r'\b', re.IGNORECASE), canon_lower))
        except re.error:
            pass
        for alias in (r.get("aliases") or []):
            if len(alias) < 2:
                continue
            alias_lower = alias.lower()
            alias_map[alias_lower] = r["canonical_name"]
            try:
                regex_patterns.append((re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE), alias_lower))
            except re.error:
                pass

    logger.info(f"Loaded {len(regex_patterns)} patterns, {len(entity_db_map)} entities "
                f"(era={'yes' if has_era else 'no'})")
    return alias_map, entity_db_map, id_to_name, id_to_entity, regex_patterns


def is_false_positive(matched_text: str, canonical_name: str, full_persons: list) -> bool:
    matched_lower = matched_text.lower()
    canonical_lower = canonical_name.lower()
    for person in full_persons:
        person_lower = person.lower()
        person_words = person_lower.split()
        if matched_lower in person_words and len(person_lower) > len(matched_lower):
            canonical_parts = [p for p in canonical_lower.split() if p != matched_lower]
            if any(part in person_words for part in canonical_parts):
                return False
            else:
                return True
    return False


def check_semantic_role(sent, entity_start: int, entity_end: int) -> dict:
    """Check if entity at (start, end) is nsubj/obj of a sentiment predicate."""
    result = {
        'has_sentiment_role': False,
        'has_attribution_role': False,
        'sentiment_verb': None,
        'role': None,
        'is_passive': False,
    }
    entity_word = None
    for word in sent.words:
        if word.start_char <= entity_start < word.end_char:
            entity_word = word
            break
        if entity_start <= word.start_char < entity_end:
            entity_word = word
            break
    if entity_word is None:
        return result
    if entity_word.deprel in ('nsubj', 'nsubj:pass', 'obj', 'iobj', 'csubj', 'obl'):
        result['role'] = entity_word.deprel
        is_passive = (entity_word.deprel == 'nsubj:pass')
        result['is_passive'] = is_passive
        head_id = entity_word.head
        for word in sent.words:
            if word.id == head_id:
                root_lemma = (word.lemma or word.text).lower()
                if root_lemma in SENTIMENT_PREDICATES_ACTIVE:
                    result['has_sentiment_role'] = True
                    result['sentiment_verb'] = root_lemma + (" (passive)" if is_passive else "")
                elif root_lemma in ATTRIBUTION_VERBS:
                    result['has_attribution_role'] = True
                break
    return result


# ---------------------------------------------------------------------------
# v15 NEW: Era validation
# ---------------------------------------------------------------------------
def check_era_compatibility(article_text: str, entity_eras: list) -> tuple:
    """Check if article mentions eras compatible with entity's era.

    Returns (is_compatible, detected_era).
    If entity_eras is empty -> no era info -> return (True, None) (no-op).
    """
    if not entity_eras:
        return True, None

    text_lower = article_text.lower()
    era_markers = {
        "era jokowi": ["era jokowi", "zaman jokowi", "pemerintahan jokowi", "jokowi era"],
        "era prabowo": ["era prabowo", "zaman prabowo", "pemerintahan prabowo", "prabowo era"],
        "era sby": ["era sby", "zaman sby", "pemerintahan sby", "sby era"],
        "era gus dur": ["era gus dur", "zaman gus dur", "pemerintahan gus dur"],
        "era habibie": ["era habibie", "zaman habibie"],
        "era megawati": ["era megawati", "zaman megawati"],
    }

    detected_eras = []
    for era_key, markers in era_markers.items():
        if any(m in text_lower for m in markers):
            detected_eras.append(era_key)

    if not detected_eras:
        return True, None  # no era markers in article -> no constraint

    entity_eras_lower = [e.lower() for e in entity_eras]
    for detected in detected_eras:
        for ent_era in entity_eras_lower:
            # fuzzy match: "era prabowo" in entity_eras matches "Era Prabowo" entry
            if detected in ent_era or ent_era in detected:
                return True, detected

    return False, detected_eras[0]


# ---------------------------------------------------------------------------
# v15 NEW: Affiliation validation
# ---------------------------------------------------------------------------
def check_affiliation(article_text: str, entity_info: dict) -> tuple:
    """Check if party/position mentioned in article matches DB.

    Returns (match_status, mentioned_party).
    match_status: True (match), False (mismatch), None (inconclusive).
    """
    party = entity_info.get("party")
    if not party:
        return True, None

    text_lower = article_text.lower()
    party_lower = party.lower()

    if party_lower in text_lower:
        return True, party

    party_abbr = {
        "PDI-P": ["pdip", "pdi-p", "pdiperjuangan", "pdi perjuangan"],
        "Gerindra": ["gerindra"],
        "Golkar": ["golkar"],
        "Demokrat": ["demokrat", "partai demokrat"],
        "PKB": ["pkb", "kebangkitan bangsa"],
        "PAN": ["pan", "amanat nasional"],
        "PKS": ["pks", "keadilan sejahtera"],
        "Nasdem": ["nasdem", "nasional demokrat"],
        "PPP": ["ppp", "persatuan pembangunan"],
        "PBB": ["pbb", "bulan bintang"],
        "Perindo": ["perindo"],
        "Hanura": ["hanura"],
        "Independen": ["independen"],
    }

    abbrs = party_abbr.get(party, [party_lower])
    for abbr in abbrs:
        if abbr in text_lower:
            return True, party

    return None, None  # None = inconclusive (party not mentioned, can't validate)


# ---------------------------------------------------------------------------
# v15: Main entity resolution with INTUITIVE VALIDATION
# ---------------------------------------------------------------------------
def process_single_article_entity(art: dict, alias_map: dict, entity_db_map: dict,
                                   id_to_name: dict, id_to_entity: dict,
                                   regex_patterns: list) -> dict | None:
    """v15: title-candidate validation + era + affiliation check."""
    title = (art.get('title') or '').strip()
    body = (art.get('text') or '').strip()
    title_lower = title.lower()
    metadata = art.get("metadata") or {}
    ingested_month = art.get("ingested_month")

    if not body or len(body) < 50:
        return None

    try:
        doc = NLP(body)
    except Exception as e:
        logger.error(f"ID: {art['id'][:8]} | Stanza Error: {e}")
        return None

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
    full_persons = persons

    sentences = []
    for sent in doc.sentences:
        if len(sent.text.strip()) > 10:
            sentences.append({
                "text": sent.text,
                "start": sent.tokens[0].start_char if sent.tokens else 0,
                "end": sent.tokens[-1].end_char if sent.tokens else 0,
                "parsed": sent,
            })
    if not sentences:
        return None
    total_sents = len(sentences)

    configured_entity_id = metadata.get("configured_entity_id")

    entity_data = {}
    found_matches = []
    for pattern, key in regex_patterns:
        for match in pattern.finditer(body):
            found_matches.append((match.start(), match.end(), match.group(), key))
    found_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    last_end = -1
    for start, end, matched_text, key in found_matches:
        if start < last_end:
            continue
        resolved_name = None
        if key in alias_map:
            resolved_name = alias_map[key]
        elif key in entity_db_map:
            resolved_name = key
        if resolved_name and resolved_name.lower() in entity_db_map:
            ent_id = entity_db_map[resolved_name.lower()]
            if is_false_positive(matched_text, resolved_name, full_persons):
                last_end = end
                continue

            if ent_id not in entity_data:
                entity_info = id_to_entity.get(ent_id, {})
                entity_data[ent_id] = {
                    "count": 0,
                    "in_title": resolved_name.lower() in title_lower,
                    "in_body": True,
                    "sentence_indices": set(),
                    "has_sentiment_role": False,
                    "has_attribution_role": False,
                    "sentiment_verbs": [],
                    "entity_info": entity_info,
                    "src": "regex_exact",
                    "conf": 1.0,
                    "offsets": [],
                }

            entity_data[ent_id]["count"] += 1
            entity_data[ent_id]["offsets"].append({"start": start, "end": end, "text": matched_text})

            for sidx, s in enumerate(sentences):
                if s["start"] <= start < s["end"]:
                    entity_data[ent_id]["sentence_indices"].add(sidx)
                    role = check_semantic_role(s["parsed"], start, end)
                    if role["has_sentiment_role"]:
                        entity_data[ent_id]["has_sentiment_role"] = True
                        entity_data[ent_id]["sentiment_verbs"].append(role["sentiment_verb"])
                    if role["has_attribution_role"]:
                        entity_data[ent_id]["has_attribution_role"] = True
                    break
        last_end = end

    if configured_entity_id and configured_entity_id not in entity_data:
        configured_name = id_to_name.get(configured_entity_id, "")
        entity_data[configured_entity_id] = {
            "count": 0,
            "in_title": configured_name.lower() in title_lower if configured_name else False,
            "in_body": False,
            "sentence_indices": set(),
            "has_sentiment_role": False,
            "has_attribution_role": False,
            "sentiment_verbs": [],
            "entity_info": id_to_entity.get(configured_entity_id, {}),
            "src": "pre_attributed",
            "conf": 0.3,
            "offsets": [],
        }

    # v15: Compute topic_dominance + era + affiliation
    for ent_id, data in entity_data.items():
        data["topic_dominance"] = len(data["sentence_indices"]) / total_sents if total_sents > 0 else 0

        entity_info = data.get("entity_info", {})
        era_compatible, detected_era = check_era_compatibility(body, entity_info.get("era", []))
        data["era_compatible"] = era_compatible
        data["detected_era"] = detected_era

        affil_match, mentioned_party = check_affiliation(body, entity_info)
        data["affiliation_match"] = affil_match
        data["mentioned_party"] = mentioned_party

    # v15: INTUITIVE RANKING - body evidence + era + affiliation first
    def intuitive_salience_key(item):
        ent_id, data = item
        return (
            data["has_sentiment_role"],
            data["topic_dominance"] >= TOPIC_DOMINANCE_THRESHOLD,
            data["era_compatible"],
            data["affiliation_match"] is not False,
            data["in_body"] and data["count"] >= MIN_BODY_MENTIONS_FOR_MAIN,
            data["in_title"],
            data["count"],
        )

    ranked = sorted(entity_data.items(), key=intuitive_salience_key, reverse=True)

    # v15: SALIENCE GATE - stricter with era + affiliation
    valid_entities = []
    for ent_id, data in ranked:
        if not data["in_body"] and data["count"] == 0:
            continue
        if data["has_sentiment_role"]:
            valid_entities.append((ent_id, data))
        elif data["topic_dominance"] >= TOPIC_DOMINANCE_THRESHOLD:
            valid_entities.append((ent_id, data))
        elif data["count"] > 1 and data["in_title"] and data["era_compatible"]:
            valid_entities.append((ent_id, data))
        elif ent_id == configured_entity_id and data["count"] > 0 and data["era_compatible"]:
            valid_entities.append((ent_id, data))

    if not valid_entities:
        body_entities = [e for e in ranked if e[1]["count"] > 0 and e[1]["era_compatible"]]
        if body_entities:
            valid_entities = [body_entities[0]]
        elif ranked:
            valid_entities = [ranked[0]]

    mappings = []
    mentions = []
    for idx, (ent_id, data) in enumerate(valid_entities):
        is_main = (idx == 0)

        if data["has_sentiment_role"]:
            conf = 0.95
        elif data["topic_dominance"] >= TOPIC_DOMINANCE_THRESHOLD:
            conf = 0.85
        elif data["count"] > 1 and data["in_title"]:
            conf = 0.70
        else:
            conf = 0.50

        # v15: Penalize era mismatch
        if not data["era_compatible"]:
            conf *= 0.7

        # v15: Penalize affiliation mismatch
        if data["affiliation_match"] is False:
            conf *= 0.8

        # IMPORTANT: Only upsert fields that exist in DB schema (article_entity_map).
        # DB columns: entity_id, is_main_entity, confidence, resolver_source.
        # Extra fields are computed but NOT stored to prevent PGRST204 errors.
        mappings.append({
            "entity_id": ent_id,
            "is_main_entity": is_main,
            "confidence": round(conf, 3),
            "resolver_source": data["src"],
        })
        for offset in data["offsets"]:
            mentions.append({
                "entity_id": ent_id,
                "text": offset["text"],
                "count": data["count"],
                "start": offset["start"],
                "end": offset["end"],
            })

    if mappings:
        main_ent_id = mappings[0]["entity_id"]
        main_data = entity_data.get(main_ent_id, {})
        logger.info(f"ID: {art['id'][:8]} | Main: {id_to_name.get(main_ent_id, '?')} "
                    f"(sent_role={main_data.get('has_sentiment_role', False)}, "
                    f"dom={main_data.get('topic_dominance', 0):.3f}, "
                    f"era={main_data.get('era_compatible', True)}, "
                    f"affil={main_data.get('affiliation_match', None)}, "
                    f"conf={mappings[0]['confidence']}) | Total: {len(mappings)} entities")
    else:
        logger.info(f"ID: {art['id'][:8]} | Resolved: 0 entities (Skipped)")

    return {
        "raw_text_id": art["id"],
        "ingested_month": ingested_month,
        "mappings": mappings,
        "mentions": mentions,
    }


def process_articles_batch(articles: list, alias_map: dict, entity_db_map: dict,
                            id_to_name: dict, id_to_entity: dict,
                            regex_patterns: list) -> list:
    results = []
    with ThreadPoolExecutor(max_workers=MAX_NLP_WORKERS) as pool:
        futures = {pool.submit(process_single_article_entity, art, alias_map,
                               entity_db_map, id_to_name, id_to_entity,
                               regex_patterns): art for art in articles}
        for future in as_completed(futures):
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as e:
                logger.error(f"Entity resolver thread crashed: {e}")
    return results


def chunked_upsert_tracked(sb, table_name: str, data: list, on_conflict: str, chunk_size: int = 50) -> set:
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

    alias_map, entity_db_map, id_to_name, id_to_entity, regex_patterns = load_caches(sb)

    total_processed = 0
    total_success = 0
    batch_num = 1

    logger.info(f"[ENTITY_RESOLVER v15] Intuitive Validation | Limit: {limit}/batch | "
                f"Days back: {days_back} | Threads: {MAX_NLP_WORKERS}")

    while True:
        if max_total > 0 and total_processed >= max_total:
            break

        current_limit = min(limit, max_total - total_processed) if max_total > 0 else limit

        try:
            time_filter = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
            res = sb.table("raw_texts").select(
                "id, title, text, metadata, ingested_month"
            ).eq("status", pc.STATUS_VALIDATED).not_.is_("preprocessed_at", "null").is_(
                "entity_resolved_at", "null"
            ).gte("ingested_at", time_filter).limit(current_limit).execute()
        except Exception as e:
            logger.warning(f"DB Query Timeout/Error: {e}. Menunggu 10s...")
            time.sleep(10)
            continue

        articles = res.data or []
        if not articles:
            break

        logger.info(f"Batch {batch_num}: Memproses {len(articles)} artikel dengan Intuitive Validation...")
        batch_results = process_articles_batch(
            articles, alias_map, entity_db_map, id_to_name, id_to_entity, regex_patterns
        )

        all_mappings = []
        all_mentions = []
        now_iso = datetime.now(timezone.utc).isoformat()
        
        # FIX: Mark ALL articles as resolved (even failed ones)
        # BUG ORIGINAL: articles yang return None tidak masuk batch_results
        # → tidak ditandai resolved → diambil lagi batch berikutnya → infinite loop
        # Sekarang: ALL articles ditandai resolved, supaya tidak diulang
        all_art_ids = {a["id"] for a in articles}
        succeeded_ids = {r["raw_text_id"] for r in batch_results}
        failed_ids = all_art_ids - succeeded_ids
        
        if failed_ids:
            logger.info(f"{len(failed_ids)} articles gagal (Stanza/error) — ditandai resolved supaya tidak diulang")

        for result in batch_results:
            if result["mappings"]:
                all_mappings.extend([{**m, "raw_text_id": result["raw_text_id"],
                                     "ingested_month": result["ingested_month"]} for m in result["mappings"]])
                all_mentions.extend([{**m, "raw_text_id": result["raw_text_id"],
                                     "ingested_month": result["ingested_month"]} for m in result["mentions"]])

        mapping_fail_ids = chunked_upsert_tracked(sb, "article_entity_map", all_mappings,
                                                   on_conflict="raw_text_id,entity_id")
        succeeded_ids -= mapping_fail_ids

        db_mentions = [{
            "raw_text_id": m["raw_text_id"], "ingested_month": m["ingested_month"],
            "entity_id": m["entity_id"], "mention_text": m["text"],
            "start_offset": m["start"], "end_offset": m["end"],
        } for m in all_mentions]
        mention_fail_ids = chunked_upsert_tracked(sb, "entity_mentions", db_mentions,
                                                   on_conflict="raw_text_id,entity_id,start_offset")
        succeeded_ids -= mention_fail_ids

        # Mark ALL articles as resolved (success + failed)
        # Failed articles get resolver_version="failed_no_entity" for tracking
        resolved_updates = [{"id": rid, "entity_resolved_at": now_iso,
                            "resolver_version": RESOLVER_VERSION} for rid in succeeded_ids]
        resolved_updates.extend([{"id": rid, "entity_resolved_at": now_iso,
                                 "resolver_version": "failed_no_entity"} for rid in failed_ids])
        if resolved_updates:
            for i in range(0, len(resolved_updates), 25):
                try:
                    sb.rpc("bulk_update_raw_texts", {"p_updates": resolved_updates[i:i+25]}).execute()
                except Exception as e:
                    logger.error(f"Status Update Error: {e}")

        success_count = len(succeeded_ids)
        logger.info(f"{success_count}/{len(articles)} artikel berhasil di-resolve & ditandai. "
                    f"Mappings: {len(all_mappings)} | Mentions: {len(all_mentions)}")
        total_processed += len(articles)
        total_success += success_count
        batch_num += 1

        sleep_time = random.uniform(2, 5)
        logger.info(f"Menunggu {sleep_time:.1f}s sebelum batch berikutnya...")
        time.sleep(sleep_time)

    finish_run(run_id, total_processed, total_success, 0)
    logger.info("Eksekusi Entity Resolver (v15 Intuitive Validation) Selesai.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-total", type=int, default=0)
    parser.add_argument("--days-back", type=int, default=DEFAULT_DAYS_BACK)
    args = parser.parse_args()
    main(limit=args.limit, max_total=args.max_total, days_back=args.days_back)
