"""
entity_resolution_worker.py v15 — Intuitive Main-Entity Validation
===================================================================
User's intuitive logic:
  1. Title contains entity candidate, BUT must verify in body if truly main.
  2. Check if entity is from correct era (time period).
  3. Check if entity affiliation (party/position) matches article.
  4. "Naluriah" approach: title is HINT, body is PROOF.

v15 improvements over v14.2:
  1. TITLE CANDIDATE VALIDATION: title entities are CANDIDATES, must confirm in body.
  2. ERA VALIDATION: check entity era vs article era markers.
  3. AFFILIATION VALIDATION: check party/position match.
  4. INTUITIVE SCORING: body evidence > title evidence.
"""
import re, time, random, logging, argparse, json
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

RESOLVER_VERSION = "v15_intuitive_validation"
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

# v15: EXPANDED verb sets (lemma forms)
SENTIMENT_PRED_ACTIVE = {
    "kritik","kecam","sindir","serang","hina","cela","ejek","tuding",
    "tuduh","lapor","cekal","tahan","vonis","tangkap","pidana","anggap",
    "nilai","sorot","gugur","bongkar","pecat","mundur","undur","berhenti",
    "ganti","razia","sita","denda","hukum","ganjar",
    "puji","dukung","apresiasi","restui","sahkan","setuju","kukuhkan",
    "akui","legitimasi","bela","tolak","keberatan","menentang",
    "pandang","sikapi","persepsi","ungkap",
}
ATTRIBUTION_VERBS = {
    "kata","nyata","tegas","jelaskan","tambah","imbau","ingat","sampai",
    "aku","klaim","nilai","ungkap","jawab","ujar","tutur","sebut","papar",
    "ucap","sampaikan","katakan","ungkapkan","nyatakan","tegaskan",
    "tambahkan","imbaukan","ingatkan","balas","tanggapi",
    "saran","menyaran","rekomendasi","usul","ajak","mengajak",
    "pinta","minta","meminta","perintah","wantiwanti",
    "tekan","tekankan","menekankan","sorot","soroti","tandai","tanda",
    "tunjuk","menunjuk",
}

TOPIC_DOMINANCE_THRESHOLD = 0.25
MIN_BODY_MENTIONS_FOR_MAIN = 1


def normalize_name(name):
    return re.sub(r'\s+', ' ', name).strip()


def load_caches(sb):
    """Load entity DB with era + affiliation for v15 validation."""
    logger.info("Loading caches (dengan era + affiliation)...")
    pe_res = sb.table("political_entities").select(
        "id, canonical_name, aliases, entity_type, party_affiliation, position, era"
    ).execute()

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
            "era": r.get("era") or [],
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

    return alias_map, entity_db_map, id_to_name, id_to_entity, regex_patterns


def is_false_positive(matched_text, canonical_name, full_persons):
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


def check_semantic_role(sent, entity_start, entity_end):
    result = {"has_sentiment": False, "has_attribution": False, "verb": None, "role": None}
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
        result["role"] = entity_word.deprel
        head_id = entity_word.head
        for word in sent.words:
            if word.id == head_id:
                lemma = (word.lemma or word.text).lower()
                if lemma in SENTIMENT_PRED_ACTIVE:
                    result["has_sentiment"] = True
                    result["verb"] = lemma
                elif lemma in ATTRIBUTION_VERBS:
                    result["has_attribution"] = True
                break
    return result


# ---------------------------------------------------------------------------
# v15 NEW: Era validation
# ---------------------------------------------------------------------------
def check_era_compatibility(article_text, entity_eras):
    """Check if article mentions eras compatible with entity's era."""
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
        return True, None

    entity_eras_lower = [e.lower() for e in entity_eras]
    for detected in detected_eras:
        if any(detected in e for e in entity_eras_lower):
            return True, detected

    return False, detected_eras[0]


# ---------------------------------------------------------------------------
# v15 NEW: Affiliation validation
# ---------------------------------------------------------------------------
def check_affiliation(article_text, entity_info):
    """Check if party/position mentioned in article matches DB."""
    party = entity_info.get("party")
    if not party:
        return True, None

    text_lower = article_text.lower()
    party_lower = party.lower()

    if party_lower in text_lower:
        return True, party

    party_abbr = {
        "PDI-P": ["pdip", "pdi-p", "pdiperjuangan"],
        "Gerindra": ["gerindra"],
        "Golkar": ["golkar"],
        "Demokrat": ["demokrat", "partai demokrat"],
        "PKB": ["pkb", "kebangkitan bangsa"],
        "PAN": ["pan", "amanat nasional"],
        "PKS": ["pks", "keadilan sejahtera"],
        "Nasdem": ["nasdem", "nasional demokrat"],
        "PPP": ["ppp", "persatuan pembangunan"],
        "PBB": ["pbb", "bulan bintang"],
        "Independen": ["independen"],
    }

    abbrs = party_abbr.get(party, [party_lower])
    for abbr in abbrs:
        if abbr in text_lower:
            return True, party

    return None, None  # None = inconclusive


# ---------------------------------------------------------------------------
# v15: Main entity resolution with INTUITIVE VALIDATION
# ---------------------------------------------------------------------------
def process_single_article_entity(art, alias_map, entity_db_map, id_to_name,
                                   id_to_entity, regex_patterns):
    """v15: title-candidate validation + era + affiliation check."""
    title = (art.get('title') or '').strip()
    body = (art.get('text') or '').strip()
    title_lower = title.lower()
    metadata = art.get("metadata") or {}
    ingested_month = art.get("ingested_month")

    if not body or len(body) < 50:
        return None

    full_text = body  # detection on body only

    try:
        doc = NLP(full_text)
    except Exception as e:
        logger.error(f"Stanza Error: {e}")
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
        for match in pattern.finditer(full_text):
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
                    "has_sentiment": False,
                    "has_attribution": False,
                    "sentiment_verbs": [],
                    "roles": [],
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
                    if role["has_sentiment"]:
                        entity_data[ent_id]["has_sentiment"] = True
                        entity_data[ent_id]["sentiment_verbs"].append(role["verb"])
                    if role["has_attribution"]:
                        entity_data[ent_id]["has_attribution"] = True
                    if role["role"]:
                        entity_data[ent_id]["roles"].append(role["role"])
                    break
        last_end = end

    if configured_entity_id and configured_entity_id not in entity_data:
        entity_data[configured_entity_id] = {
            "count": 0,
            "in_title": id_to_name.get(configured_entity_id, "").lower() in title_lower,
            "in_body": False,
            "sentence_indices": set(),
            "has_sentiment": False,
            "has_attribution": False,
            "sentiment_verbs": [],
            "roles": [],
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

    # v15: INTUITIVE RANKING — body evidence + era + affiliation first
    def intuitive_salience_key(item):
        ent_id, data = item
        return (
            data["has_sentiment"],
            data["topic_dominance"] >= TOPIC_DOMINANCE_THRESHOLD,
            data["era_compatible"],
            data["affiliation_match"] is not False,
            data["in_body"] and data["count"] >= MIN_BODY_MENTIONS_FOR_MAIN,
            data["in_title"],
            data["count"],
        )

    ranked = sorted(entity_data.items(), key=intuitive_salience_key, reverse=True)

    # v15: SALIENCE GATE — stricter with era + affiliation
    valid_entities = []
    for ent_id, data in ranked:
        if not data["in_body"] and data["count"] == 0:
            continue
        if data["has_sentiment"]:
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

        if data["has_sentiment"]:
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

        entity_info = data.get("entity_info", {})
        mappings.append({
            "entity_id": ent_id,
            "is_main_entity": is_main,
            "confidence": round(conf, 3),
            "resolver_source": data["src"],
            "has_sentiment_role": data["has_sentiment"],
            "has_attribution_role": data["has_attribution"],
            "topic_dominance": round(data["topic_dominance"], 3),
            "sentiment_verbs": list(set(data["sentiment_verbs"])),
            "era_compatible": data["era_compatible"],
            "detected_era": data["detected_era"],
            "affiliation_match": data["affiliation_match"],
            "entity_era": entity_info.get("era", []),
            "entity_party": entity_info.get("party"),
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
        main = mappings[0]
        logger.info(f"Main: {id_to_name.get(main['entity_id'],'?')} "
                    f"(sent={main['has_sentiment_role']}, dom={main['topic_dominance']}, "
                    f"era={main['era_compatible']}, affil={main['affiliation_match']}, "
                    f"conf={main['confidence']})")
    else:
        logger.info("Resolved: 0 entities (Skipped)")

    return {
        "raw_text_id": art["id"],
        "ingested_month": ingested_month,
        "mappings": mappings,
        "mentions": mentions,
    }


def process_articles_batch(articles, alias_map, entity_db_map, id_to_name,
                            id_to_entity, regex_patterns):
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
                logger.error(f"Thread crashed: {e}")
    return results


def chunked_upsert_tracked(sb, table_name, data, on_conflict, chunk_size=50):
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


def main(limit=50, max_total=0, days_back=DEFAULT_DAYS_BACK):
    sb = get_client()
    run_id = start_run("entity_resolution_worker", RESOLVER_VERSION)

    alias_map, entity_db_map, id_to_name, id_to_entity, regex_patterns = load_caches(sb)
    logger.info(f"Loaded {len(regex_patterns)} patterns, {len(entity_db_map)} entities (with era+affiliation)")

    total_processed = 0
    total_success = 0

    logger.info(f"[ENTITY_RESOLVER v15] Intuitive Validation | Threads: {MAX_NLP_WORKERS}")

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
            logger.warning(f"DB Query Error: {e}. Menunggu 10s...")
            time.sleep(10)
            continue

        articles = res.data or []
        if not articles:
            break

        logger.info(f"Memproses {len(articles)} artikel dengan Intuitive Validation...")
        batch_results = process_articles_batch(
            articles, alias_map, entity_db_map, id_to_name, id_to_entity, regex_patterns
        )

        all_mappings = []
        all_mentions = []
        now_iso = datetime.now(timezone.utc).isoformat()
        succeeded_ids = {r["raw_text_id"] for r in batch_results}

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

        resolved_updates = [{"id": rid, "entity_resolved_at": now_iso,
                            "resolver_version": RESOLVER_VERSION} for rid in succeeded_ids]
        if resolved_updates:
            for i in range(0, len(resolved_updates), 25):
                try:
                    sb.rpc("bulk_update_raw_texts", {"p_updates": resolved_updates[i:i+25]}).execute()
                except Exception as e:
                    logger.error(f"Status Update Error: {e}")

        logger.info(f"{len(succeeded_ids)}/{len(articles)} resolved. Mappings: {len(all_mappings)} | Mentions: {len(all_mentions)}")
        total_processed += len(articles)
        total_success += len(succeeded_ids)

        sleep_time = random.uniform(2, 5)
        time.sleep(sleep_time)

    finish_run(run_id, total_processed, total_success, 0)
    logger.info("Entity Resolver v15 (Intuitive Validation) Selesai.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-total", type=int, default=0)
    parser.add_argument("--days-back", type=int, default=DEFAULT_DAYS_BACK)
    args = parser.parse_args()
    main(limit=args.limit, max_total=args.max_total, days_back=args.days_back)
