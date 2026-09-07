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
import threading  # FIX RC#1: for lazy-loading lock
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

# v4.3: Stanza Coref Pipeline untuk attribution check
# Coref resolver — untuk filter "speaker_not_target" cases
# (mis. "Erick mengatakan X" → Erick = speaker, bukan target sentiment)
# FIX RC#1 (CRITICAL): Add threading.Lock to prevent concurrent model loading.
# Before: 4 threads could simultanously pass `if NLP_COREF is None` check →
# load 4x Stanza Coref (~1GB each) = ~4GB OOM crash.
# After: Double-checked locking with threading.Lock ensures single load.
_MODEL_LOCK = threading.Lock()
NLP_COREF = None
def get_coref_pipeline():
    global NLP_COREF
    if NLP_COREF is None:  # fast path (no lock) — already loaded
        with _MODEL_LOCK:  # slow path — acquire lock
            if NLP_COREF is None:  # double-check inside lock
                try:
                    logger.info("Memuat Stanza Coref Pipeline (tokenize,pos,lemma,depparse,coref)...")
                    use_gpu = torch.cuda.is_available()
                    NLP_COREF = stanza.Pipeline(
                        'id',
                        processors='tokenize,pos,lemma,depparse,coref',
                        verbose=False, use_gpu=use_gpu, batch_size=16
                    )
                except Exception as e:
                    logger.warning(f"Coref pipeline load failed (attribution check disabled): {e}")
                    NLP_COREF = None
    return NLP_COREF

# v4.3: KeyBERT untuk topic dominance check
# Cek apakah entity adalah topik utama context (bukan cuma disebut)
_KW_MODEL = None
def get_keybert_model():
    global _KW_MODEL
    if _KW_MODEL is None:  # fast path
        with _MODEL_LOCK:  # slow path
            if _KW_MODEL is None:  # double-check
                try:
                    from keybert import KeyBERT
                    logger.info("Memuat KeyBERT model (indobenchmark/indobert-base-p1)...")
                    _KW_MODEL = KeyBERT(model="indobenchmark/indobert-base-p1")
                except Exception as e:
                    logger.warning(f"KeyBERT load failed (topic check disabled): {e}")
                    _KW_MODEL = None
    return _KW_MODEL

# v4.4: Scoring constants (FIX cacat #4 — documented & configurable)
# Justification: these weights were tuned empirically on v17→v18 dataset.
# attr_score=40 because sentiment predicates (kritik/puji) are the strongest
#   signal that context has entity-level sentiment.
# actor_score=30 because is_main_actor (entity = primary subject) is the
#   second strongest signal for attribution.
# pos_score=20/12/5 because paragraph 0 (lead) has highest editorial weight,
#   paragraphs 1-2 medium, later paragraphs lower.
# exclusivity_score=10/5/0 because single-entity sentences are cleaner than
#   crowded multi-entity sentences.
#
# Env var overrides for tuning without code change:
import os as _os
ATTR_SCORE_SENTIMENT = int(_os.environ.get("ATTR_SCORE_SENTIMENT", "40"))
ATTR_SCORE_ATTRIBUTION = int(_os.environ.get("ATTR_SCORE_ATTRIBUTION", "10"))
ACTOR_SCORE_MAIN = int(_os.environ.get("ACTOR_SCORE_MAIN", "30"))
POS_SCORE_LEAD = int(_os.environ.get("POS_SCORE_LEAD", "20"))
PRECISION_BONUS_TARGET = int(_os.environ.get("PRECISION_BONUS_TARGET", "15"))
PRECISION_BONUS_DOMINANT = int(_os.environ.get("PRECISION_BONUS_DOMINANT", "10"))
PRECISION_PENALTY_DOER = int(_os.environ.get("PRECISION_PENALTY_DOER", "-20"))

# v18: Load relevancy model for pre-filtering
RELEVANCY_MODEL_ID = "apriandito/indobert-relevancy-classifier"
RELEVANCY_THRESHOLD = 0.5
_relevancy_pipeline = None

# v23: Quality filter constants
PROFILE_PATTERNS_V23 = [
    r'(?i)merupakan\s+(seorang\s+)?(tokoh|ulama|politisi|ekonom|pengusaha|aktivis|jurnalis|akademisi)',
    r'(?i)lahir\s+(pada|di)\s+[\w\d]',  # FIX EC#8: match word OR digit (was \d only)
    r'(?i)putra\s+(dari|ke-)',
    r'(?i)menjabat\s+sebagai\s+(Menteri|Gubernur|Walikota|Bupati|Ketua|Direktur)\s+(pada|di|tahun)\s+\d',
    r'(?i)perjalanan\s+(karier|politik)',
    r'(?i)berikut\s+(profil|biografi|perjalanan)',
    r'(?i)\binformasi\s+pribadi\b',
]

SENTIMENT_KEYWORDS_V23 = {
    'puji', 'dipuji', 'memuji', 'apresiasi', 'berhasil', 'menang', 'prestasi',
    'penghargaan', 'mendukung', 'kritik', 'dikritik', 'mengkritik', 'korupsi',
    'tersangka', 'divonis', 'ditahan', 'dicopot', 'mundur', 'gagal', 'skandal',
    'menolak', 'kecewa', 'menyatakan', 'mengatakan', 'menegaskan', 'mengimbau',
    'klarifikasi', 'menjelaskan', 'melantik',
}

def is_profile_sentence_v23(sentence):
    import re
    for pattern in PROFILE_PATTERNS_V23:
        if re.search(pattern, sentence):
            return True
    return False

def is_redundant_v23(sentence, previous_sentences, threshold=0.5):
    if not previous_sentences:
        return False
    def get_words(s):
        return set(w.lower().strip('.,;:!?()\"\'[]{}') for w in s.split() if len(w) > 2)
    sent_words = get_words(sentence)
    if not sent_words:
        return False
    # FIX OB#8 (LOW): Only apply overlap check for sentences > 5 words.
    # Before: short sentences (3-4 words) could false-match on 2 shared words.
    # After: skip overlap check for short sentences (use Jaccard only).
    sent_is_short = len(sent_words) <= 5
    for prev in previous_sentences:
        prev_words = get_words(prev)
        if not prev_words:
            continue
        intersection = len(sent_words & prev_words)
        union = len(sent_words | prev_words)
        if union > 0 and intersection / union >= threshold:
            return True
        # Only apply overlap check if both sentences are long enough
        if not sent_is_short and len(prev_words) > 5:
            overlap = intersection / min(len(sent_words), len(prev_words))
            if overlap >= 0.6:
                return True
    return False

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
    """Run relevancy model on (entity, context) pair. Returns prob[relevant].

    BUG N6 FIX: normalize entity_name ke format "Tentang {entity}" untuk
    match dengan training v4. Sebelumnya kirim entity_name langsung.
    """
    pipe = get_relevancy_pipeline()
    if pipe is None:
        return 1.0  # if model unavailable, don't filter (fail-open)
    tok, model = pipe
    device = next(model.parameters()).device
    # BUG N6 FIX: normalize premise ke format training v4
    # Training: premise = "Tentang Erick Thohir"
    # Production: entity_name = "Erick Thohir" → normalize → "Tentang Erick Thohir"
    import os as _os
    prefix = _os.environ.get("NLP_PREMISE_PREFIX", "Tentang ")
    premise = entity_name
    if prefix and not entity_name.strip().lower().startswith(prefix.strip().lower()):
        premise = f"{prefix}{entity_name.strip()}"
    enc = tok(premise, context_text, truncation=True, max_length=256, return_tensors="pt").to(device)
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


# ---------------------------------------------------------------------------
# v4.4: PRECISION BOOST — Coreference + Topic Dominance (FIXED LOGIC)
# ---------------------------------------------------------------------------
# Cacat logika yang DIPERBAIKI:
#
# Cacat #2 (FIXED): HAPUS manual verb lists (SENTIMENT_PREDICATES, ATTRIBUTION_VERBS).
#   Sebelumnya: 20 sentiment verbs + 10 attribution verbs hardcoded.
#   Sekarang: Pure Stanza dependency parsing — find entity's grammatical role
#   (subject/object/unknown). No manual verb list needed.
#
# Cacat #3 (FIXED): Fail-open ambiguity.
#   Sebelumnya: is_target=True untuk "no_predicate_found" (ambiguous).
#   Sekarang: 3-state return: "object" (target), "subject" (doer), "unknown".
#
# Cacat #1 (FIXED below in integration): AND logic terlalu rigid.
#   Sebelumnya: Layer1 AND Layer3 (redundant topic checks).
#   Sekarang: (Layer1 OR Layer3) AND Layer2.
# ---------------------------------------------------------------------------

def analyze_entity_role(entity_name: str, context_text: str) -> tuple[str, str]:
    """Determine entity's grammatical role via Stanza dependency parsing.

    PURE LIBRARY APPROACH — no manual verb lists needed.

    Uses Stanza coref + depparse to find entity's role relative to the
    root verb of each sentence:
      - "subject": entity = doer/actor (mis. "Erick mengkritik X")
        → sentiment in context is BY entity, not TOWARD entity
        → is_target = False (entity is speaker/actor)
      - "object": entity = target/patient (mis. "Erick dikritik X")
        → sentiment in context is TOWARD entity
        → is_target = True (entity is sentiment target)
      - "unknown": can't determine role
        → fail-open: is_target = True (don't block)

    Returns:
        (role, verb_lemma): role in {"subject", "object", "unknown"}
    """
    nlp_coref = get_coref_pipeline()
    if nlp_coref is None:
        return "unknown", "coref_unavailable"  # fail-open

    try:
        doc = nlp_coref(context_text)
        entity_lower = entity_name.lower().strip()

        # FIX OB#7: Coref cluster tracking moved below (replaced old single-cluster map).

        def is_entity_match(word_text: str) -> bool:
            """Check if a word refers to the entity (direct or via coref).

            FIX OB#6 (MEDIUM): Tighter matching to prevent false positives.
            Before: `entity_lower in wt or wt in entity_lower` — substring check.
              Bug: "Anies" matched "anieskan" (false positive).
                   "Joko Widodo" matched "joko" (different person).
            After: Word-level token overlap for multi-word entities.
            """
            wt = word_text.lower().strip()
            if not wt:
                return False
            # For single-word entities, require exact word match (not substring)
            entity_words = entity_lower.split()
            wt_words = wt.split()
            # Exact match (fast path)
            if entity_lower == wt:
                return True
            # Multi-word entity: all entity words must be in word_text's tokens
            if len(entity_words) > 1:
                return all(ew in wt_words for ew in entity_words)
            # Single-word entity: exact match only (no substring)
            return entity_words[0] == wt

        # FIX OB#7 (MEDIUM): Handle coref cluster collision for common pronouns.
        # Before: mention_to_cluster[t] = cid → last cluster wins for "dia"/"ia".
        # After: Track all clusters per mention; if ambiguous, return "unknown".
        cluster_to_mentions = {}  # cluster_id → set of mention texts
        mention_to_clusters = {}  # mention_text → set of cluster_ids
        if hasattr(doc, 'coref') and doc.coref:
            for cid, cluster in enumerate(doc.coref):
                texts = set()
                for mention in cluster.mentions:
                    texts.add(mention.text.lower())
                cluster_to_mentions[cid] = texts
                for t in texts:
                    mention_to_clusters.setdefault(t, set()).add(cid)

        def resolve_coref_cluster(word_text: str):
            """Return cluster_id if unambiguous, None if ambiguous or not in any cluster."""
            wt = word_text.lower().strip()
            clusters = mention_to_clusters.get(wt, set())
            if len(clusters) == 0:
                return None
            if len(clusters) > 1:
                return None  # ambiguous — multiple clusters claim this pronoun
            return next(iter(clusters))  # exactly one cluster

        for sent in doc.sentences:
            words = sent.words

            # Find root verb of sentence
            root_verb = None
            for word in words:
                if word.deprel == "root" and word.upos in ("VERB", "AUX"):
                    root_verb = word
                    break
            if root_verb is None:
                # Try any root (might be noun phrase)
                for word in words:
                    if word.deprel == "root":
                        root_verb = word
                        break
            if root_verb is None:
                continue

            root_lemma = root_verb.lemma or root_verb.text or "unknown"

            # Check entity's role relative to root verb
            for word in words:
                # FIX OB#6/OB#7: Check direct match OR coref resolution
                matched = is_entity_match(word.text)
                if not matched:
                    # Try coref: is this word a pronoun that refers to entity?
                    cid = resolve_coref_cluster(word.text)
                    if cid is not None:
                        # Check if entity name is in this cluster's mentions
                        cluster_texts = cluster_to_mentions.get(cid, set())
                        if any(is_entity_match(ct) for ct in cluster_texts):
                            matched = True
                if not matched:
                    continue
                # Entity found — check its dependency role
                if word.deprel in ("nsubj", "nsubj:pass", "csubj"):
                    # Entity is subject (doer/actor)
                    # For passive (nsubj:pass): entity is actually the PATIENT
                    if word.deprel == "nsubj:pass":
                        return "object", f"passive_subject({root_lemma})"
                    return "subject", f"subject({root_lemma})"
                elif word.deprel in ("obj", "obl", "nmod", "iobj"):
                    # Entity is object (target/patient)
                    return "object", f"object({root_lemma})"

        return "unknown", "no_role_found"  # fail-open
    except Exception as e:
        # FIX SF#4 (MEDIUM): upgrade debug→warning so errors are visible.
        # Before: logger.debug() → invisible at default INFO level.
        # After: logger.warning() → user sees when coref layer is disabled.
        logger.warning(f"Coref role analysis error for entity '{entity_name}': {e}")
        return "unknown", "coref_error"  # fail-open


def is_dominant_topic(entity_name: str, context_text: str,
                      top_n: int = 5, threshold: float = 0.25) -> tuple[bool, float]:
    """Cek apakah entity adalah topik utama context (bukan cuma disebut).

    Uses KeyBERT untuk extract keywords, lalu cek apakah entity muncul
    di top-N keywords dengan score >= threshold.

    Returns:
        (is_dominant, top_score): is_dominant=True jika entity topik utama
    """
    kw_model = get_keybert_model()
    if kw_model is None:
        return True, 1.0  # fail-open: assume dominant

    try:
        entity_lower = entity_name.lower().strip()
        # Extract keywords (1-2 gram untuk capture "Erick Thohir")
        keywords = kw_model.extract_keywords(
            context_text,
            keyphrase_ngram_range=(1, 2),
            top_n=top_n,
            stop_words=None  # Indonesian not in default stop words
        )

        if not keywords:
            return True, 0.5  # fail-open

        # Cek apakah entity muncul di top keywords
        for kw_text, kw_score in keywords:
            if entity_lower in kw_text.lower() or kw_text.lower() in entity_lower:
                return kw_score >= threshold, kw_score

        # Entity tidak ada di top-N keywords → not dominant
        return False, 0.0
    except Exception as e:
        # FIX SF#4 (MEDIUM): upgrade debug→warning so errors are visible.
        logger.warning(f"KeyBERT topic analysis error for entity '{entity_name}': {e}")
        return True, 0.5  # fail-open

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
    """Return paragraph index for the given character offset.

    FIX EC#7 (MEDIUM): Fallback for single-paragraph articles (no \\n\\n).
    Before: `text[:offset].count('\\n\\n')` → always 0 for enriched articles
    (enricher_worker joins sentences with spaces, not \\n\\n).
    After: Estimate paragraph from sentence boundaries if no \\n\\n found.
    """
    para_count = text[:offset].count('\n\n')
    if para_count > 0:
        return para_count
    # FIX EC#7: Estimate paragraph index from sentence count.
    # Assume ~5 sentences per paragraph as fallback.
    text_up_to = text[:offset]
    sentence_count = text_up_to.count('. ') + text_up_to.count('! ') + text_up_to.count('? ')
    return sentence_count // 5

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
                # FIX CW#6 (MEDIUM): always advance next_idx, even if sentence skipped.
                # Before: short sentence → next_idx not incremented → loop stuck.
                if len(next_sent["text"]) > 20:
                    context_parts.append(next_sent["text"])
                    next_added += 1
                    added_this_round = True
                next_idx += 1  # FIX: always advance regardless of append

            if prev_idx >= 0 and prev_added < max_each_side:
                prev_sent = sentences[prev_idx]
                if len(prev_sent["text"]) > 20:
                    context_parts.insert(0, prev_sent["text"])
                    prev_added += 1
                    added_this_round = True
                # FIX CW#6: same fix for prev_idx
                prev_idx -= 1

            # FIX OB#5 (MEDIUM): Don't break prematurely when both sentences are short.
            # Before: `if not added_this_round: break` stops loop even if longer
            # sentences exist further away. Now only break if we've exhausted
            # both sides (prev_idx < 0 AND next_idx >= len(sentences)).
            if next_idx >= len(sentences) and prev_idx < 0:
                break
            # Also break if max capacity reached (handled by current_chars check above)

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

        # v18: QUALITY_SCORE — sentiment predicate gets 40, attribution gets 10, none gets 5
        # FIX CW#4 (MEDIUM): dead code — both branches returned ATTR_SCORE_ATTRIBUTION.
        # Now: no predicate → lower score (5), differentiating from attribution (10).
        attr_score = ATTR_SCORE_SENTIMENT if has_sentiment_predicate else (ATTR_SCORE_ATTRIBUTION if has_attribution else 5)
        actor_score = ACTOR_SCORE_MAIN if is_main_actor else 10
        pos_score = POS_SCORE_LEAD if para_idx == 0 else (12 if para_idx <= 2 else 5)
        exclusivity_score = 10 if not is_crowded else (5 if used_local_clause else 0)
        quality_score = attr_score + actor_score + pos_score + exclusivity_score

        # v4.4: Coref-based role analysis (PURE LIBRARY — no manual verb lists)
        # Determine entity's grammatical role: subject (doer) / object (target) / unknown
        entity_role, role_reason = analyze_entity_role(entity_name, ctx_text)

        # v4.4: KeyBERT topic dominance check
        is_dominant, topic_score = is_dominant_topic(entity_name, ctx_text)

        # v4.4: FIX CACAT #3 — 3-state attribution (no ambiguity)
        # role="object" → confirmed target (is_target=True, high confidence)
        # role="subject" → confirmed doer (is_target=False, entity is speaker/actor)
        # role="unknown" → can't determine (is_target=True, fail-open but flagged)
        is_target = entity_role != "subject"  # True for "object" and "unknown"
        is_confirmed_target = entity_role == "object"  # only True for confirmed

        # v4.4: Precision bonus (configurable via env vars)
        precision_bonus = 0
        if is_confirmed_target:
            precision_bonus += PRECISION_BONUS_TARGET  # +15: entity confirmed as target
        elif entity_role == "subject":
            precision_bonus += PRECISION_PENALTY_DOER  # -20: entity is doer (speaker)
        if is_dominant:
            precision_bonus += PRECISION_BONUS_DOMINANT  # +10: entity is dominant topic
        quality_score = max(0, quality_score + precision_bonus)

        # v18: relevancy pre-filter (Layer 1 — kept, not deleted per user request)
        relevancy_score = check_relevancy(entity_name, ctx_text)

        # v4.4: FIX CACAT #1 — correct AND/OR logic
        #
        # CACAT SEBELUMNYA (FLAWED):
        #   is_relevant = model_relevant AND not_speaker AND (is_dominant OR has_sentiment_predicate)
        #   → Layer 1 (model) AND Layer 3 (KeyBERT) = REDUNDANT (both check topic)
        #   → If model says relevant but KeyBERT says not dominant → is_relevant=False (WRONG)
        #
        # FIX (CORRECT):
        #   Layer 1 (model) OR Layer 3 (KeyBERT) → topic_relevant (either signal suffices)
        #   Layer 2 (coref role) → attribution check (hard filter: subject = doer = not target)
        #   is_relevant = topic_relevant AND attribution_ok
        #
        # Logic:
        #   - topic_relevant: True if relevancy model OR KeyBERT confirms topic
        #   - attribution_ok: True if entity is target (object) or unknown (fail-open)
        #                     False if entity is confirmed doer (subject = speaker)
        model_relevant = relevancy_score >= RELEVANCY_THRESHOLD
        topic_relevant = model_relevant or is_dominant  # FIX: OR not AND
        attribution_ok = is_target  # True for object/unknown, False for subject
        is_relevant_final = topic_relevant and attribution_ok

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
            "is_relevant": is_relevant_final,
            # v4.4: precision boost metadata (3-state, no ambiguity)
            "entity_role": entity_role,  # "subject" | "object" | "unknown"
            "role_reason": role_reason,
            "is_target": is_target,
            "is_confirmed_target": is_confirmed_target,
            "is_dominant_topic": is_dominant,
            "topic_score": round(topic_score, 3),
            "precision_bonus": precision_bonus,
            # v4.4: layer breakdown for debugging
            "layer1_model_relevant": model_relevant,
            "layer2_attribution_ok": attribution_ok,
            "layer3_topic_dominant": is_dominant,
        }

        # v23: Apply quality filter — remove profile & redundant sentences
        # FIX CW#7 (MEDIUM): split on all sentence terminators (. ! ?), not just ". "
        import re as _re
        ctx_sentences = _re.split(r'(?<=[.!?])\s+', ctx_text)
        # FIX CW#5 (HIGH): protect anchor sentence from being filtered out.
        # Before: v23 filter could remove anchor sentence (which contains entity
        # mention), leaving context without entity → sentiment analysis fails.
        # After: anchor sentence is always kept, even if it matches profile/redundant.
        anchor_lower = anchor_text_for_context.lower().strip()
        filtered_sentences = []
        for sent in ctx_sentences:
            sent = sent.strip()
            if not sent:
                continue
            # FIX CW#5: never filter anchor sentence (contains entity mention)
            is_anchor = sent.lower().strip() == anchor_lower or anchor_lower in sent.lower()
            if is_anchor:
                filtered_sentences.append(sent)
                continue
            # Skip profile sentences
            if is_profile_sentence_v23(sent):
                continue
            # Skip redundant sentences
            if is_redundant_v23(sent, filtered_sentences):
                continue
            filtered_sentences.append(sent)
        ctx_text = '. '.join(filtered_sentences)
        if ctx_text and not ctx_text.endswith('.'):
            ctx_text += '.'

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
    total_failed = 0  # FIX CW#2: track failed count for finish_run
    batch_num = 1
    logger.info(f"[CONTEXT_WORKER v18] Precision Multi | Limit: {limit}/batch | Days back: {days_back} | Threads: {MAX_NLP_WORKERS}")
    while True:
        if max_total > 0 and total_processed >= max_total: break
        current_limit = min(limit, max_total - total_processed) if max_total > 0 else limit
        try:
            time_filter = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
            res = sb.table("raw_texts") \
                    .select("id, title, text, ingested_month, content_type") \
                    .eq("status", pc.STATUS_VALIDATED) \
                    .not_.is_("entity_resolved_at", "null") \
                    .is_("context_extracted_at", "null") \
                    .neq("text", "") \
                    .neq("content_type", "SNIPPET") \
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
            # FIX SF#6 (MEDIUM): Log error + max retries to prevent infinite loop.
            # Before: silent `time.sleep(5); continue` → infinite loop if query persistently fails.
            # After: log error, increment retry counter, break after 3 consecutive failures.
            logger.error(f"Mentions fetch failed for batch {len(art_ids)} articles: {e}")
            _mentions_fetch_failures = getattr(main, '_mentions_fetch_failures', 0) + 1
            setattr(main, '_mentions_fetch_failures', _mentions_fetch_failures)
            if _mentions_fetch_failures >= 3:
                logger.error(f"Mentions fetch failed {_mentions_fetch_failures} consecutive times — breaking to prevent infinite loop")
                break
            time.sleep(5)
            continue
        # Reset failure counter on success
        setattr(main, '_mentions_fetch_failures', 0)
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
        # OBSERVABILITY: Detailed batch summary
        failed_ctx = len(articles) - len(succeeded_art_ids)
        total_spans = sum(len(c.get("metadata", {}).get("all_spans", [])) for c in context_inserts)
        relevant_count = sum(1 for c in context_inserts if c.get("metadata", {}).get("is_relevant", True))
        avg_quality = sum(c.get("metadata", {}).get("quality_score", 0) for c in context_inserts) / max(1, len(context_inserts))
        
        logger.info(f"{'='*60}")
        logger.info(f"BATCH {batch_num} CONTEXT SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"  Articles processed:  {len(articles)}")
        logger.info(f"  ✅ Succeeded:         {len(succeeded_art_ids)}")
        logger.info(f"  ❌ Failed/Skipped:     {failed_ctx}")
        logger.info(f"  📝 Contexts created:  {len(context_inserts)}")
        logger.info(f"  📍 Total spans:      {total_spans}")
        logger.info(f"  ✅ Relevant:          {relevant_count}")
        logger.info(f"  🎯 Avg quality:       {avg_quality:.1f}")
        if failed_ctx > 0:
            logger.warning(f"  ⚠️ {failed_ctx} articles failed — see SKIP/ERROR logs above")
        logger.info(f"{'='*60}")
        total_processed += len(articles)
        # FIX CW#1 (HIGH): total_success should count articles (succeeded_art_ids),
        # not entity contexts (context_inserts can be > articles due to multi-entity).
        total_success += len(succeeded_art_ids)
        # FIX CW#2 (MEDIUM): track actual failed count, not hardcoded 0.
        total_failed += failed_ctx
        batch_num += 1
    # FIX CW#1/CW#2: pass correct succeeded (article-level) and failed counts.
    finish_run(run_id, total_processed, total_success, total_failed)
    logger.info("Eksekusi Context Worker (v18 Precision Multi) Selesai.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-total", type=int, default=0)
    parser.add_argument("--days-back", type=int, default=DEFAULT_DAYS_BACK)
    args = parser.parse_args()
    main(limit=args.limit, max_total=args.max_total, days_back=args.days_back)
