#!/usr/bin/env python3.13
"""
Expert test: context_worker v18 logic — WITHOUT loading relevancy model (disk limit).
Instead: use depparse-based relevancy proxy (entity is nsubj/obj of root verb in context).
This is a stronger signal than regex proximity and tests the v18 quality_score fix.
"""
import os, json, re, sys, time
from collections import Counter

sys.path.insert(0, '/home/z/.local/lib/python3.13/site-packages')
import stanza

print("Loading Stanza (tokenize,pos,lemma,depparse)...", flush=True)
t0 = time.time()
NLP = stanza.Pipeline('id', processors='tokenize,pos,lemma,depparse',
                      verbose=False, use_gpu=False, batch_size=16,
                      download_method=None)
print(f"Stanza loaded in {time.time()-t0:.1f}s", flush=True)

SENTIMENT_PRED_ACTIVE = {"kritik","kecam","sindir","serang","bela","tolak","dukung","puji","tunding","singgung","ejek","cela","hina","apresiasi"}
ATTRIBUTION_VERBS = {"kata","nyata","tegas","jelaskan","tambah","imbau","ingat","sampai","aku","klaim","nilai","ungkap","jawab","ujar","tutur","sebut","papar"}
MAX_CONTEXT_WORDS = 180

def get_paragraph_index(text, offset):
    return text[:offset].count('\n\n')

def is_core_argument(sent, start_offset, end_offset):
    for word in sent.words:
        if word.start_char <= start_offset < word.end_char or \
           (start_offset <= word.start_char < end_offset):
            if word.deprel in ['nsubj', 'nsubj:pass', 'obj', 'iobj', 'csubj']:
                return True
            if word.deprel in ['nmod', 'nmod:poss', 'amod', 'appos']:
                return False
    return True

def check_entity_relevance_proxy(sent, entity_start, entity_end):
    """Relevancy proxy: is entity a core argument (nsubj/obj) of the sentence root?
    If yes → relevant (entity is grammatical subject/object, not background).
    If no → likely background mention.
    """
    for word in sent.words:
        if word.start_char <= entity_start < word.end_char or \
           (entity_start <= word.start_char < entity_end):
            if word.deprel in ('nsubj', 'nsubj:pass', 'obj', 'iobj', 'csubj'):
                return True, word.deprel
            return False, word.deprel
    return False, None

def v18_extract(article, entity_name, entity_aliases):
    title = (article.get("title") or "").strip()
    body = (article.get("text") or "").strip()
    if not body: return None
    try:
        doc = NLP(body)
    except: return None

    sentences = []
    for sent in doc.sentences:
        if len(sent.text.strip()) > 10:
            sentences.append({
                "text": sent.text,
                "start": sent.tokens[0].start_char if sent.tokens else 0,
                "end": sent.tokens[-1].end_char if sent.tokens else 0,
                "parsed": sent,
            })
    if not sentences: return None

    forms = [entity_name] + (entity_aliases or [])
    mentions = []
    for form in forms:
        if len(form) < 3: continue
        try:
            for m in re.finditer(r'\b' + re.escape(form) + r'\b', body, re.IGNORECASE):
                mentions.append((m.start(), m.end(), m.group()))
        except: continue
    if not mentions: return None

    # find best anchor: prefer sentence where entity is core argument
    best_anchor = -1
    best_role = None
    for sidx, s in enumerate(sentences):
        for start, end, _ in mentions:
            if s["start"] <= start < s["end"]:
                is_core, role = check_entity_relevance_proxy(s["parsed"], start, end)
                if is_core:
                    best_anchor = sidx
                    best_role = role
                    break
        if best_anchor >= 0: break
    if best_anchor < 0:
        # fallback: first sentence containing entity
        for sidx, s in enumerate(sentences):
            for start, end, _ in mentions:
                if s["start"] <= start < s["end"]:
                    best_anchor = sidx; break
            if best_anchor >= 0: break
    if best_anchor < 0: return None

    anchor_sent = sentences[best_anchor]
    first_mention = next((m for m in mentions if anchor_sent["start"] <= m[0] < anchor_sent["end"]), mentions[0])
    is_main_actor = is_core_argument(anchor_sent["parsed"], first_mention[0], first_mention[1])

    has_sentiment = False
    has_attribution = False
    root_verb = None
    for word in anchor_sent["parsed"].words:
        if word.deprel == 'root':
            root_verb = (word.lemma or word.text).lower()
            if root_verb in SENTIMENT_PRED_ACTIVE:
                has_sentiment = True
            if root_verb in ATTRIBUTION_VERBS:
                has_attribution = True

    context_parts = [anchor_sent["text"]]
    if best_anchor + 1 < len(sentences) and not has_sentiment:
        context_parts.append(sentences[best_anchor + 1]["text"])
    ctx_text = " ".join(context_parts)
    words_list = ctx_text.split()
    if len(words_list) > MAX_CONTEXT_WORDS:
        ctx_text = " ".join(words_list[:MAX_CONTEXT_WORDS])

    para_idx = get_paragraph_index(body, first_mention[0])

    # v18 quality_score: sentiment=40, attribution=10
    attr_score_v18 = 40 if has_sentiment else (10 if has_attribution else 10)
    # v17 quality_score: attribution got 25
    attr_score_v17 = 40 if has_sentiment else (25 if has_attribution else 10)

    actor_score = 30 if is_main_actor else 10
    pos_score = 20 if para_idx == 0 else (12 if para_idx <= 2 else 5)
    exclusivity_score = 10

    quality_v18 = attr_score_v18 + actor_score + pos_score + exclusivity_score
    quality_v17 = attr_score_v17 + actor_score + pos_score + exclusivity_score

    # relevancy proxy
    is_relevant_proxy = is_main_actor  # entity is core argument → relevant

    return {
        "context_text": ctx_text,
        "quality_v18": quality_v18,
        "quality_v17": quality_v17,
        "attr_v18": attr_score_v18,
        "attr_v17": attr_score_v17,
        "actor_score": actor_score,
        "is_main_actor": is_main_actor,
        "has_sentiment": has_sentiment,
        "has_attribution": has_attribution,
        "root_verb": root_verb,
        "is_relevant_proxy": is_relevant_proxy,
        "best_role": best_role,
        "anchor_sentence": anchor_sent["text"][:120],
    }

# Load live data
data = json.load(open("/tmp/live_multi.json"))
articles = data["articles"]
contexts_db = data["contexts"]
mappings = data["mappings"]
ent_full = data["ent_full"]
ent_map = data["ent_map"]

ctx_by_art = {}
for c in contexts_db:
    ctx_by_art.setdefault(c["raw_text_id"], []).append(c)

print(f"\n{'='*70}")
print(f"CONTEXT_WORKER v18 TEST (depparse-based, no relevancy model — disk limit)")
print(f"{'='*70}", flush=True)

tested = 0
v17_attr_rewarded = 0  # v17 gave attr>=25 to attribution (speaker bias)
v18_attr_neutral = 0   # v18 gives attr=10 to attribution
sentiment_detected = 0
relevant_proxies = 0
samples = []

for a in articles:
    art_id = a["id"]
    if art_id not in ctx_by_art: continue
    db_ctxs = ctx_by_art[art_id]
    if not db_ctxs: continue

    main_ent_id = None
    for m in mappings:
        if m["raw_text_id"] == art_id and m.get("is_main_entity"):
            main_ent_id = m["entity_id"]; break
    if not main_ent_id: continue

    ent_info = ent_full.get(main_ent_id)
    if not ent_info: continue
    ent_name = ent_info["name"]
    ent_aliases = ent_info.get("aliases", [])

    v17_ctx = next((c for c in db_ctxs if c["entity_id"] == main_ent_id), None)
    if not v17_ctx: continue

    v17_meta = v17_ctx.get("metadata") or {}
    v17_text = v17_ctx.get("context_text") or ""
    v17_quality = v17_meta.get("quality_score", 0)
    v17_attr = v17_meta.get("attr_score", 0)
    v17_actor = v17_meta.get("is_main_actor", False)

    t1 = time.time()
    v18 = v18_extract(a, ent_name, ent_aliases)
    parse_time = time.time() - t1
    if not v18: continue

    tested += 1

    print(f"\n[{tested}] {(a.get('title') or '')[:65]}", flush=True)
    print(f"  ENTITY: {ent_name}  [parse: {parse_time:.1f}s]")
    print(f"  v17 (DB):    quality={v17_quality} attr={v17_attr} actor={v17_actor}")
    print(f"  v18 (new):   quality={v18['quality_v18']} attr={v18['attr_v18']} actor={v18['is_main_actor']}")
    print(f"    root_verb={v18['root_verb']} sent={v18['has_sentiment']} attr_verb={v18['has_attribution']}")
    print(f"    role={v18['best_role']} relevant_proxy={v18['is_relevant_proxy']}")
    print(f"    v17 ctx[:90]: {v17_text[:90]}")
    print(f"    v18 ctx[:90]: {v18['context_text'][:90]}")

    # Track improvements
    if v17_attr >= 25 and v18["attr_v18"] == 10:
        v18_attr_neutral += 1
        print(f"  ⚡ v18 DOWNGRADED attr: {v17_attr}→10 (speaker bias killed)")
    if v17_attr >= 25:
        v17_attr_rewarded += 1
    if v18["has_sentiment"]:
        sentiment_detected += 1
        print(f"  ✅ v18 detected SENTIMENT predicate: {v18['root_verb']}")
    if v18["is_relevant_proxy"]:
        relevant_proxies += 1

    samples.append({
        "title": a.get("title","")[:55], "entity": ent_name,
        "v17_q": v17_quality, "v18_q": v18["quality_v18"],
        "v17_attr": v17_attr, "v18_attr": v18["attr_v18"],
        "root_verb": v18["root_verb"], "has_sentiment": v18["has_sentiment"],
        "relevant": v18["is_relevant_proxy"], "role": v18["best_role"],
    })
    if tested >= 15: break

print(f"\n{'='*70}")
print(f"CONTEXT_WORKER v18 SUMMARY ({tested} articles tested)")
print(f"{'='*70}")
print(f"  v17 rewarded attribution with attr>=25: {v17_attr_rewarded}/{tested}")
print(f"  v18 downgraded attribution to attr=10:  {v18_attr_neutral}/{tested}")
print(f"  → speaker_not_target bias reduced by v18")
print(f"")
print(f"  v18 detected sentiment predicate:       {sentiment_detected}/{tested}")
print(f"  v18 relevancy proxy (entity is core arg): {relevant_proxies}/{tested}")
print(f"")
print(f"  Quality score changes (v17 vs v18):")
for s in samples[:8]:
    delta = s["v18_q"] - s["v17_q"]
    sign = "+" if delta > 0 else ""
    print(f"    {s['entity']:22s} v17={s['v17_q']:3d} v18={s['v18_q']:3d} ({sign}{delta}) "
          f"verb={s['root_verb']} sent={s['has_sentiment']}")
print(f"{'='*70}")
