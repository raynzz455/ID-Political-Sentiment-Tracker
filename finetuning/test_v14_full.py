#!/usr/bin/env python3.13
"""
Expert test: run FULL v14.1 entity resolution with Stanza depparse on live DB articles.
"""
import os, json, re, sys, time
from collections import Counter, defaultdict

# Setup path for stanza
sys.path.insert(0, '/home/z/.local/lib/python3.13/site-packages')
import stanza
import torch

# Load Stanza pipeline ONCE (expensive)
print("Loading Stanza pipeline (tokenize,pos,lemma,depparse)...", flush=True)
t0 = time.time()
NLP = stanza.Pipeline('id', processors='tokenize,pos,lemma,depparse',
                      verbose=False, use_gpu=False, batch_size=16,
                      download_method=None)
print(f"Stanza loaded in {time.time()-t0:.1f}s", flush=True)

# v14.1 verb sets — use LEMMA forms (Stanza returns root lemmas: dikritik→kritik, mengecam→kecam)
SENTIMENT_PRED_ACTIVE = {
    "kritik","kecam","sindir","serang","bela","tolak",
    "dukung","puji","tuding","singgung","ejek","cela",
    "hina","apresiasi",
}
SENTIMENT_PRED_PASSIVE = {
    # passive forms are same lemma in Stanza (dikritik→kritik, dikecam→kecam)
    # so we check the root lemma + deprel=nsubj:pass to detect passive
}
ATTRIBUTION_VERBS = {
    "kata","nyata","tegas","jelaskan","tambah",
    "imbau","ingat","sampai","aku","klaim",
    "nilai","ungkap","jawab","ujar","tutur","sebut","papar",
}
TOPIC_DOMINANCE_THRESHOLD = 0.25

def check_semantic_role(sent, entity_start, entity_end):
    """FULL depparse-based semantic role check (v14.1 proper).
    Stanza returns root lemmas: dikritik→kritik, mengecam→kecam.
    So passive detection = (lemma in ACTIVE set) AND (deprel=nsubj:pass).
    """
    result = {"has_sentiment": False, "has_attribution": False, "verb": None, "role": None, "is_passive": False}
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
        is_passive = (entity_word.deprel == 'nsubj:pass')
        result["is_passive"] = is_passive
        head_id = entity_word.head
        for word in sent.words:
            if word.id == head_id:
                lemma = (word.lemma or word.text).lower()
                if lemma in SENTIMENT_PRED_ACTIVE:
                    # sentiment predicate: entity is target (passive patient OR active object)
                    result["has_sentiment"] = True
                    result["verb"] = lemma + (" (passive)" if is_passive else "")
                elif lemma in ATTRIBUTION_VERBS:
                    result["has_attribution"] = True
                break
    return result

def find_mentions(name, aliases, text):
    forms = [name] + aliases
    mentions = []
    for form in forms:
        if len(form) < 3: continue
        try:
            for m in re.finditer(r'\b' + re.escape(form) + r'\b', text, re.IGNORECASE):
                mentions.append((m.start(), m.end(), m.group()))
        except re.error:
            continue
    # dedupe overlapping (keep longest)
    mentions.sort(key=lambda x: (x[0], -(x[1]-x[0])))
    deduped = []
    last_end = -1
    for s, e, t in mentions:
        if s < last_end: continue
        deduped.append((s, e, t))
        last_end = e
    return deduped

def v14_1_resolve(article, entities):
    """Full v14.1 resolution with depparse."""
    title = (article.get("title") or "").strip()
    body = (article.get("text") or "").strip()
    if not body or len(body) < 50:
        return None, []
    title_lower = title.lower()

    try:
        doc = NLP(body)
    except Exception as e:
        return None, []

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
        return None, []
    total_sents = len(sentences)

    results = []
    for eid, e in entities.items():
        name = e["name"]
        aliases = e.get("aliases", [])
        mentions = find_mentions(name, aliases, body)
        if not mentions: continue

        in_title = name.lower() in title_lower or any(a.lower() in title_lower for a in aliases)
        sent_indices = set()
        has_sentiment = False
        has_attribution = False
        sentiment_verbs = []
        roles_found = []

        for start, end, _ in mentions:
            for sidx, s in enumerate(sentences):
                if s["start"] <= start < s["end"]:
                    sent_indices.add(sidx)
                    role = check_semantic_role(s["parsed"], start, end)
                    if role["has_sentiment"]:
                        has_sentiment = True
                        sentiment_verbs.append(role["verb"])
                    if role["has_attribution"]:
                        has_attribution = True
                    if role["role"]:
                        roles_found.append(role["role"])
                    break

        topic_dominance = len(sent_indices) / total_sents if total_sents > 0 else 0
        results.append({
            "entity_id": eid, "name": name,
            "in_title": in_title, "in_body": True,
            "count": len(mentions),
            "sentence_count": len(sent_indices),
            "topic_dominance": round(topic_dominance, 3),
            "has_sentiment_role": has_sentiment,
            "has_attribution_role": has_attribution,
            "sentiment_verbs": list(set(sentiment_verbs)),
            "roles_found": list(set(roles_found)),
        })

    if not results:
        return None, []

    # v14.1 sort: sentiment > dominance > in_title > count
    results.sort(key=lambda x: (
        x["has_sentiment_role"],
        x["topic_dominance"] >= TOPIC_DOMINANCE_THRESHOLD,
        x["in_title"],
        x["count"],
    ), reverse=True)
    return results[0], results

# Main: load live data and run
print("\nLoading live data...", flush=True)
data = json.load(open("/tmp/live_multi.json"))
articles = data["articles"]
mappings = data["mappings"]
ent_full = data["ent_full"]
ent_map = data["ent_map"]

# Find multi-entity articles
ent_per_art = Counter()
for m in mappings:
    ent_per_art[m["raw_text_id"]] += 1
multi_ids = {aid for aid, c in ent_per_art.items() if c >= 2}
print(f"Multi-entity articles: {len(multi_ids)}")

print("\n" + "="*70)
print("FULL v14.1 TEST (with Stanza depparse) on multi-entity articles")
print("="*70, flush=True)

results_log = []
agree = 0
disagree = 0
sentiment_role_hits = 0

for i, a in enumerate(articles):
    if a["id"] not in multi_ids: continue
    art_id = a["id"]
    title = (a.get("title") or "")
    print(f"\n[{i+1}] TITLE: {title[:80]}", flush=True)

    # v12 (DB) main entity
    v12_main = None
    v12_all = []
    for m in mappings:
        if m["raw_text_id"] == art_id:
            nm = ent_map.get(m["entity_id"], "?")
            if m.get("is_main_entity"): v12_main = nm
            v12_all.append((nm, m.get("is_main_entity"), m.get("confidence")))

    # v14.1 main entity
    t1 = time.time()
    main14, all14 = v14_1_resolve(a, ent_full)
    parse_time = time.time() - t1

    print(f"  v12 (DB):    {v12_main}", flush=True)
    if main14:
        print(f"  v14.1 (new): {main14['name']}  [parse: {parse_time:.1f}s]", flush=True)
        print(f"    sent_role={main14['has_sentiment_role']}({main14['sentiment_verbs']}) "
              f"attr={main14['has_attribution_role']} dom={main14['topic_dominance']} "
              f"in_title={main14['in_title']} cnt={main14['count']} roles={main14['roles_found']}")
        if main14['has_sentiment_role']:
            sentiment_role_hits += 1

        if v12_main == main14["name"]:
            print(f"  ✅ AGREE")
            agree += 1
        else:
            print(f"  ⚡ DELTA: v12='{v12_main}' → v14.1='{main14['name']}'")
            v12r = [r for r in all14 if r["name"] == v12_main]
            if v12r:
                v = v12r[0]
                reasons = []
                if v['has_attribution_role'] and not v['has_sentiment_role']:
                    reasons.append(f"v12 picked SPEAKER (attr={v['has_attribution_role']}) not TARGET")
                if v['topic_dominance'] < 0.25:
                    reasons.append(f"low dominance ({v['topic_dominance']})")
                if v['count'] < main14['count']:
                    reasons.append(f"fewer mentions ({v['count']}<{main14['count']})")
                if v['has_sentiment_role'] and not main14['has_sentiment_role']:
                    reasons.append("v14.1's pick lacks sentiment role")
                if reasons:
                    print(f"    WHY: {'; '.join(reasons)}")
            disagree += 1
            results_log.append({"title": title, "v12": v12_main, "v14_1": main14["name"],
                                "sent_role": main14["has_sentiment_role"],
                                "sent_verbs": main14["sentiment_verbs"]})
    else:
        print(f"  v14.1: no entity in body")
        disagree += 1

    # show all entities ranked
    if all14 and len(all14) > 1:
        for r in all14[:4]:
            tag = " ← v14.1" if main14 and r["name"] == main14["name"] else ""
            v12tag = " ← v12" if r["name"] == v12_main else ""
            print(f"    {r['name']:25s} sent={r['has_sentiment_role']!s:5} dom={r['topic_dominance']:.2f} "
                  f"in_title={r['in_title']!s:5} cnt={r['count']:2d} roles={r['roles_found']}{tag}{v12tag}")

print(f"\n{'='*70}")
print(f"FULL v14.1 (depparse) SUMMARY on {len(multi_ids)} multi-entity articles:")
print(f"  Agree with v12:        {agree}")
print(f"  Disagree (override):   {disagree}")
print(f"  Articles with sentiment role detected: {sentiment_role_hits}")
print(f"  Override rate: {disagree}/{agree+disagree} = {disagree/(agree+disagree)*100:.0f}%")
print(f"\n  Articles where v14.1 OVERRIDES v12:")
for d in results_log:
    print(f"    '{d['title'][:55]}' | v12={d['v12']} → v14.1={d['v14_1']} (sent={d['sent_role']}, verbs={d['sent_verbs']})")
print(f"{'='*70}")

# Save full results
with open("/tmp/v14_1_full_results.json", "w") as f:
    json.dump({"agree": agree, "disagree": disagree, "overrides": results_log,
               "sentiment_role_hits": sentiment_role_hits}, f, indent=2, ensure_ascii=False)
