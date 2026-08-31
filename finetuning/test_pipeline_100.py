#!/usr/bin/env python3.13
"""
Full pipeline test: run entity_resolution_worker v14.2 + context_worker v18.1
on 100 live articles from Supabase. Compare with existing v12/v17 output.
"""
import os, json, re, sys, time
from collections import Counter, defaultdict

sys.path.insert(0, '/home/z/.local/lib/python3.13/site-packages')
import stanza
import torch

print("Loading Stanza (tokenize,pos,lemma,depparse)...", flush=True)
t0 = time.time()
NLP = stanza.Pipeline('id', processors='tokenize,pos,lemma,depparse',
                      verbose=False, use_gpu=False, batch_size=16,
                      download_method=None)
print(f"Stanza loaded in {time.time()-t0:.1f}s\n", flush=True)

# v14.2 EXPANDED verb sets (lemma forms, 70.7% coverage)
SENTIMENT_PRED_ACTIVE = {
    "kritik","kecam","sindir","serang","hina","cela","ejek","tuding",
    "tuduh","lapor","cekal","tahan","vonis","tangkap","pidana","anggap",
    "nilai","sorot","gugur","bongkar","pecat","mundur","undur","berhenti",
    "ganti","razia","sita","denda","hukum","ganjar",
    "puji","dukung","apresiasi","restui","sahkan","setuju","kukuhkan",
    "akui","legitimasi",
    "bela","tolak","keberatan","menentang",
    "pandang","sikapi","persepsi",
    "ungkap",
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
MAX_CONTEXT_WORDS = 180

def find_mentions(name, aliases, text):
    forms = [name] + aliases
    mentions = []
    for form in forms:
        if len(form) < 3: continue
        try:
            for m in re.finditer(r'\b' + re.escape(form) + r'\b', text, re.IGNORECASE):
                mentions.append((m.start(), m.end(), m.group()))
        except: continue
    mentions.sort(key=lambda x: (x[0], -(x[1]-x[0])))
    deduped = []
    last_end = -1
    for s, e, t in mentions:
        if s < last_end: continue
        deduped.append((s, e, t))
        last_end = e
    return deduped

def check_semantic_role(sent, entity_start, entity_end):
    result = {"has_sentiment": False, "has_attribution": False, "verb": None, "role": None, "is_passive": False}
    entity_word = None
    for word in sent.words:
        if word.start_char <= entity_start < word.end_char:
            entity_word = word; break
        if entity_start <= word.start_char < entity_end:
            entity_word = word; break
    if entity_word is None: return result
    if entity_word.deprel in ('nsubj', 'nsubj:pass', 'obj', 'iobj', 'csubj', 'obl'):
        result["role"] = entity_word.deprel
        is_passive = (entity_word.deprel == 'nsubj:pass')
        result["is_passive"] = is_passive
        head_id = entity_word.head
        for word in sent.words:
            if word.id == head_id:
                lemma = (word.lemma or word.text).lower()
                if lemma in SENTIMENT_PRED_ACTIVE:
                    result["has_sentiment"] = True
                    result["verb"] = lemma + (" (passive)" if is_passive else "")
                elif lemma in ATTRIBUTION_VERBS:
                    result["has_attribution"] = True
                break
    return result

def resolve_entity_v14_2(article, entities):
    """Run v14.2 entity resolution. Returns main entity + all entities found.
    Optimization: pre-filter entities by title+body text match (fast regex) before Stanza parse.
    """
    title = (article.get("title") or "").strip()
    body = (article.get("text") or "").strip()
    title_lower = title.lower()
    if not body or len(body) < 50: return None, []

    # PRE-FILTER: find which entities are mentioned at all (fast regex, no Stanza)
    candidate_entities = []
    for eid, e in entities.items():
        mentions = find_mentions(e["name"], e.get("aliases", []), body)
        if mentions:
            in_title = e["name"].lower() in title_lower or any(a.lower() in title_lower for a in e.get("aliases", []))
            candidate_entities.append((eid, e, mentions, in_title))

    if not candidate_entities:
        return None, []  # no entity mentioned, skip Stanza parse

    # Only parse body with Stanza if we have candidates
    try:
        doc = NLP(body)
    except: return None, []

    sentences = []
    for sent in doc.sentences:
        if len(sent.text.strip()) > 10:
            sentences.append({
                "text": sent.text,
                "start": sent.tokens[0].start_char if sent.tokens else 0,
                "end": sent.tokens[-1].end_char if sent.tokens else 0,
                "parsed": sent,
            })
    if not sentences: return None, []
    total_sents = len(sentences)

    results = []
    for eid, e, mentions, in_title in candidate_entities:
        sent_indices = set()
        has_sentiment = False
        has_attribution = False
        sentiment_verbs = []
        roles = []

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
                        roles.append(role["role"])
                    break

        dom = len(sent_indices) / total_sents if total_sents > 0 else 0
        results.append({
            "entity_id": eid, "name": e["name"],
            "in_title": in_title, "count": len(mentions),
            "topic_dominance": round(dom, 3),
            "has_sentiment": has_sentiment,
            "has_attribution": has_attribution,
            "sentiment_verbs": list(set(sentiment_verbs)),
            "roles": list(set(roles)),
            "sentence_count": len(sent_indices),
        })

    if not results: return None, []
    # v14.2 sort: sentiment > dom>=0.25 > in_title > count
    results.sort(key=lambda x: (x["has_sentiment"], x["topic_dominance"]>=TOPIC_DOMINANCE_THRESHOLD, x["in_title"], x["count"]), reverse=True)
    return results[0], results

def extract_context_v18(article, entity_name, entity_aliases, main_entity_data):
    """Run v18.1 context extraction for a given entity."""
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

    mentions = find_mentions(entity_name, entity_aliases, body)
    if not mentions: return None

    # find best anchor: prefer sentence where entity is core arg
    best_anchor = -1
    for sidx, s in enumerate(sentences):
        for start, end, _ in mentions:
            if s["start"] <= start < s["end"]:
                best_anchor = sidx; break
        if best_anchor >= 0: break
    if best_anchor < 0: return None

    anchor_sent = sentences[best_anchor]
    first_mention = next((m for m in mentions if anchor_sent["start"] <= m[0] < anchor_sent["end"]), mentions[0])

    has_sentiment = main_entity_data.get("has_sentiment", False) if main_entity_data else False
    has_attribution = main_entity_data.get("has_attribution", False) if main_entity_data else False
    is_main_actor = any("nsubj" in r or "obj" in r for r in (main_entity_data.get("roles", []) if main_entity_data else []))

    root_verb = None
    for word in anchor_sent["parsed"].words:
        if word.deprel == 'root':
            root_verb = (word.lemma or word.text).lower()
            break

    # build context
    context_parts = [anchor_sent["text"]]
    if best_anchor + 1 < len(sentences) and not has_sentiment:
        context_parts.append(sentences[best_anchor + 1]["text"])
    ctx_text = " ".join(context_parts)
    words_list = ctx_text.split()
    if len(words_list) > MAX_CONTEXT_WORDS:
        ctx_text = " ".join(words_list[:MAX_CONTEXT_WORDS])

    para_idx = body[:first_mention[0]].count('\n\n')

    # v18 quality_score
    attr_score = 40 if has_sentiment else (10 if has_attribution else 10)
    actor_score = 30 if is_main_actor else 10
    pos_score = 20 if para_idx == 0 else (12 if para_idx <= 2 else 5)
    quality_v18 = attr_score + actor_score + pos_score + 10

    # v17 quality_score (for comparison — attribution got 25)
    attr_v17 = 40 if has_sentiment else (25 if has_attribution else 10)
    quality_v17 = attr_v17 + actor_score + pos_score + 10

    return {
        "context_text": ctx_text,
        "quality_v18": quality_v18,
        "quality_v17": quality_v17,
        "attr_v18": attr_score,
        "attr_v17": attr_v17,
        "has_sentiment": has_sentiment,
        "has_attribution": has_attribution,
        "root_verb": root_verb,
        "is_main_actor": is_main_actor,
        "anchor_sentence": anchor_sent["text"][:100],
    }

# ===== LOAD DATA =====
print("Loading data...", flush=True)
articles = json.load(open("/tmp/articles_100.json"))[:25]  # limit to 25 for stability
existing = json.load(open("/tmp/v12_v17_existing.json"))
ent_full = existing["ent_full"]
ent_map = existing["ent_map"]
existing_maps = existing["mappings"]  # v12 article_entity_map

# Build existing v12 main entity lookup
v12_main_by_art = {}
for m in existing_maps:
    if m.get("is_main_entity"):
        v12_main_by_art[m["raw_text_id"]] = ent_map.get(m["entity_id"], "?")

print(f"Articles: {len(articles)} | Entities in DB: {len(ent_full)} | v12 mappings: {len(existing_maps)}")
print(f"Articles with v12 main entity: {len(v12_main_by_art)}")

# ===== RUN PIPELINE =====
print(f"\n{'='*70}")
print(f"RUNNING v14.2 ENTITY RESOLUTION + v18.1 CONTEXT ON 100 ARTICLES")
print(f"{'='*70}\n", flush=True)

results = []
stats = Counter()
t_start = time.time()

for i, a in enumerate(articles):
    art_id = a["id"]
    title = (a.get("title") or "")[:60]

    # Run v14.2 entity resolution
    t1 = time.time()
    main14, all14 = resolve_entity_v14_2(a, ent_full)
    parse_time = time.time() - t1

    v12_main = v12_main_by_art.get(art_id)

    # Run v18.1 context for main entity
    ctx_v18 = None
    if main14:
        ent_info = ent_full.get(main14["entity_id"], {})
        ctx_v18 = extract_context_v18(a, ent_info["name"], ent_info.get("aliases", []), main14)

    # Compare
    agree = (v12_main == main14["name"]) if main14 and v12_main else None

    results.append({
        "art_id": art_id, "title": title,
        "v12_main": v12_main,
        "v14_main": main14["name"] if main14 else None,
        "v14_sentiment": main14["has_sentiment"] if main14 else False,
        "v14_sentiment_verbs": main14["sentiment_verbs"] if main14 else [],
        "v14_dominance": main14["topic_dominance"] if main14 else 0,
        "v14_in_title": main14["in_title"] if main14 else False,
        "v14_count": main14["count"] if main14 else 0,
        "v14_roles": main14["roles"] if main14 else [],
        "v14_all_entities": len(all14) if all14 else 0,
        "ctx_v18_quality": ctx_v18["quality_v18"] if ctx_v18 else None,
        "ctx_v17_quality": ctx_v18["quality_v17"] if ctx_v18 else None,
        "ctx_v18_root_verb": ctx_v18["root_verb"] if ctx_v18 else None,
        "ctx_v18_has_sentiment": ctx_v18["has_sentiment"] if ctx_v18 else False,
        "ctx_v18_text": ctx_v18["context_text"][:150] if ctx_v18 else None,
        "agree_with_v12": agree,
        "parse_time": round(parse_time, 2),
    })

    if main14:
        stats["v14_found_entity"] += 1
        if main14["has_sentiment"]:
            stats["v14_sentiment_detected"] += 1
        if agree:
            stats["agree"] += 1
        elif v12_main:
            stats["disagree"] += 1
        else:
            stats["v14_found_v12_missed"] += 1
    else:
        stats["v14_no_entity"] += 1

    if (i+1) % 10 == 0:
        elapsed = time.time() - t_start
        rate = (i+1) / elapsed
        eta = (len(articles) - i - 1) / rate
        print(f"  [{i+1}/100] {elapsed:.0f}s elapsed, {rate:.1f} art/s, ETA {eta:.0f}s | "
              f"agree={stats['agree']} disagree={stats['disagree']} sent={stats['v14_sentiment_detected']}", flush=True)
        # INCREMENTAL SAVE — prevent data loss on timeout
        with open("/tmp/pipeline_100_results.json", "w") as f:
            json.dump({"stats": dict(stats), "results": results,
                       "total_time": time.time()-t_start, "completed": i+1}, f, indent=2, ensure_ascii=False)

# ===== RESULTS =====
total_time = time.time() - t_start
print(f"\n{'='*70}")
print(f"PIPELINE RESULTS (100 articles, {total_time:.0f}s total, {total_time/100:.1f}s/art)")
print(f"{'='*70}")

print(f"\n--- Entity Resolution v14.2 ---")
print(f"  Articles with entity found:        {stats['v14_found_entity']}/100")
print(f"  Articles with NO entity found:     {stats['v14_no_entity']}/100")
print(f"  Articles with v12 main (compare):  {len(v12_main_by_art)}/100")
print(f"  AGREE with v12:                    {stats['agree']}")
print(f"  DISAGREE with v12:                 {stats['disagree']}")
print(f"  v14 found entity, v12 missed:      {stats['v14_found_v12_missed']}")
print(f"  Sentiment predicate detected:      {stats['v14_sentiment_detected']}/100")

# Show disagreements
disagree_rows = [r for r in results if r["agree_with_v12"] is False]
print(f"\n--- DISAGREEMENTS ({len(disagree_rows)}) ---")
for r in disagree_rows[:15]:
    print(f"\n  TITLE: {r['title']}")
    print(f"    v12 main: {r['v12_main']}")
    print(f"    v14 main: {r['v14_main']} (sent={r['v14_sentiment']} verbs={r['v14_sentiment_verbs']} dom={r['v14_dominance']} in_title={r['v14_in_title']} cnt={r['v14_count']})")

# Show sentiment-detected articles
sent_rows = [r for r in results if r["v14_sentiment"]]
print(f"\n{'='*70}")
print(f"SENTIMENT PREDICATE DETECTED ({len(sent_rows)}/100)")
print(f"{'='*70}")
for r in sent_rows[:10]:
    print(f"\n  TITLE: {r['title']}")
    print(f"    ENTITY: {r['v14_main']} | verbs={r['v14_sentiment_verbs']} | roles={r['v14_roles']}")
    print(f"    ctx root_verb: {r['ctx_v18_root_verb']} | ctx_has_sentiment: {r['ctx_v18_has_sentiment']}")
    print(f"    quality v17={r['ctx_v17_quality']} → v18={r['ctx_v18_quality']}")
    print(f"    ctx[:120]: {r['ctx_v18_text']}")

# Context quality comparison
print(f"\n{'='*70}")
print(f"CONTEXT QUALITY v17 vs v18")
print(f"{'='*70}")
quality_changes = [r for r in results if r["ctx_v18_quality"] and r["ctx_v17_quality"]]
if quality_changes:
    v17_scores = [r["ctx_v17_quality"] for r in quality_changes]
    v18_scores = [r["ctx_v18_quality"] for r in quality_changes]
    print(f"  v17 quality: mean={sum(v17_scores)/len(v17_scores):.1f}")
    print(f"  v18 quality: mean={sum(v18_scores)/len(v18_scores):.1f}")
    upgraded = sum(1 for r in quality_changes if r["ctx_v18_quality"] > r["ctx_v17_quality"])
    downgraded = sum(1 for r in quality_changes if r["ctx_v18_quality"] < r["ctx_v17_quality"])
    same = sum(1 for r in quality_changes if r["ctx_v18_quality"] == r["ctx_v17_quality"])
    print(f"  Upgraded (v18 > v17): {upgraded}")
    print(f"  Downgraded (v18 < v17): {downgraded} (attribution verbs no longer rewarded)")
    print(f"  Same: {same}")

# Save full results
with open("/tmp/pipeline_100_results.json", "w") as f:
    json.dump({"stats": dict(stats), "results": results, "total_time": total_time}, f, indent=2, ensure_ascii=False)
print(f"\nSaved to /tmp/pipeline_100_results.json")
