#!/usr/bin/env python3.13
"""Test v18.1 context_worker on articles with attribution verbs (attr_score=25)."""
import sys, json, re, time, gc
sys.path.insert(0, '/home/z/.local/lib/python3.13/site-packages')
import stanza

print("Loading Stanza...", flush=True)
NLP = stanza.Pipeline('id', processors='tokenize,pos,lemma,depparse',
                      verbose=False, use_gpu=False, batch_size=8, download_method=None)
print("Stanza ready\n", flush=True)

# v18.1 EXPANDED verb sets (lemma forms)
SENTIMENT_PRED = {
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
MAX_CTX_WORDS = 180

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
    deduped, last_end = [], -1
    for s,e,t in mentions:
        if s < last_end: continue
        deduped.append((s,e,t)); last_end = e
    return deduped

def extract_v18(article, entity_name, entity_aliases):
    body = (article.get("body") or "").strip()
    title = (article.get("title") or "").strip()
    if not body: return None
    try:
        doc = NLP(body)
    except: return None

    sentences = []
    for sent in doc.sentences:
        if len(sent.text.strip()) > 10:
            sentences.append({"text": sent.text, "start": sent.tokens[0].start_char if sent.tokens else 0,
                              "end": sent.tokens[-1].end_char if sent.tokens else 0, "parsed": sent})
    if not sentences: return None

    mentions = find_mentions(entity_name, entity_aliases, body)
    if not mentions: return None

    # find anchor
    anchor_idx = -1
    for sidx, s in enumerate(sentences):
        for start, end, _ in mentions:
            if s["start"] <= start < s["end"]:
                anchor_idx = sidx; break
        if anchor_idx >= 0: break
    if anchor_idx < 0: return None

    anchor_sent = sentences[anchor_idx]
    first_mention = next((m for m in mentions if anchor_sent["start"] <= m[0] < anchor_sent["end"]), mentions[0])

    # detect root verb + semantic role
    has_sentiment = False
    has_attribution = False
    root_verb = None
    entity_role = None
    for word in anchor_sent["parsed"].words:
        if word.deprel == 'root':
            root_verb = (word.lemma or word.text).lower()
            if root_verb in SENTIMENT_PRED:
                has_sentiment = True
            if root_verb in ATTRIBUTION_VERBS:
                has_attribution = True
        # check entity's role
        if word.start_char <= first_mention[0] < word.end_char or \
           (first_mention[0] <= word.start_char < first_mention[1]):
            if word.deprel in ('nsubj','nsubj:pass','obj','iobj','obl','csubj'):
                entity_role = word.deprel

    is_main_actor = entity_role in ('nsubj','nsubj:pass','obj','iobj','csubj')

    # build context
    context_parts = [anchor_sent["text"]]
    if anchor_idx + 1 < len(sentences) and not has_sentiment:
        context_parts.append(sentences[anchor_idx + 1]["text"])
    ctx_text = " ".join(context_parts)
    words_list = ctx_text.split()
    if len(words_list) > MAX_CTX_WORDS:
        ctx_text = " ".join(words_list[:MAX_CTX_WORDS])

    para_idx = body[:first_mention[0]].count('\n\n')

    # v18 quality_score: sentiment=40, attribution=10
    attr_v18 = 40 if has_sentiment else (10 if has_attribution else 10)
    # v17 quality_score: attribution got 25
    attr_v17 = 40 if has_sentiment else (25 if has_attribution else 10)

    actor_score = 30 if is_main_actor else 10
    pos_score = 20 if para_idx == 0 else (12 if para_idx <= 2 else 5)
    exclusivity = 10

    return {
        "ctx_v18": ctx_text[:200],
        "quality_v18": attr_v18 + actor_score + pos_score + exclusivity,
        "quality_v17": attr_v17 + actor_score + pos_score + exclusivity,
        "attr_v18": attr_v18,
        "attr_v17": attr_v17,
        "root_verb": root_verb,
        "has_sentiment": has_sentiment,
        "has_attribution": has_attribution,
        "entity_role": entity_role,
        "is_main_actor": is_main_actor,
    }

# Load test data
test_data = json.load(open("/tmp/attr_verb_articles.json"))
print(f"Testing {len(test_data)} articles with attribution verbs\n", flush=True)

print(f"{'='*90}")
print(f"{'TITLE':40s} {'ENTITY':18s} {'v17_q':>5} {'v18_q':>5} {'Δ':>4} {'v17_a':>5} {'v18_a':>5} {'root_verb':12s} {'role':8s}")
print(f"{'='*90}")

results = []
for t in test_data:
    art = {"title": t["title"], "body": t["body"]}
    v18 = extract_v18(art, t["entity_name"], t["entity_aliases"])
    if not v18:
        print(f"{t['title'][:40]:40s} {t['entity_name'][:18]:18s} --- FAILED ---")
        continue
    delta = v18["quality_v18"] - v18["quality_v17"]
    flag = "⚡" if delta < 0 else "✅" if delta > 0 else "="
    print(f"{t['title'][:40]:40s} {t['entity_name'][:18]:18s} "
          f"{t['v17_quality']:>5} {v18['quality_v18']:>5} {delta:>+4} "
          f"{v18['attr_v17']:>5} {v18['attr_v18']:>5} "
          f"{(v18['root_verb'] or '')[:12]:12s} {(v18['entity_role'] or '')[:8]:8s} {flag}")
    results.append({
        "title": t["title"][:60], "entity": t["entity_name"],
        "v17_quality": t["v17_quality"], "v18_quality": v18["quality_v18"],
        "v17_attr": v18["attr_v17"], "v18_attr": v18["attr_v18"],
        "root_verb": v18["root_verb"], "has_sentiment": v18["has_sentiment"],
        "has_attribution": v18["has_attribution"], "entity_role": v18["entity_role"],
        "delta": delta,
    })
    gc.collect()  # prevent memory leak

# Summary
print(f"\n{'='*90}")
print(f"SUMMARY ({len(results)} articles)")
print(f"{'='*90}")
downgraded = sum(1 for r in results if r["delta"] < 0)
upgraded = sum(1 for r in results if r["delta"] > 0)
same = sum(1 for r in results if r["delta"] == 0)
sentiment_found = sum(1 for r in results if r["has_sentiment"])
attr_found = sum(1 for r in results if r["has_attribution"])
print(f"  Downgraded (attr 25→10): {downgraded}  ← v18 kills speaker bias here")
print(f"  Upgraded (sentiment found): {upgraded}")
print(f"  Same: {same}")
print(f"  Sentiment predicate found: {sentiment_found}")
print(f"  Attribution verb found: {attr_found}")

print(f"\n--- Articles downgraded by v18 (speaker bias killed) ---")
for r in results:
    if r["delta"] < 0:
        print(f"  '{r['title'][:50]}' | {r['entity']} | verb={r['root_verb']} | role={r['entity_role']}")
        print(f"    quality {r['v17_quality']}→{r['v18_quality']} (attr {r['v17_attr']}→{r['v18_attr']})")

with open("/tmp/v18_attr_test_results.json","w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to /tmp/v18_attr_test_results.json")
