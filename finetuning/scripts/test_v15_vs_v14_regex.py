#!/usr/bin/env python3.13
"""Test v15 vs v14.2 using REGEX simulation (no Stanza needed).
Measures if era + affiliation checks improve entity resolution."""
import os, json, re, gc
from collections import Counter

# Load live data
from supabase import create_client
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

print("Fetching articles + entities...", flush=True)
res = sb.table("raw_texts").select("id, title, text, status").eq("status","processed").limit(30).execute()
articles = res.data
print(f"Articles: {len(articles)}")

res_ent = sb.table("political_entities").select("id, canonical_name, aliases, party_affiliation, position, era").execute()
ent_full = {}
for e in res_ent.data:
    ent_full[e["id"]] = {
        "name": e["canonical_name"], "aliases": e.get("aliases") or [],
        "party": e.get("party_affiliation"), "position": e.get("position"),
        "era": e.get("era") or [],
    }
ent_id_to_name = {eid: e["name"] for eid, e in ent_full.items()}
print(f"Entities: {len(ent_full)}")

art_ids = [a["id"] for a in articles]
res_map = sb.table("article_entity_map").select("raw_text_id, entity_id, is_main_entity").in_("raw_text_id", art_ids).execute()
v12_main = {}
for m in res_map.data:
    if m.get("is_main_entity"):
        v12_main[m["raw_text_id"]] = ent_id_to_name.get(m["entity_id"], "?")
print(f"v12 main: {len(v12_main)}")

# Verb sets (lemma forms)
SENTIMENT_PRED = {
    "kritik","kecam","sindir","serang","hina","cela","ejek","tuding",
    "tuduh","lapor","cekal","tahan","vonis","tangkap","pidana","anggap",
    "nilai","sorot","gugur","bongkar","pecat","mundur","undur","berhenti",
    "ganti","razia","sita","denda","hukum","ganjar","puji","dukung","apresiasi",
    "restui","sahkan","setuju","kukuhkan","akui","legitimasi","bela","tolak",
    "keberatan","menentang","pandang","sikapi","persepsi","ungkap",
}
ATTRIBUTION = {
    "kata","nyata","tegas","jelaskan","tambah","imbau","ingat","sampai",
    "aku","klaim","nilai","ungkap","jawab","ujar","tutur","sebut","papar",
    "ucap","sampaikan","katakan","ungkapkan","nyatakan","tegaskan","tambahkan",
    "imbaukan","ingatkan","balas","tanggapi","saran","menyaran","rekomendasi",
    "usul","ajak","mengajak","pinta","minta","meminta","perintah","wantiwanti",
    "tekan","tekankan","menekankan","sorot","soroti","tandai","tanda","tunjuk","menunjuk",
}

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

def check_sentiment_proximity(text, start, end):
    """Regex proximity: check if sentiment verb near entity (within 60 chars)."""
    window = text[max(0,start-60):end+60].lower()
    for v in SENTIMENT_PRED:
        if v in window: return True, v
    return False, None

def check_attr_proximity(text, start, end):
    window = text[max(0,start-60):end+60].lower()
    for v in ATTRIBUTION:
        if v in window: return True, v
    return False, None

def check_era(text, entity_eras):
    if not entity_eras: return True, None
    tl = text.lower()
    era_markers = {
        "era jokowi": ["era jokowi","zaman jokowi","pemerintahan jokowi"],
        "era prabowo": ["era prabowo","zaman prabowo","pemerintahan prabowo"],
        "era sby": ["era sby","zaman sby","pemerintahan sby"],
        "era gus dur": ["era gus dur","zaman gus dur"],
        "era megawati": ["era megawati","zaman megawati"],
    }
    detected = []
    for k, ms in era_markers.items():
        if any(m in tl for m in ms): detected.append(k)
    if not detected: return True, None
    entity_eras_lower = [e.lower() for e in entity_eras]
    for d in detected:
        if any(d in e for e in entity_eras_lower): return True, d
    return False, detected[0]

def check_affil(text, entity_info):
    party = entity_info.get("party")
    if not party: return True, None
    tl = text.lower()
    if party.lower() in tl: return True, party
    abbrs = {"PDI-P":["pdip","pdi-p"],"Gerindra":["gerindra"],"Golkar":["golkar"],
             "Demokrat":["demokrat"],"PKB":["pkb"],"PAN":["pan"],"PKS":["pks"],
             "Nasdem":["nasdem"],"Independen":["independen"]}
    for a in abbrs.get(party, [party.lower()]):
        if a in tl: return True, party
    return None, None

def split_sents(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z""])', text) if len(s.strip())>10]

def resolve_v14_2(article, entities):
    title = (article.get("title") or "").strip()
    body = (article.get("text") or "").strip()
    title_lower = title.lower()
    if not body or len(body) < 50: return None, []
    
    candidates = []
    for eid, e in entities.items():
        mentions = find_mentions(e["name"], e.get("aliases",[]), body)
        if mentions:
            in_title = e["name"].lower() in title_lower or any(a.lower() in title_lower for a in e.get("aliases",[]))
            candidates.append((eid, e, mentions, in_title))
    if not candidates: return None, []
    
    sents = split_sents(body)
    total = len(sents)
    
    results = []
    for eid, e, mentions, in_title in candidates:
        sent_idxs = set()
        has_sent = has_attr = False
        sverbs = []
        for s, en, _ in mentions:
            for si, sent in enumerate(sents):
                sp = body.find(sent)
                if sp >= 0 and sp <= s < sp+len(sent): sent_idxs.add(si); break
            sr, v = check_sentiment_proximity(body, s, en)
            if sr: has_sent = True; sverbs.append(v)
            ar, _ = check_attr_proximity(body, s, en)
            if ar: has_attr = True
        dom = len(sent_idxs) / total if total > 0 else 0
        results.append({"name": e["name"], "in_title": in_title, "count": len(mentions),
                        "topic_dominance": round(dom,3), "has_sentiment": has_sent,
                        "has_attribution": has_attr, "sentiment_verbs": list(set(sverbs))})
    
    results.sort(key=lambda x: (x["has_sentiment"], x["topic_dominance"]>=0.25, x["in_title"], x["count"]), reverse=True)
    return results[0], results

def resolve_v15(article, entities):
    """v15: with era + affiliation validation."""
    title = (article.get("title") or "").strip()
    body = (article.get("text") or "").strip()
    title_lower = title.lower()
    if not body or len(body) < 50: return None, []
    
    candidates = []
    for eid, e in entities.items():
        mentions = find_mentions(e["name"], e.get("aliases",[]), body)
        if mentions:
            in_title = e["name"].lower() in title_lower or any(a.lower() in title_lower for a in e.get("aliases",[]))
            candidates.append((eid, e, mentions, in_title))
    if not candidates: return None, []
    
    sents = split_sents(body)
    total = len(sents)
    
    results = []
    for eid, e, mentions, in_title in candidates:
        sent_idxs = set()
        has_sent = has_attr = False
        sverbs = []
        for s, en, _ in mentions:
            for si, sent in enumerate(sents):
                sp = body.find(sent)
                if sp >= 0 and sp <= s < sp+len(sent): sent_idxs.add(si); break
            sr, v = check_sentiment_proximity(body, s, en)
            if sr: has_sent = True; sverbs.append(v)
            ar, _ = check_attr_proximity(body, s, en)
            if ar: has_attr = True
        dom = len(sent_idxs) / total if total > 0 else 0
        
        # v15 NEW: Era + Affiliation
        era_ok, detected_era = check_era(body, e.get("era", []))
        affil_ok, mentioned_party = check_affil(body, e)
        
        results.append({
            "name": e["name"], "in_title": in_title, "count": len(mentions),
            "topic_dominance": round(dom,3), "has_sentiment": has_sent,
            "has_attribution": has_attr, "sentiment_verbs": list(set(sverbs)),
            "era_compatible": era_ok, "detected_era": detected_era,
            "affiliation_match": affil_ok, "entity_era": e.get("era",[]),
            "entity_party": e.get("party"),
        })
    
    # v15: INTUITIVE RANKING with era + affiliation
    results.sort(key=lambda x: (
        x["has_sentiment"], x["topic_dominance"]>=0.25, x["era_compatible"],
        x["affiliation_match"] is not False, x["in_title"], x["count"],
    ), reverse=True)
    
    # v15: Confidence with era + affiliation penalty
    main = results[0]
    if main["has_sentiment"]: conf = 0.95
    elif main["topic_dominance"] >= 0.25: conf = 0.85
    elif main["count"] > 1 and main["in_title"]: conf = 0.70
    else: conf = 0.50
    if not main["era_compatible"]: conf *= 0.7
    if main["affiliation_match"] is False: conf *= 0.8
    main["confidence"] = round(conf, 3)
    
    return main, results

# Run comparison
print(f"\n{'='*80}")
print(f"v14.2 vs v15 COMPARISON (regex simulation, {len(articles)} articles)")
print(f"{'='*80}\n", flush=True)

v14_agree = v14_disagree = v15_agree = v15_disagree = 0
v15_changed = era_mismatches = affil_mismatches = sentiment_found = 0
results_log = []

for i, a in enumerate(articles):
    art_id = a["id"]
    v12_m = v12_main.get(art_id)
    
    main14, _ = resolve_v14_2(a, ent_full)
    main15, _ = resolve_v15(a, ent_full)
    
    v14_name = main14["name"] if main14 else None
    v15_name = main15["name"] if main15 else None
    
    if v12_m and v14_name:
        if v14_name == v12_m: v14_agree += 1
        else: v14_disagree += 1
    if v12_m and v15_name:
        if v15_name == v12_m: v15_agree += 1
        else: v15_disagree += 1
    
    if v14_name != v15_name:
        v15_changed += 1
        print(f"\n⚡ DELTA v14.2 vs v15:")
        print(f"  TITLE: {(a.get('title') or '')[:65]}")
        print(f"  v12:   {v12_m}")
        print(f"  v14.2: {v14_name}")
        print(f"  v15:   {v15_name}")
        if main15:
            print(f"  v15: sent={main15['has_sentiment']} dom={main15['topic_dominance']} "
                  f"era={main15['era_compatible']} affil={main15['affiliation_match']} conf={main15['confidence']}")
            if not main15["era_compatible"]:
                era_mismatches += 1
                print(f"  ⚠️  ERA MISMATCH: entity era={main15.get('entity_era',[])} vs detected={main15['detected_era']}")
            if main15["affiliation_match"] is False:
                affil_mismatches += 1
                print(f"  ⚠️  AFFIL MISMATCH: entity party={main15.get('entity_party')} not in article")
        results_log.append({"title": a.get("title",""), "v12": v12_m, "v14": v14_name, "v15": v15_name})
    
    if main15 and main15["has_sentiment"]:
        sentiment_found += 1

# Summary
print(f"\n{'='*80}")
print(f"SUMMARY — v14.2 vs v15 on {len(articles)} articles (regex simulation)")
print(f"{'='*80}")
print(f"  v12 main available: {len(v12_main)}/{len(articles)}")
print(f"\n  v14.2 agree with v12:    {v14_agree}")
print(f"  v14.2 disagree with v12: {v14_disagree}")
print(f"  v15 agree with v12:      {v15_agree}")
print(f"  v15 disagree with v12:   {v15_disagree}")
print(f"\n  v15 CHANGED vs v14.2:    {v15_changed}/{len(articles)}")
print(f"  Era mismatches detected: {era_mismatches}")
print(f"  Affiliation mismatches:   {affil_mismatches}")
print(f"  Sentiment predicates:     {sentiment_found}")

if v14_agree + v14_disagree > 0:
    print(f"\n  v14.2 accuracy: {v14_agree}/{v14_agree+v14_disagree} = {v14_agree/(v14_agree+v14_disagree)*100:.0f}%")
if v15_agree + v15_disagree > 0:
    print(f"  v15 accuracy:   {v15_agree}/{v15_agree+v15_disagree} = {v15_agree/(v15_agree+v15_disagree)*100:.0f}%")

if v15_changed > 0:
    print(f"\n  v15 OVERRIDES v14.2 in {v15_changed} articles:")
    for r in results_log:
        print(f"    '{r['title'][:50]}' | v14={r['v14']} → v15={r['v15']}")

with open("/tmp/v15_vs_v14_results.json", "w") as f:
    json.dump({"v14_agree": v14_agree, "v14_disagree": v14_disagree,
               "v15_agree": v15_agree, "v15_disagree": v15_disagree,
               "v15_changed": v15_changed, "era_mismatches": era_mismatches,
               "affil_mismatches": affil_mismatches, "sentiment_found": sentiment_found,
               "results": results_log}, f, indent=2, ensure_ascii=False)
print(f"\nSaved to /tmp/v15_vs_v14_results.json")
