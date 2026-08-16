#!/usr/bin/env python3
"""
================================================================
GOOGLE COLAB RUNNER — ID-Political-Sentiment-Tracker Pipeline Test
================================================================
Run this in Google Colab (with GPU) to test the 3 patch files
(entity_v14.2, context_v18.1, nlp_v15) on live production data.

INSTRUCTIONS (copy-paste to Colab):
1. Upload this file + 3 patch files to Colab
2. Set your Supabase env vars (cell below)
3. Run all cells
4. Download the 3 output JSON files
5. Send them back to me for analysis

OUTPUT FILES (send these back):
  - colab_entity_results.json    (entity resolution comparison)
  - colab_context_results.json   (context worker comparison)
  - colab_nlp_results.json       (nlp worker predictions sample)
"""

# ================================================================
# CELL 1: Install dependencies (run once)
# ================================================================
# !pip install stanza torch transformers supabase --quiet
# !python -c "import stanza; stanza.download('id', processors='tokenize,pos,lemma,depparse')"

# ================================================================
# CELL 2: Set environment variables
# ================================================================
import os
os.environ["SUPABASE_URL"] = "https://bawvxtivogcuwvqdqoae.supabase.co"
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "YOUR_SERVICE_ROLE_KEY_HERE"  # ← paste your key

# ================================================================
# CELL 3: Configuration
# ================================================================
NUM_ARTICLES = 100  # how many articles to test (recommend 100-200)
BATCH_SAVE_EVERY = 10  # save progress every N articles (crash recovery)
OUTPUT_DIR = "/content"  # Colab default

# ================================================================
# CELL 4: Load patch code (paste the 3 patch files content here, OR upload them)
# ================================================================
# Option A: Upload files
# from google.colab import files
# uploaded = files.upload()  # upload entity_resolution_worker_v14.py, context_worker_v18.py, nlp_worker_v15.py

# Option B: Paste content inline (copy from finetuning/patches/)
# For now, inline the key functions:

import sys, json, re, time, gc
from collections import Counter
import stanza
import torch

print("Loading Stanza (tokenize,pos,lemma,depparse) with GPU...")
NLP = stanza.Pipeline('id', processors='tokenize,pos,lemma,depparse',
                      verbose=False, use_gpu=torch.cuda.is_available(),
                      batch_size=32, download_method=None)
print(f"Stanza ready. GPU: {torch.cuda.is_available()}\n")

# v14.2 EXPANDED verb sets (from patch file)
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

# ================================================================
# CELL 5: Entity Resolution v14.2 (from patch)
# ================================================================
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

def check_semantic_role(sent, entity_start, entity_end):
    result = {"has_sentiment": False, "has_attribution": False, "verb": None, "role": None}
    entity_word = None
    for word in sent.words:
        if word.start_char <= entity_start < word.end_char:
            entity_word = word; break
        if entity_start <= word.start_char < entity_end:
            entity_word = word; break
    if entity_word is None: return result
    if entity_word.deprel in ('nsubj','nsubj:pass','obj','iobj','csubj','obl'):
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

def resolve_entity_v14_2(article, entities):
    title = (article.get("title") or "").strip()
    body = (article.get("text") or "").strip()
    title_lower = title.lower()
    if not body or len(body) < 50: return None, []
    # pre-filter
    candidates = []
    for eid, e in entities.items():
        mentions = find_mentions(e["name"], e.get("aliases",[]), body)
        if mentions:
            in_title = e["name"].lower() in title_lower or any(a.lower() in title_lower for a in e.get("aliases",[]))
            candidates.append((eid, e, mentions, in_title))
    if not candidates: return None, []
    try:
        doc = NLP(body)
    except: return None, []
    sentences = []
    for sent in doc.sentences:
        if len(sent.text.strip()) > 10:
            sentences.append({"text": sent.text, "start": sent.tokens[0].start_char if sent.tokens else 0,
                              "end": sent.tokens[-1].end_char if sent.tokens else 0, "parsed": sent})
    if not sentences: return None, []
    total_sents = len(sentences)
    results = []
    for eid, e, mentions, in_title in candidates:
        sent_indices = set()
        has_sent = has_attr = False
        sverbs = []
        roles = []
        for start, end, _ in mentions:
            for sidx, s in enumerate(sentences):
                if s["start"] <= start < s["end"]:
                    sent_indices.add(sidx)
                    role = check_semantic_role(s["parsed"], start, end)
                    if role["has_sentiment"]: has_sent = True; sverbs.append(role["verb"])
                    if role["has_attribution"]: has_attr = True
                    if role["role"]: roles.append(role["role"])
                    break
        dom = len(sent_indices) / total_sents if total_sents > 0 else 0
        results.append({"entity_id": eid, "name": e["name"], "in_title": in_title,
                        "count": len(mentions), "topic_dominance": round(dom,3),
                        "has_sentiment": has_sent, "has_attribution": has_attr,
                        "sentiment_verbs": list(set(sverbs)), "roles": list(set(roles))})
    results.sort(key=lambda x: (x["has_sentiment"], x["topic_dominance"]>=0.25, x["in_title"], x["count"]), reverse=True)
    return results[0], results

# ================================================================
# CELL 6: Context Worker v18.1 (from patch)
# ================================================================
MAX_CTX_WORDS = 180

def extract_context_v18(article, entity_name, entity_aliases, main_data):
    body = (article.get("text") or "").strip()
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
    anchor_idx = -1
    for sidx, s in enumerate(sentences):
        for start, end, _ in mentions:
            if s["start"] <= start < s["end"]:
                anchor_idx = sidx; break
        if anchor_idx >= 0: break
    if anchor_idx < 0: return None
    anchor_sent = sentences[anchor_idx]
    first_mention = next((m for m in mentions if anchor_sent["start"] <= m[0] < anchor_sent["end"]), mentions[0])
    has_sent = main_data.get("has_sentiment", False) if main_data else False
    has_attr = main_data.get("has_attribution", False) if main_data else False
    is_main_actor = any("nsubj" in r or "obj" in r for r in (main_data.get("roles",[]) if main_data else []))
    root_verb = None
    for word in anchor_sent["parsed"].words:
        if word.deprel == 'root':
            root_verb = (word.lemma or word.text).lower(); break
    context_parts = [anchor_sent["text"]]
    if anchor_idx + 1 < len(sentences) and not has_sent:
        context_parts.append(sentences[anchor_idx + 1]["text"])
    ctx_text = " ".join(context_parts)
    wl = ctx_text.split()
    if len(wl) > MAX_CTX_WORDS:
        ctx_text = " ".join(wl[:MAX_CTX_WORDS])
    para_idx = body[:first_mention[0]].count('\n\n')
    attr_v18 = 40 if has_sent else (10 if has_attr else 10)
    attr_v17 = 40 if has_sent else (25 if has_attr else 10)
    actor_score = 30 if is_main_actor else 10
    pos_score = 20 if para_idx == 0 else (12 if para_idx <= 2 else 5)
    return {"ctx_text": ctx_text[:300], "quality_v18": attr_v18+actor_score+pos_score+10,
            "quality_v17": attr_v17+actor_score+pos_score+10, "attr_v18": attr_v18,
            "attr_v17": attr_v17, "root_verb": root_verb, "has_sentiment": has_sent,
            "has_attribution": has_attr, "is_main_actor": is_main_actor}

# ================================================================
# CELL 7: Fetch articles from Supabase + run pipeline
# ================================================================
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

# Fetch articles
print(f"Fetching {NUM_ARTICLES} processed articles...")
res = sb.table("raw_texts").select("id, title, text, source_url, status, resolver_version").eq("status","processed").limit(NUM_ARTICLES).execute()
articles = res.data
print(f"Got {len(articles)} articles")

# Fetch existing v12/v17 data for comparison
art_ids = [a["id"] for a in articles]
res_ctx = sb.table("entity_contexts").select("raw_text_id, entity_id, context_text, metadata").in_("raw_text_id", art_ids).execute()
res_map = sb.table("article_entity_map").select("raw_text_id, entity_id, is_main_entity, confidence").in_("raw_text_id", art_ids).execute()

# Fetch ALL entities
res_ent = sb.table("political_entities").select("id, canonical_name, aliases").execute()
ent_full = {e["id"]: {"name": e["canonical_name"], "aliases": e.get("aliases") or []} for e in res_ent.data}
ent_map = {e["id"]: e["canonical_name"] for e in res_ent.data}
print(f"Entities: {len(ent_full)} | Contexts: {len(res_ctx.data)} | Mappings: {len(res_map.data)}")

# Build lookup
v12_main = {}
for m in res_map.data:
    if m.get("is_main_entity"):
        v12_main[m["raw_text_id"]] = ent_map.get(m["entity_id"], "?")
v17_ctx = {}
for c in res_ctx.data:
    v17_ctx[(c["raw_text_id"], c["entity_id"])] = c

# ================================================================
# CELL 8: RUN PIPELINE (this is the main loop)
# ================================================================
entity_results = []
context_results = []
stats = Counter()
t0 = time.time()

for i, a in enumerate(articles):
    art_id = a["id"]
    # Run v14.2 entity resolution
    main14, all14 = resolve_entity_v14_2(a, ent_full)
    v12_m = v12_main.get(art_id)

    # Run v18.1 context for main entity
    ctx_v18 = None
    if main14:
        ent_info = ent_full.get(main14["entity_id"], {})
        ctx_v18 = extract_context_v18(a, ent_info["name"], ent_info.get("aliases",[]), main14)
        # get v17 context for comparison
        v17 = v17_ctx.get((art_id, main14["entity_id"]))
        v17_quality = (v17.get("metadata") or {}).get("quality_score") if v17 else None
        v17_attr = (v17.get("metadata") or {}).get("attr_score") if v17 else None

    # Record entity result
    entity_results.append({
        "art_id": art_id,
        "title": (a.get("title") or "")[:80],
        "v12_main": v12_m,
        "v14_main": main14["name"] if main14 else None,
        "v14_sentiment": main14["has_sentiment"] if main14 else False,
        "v14_sentiment_verbs": main14["sentiment_verbs"] if main14 else [],
        "v14_dominance": main14["topic_dominance"] if main14 else 0,
        "v14_in_title": main14["in_title"] if main14 else False,
        "v14_count": main14["count"] if main14 else 0,
        "v14_roles": main14["roles"] if main14 else [],
        "v14_all_entities": len(all14) if all14 else 0,
        "agree_with_v12": (v12_m == main14["name"]) if main14 and v12_m else None,
    })

    # Record context result
    if ctx_v18:
        context_results.append({
            "art_id": art_id,
            "title": (a.get("title") or "")[:80],
            "entity": main14["name"],
            "v17_quality": v17_quality,
            "v18_quality": ctx_v18["quality_v18"],
            "v17_attr": v17_attr,
            "v18_attr": ctx_v18["attr_v18"],
            "root_verb": ctx_v18["root_verb"],
            "has_sentiment": ctx_v18["has_sentiment"],
            "has_attribution": ctx_v18["has_attribution"],
            "is_main_actor": ctx_v18["is_main_actor"],
            "ctx_text": ctx_v18["ctx_text"],
        })

    if main14:
        stats["v14_found"] += 1
        if main14["has_sentiment"]: stats["sentiment_detected"] += 1
        if v12_m:
            if v12_m == main14["name"]: stats["agree"] += 1
            else: stats["disagree"] += 1
        else:
            stats["v14_found_v12_missed"] += 1
    else:
        stats["v14_no_entity"] += 1

    gc.collect()  # prevent memory leak

    if (i+1) % BATCH_SAVE_EVERY == 0:
        elapsed = time.time() - t0
        rate = (i+1) / elapsed
        print(f"  [{i+1}/{len(articles)}] {elapsed:.0f}s, {rate:.1f} art/s | "
              f"found={stats['v14_found']} sent={stats['sentiment_detected']} "
              f"agree={stats['agree']} disagree={stats['disagree']}", flush=True)
        # incremental save
        with open(f"{OUTPUT_DIR}/colab_entity_results.json","w") as f:
            json.dump({"stats": dict(stats), "results": entity_results,
                       "total_time": time.time()-t0, "completed": i+1}, f, indent=2, ensure_ascii=False)
        with open(f"{OUTPUT_DIR}/colab_context_results.json","w") as f:
            json.dump({"results": context_results, "completed": i+1}, f, indent=2, ensure_ascii=False)

# Final save
total_time = time.time() - t0
with open(f"{OUTPUT_DIR}/colab_entity_results.json","w") as f:
    json.dump({"stats": dict(stats), "results": entity_results,
               "total_time": total_time, "completed": len(articles)}, f, indent=2, ensure_ascii=False)
with open(f"{OUTPUT_DIR}/colab_context_results.json","w") as f:
    json.dump({"results": context_results, "completed": len(articles)}, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"DONE! {len(articles)} articles in {total_time:.0f}s ({total_time/len(articles):.1f}s/art)")
print(f"{'='*60}")
print(f"Stats: {dict(stats)}")
print(f"\nOutput files:")
print(f"  {OUTPUT_DIR}/colab_entity_results.json")
print(f"  {OUTPUT_DIR}/colab_context_results.json")
print(f"\nDownload these 2 files and send to me for analysis!")

# ================================================================
# CELL 9 (optional): Test NLP Worker v15 if you want
# ================================================================
# This requires loading the sentiment model. Only run if you want to test
# the full end-to-end NLP pipeline. Uncomment below:
#
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch.nn.functional as F
#
# print("Loading sentiment model...")
# sent_tok = AutoTokenizer.from_pretrained("apriandito/indobert-sentiment-classifier")
# sent_model = AutoModelForSequenceClassification.from_pretrained("apriandito/indobert-sentiment-classifier")
# sent_model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
#
# # Test on 20 sample contexts from context_results
# nlp_results = []
# for cr in context_results[:20]:
#     enc = sent_tok(cr["entity"], cr["ctx_text"], truncation=True, max_length=256, return_tensors="pt").to(sent_model.device)
#     with torch.no_grad():
#         probs = F.softmax(sent_model(**enc).logits, dim=-1)[0]
#     pred_idx = probs.argmax().item()
#     labels = sent_model.config.id2label
#     nlp_results.append({
#         "art_id": cr["art_id"], "entity": cr["entity"],
#         "label": labels[pred_idx], "confidence": round(probs[pred_idx].item(), 3),
#         "probs": [round(p,3) for p in probs.tolist()],
#         "ctx_text": cr["ctx_text"][:100],
#     })
#
# with open(f"{OUTPUT_DIR}/colab_nlp_results.json","w") as f:
#     json.dump(nlp_results, f, indent=2, ensure_ascii=False)
# print(f"Saved {OUTPUT_DIR}/colab_nlp_results.json")
