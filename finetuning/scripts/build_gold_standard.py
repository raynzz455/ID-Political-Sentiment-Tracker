#!/usr/bin/env python3
"""
build_gold_standard.py
=====================
Complete pipeline: clean dataset → validate entity → build gold standard.

Input: dataset_v10_final.jsonl (with LLM labels applied)
Output: dataset_gold_standard.jsonl (production-ready)

Steps:
  1. Re-extract context with SENTENCE-BOUNDARY alignment (no mid-word truncation)
  2. Validate entity presence in text (with short form support)
  3. Add entity_premise: "Tentang {entity}" for sentence-pair format
  4. Remove rows where entity genuinely not in text
  5. Remove rows with text < 80 chars
  6. Output gold standard dataset
"""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
INPUT  = BASE / "datasets" / "dataset_v10_final.jsonl"
OUTPUT = BASE / "datasets" / "dataset_gold_standard.jsonl"

SENTENCE_END_PATTERN = re.compile(r'[.!?]["\')\]]?\s+')
MAX_CONTEXT_SENTENCES = 3
MAX_CONTEXT_CHARS = 600
MIN_CONTEXT_CHARS = 80

SHORT_FORMS = {
    "joko widodo": ["jokowi"], "prabowo subianto": ["prabowo"],
    "megawati soekarnoputri": ["megawati"], "susilo bambang yudhoyono": ["sby"],
    "basuki tjahaja purnama": ["ahok"], "abdurrahman wahid": ["gus dur"],
    "ma'ruf amin": ["ma'ruf", "maruf"], "muhaimin iskandar": ["cak imin"],
    "erick thohir": ["erick"], "bima arya sugiarto": ["bima"],
    "sri mulyani indrawati": ["sri mulyani"], "ridwan kamil": ["rk", "kang emil"],
    "anies baswedan": ["anies"], "pramono anung": ["pram"],
    "puan maharani": ["puan"], "agus harimurti yudhoyono": ["ahy"],
    "sandiaga uno": ["sandi"], "bahlil lahadalia": ["bahlil"],
    "setya novanto": ["setya"], "refly harun": ["refly"],
    "sufmi dasco ahmad": ["dasco"], "khofifah indar parawansa": ["khofifah"],
    "yusril ihza mahendra": ["yusril"], "dedi mulyadi": ["dedi"],
    "deddy corbuzier": ["deddy"], "thomas lembong": ["tom lembong"],
    "rocky gerung": ["rocky"], "ganjar pranowo": ["ganjar"],
    "mahfud md": ["mahfud"], "gibran rakabuming raka": ["gibran"],
    "bobby nasution": ["bobby"], "tito karnavian": ["tito"],
    "jusuf kalla": ["jk"], "bacharuddin jusuf habibie": ["habibie"],
    "bj habibie": ["habibie"], "soekarno": ["bung karno", "soekarno"],
    "soeharto": ["pak harto"],
}


def find_entity_in_text(entity, text):
    """Find entity position in text, trying various forms."""
    if not entity or not text: return -1, "not_found"
    el = entity.lower().strip()
    tl = text.lower()
    pos = tl.find(el)
    if pos >= 0: return pos, "full_match"
    parts = entity.split()
    if len(parts) >= 2:
        fl = f"{parts[0]} {parts[-1]}".lower()
        pos = tl.find(fl)
        if pos >= 0: return pos, "first_last"
        last = parts[-1].lower()
        if len(last) >= 4 and last in tl: return tl.find(last), "last_name"
        first = parts[0].lower()
        if len(first) >= 4 and first in tl: return tl.find(first), "first_name"
    if el in SHORT_FORMS:
        for sf in SHORT_FORMS[el]:
            if sf in tl: return tl.find(sf), f"short_form:{sf}"
    return -1, "not_found"


def extract_sentence_context(article, entity_pos, entity_len):
    """Extract context with sentence-boundary alignment."""
    if not article: return ""
    start = entity_pos
    matches = list(SENTENCE_END_PATTERN.finditer(article[:entity_pos]))
    if matches: start = matches[-1].end()
    else: start = 0

    end = entity_pos + entity_len
    sentences_found = 0
    for match in SENTENCE_END_PATTERN.finditer(article[end:]):
        end = end + match.end()
        sentences_found += 1
        if sentences_found >= MAX_CONTEXT_SENTENCES - 1: break

    if end == entity_pos + entity_len:
        last_end = None
        for match in SENTENCE_END_PATTERN.finditer(article[end:]):
            last_end = match
        if last_end: end = end + last_end.end()
        else:
            available = min(len(article) - entity_pos, MAX_CONTEXT_CHARS)
            chunk = article[entity_pos:entity_pos + available]
            last_space = chunk.rfind(' ')
            end = entity_pos + (last_space if last_space > 50 else available)

    context = article[start:end]
    # Ensure ends with punctuation
    if context and context[-1] not in {'.','!','?','"',"'",')',']'}:
        matches = list(SENTENCE_END_PATTERN.finditer(context))
        if matches: context = context[:matches[-1].end()]
        elif len(context) > 50:
            last_space = context.rfind(' ')
            context = (context[:last_space] + '.') if last_space > 30 else (context + '.')

    if len(context) > MAX_CONTEXT_CHARS:
        sub = context[:MAX_CONTEXT_CHARS]
        matches = list(SENTENCE_END_PATTERN.finditer(sub))
        if matches: context = context[:matches[-1].end()]
        else:
            last_space = sub.rfind(' ')
            context = (sub[:last_space] + '.') if last_space > 50 else sub

    return context.strip()


def clean_text(text):
    if not text: return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('&nbsp;',' ').replace('&amp;','&').replace('&quot;','"').replace('&#39;',"'")
    text = re.sub(r'\[\d+\]', '', text)
    return text.strip()


def main():
    print("=" * 64)
    print("BUILD GOLD STANDARD DATASET")
    print("=" * 64)

    if not INPUT.exists():
        print(f"ERROR: {INPUT} not found. Run apply_llm_pseudo_labels.py first.")
        sys.exit(1)

    rows = [json.loads(l) for l in open(INPUT) if l.strip()]
    print(f"Input: {len(rows)} rows from {INPUT.name}")

    cleaned = []
    stats = {"extracted_ok": 0, "entity_not_found": 0, "too_short": 0, "no_label": 0}
    match_types = Counter()

    for i, r in enumerate(rows):
        entity = r.get("entity_name", "").strip()
        article = clean_text(r.get("article_text", ""))
        context_orig = clean_text(r.get("context_text", ""))
        label = r.get("gold_label") or r.get("pseudo_label") or r.get("label", "")

        if not label or label not in ("positive", "neutral", "negative"):
            stats["no_label"] += 1
            continue

        # Try article first, then context_text
        pos, mt = find_entity_in_text(entity, article)
        if pos >= 0:
            entity_len = len(entity) if mt == "full_match" else len(entity.split()[-1])
            new_context = extract_sentence_context(article, pos, entity_len)
            new_context = clean_text(new_context)
        else:
            # Fallback to context_text
            pos2, mt2 = find_entity_in_text(entity, context_orig)
            if pos2 >= 0:
                entity_len = len(entity) if mt2 == "full_match" else len(entity.split()[-1])
                new_context = extract_sentence_context(context_orig, pos2, entity_len)
                new_context = clean_text(new_context)
                if len(new_context) < MIN_CONTEXT_CHARS:
                    new_context = context_orig
                mt = "context_fallback"
            else:
                stats["entity_not_found"] += 1
                continue

        if len(new_context) < MIN_CONTEXT_CHARS:
            stats["too_short"] += 1
            continue

        match_types[mt] += 1
        stats["extracted_ok"] += 1

        cleaned.append({
            "text": new_context,
            "entity_name": entity,
            "entity_premise": f"Tentang {entity}",
            "label": label,
            "raw_text_id": r.get("raw_text_id", ""),
            "source_url": r.get("source_url", ""),
            "gold_relevancy": r.get("gold_relevancy", "relevant"),
            "label_source": r.get("label_source", "unknown"),
            "label_confidence": r.get("label_confidence", 0.5),
            "match_type": mt,
            "context_chars": len(new_context),
            "verification_reasoning": r.get("verification_reasoning", r.get("reasoning", "")),
        })

    with open(OUTPUT, "w") as f:
        for r in cleaned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    labels = Counter(r["label"] for r in cleaned)
    rels = Counter(r.get("gold_relevancy","") for r in cleaned)

    print(f"\nResults:")
    print(f"  Input:           {len(rows)}")
    print(f"  Cleaned:         {stats['extracted_ok']}")
    print(f"  Entity not found: {stats['entity_not_found']}")
    print(f"  Too short:       {stats['too_short']}")
    print(f"  No label:        {stats['no_label']}")
    print(f"\nMatch types:")
    for k, v in match_types.most_common():
        print(f"  {k:20s}: {v}")
    print(f"\nLabel distribution:")
    for k, v in sorted(labels.items(), key=lambda x: -x[1]):
        print(f"  {k:10s}: {v:5d} ({v/len(cleaned)*100:.1f}%)")
    imbalance = max(labels.values()) / min(labels.values()) if labels else 0
    print(f"  Imbalance: {imbalance:.1f}x")
    print(f"\nRelevancy:")
    for k, v in sorted(rels.items(), key=lambda x: -x[1]):
        print(f"  {k:15s}: {v:5d}")

    lens = [r["context_chars"] for r in cleaned]
    print(f"\nContext length: min={min(lens)}, max={max(lens)}, avg={sum(lens)/len(lens):.0f}")
    print(f"\nOutput: {OUTPUT} ({len(cleaned)} rows)")


if __name__ == "__main__":
    main()
