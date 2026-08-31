#!/usr/bin/env python3
"""
finalize_dataset_v11.py
======================
Final cleanup pass on dataset_train_v11.jsonl:
  1. Remove rows where entity (or short form) is genuinely NOT in final text
  2. Remove rows with text < 80 chars (too short for meaningful context)
  3. Remove rows with text > 700 chars (re-extract if needed)
  4. Validate all fields are present and non-empty
  5. Output: dataset_train_v11_final.jsonl (production-ready)
"""
import json
import re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
INPUT  = BASE / "datasets" / "dataset_train_v11.jsonl"
OUTPUT = BASE / "datasets" / "dataset_train_v11_final.jsonl"

SHORT_FORMS = {
    "joko widodo": ["jokowi"],
    "prabowo subianto": ["prabowo"],
    "megawati soekarnoputri": ["megawati"],
    "susilo bambang yudhoyono": ["sby"],
    "basuki tjahaja purnama": ["ahok"],
    "abdurrahman wahid": ["gus dur"],
    "ma'ruf amin": ["ma'ruf", "maruf"],
    "muhaimin iskandar": ["cak imin"],
    "erick thohir": ["erick"],
    "bima arya sugiarto": ["bima"],
    "sri mulyani indrawati": ["sri mulyani"],
    "ridwan kamil": ["rk", "kang emil"],
    "anies baswedan": ["anies"],
    "pramono anung": ["pram"],
    "puan maharani": ["puan"],
    "agus harimurti yudhoyono": ["ahy"],
    "sandiaga uno": ["sandi"],
    "bahlil lahadalia": ["bahlil"],
    "setya novanto": ["setya"],
    "refly harun": ["refly"],
    "sufmi dasco ahmad": ["dasco"],
    "khofifah indar parawansa": ["khofifah"],
    "yusril ihza mahendra": ["yusril"],
    "miftachul akhyar": ["miftach"],
    "soekarno": ["bung karno", "soekarno"],
    "soeharto": ["pak harto"],
    "b j habibie": ["habibie"],
    "bj habibie": ["habibie"],
    "sri sultan hamengkubuwono ix": ["hamengkubuwono", "sri sultan"],
    "deddy corbuzier": ["deddy"],
    "thomas lembong": ["tom lembong"],
    "rocky gerung": ["rocky"],
    "rhenald kasali": ["rhenald"],
    "chatib basri": ["chatib"],
    "gunarso": ["gunarso"],
    "dito ariotedjo": ["dito"],
    "umar wirahadikusumah": ["umar"],
    "try sutrisno": ["try"],
    "hamengkubuwono x": ["sultan"],
    "ganjar pranowo": ["ganjar"],
    "mahfud md": ["mahfud"],
    "fenty noverita": ["fenty"],
    "bambang soesatyo": ["bamsoet"],
    "djamari chaniago": ["djamari"],
    "sudaryono": ["sudaryono"],
    "teddy indra wijaya": ["teddy"],
    "gibran rakabuming": ["gibran"],
    "bobby nasution": ["bobby"],
    "ridwan kamil": ["kang emil"],
}


def entity_in_text(entity, text):
    """Check if entity (or short form) appears in text."""
    if not entity or not text:
        return False
    entity_lower = entity.lower().strip()
    text_lower = text.lower()

    # Direct full match
    if entity_lower in text_lower:
        return True

    # First + last name
    parts = entity.split()
    if len(parts) >= 2:
        if parts[-1].lower() in text_lower and len(parts[-1]) >= 4:
            return True
        if parts[0].lower() in text_lower and len(parts[0]) >= 4:
            return True

    # Short forms
    if entity_lower in SHORT_FORMS:
        for sf in SHORT_FORMS[entity_lower]:
            if sf in text_lower:
                return True

    return False


def main():
    print("=" * 64)
    print("FINALIZE DATASET v11 → v11_final")
    print("=" * 64)

    rows = [json.loads(l) for l in open(INPUT) if l.strip()]
    print(f"Input: {len(rows)} rows\n")

    kept = []
    removed = {
        "entity_not_in_text": [],
        "too_short": [],
        "too_long": [],
        "missing_fields": [],
    }

    for i, r in enumerate(rows):
        entity = r.get("entity_name", "")
        text = r.get("text", "")
        label = r.get("label", "")
        premise = r.get("entity_premise", "")

        # Validate required fields
        if not entity or not text or not label or not premise:
            removed["missing_fields"].append(i)
            continue

        # Check entity in text
        if not entity_in_text(entity, text):
            removed["entity_not_in_text"].append({
                "row": i,
                "entity": entity,
                "text_first_100": text[:100],
            })
            continue

        # Check text length
        if len(text) < 80:
            removed["too_short"].append({"row": i, "entity": entity, "len": len(text)})
            continue

        if len(text) > 700:
            # Truncate to last complete sentence
            sub = text[:700]
            matches = list(re.finditer(r'[.!?]["\')\]]?\s+', sub))
            if matches:
                text = text[:matches[-1].end()]
            else:
                last_space = sub.rfind(' ')
                if last_space > 100:
                    text = sub[:last_space] + '.'
                else:
                    text = sub + '.'
                r["text"] = text
                r["context_chars"] = len(text)

        kept.append(r)

    # Write output
    with open(OUTPUT, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats
    label_dist = Counter(r["label"] for r in kept)
    rel_dist = Counter(r["gold_relevancy"] for r in kept)
    match_types = Counter(r.get("match_type", "none") for r in kept)

    print("REMOVAL SUMMARY:")
    print(f"  Entity not in text: {len(removed['entity_not_in_text'])}")
    print(f"  Too short (<80):    {len(removed['too_short'])}")
    print(f"  Too long (>700):   {len(removed['too_long'])} (truncated)")
    print(f"  Missing fields:    {len(removed['missing_fields'])}")
    print(f"\nKept: {len(kept)} / {len(rows)} ({len(kept)/len(rows)*100:.1f}%)")

    print(f"\nFINAL LABEL DISTRIBUTION:")
    for k, v in sorted(label_dist.items(), key=lambda x: -x[1]):
        print(f"  {k:10s}: {v:5d} ({v/len(kept)*100:5.1f}%)")

    print(f"\nRELEVANCY:")
    for k, v in sorted(rel_dist.items(), key=lambda x: -x[1]):
        print(f"  {k:15s}: {v:5d}")

    print(f"\nMATCH TYPES:")
    for k, v in match_types.most_common():
        print(f"  {k:20s}: {v}")

    print(f"\nRemoved rows (entity not in text):")
    for item in removed["entity_not_in_text"]:
        print(f"  Row {item['row']} [{item['entity']}]: {item['text_first_100'][:80]}")

    print(f"\nOutput: {OUTPUT} ({len(kept)} rows)")

    # Context length stats
    lens = [len(r["text"]) for r in kept]
    print(f"\nContext length: min={min(lens)}, max={max(lens)}, avg={sum(lens)/len(lens):.0f}")


if __name__ == "__main__":
    main()
