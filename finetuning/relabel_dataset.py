#!/usr/bin/env python3
"""
relabel_dataset.py
==================
Turns the raw 909-row extraction dataset (dataset.jsonl) into TWO train-ready,
sentence-pair datasets that match the base-model architecture in
packages/nlp/sentiment_model.py:

  tokenizer(entity_name, context_text)  ->  label

Outputs:
  - dataset_relevancy.jsonl   : (entity_name, context_text) -> {relevant, not_relevant}
  - dataset_sentiment.jsonl   : (entity_name, context_text) -> {positive, neutral, negative}
                                 (only rows where relevancy == relevant)

Cleaning pipeline (each step is auditable in the `meta` field of every output row):

  1. BYLINE STRIP   — remove journalist bylines / datelines that leaked into
                      context_text:  "(Mir/P-3)", "JAKARTA -", "INFO TEMPO -",
                      "TRIBUNNEWS.COM -", "Penulis, ... - Waktu membaca: N menit", ...
  2. CORRUPTION FLAG — detect stitched / mismatched context+article pairs
                      (byline inside body, common-prefix length == 0 AND
                       context not in article even after normalisation).
                      These are emitted with relevancy=not_relevant and a
                      `needs_reextract` flag so they can be re-fetched.
  3. GOLD OVERRIDE   — for rows present in gold_labels.jsonl, the human
                      gold_label / gold_relevancy wins over everything.
  4. HEURISTIC RELABEL — for the unreviewed majority, apply the defect
                      patterns discovered in the critical review:
                        a. speaker_vs_target   -> neutral  + relevant
                        b. misattribution_bg   -> neutral  + not_relevant
                        c. wrong_polarity cues -> flip with low confidence
                      Rows that no rule fires on keep their pseudo_label but
                      get confidence=0.5 (heuristic) so the finetune script
                      can down-weight them.
  5. SENTENCE-PAIR FORM — emit {premise, hypothesis, label, meta} where
                      premise = entity_name (+ alias hint) and
                      hypothesis = cleaned context_text, truncated to 256
                      tokens at TRAIN time (not here — we keep full text and
                      let the tokenizer truncate, matching production).

The script is deterministic (seeded) and prints a full audit report.
"""
from __future__ import annotations
import json, re, random, hashlib
from pathlib import Path
from collections import Counter, defaultdict

HERE = Path(__file__).parent
RAW = HERE / "dataset.jsonl"
GOLD = HERE / "gold_labels.jsonl"
OUT_REL = HERE / "dataset_relevancy.jsonl"
OUT_SENT = HERE / "dataset_sentiment.jsonl"
OUT_AUDIT = HERE / "relabel_audit.json"

random.seed(42)

# ---------------------------------------------------------------------------
# 1. Byline / dateline patterns that leak into context_text
# ---------------------------------------------------------------------------
BYLINE_PATTERNS = [
    # Tempo-style journalist initials in parentheses  "(Mir/P-3)" "(abc/X-2)"
    re.compile(r"\s*\([A-Za-z]{2,5}/[A-Za-z0-9-]+\)\s*"),
    # Outlet datelines at start: "JAKARTA -", "MEDAN -", "INFO TEMPO -",
    # "TRIBUNNEWS.COM -", "JATIMTIMES -", "INDORAYA -", "ANTARA -"
    re.compile(r"^(?:[A-Z][A-Z0-9.\- ]{2,30}?(?:COM|TIMES|TEMPO|RAYA|NEWS)?\s*-\s+)", re.MULTILINE),
    # "Penulis, NAME - Peranan, Jurnalis ... - Waktu membaca: N menit"
    re.compile(r"^Penulis,\s.*?Waktu membaca:\s*\d+\s*menit\s*", re.MULTILINE | re.DOTALL),
    # "Headline " prefix that leaked from a stitched article
    re.compile(r"^Headline\s+", re.MULTILINE),
]

def strip_bylines(text: str) -> tuple[str, list[str]]:
    """Return (cleaned_text, list_of_removed_fragments)."""
    removed = []
    for pat in BYLINE_PATTERNS:
        def _sub(m):
            removed.append(m.group(0).strip())
            return " "
        text = pat.sub(_sub, text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text, removed

# ---------------------------------------------------------------------------
# 2. Corruption detection: stitched / mismatched context+article
# ---------------------------------------------------------------------------
def is_context_in_article(context: str, article: str) -> bool:
    """Whitespace-normalised substring test."""
    c = re.sub(r"\s+", " ", context).strip()
    a = re.sub(r"\s+", " ", article)
    return bool(c) and c in a

def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

# Strong corruption signal: a Tempo-style journalist byline "(ABC/X-Y)" appears
# INSIDE the body (not at the very start). This only happens when sentences
# from different articles have been stitched together.
BYLINE_IN_BODY_RE = re.compile(r"\([A-Za-z]{2,5}/[A-Za-z0-9-]+\)")

# ---------------------------------------------------------------------------
# 3. Heuristic defect detectors (informed by the critical review)
# ---------------------------------------------------------------------------
# Verbs where the entity is the SPEAKER, not the target. When the context is
# an attribution sentence built around one of these, the sentiment expressed
# is the entity's, not toward the entity  ->  sentiment toward entity = neutral.
SPEAKER_VERBS = {
    "mengatakan", "menyatakan", "menegaskan", "menjelaskan", "menambahkan",
    "mengimbau", "mengingatkan", "menyampaikan", "mengaku", "mengklaim",
    "menuding", "menilai", "menyebut", "mengungkapkan", "menjawab",
    "menyindir", "mengkritik", "mengecam", "menyerang", "membela",
    "ujar", "tutur", "tegas", "kata", "sebut", "ungkap", "papar",
    "nyatakan", "imbau", "sampaikan", "tambah", "jelaskan",
}

# Markers that the entity is only a TEMPORAL/BACKGROUND anchor (misattribution).
BACKGROUND_MARKERS = [
    "era presiden", "era pemerintahan", "zaman pemerintahan",
    "pada masa pemerintahan", "di bawah pemerintahan",
    "pada zaman", "seperti era",
]

# Strong-negative lexical cues toward a person (conviction, arrest, etc.)
NEG_CUES = [
    "divonis", "ditahan", "dicekal", "ditangkap", "dipidana", "terbukti bersalah",
    "korupsi", "suap", "penjara", "pidana", "dakwaan", "tersangka",
    "skandal", "dugaan korupsi", "tindak pidana",
]
POS_CUES = [
    "dipuji", "diekskan", "apresiasi", "dukungan", "mendukung",
    "berprestasi", "sukses", "keberhasilan", "optimistis", "mengapresiasi",
    "komitmen kuat", "terus mendorong", "menjunjung tinggi",
]

def detect_speaker_vs_target(context: str, entity_lower: str) -> bool:
    """CONSERVATIVE: True only when the context is a PURE attribution sentence
    (entity is the speaker) AND no sentiment cues target the entity.

    We require BOTH:
      (1) entity is the grammatical subject of a speaking verb, AND
      (2) the context contains NO positive/negative cues that could be
          directed at the entity (otherwise the sentiment may legitimately
          be toward them, e.g. "Prabowo was praised for...").
    """
    cl = context.lower()
    # (2) gate: if any sentiment cue present, do NOT fire (too ambiguous).
    if any(c in cl for c in NEG_CUES + POS_CUES):
        return False
    # (1) entity subject of a speaking verb
    first_token = entity_lower.split()[0]
    for verb in SPEAKER_VERBS:
        for m in re.finditer(rf"\b{re.escape(verb)}\b", cl):
            window = cl[max(0, m.start()-60):m.end()+10]
            if entity_lower in window or first_token in window:
                return True
    return False

def detect_background_mention(context: str, entity_lower: str) -> bool:
    """True if entity is only a temporal/possessive anchor (misattribution).

    Fires when:
      - a background temporal marker is present ('era presiden X', 'pada masa
        pemerintahan X', ...) and the entity appears exactly once, OR
      - the entity appears exactly once AND the context opens with a different
        proper noun (the context is 'about' someone/something else).
    """
    cl = context.lower()
    occ = cl.count(entity_lower)
    if occ == 0:
        # entity not literally present (alias) -> cannot decide, don't fire
        return False
    # temporal anchor pattern
    for marker in BACKGROUND_MARKERS:
        if marker in cl and occ == 1:
            return True
    # single-mention + context starts with a different PROPN-ish token
    if occ == 1:
        first_tokens = context.strip().split()[:6]
        # if the entity is not among the first 6 tokens, it's likely background
        first_tokens_lower = " ".join(t.lower() for t in first_tokens)
        first_name = entity_lower.split()[0]
        if entity_lower not in first_tokens_lower and first_name not in first_tokens_lower:
            # but only fire if a different salient person/party is foregrounded
            if re.search(r"\b(Presiden|Menteri|Gubernur|Wali Kota|Bupati|Ketua|Sekretaris|Jenderal|Kapolri|Jaksa)\s+[A-Z]", context):
                return True
    return False

def detect_wrong_polarity(context: str, pseudo: str) -> tuple[str | None, float]:
    """If pseudo is positive but strong negative cues present (or vice versa),
    suggest a flip with a confidence."""
    cl = context.lower()
    neg_hits = sum(1 for c in NEG_CUES if c in cl)
    pos_hits = sum(1 for c in POS_CUES if c in cl)
    if pseudo == "positive" and neg_hits >= 1 and pos_hits == 0:
        return "negative", 0.6 + 0.1 * (neg_hits - 1)
    if pseudo == "negative" and pos_hits >= 2 and neg_hits == 0:
        return "positive", 0.6
    return None, 0.0

# ---------------------------------------------------------------------------
# 4. Load inputs
# ---------------------------------------------------------------------------
rows = []
with open(RAW) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

gold = {}
with open(GOLD) as f:
    for line in f:
        line = line.strip()
        if line:
            g = json.loads(line)
            gold[g["row_index"]] = g

print(f"Loaded {len(rows)} raw rows, {len(gold)} gold labels.")

# ---------------------------------------------------------------------------
# 5. Build alias hints per entity (short form for the premise)
# ---------------------------------------------------------------------------
entity_aliases = defaultdict(set)
for r in rows:
    name = r["entity_name"]
    # crude short form: last 1-2 tokens of canonical name
    parts = name.split()
    if len(parts) >= 2:
        entity_aliases[name].add(parts[0])      # "Prabowo"
        if len(parts) >= 3:
            entity_aliases[name].add(parts[-1])  # "Subianto"

# ---------------------------------------------------------------------------
# 6. Relabel every row
# ---------------------------------------------------------------------------
rel_out, sent_out = [], []
audit = {
    "total": len(rows),
    "byline_stripped": 0,
    "corruption_flagged": 0,
    "gold_overridden": 0,
    "heuristic_speaker_vs_target": 0,
    "heuristic_background": 0,
    "heuristic_wrong_polarity": 0,
    "kept_pseudo_low_conf": 0,
    "relevancy_relevant": 0,
    "relevancy_not_relevant": 0,
    "sentiment_positive": 0,
    "sentiment_neutral": 0,
    "sentiment_negative": 0,
}

for idx, r in enumerate(rows):
    entity = r["entity_name"]
    entity_lower = entity.lower()
    ctx_raw = r["context_text"] or ""
    article = r["article_text"] or ""
    pseudo = r["pseudo_label"]

    # 1. byline strip
    ctx_clean, removed_bylines = strip_bylines(ctx_raw)
    if removed_bylines:
        audit["byline_stripped"] += 1

    # 2. corruption detection — CONSERVATIVE: only flag on strong evidence.
    #    "context not in truncated article" alone is NOT corruption (the
    #    article_text field is hard-truncated to 1000 chars in the export,
    #    so ~33% of legitimate contexts naturally fail the substring test).
    #    Real corruption signals:
    #      (a) a journalist byline "(ABC/X-Y)" appears INSIDE the body, OR
    #      (b) the "Headline " prefix leaked into the context.
    in_article = is_context_in_article(ctx_clean, article)
    byline_in_body = bool(BYLINE_IN_BODY_RE.search(ctx_raw.strip()[60:]))  # skip dateline
    headline_leak = bool(re.match(r"^Headline\s+", ctx_raw))
    corruption = byline_in_body or headline_leak
    if corruption:
        audit["corruption_flagged"] += 1

    # default label/relevancy
    label = pseudo
    relevancy = "relevant"
    confidence = 0.5      # heuristic baseline
    source = "pseudo"
    defect = "none"
    needs_reextract = False

    # 3. gold override (highest priority)
    if idx in gold:
        g = gold[idx]
        label = g["gold_label"]
        relevancy = g["gold_relevancy"]
        confidence = 1.0
        source = "gold"
        defect = g["defect_class"]
        audit["gold_overridden"] += 1
    else:
        # 4. heuristic relabel
        if corruption:
            relevancy = "not_relevant"
            label = "neutral"
            confidence = 0.9
            source = "heuristic_corruption"
            defect = "corruption_stitch"
            needs_reextract = True
            audit["heuristic_wrong_polarity"] += 0
        elif detect_background_mention(ctx_clean, entity_lower):
            relevancy = "not_relevant"
            label = "neutral"
            confidence = 0.7
            source = "heuristic_background"
            defect = "misattribution_background"
            audit["heuristic_background"] += 1
        else:
            flip, flip_conf = detect_wrong_polarity(ctx_clean, pseudo)
            if flip is not None:
                label = flip
                confidence = flip_conf
                source = "heuristic_polarity"
                defect = "wrong_polarity"
                audit["heuristic_wrong_polarity"] += 1
            elif detect_speaker_vs_target(ctx_clean, entity_lower):
                # speaker vs target -> sentiment TOWARD entity is neutral,
                # but the context IS about the entity (they're speaking) so
                # relevancy stays relevant.
                label = "neutral"
                confidence = 0.7
                source = "heuristic_speaker"
                defect = "speaker_vs_target"
                audit["heuristic_speaker_vs_target"] += 1
            else:
                # no rule fired -> keep pseudo but low confidence
                confidence = 0.5
                source = "pseudo_kept"
                audit["kept_pseudo_low_conf"] += 1

    # tally
    if relevancy == "relevant":
        audit["relevancy_relevant"] += 1
    else:
        audit["relevancy_not_relevant"] += 1
    audit[f"sentiment_{label}"] = audit.get(f"sentiment_{label}", 0) + 1

    # alias hint for premise: "Prabowo Subianto (Prabowo)"
    alias_hint = ""
    if entity in entity_aliases and entity_aliases[entity]:
        alias_hint = f" ({', '.join(sorted(entity_aliases[entity])[:2])})"

    premise = f"{entity}{alias_hint}"
    hypothesis = ctx_clean

    meta = {
        "row_index": idx,
        "raw_text_id": r["raw_text_id"],
        "source_url": r["source_url"],
        "pseudo_label": pseudo,
        "label_source": source,
        "confidence": round(confidence, 3),
        "defect_class": defect,
        "bylines_removed": removed_bylines,
        "context_in_article": in_article,
        "needs_reextract": needs_reextract,
        "context_len": len(hypothesis),
    }

    # relevancy dataset (ALL rows, both classes)
    rel_out.append({
        "premise": premise,
        "hypothesis": hypothesis,
        "label": relevancy,
        "meta": meta,
    })
    # sentiment dataset (only relevant rows)
    if relevancy == "relevant":
        sent_out.append({
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label,
            "meta": meta,
        })

# ---------------------------------------------------------------------------
# 7. Write outputs
# ---------------------------------------------------------------------------
with open(OUT_REL, "w") as f:
    for r in rel_out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(OUT_SENT, "w") as f:
    for r in sent_out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(OUT_AUDIT, "w") as f:
    json.dump(audit, f, indent=2)

# ---------------------------------------------------------------------------
# 8. Audit report
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("RELABEL AUDIT")
print("="*60)
for k, v in audit.items():
    print(f"  {k:35s} {v}")
print()
print(f"Relevancy dataset -> {OUT_REL}  ({len(rel_out)} rows)")
print(f"Sentiment dataset -> {OUT_SENT}  ({len(sent_out)} rows)")
print(f"Audit JSON        -> {OUT_AUDIT}")

# class balance of the sentiment set
sc = Counter(r["label"] for r in sent_out)
print("\nSentiment dataset class balance (relevant rows only):")
for k in ["positive", "neutral", "negative"]:
    print(f"  {k:10s} {sc.get(k,0)}  ({sc.get(k,0)/len(sent_out)*100:.1f}%)")

# class balance of relevancy set
rc = Counter(r["label"] for r in rel_out)
print("\nRelevancy dataset class balance:")
for k in ["relevant", "not_relevant"]:
    print(f"  {k:15s} {rc.get(k,0)}  ({rc.get(k,0)/len(rel_out)*100:.1f}%)")
