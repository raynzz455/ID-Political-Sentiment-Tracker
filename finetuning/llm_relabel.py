#!/usr/bin/env python3
"""
llm_relabel.py
==============
Second-pass LLM labeling for the rows that the heuristic relabeler could not
confidently fix (the ~412 "pseudo_kept" rows from relabel_dataset.py).

Strategy:
  - Batch 5 rows per LLM call (balances token cost vs. parse reliability).
  - Strict system prompt with FEW-SHOT examples of every hard defect class:
      * speaker_vs_target   (entity is the speaker, not the target)
      * misattribution_bg   (entity only a temporal/background anchor)
      * wrong_polarity      (subtle cues that flip the pseudo-label)
      * corruption_stitch   (byline inside body, stitched articles)
  - Output enforced as a JSON array; parser strips ```json fences.
  - Per-row fields collected (the NEW schema the user asked for):
      gold_label, gold_relevancy,
      entity_in_context, entity_is_main_subject, entity_corrected,
      context_flag, reasoning
  - Robust: if a batch fails to parse, retry once with 1-row-per-call;
    if that also fails, mark the row as llm_failed (kept as pseudo, low conf).

Output: llm_labels.jsonl  (one row per input pseudo_kept row, with all fields)

Usage:
    python llm_relabel.py [--max-rows N] [--batch-size 5] [--dry-run]
"""
from __future__ import annotations
import json, re, subprocess, sys, time, argparse, hashlib
from pathlib import Path
from collections import Counter

HERE = Path(__file__).parent
RAW = HERE / "dataset.jsonl"
GOLD = HERE / "gold_labels.jsonl"
RELEVANCY_DS = HERE / "dataset_relevancy.jsonl"   # has label_source for each row
OUT = HERE / "llm_labels.jsonl"
PROGRESS = HERE / "llm_progress.json"             # resume support

# ---------------------------------------------------------------------------
# System prompt — strict, with few-shot examples of EVERY hard defect class.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Anda adalah annotator ahli sentimen politik Indonesia tingkat senior.
Tugas: tentukan sentimen TERHADAP entitas politik (bukan sentimen YANG DIKATAKAN entitas).

DEFINISI KUNCI:
- Sentimen TERHADAP entitas = bagaimana TONE/FRAMING media menggambarkan entitas tersebut.
- Jika entitas mengkritik sesuatu, sentimen terhadap entitas = NEUTRAL (dia pembicara, bukan target).
- Jika entitas dipuji/dicela pihak lain, sentimen terhadap entitas = POSITIF/NEGATIF.

ATURAN RELEVANSI:
- "relevant" = entitas adalah SUBJEK UTAMA kalimat/konteks (target sentimen).
- "not_relevant" = entitas hanya disebut latar (era/masa/oleh Presiden X, dll).

CONTOH (few-shot):

[CONTOH 1] entity="Rocky Gerung"
context="Rocky menyebut pasal tersebut dalam KUHP sebagai pasal yang dungu."
-> gold_label="neutral", gold_relevancy="relevant",
   entity_in_context=true, entity_is_main_subject=true,
   entity_corrected=null, context_flag="speaker_not_target",
   reasoning="Rocky adalah pembicara yang mengkritik UU; sentimen terhadap Rocky netral."

[CONTOH 2] entity="Joko Widodo"
context="Eks Menteri Pendidikan era Presiden Jokowi ini juga dituntut membayar uang pengganti Rp809 miliar."
-> gold_label="neutral", gold_relevancy="not_relevant",
   entity_in_context=true, entity_is_main_subject=false,
   entity_corrected=null, context_flag="background_only",
   reasoning="Konteks tentang kasus Nadiem; Jokowi hanya anchor temporal 'era Presiden Jokowi'."

[CONTOH 3] entity="Joko Widodo"
context="Penasihat spiritual menyarankan Jokowi mengurangi manuver politik untuk memulihkan kepercayaan publik."
-> gold_label="negative", gold_relevancy="relevant",
   entity_in_context=true, entity_is_main_subject=true,
   entity_corrected=null, context_flag="clean",
   reasoning="Implikasi 'kepercayaan publik perlu dipulihkan' = kepercayaan hilang; framing negatif."

[CONTOH 4] entity="Thomas Lembong"
context="Eks Menteri Perdagangan Thomas Trikasih Lembong divonis bersalah korupsi impor gula."
-> gold_label="negative", gold_relevancy="relevant",
   entity_in_context=true, entity_is_main_subject=true,
   entity_corrected=null, context_flag="clean",
   reasoning="Vonis bersalah korupsi = sentimen sangat negatif terhadap entitas."

[CONTOH 5] entity="Prabowo Subianto"
context='Kritik itu bagus," pungkas Prabowo. (Mir/P-3) Presiden Prabowo menegaskan pemimpin yang menganjurkan aksi pembakaran adalah pengkhianat.'
-> gold_label="neutral", gold_relevancy="not_relevant",
   entity_in_context=true, entity_is_main_subject=false,
   entity_corrected=null, context_flag="corruption_stitch",
   reasoning="Byline jurnalis (Mir/P-3) di tengah body menandakan jahitan artikel berbeda."

[CONTOH 6] entity="Tito Karnavian"
context="Menteri Dalam Negeri Tito Karnavian menilai biaya pilkada memicu kepala daerah korupsi."
-> gold_label="neutral", gold_relevancy="relevant",
   entity_in_context=true, entity_is_main_subject=true,
   entity_corrected=null, context_flag="speaker_not_target",
   reasoning="Tito pembicara yang menganalisis; sentimen terhadap Tito netral."

OUTPUT WAJIB: JSON array, setiap elemen punya field:
  id, gold_label, gold_relevancy, entity_in_context, entity_is_main_subject,
  entity_corrected, context_flag, reasoning

Output HANYA JSON array di dalam ```json ... ``` block. Tidak ada teks lain."""

# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------
def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]

rows = load_jsonl(RAW)
gold = {g["row_index"]: g for g in load_jsonl(GOLD)}

# Identify pseudo_kept rows by re-running the heuristic classification logic
# (mirrors relabel_dataset.py but we only need the label_source label).
# We import the detector functions to stay DRY.
sys.path.insert(0, str(HERE))
from relabel_dataset import (
    strip_bylines, is_context_in_article, BYLINE_IN_BODY_RE,
    detect_speaker_vs_target, detect_background_mention, detect_wrong_polarity,
)

def classify_row(idx, r):
    """Return (label_source, label, relevancy, confidence, defect) — mirrors relabel_dataset.py."""
    if idx in gold:
        g = gold[idx]
        return "gold", g["gold_label"], g["gold_relevancy"], 1.0, g["defect_class"]
    entity_lower = r["entity_name"].lower()
    ctx_raw = r["context_text"] or ""
    ctx_clean, _ = strip_bylines(ctx_raw)
    byline_in_body = bool(BYLINE_IN_BODY_RE.search(ctx_raw.strip()[60:]))
    headline_leak = bool(re.match(r"^Headline\s+", ctx_raw))
    corruption = byline_in_body or headline_leak
    if corruption:
        return "heuristic_corruption", "neutral", "not_relevant", 0.9, "corruption_stitch"
    if detect_background_mention(ctx_clean, entity_lower):
        return "heuristic_background", "neutral", "not_relevant", 0.7, "misattribution_background"
    flip, conf = detect_wrong_polarity(ctx_clean, r["pseudo_label"])
    if flip is not None:
        return "heuristic_polarity", flip, "relevant", conf, "wrong_polarity"
    if detect_speaker_vs_target(ctx_clean, entity_lower):
        return "heuristic_speaker", "neutral", "relevant", 0.7, "speaker_vs_target"
    return "pseudo_kept", r["pseudo_label"], "relevant", 0.5, "none"

# Build the list of pseudo_kept row indices (the ones we need to LLM-label)
pseudo_kept = []
for idx, r in enumerate(rows):
    src, *_ = classify_row(idx, r)
    if src == "pseudo_kept":
        pseudo_kept.append(idx)

print(f"Total rows: {len(rows)} | pseudo_kept (need LLM): {len(pseudo_kept)}")

# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------
def build_batch_prompt(batch):
    """Build the user prompt for a batch of rows."""
    lines = ["BARIS:"]
    for i, (batch_idx, row) in enumerate(batch):
        lines.append(f"[{i}] entity=\"{row['entity_name']}\"")
        # truncate context to keep prompt manageable
        ctx = (row["context_text"] or "")[:600]
        lines.append(f"context=\"{ctx}\"")
        lines.append("")
    lines.append("Output HANYA JSON array di dalam ```json ... ``` block:")
    return "\n".join(lines)

def call_llm(prompt, timeout=120, retries=3):
    """Call z-ai chat CLI with exponential backoff retry."""
    last_err = None
    for attempt in range(retries):
        try:
            proc = subprocess.run(
                ["z-ai", "chat", "-p", prompt, "-s", SYSTEM_PROMPT],
                capture_output=True, text=True, timeout=timeout,
            )
            if proc.returncode != 0:
                last_err = f"CLI exit {proc.returncode}: {proc.stderr[:200]}"
                # rate limit / network error -> backoff
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # 1s, 2s, 4s
                    continue
                return None, last_err
            out = proc.stdout
            m = re.search(r'\{[\s\S]*"choices"[\s\S]*\}', out)
            if not m:
                last_err = "no JSON envelope in stdout"
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None, last_err
            env = json.loads(m.group(0))
            content = env["choices"][0]["message"]["content"]
            return content, None
        except subprocess.TimeoutExpired:
            last_err = "timeout"
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
        except Exception as e:
            last_err = str(e)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
    return None, last_err

def parse_json_array(content):
    """Robustly extract a JSON array from LLM output (strip ```json fences)."""
    if not content:
        return None
    # strip markdown fences
    m = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', content)
    if m:
        content = m.group(1)
    else:
        # try bare array
        m = re.search(r'(\[[\s\S]*\])', content)
        if m:
            content = m.group(1)
    try:
        arr = json.loads(content)
        if isinstance(arr, list):
            return arr
    except json.JSONDecodeError:
        pass
    return None

# ---------------------------------------------------------------------------
# Resume support — load from OUT (llm_labels.jsonl), NOT PROGRESS.
# PROGRESS is just a counter; OUT holds the actual per-row records.
# ---------------------------------------------------------------------------
done = {}
if OUT.exists():
    done = {r["row_index"]: r for r in load_jsonl(OUT)}
    print(f"Resuming: {len(done)} rows already labeled.")
if PROGRESS.exists():
    pinfo = json.load(open(PROGRESS))
    print(f"(progress file says {pinfo.get('done',0)}/{pinfo.get('total_pseudo_kept',0)})")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(args):
    batch_size = args.batch_size
    max_rows = args.max_rows or len(pseudo_kept)
    # Only process rows that are NOT yet successfully LLM-labeled.
    # (llm_failed rows get retried; llm_second_pass rows are skipped.)
    to_do = []
    for i in pseudo_kept:
        if i in done and done[i].get("label_source") == "llm_second_pass":
            continue  # already done
        to_do.append(i)
    to_do = to_do[:max_rows]

    if args.dry_run:
        print(f"[DRY RUN] Would label {len(to_do)} rows in batches of {batch_size}.")
        for i in to_do[:5]:
            print(f"  row {i}: {rows[i]['entity_name']} | pseudo={rows[i]['pseudo_label']}")
        return

    batches = [to_do[i:i+batch_size] for i in range(0, len(to_do), batch_size)]
    print(f"Processing {len(to_do)} rows in {len(batches)} batches (size={batch_size})...")

    t0 = time.time()
    for bi, batch_indices in enumerate(batches):
        batch = [(idx, rows[idx]) for idx in batch_indices]
        prompt = build_batch_prompt(batch)
        content, err = call_llm(prompt)
        arr = parse_json_array(content) if content else None

        if arr is None or len(arr) != len(batch):
            # retry: 1 row per call (more reliable)
            print(f"  batch {bi+1}/{len(batches)} parse fail ({err}); retrying 1-by-1...")
            for j, (idx, r) in enumerate(batch):
                if idx in done and done[idx].get("label_source") == "llm_second_pass":
                    continue  # already successfully labeled
                single_prompt = build_batch_prompt([(idx, r)])
                c2, e2 = call_llm(single_prompt, retries=4)
                a2 = parse_json_array(c2) if c2 else None
                if a2 and len(a2) == 1:
                    item = a2[0]
                    item["id"] = 0
                    record(idx, r, item)
                else:
                    record_fail(idx, r, e2 or "parse fail 1-by-1")
        else:
            for j, (idx, r) in enumerate(batch):
                item = arr[j] if j < len(arr) else None
                if item is None:
                    record_fail(idx, r, "missing item in batch")
                else:
                    item["id"] = j
                    record(idx, r, item)

        # rate-limit friendly delay between batches
        time.sleep(0.5)

        # progress
        done_count = len(done)
        elapsed = time.time() - t0
        rate = done_count / elapsed if elapsed > 0 else 0
        eta = (len(to_do) - done_count) / rate if rate > 0 else 0
        print(f"  batch {bi+1}/{len(batches)} done | total={done_count}/{len(to_do)} | "
              f"{rate:.1f} rows/s | ETA {eta:.0f}s", flush=True)

        # flush progress every 5 batches
        if (bi+1) % 5 == 0:
            flush_progress()

    flush_progress()
    print(f"\nFinished. {len(done)} rows labeled -> {OUT}")

def record(idx, r, item):
    """Save one LLM-labeled row."""
    # validate fields
    gold_label = item.get("gold_label", r["pseudo_label"])
    if gold_label not in ("positive", "neutral", "negative"):
        gold_label = r["pseudo_label"]
    gold_rel = item.get("gold_relevancy", "relevant")
    if gold_rel not in ("relevant", "not_relevant"):
        gold_rel = "relevant"
    done[idx] = {
        "row_index": idx,
        "raw_text_id": r["raw_text_id"],
        "entity_name": r["entity_name"],
        "context_text": r["context_text"],
        "pseudo_label": r["pseudo_label"],
        "gold_label": gold_label,
        "gold_relevancy": gold_rel,
        "entity_in_context": bool(item.get("entity_in_context", True)),
        "entity_is_main_subject": bool(item.get("entity_is_main_subject", True)),
        "entity_corrected": item.get("entity_corrected"),
        "context_flag": item.get("context_flag", "clean"),
        "reasoning": item.get("reasoning", ""),
        "label_source": "llm_second_pass",
        "label_confidence": 0.85,
    }

def record_fail(idx, r, err):
    """Could not LLM-label; keep pseudo with very low confidence."""
    done[idx] = {
        "row_index": idx,
        "raw_text_id": r["raw_text_id"],
        "entity_name": r["entity_name"],
        "context_text": r["context_text"],
        "pseudo_label": r["pseudo_label"],
        "gold_label": r["pseudo_label"],
        "gold_relevancy": "relevant",
        "entity_in_context": True,
        "entity_is_main_subject": True,
        "entity_corrected": None,
        "context_flag": "llm_failed",
        "reasoning": f"LLM labeling failed: {err}",
        "label_source": "llm_failed",
        "label_confidence": 0.3,
    }

def flush_progress():
    """Atomic flush: write to temp file, then rename. Prevents data loss on kill."""
    import os
    tmp_out = str(OUT) + ".tmp"
    with open(tmp_out, "w") as f:
        for idx in sorted(done.keys()):
            f.write(json.dumps(done[idx], ensure_ascii=False) + "\n")
    os.replace(tmp_out, OUT)   # atomic on POSIX
    with open(PROGRESS, "w") as f:
        json.dump({"done": len(done), "total_pseudo_kept": len(pseudo_kept)}, f)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(args)
