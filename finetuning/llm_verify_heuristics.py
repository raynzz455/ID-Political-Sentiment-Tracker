#!/usr/bin/env python3
"""
llm_verify_heuristics.py
========================
Re-label the 464 heuristic rows via LLM second-pass.
Upgrades confidence from 0.7 (heuristic) to 0.85 (llm_second_pass).

Strategy:
  - Batch 3 rows per LLM call (token-efficient)
  - Inter-batch delay 5s (anti rate-limit)
  - Exponential backoff retry (1s, 2s, 4s, 8s)
  - Atomic save (temp + rename) every 10 rows
  - Resume support (skip already-verified)

Output: llm_verified_labels.jsonl — one row per verified heuristic
"""
import json, re, subprocess, time, sys, os
from pathlib import Path
from collections import Counter

HERE = Path(__file__).parent
HEURISTIC_INPUT = Path("/tmp/heuristic_to_verify.json")
OUT = HERE / "llm_verified_labels.jsonl"
PROGRESS = HERE / "llm_verify_progress.json"

SYSTEM_PROMPT = """Anda adalah annotator ahli sentimen politik Indonesia tingkat senior.
Tugas: tentukan sentimen TERHADAP entitas politik (bukan sentimen YANG DIKATAKAN entitas).

DEFINISI KUNCI:
- Sentimen TERHADAP entitas = bagaimana TONE/FRAMING media menggambarkan entitas tersebut.
- Jika entitas mengkritik sesuatu, sentimen terhadap entitas = NEUTRAL (dia pembicara, bukan target).
- Jika entitas dipuji/dicela pihak lain, sentimen terhadap entitas = POSITIF/NEGATIF.

ATURAN RELEVANSI:
- "relevant" = entitas adalah SUBJEK UTAMA kalimat/konteks (target sentimen).
- "not_relevant" = entitas hanya disebut latar (era/masa/oleh Presiden X, dll).

CONTOH:

[1] entity="Rocky Gerung"
context="Rocky menyebut pasal KUHP sebagai pasal yang dungu."
-> gold_label="neutral", gold_relevancy="relevant"
   reasoning="Rocky pembicara yang mengkritik UU; sentimen terhadap Rocky netral."

[2] entity="Joko Widodo"
context="Eks Menteri era Presiden Jokowi ini dituntut membayar Rp809 miliar."
-> gold_label="neutral", gold_relevancy="not_relevant"
   reasoning="Konteks tentang kasus Nadiem; Jokowi hanya anchor temporal."

[3] entity="Thomas Lembong"
context="Eks Mendag Thomas Lembong divonis bersalah korupsi impor gula."
-> gold_label="negative", gold_relevancy="relevant"
   reasoning="Vonis bersalah korupsi = sentimen negatif terhadap entitas."

[4] entity="Puan Maharani"
context="Ketua DPR Puan Maharani mengatakan akan memperkuat koperasi."
-> gold_label="neutral", gold_relevancy="relevant"
   reasoning="Puan pembicara yang menyampaikan program; sentimen netral (factual)."

[5] entity="Prabowo Subianto"
context="LBH mengecam pernyataan Presiden Prabowo soal Londo Ireng."
-> gold_label="negative", gold_relevancy="relevant"
   reasoning="Prabowo adalah target kritik (dikecam); sentimen negatif terhadap Prabowo."

OUTPUT WAJIB: JSON array di dalam ```json ... ``` block. Setiap elemen:
  id, gold_label, gold_relevancy, entity_is_main_subject, reasoning
  gold_label: "positive" | "neutral" | "negative"
  gold_relevancy: "relevant" | "not_relevant"
  entity_is_main_subject: true/false
  reasoning: satu kalimat alasan"""

def load_jsonl(p):
    if not p.exists(): return []
    return [json.loads(l) for l in open(p) if l.strip()]

def call_llm(prompt, retries=4):
    last_err = None
    for attempt in range(retries):
        try:
            proc = subprocess.run(
                ["z-ai", "chat", "-p", prompt, "-s", SYSTEM_PROMPT],
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode != 0:
                last_err = f"CLI exit {proc.returncode}: {proc.stderr[:150]}"
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None, last_err
            out = proc.stdout
            m = re.search(r'\{[\s\S]*"choices"[\s\S]*\}', out)
            if not m:
                last_err = "no JSON envelope"
                if attempt < retries - 1:
                    time.sleep(2 ** attempt); continue
                return None, last_err
            env = json.loads(m.group(0))
            content = env["choices"][0]["message"]["content"]
            return content, None
        except subprocess.TimeoutExpired:
            last_err = "timeout"
            if attempt < retries - 1:
                time.sleep(2 ** attempt); continue
        except Exception as e:
            last_err = str(e)
            if attempt < retries - 1:
                time.sleep(2 ** attempt); continue
    return None, last_err

def parse_json_array(content):
    if not content: return None
    m = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', content)
    if m: content = m.group(1)
    else:
        m = re.search(r'(\[[\s\S]*\])', content)
        if m: content = m.group(1)
    try:
        arr = json.loads(content)
        if isinstance(arr, list): return arr
    except json.JSONDecodeError: pass
    return None

def build_batch_prompt(batch):
    lines = ["BARIS:"]
    for i, (batch_idx, row) in enumerate(batch):
        ctx = (row["context_text"] or "")[:500]
        lines.append(f"[{i}] entity=\"{row['entity_name']}\"")
        lines.append(f"context=\"{ctx}\"")
        lines.append(f"current_heuristic_label=\"{row['current_label']}\" (verify or correct this)")
        lines.append("")
    lines.append("Output HANYA JSON array di dalam ```json ... ``` block:")
    return "\n".join(lines)

# Load heuristic rows to verify (JSON array, not JSONL)
if HEURISTIC_INPUT.exists():
    to_verify = json.load(open(HEURISTIC_INPUT))
else:
    to_verify = []
print(f"Total heuristic rows to verify: {len(to_verify)}")

# Resume support
done = {}
if OUT.exists():
    done = {r["row_index"]: r for r in load_jsonl(OUT)}
    print(f"Resuming: {len(done)} already verified.")

remaining = [r for r in to_verify if r["row_index"] not in done]
print(f"Remaining: {len(remaining)}")

if not remaining:
    print("All verified!")
    sys.exit(0)

# ---- record / record_fail / flush defined BEFORE use ----
def record(r, item):
    gold_label = item.get("gold_label", r["current_label"])
    if gold_label not in ("positive","neutral","negative"):
        gold_label = r["current_label"]
    gold_rel = item.get("gold_relevancy", "relevant")
    if gold_rel not in ("relevant","not_relevant"):
        gold_rel = "relevant"
    done[r["row_index"]] = {
        "row_index": r["row_index"],
        "entity_name": r["entity_name"],
        "context_text": r["context_text"],
        "pseudo_label": r["pseudo_label"],
        "heuristic_label": r["current_label"],
        "heuristic_source": r["current_source"],
        "gold_label": gold_label,
        "gold_relevancy": gold_rel,
        "entity_is_main_subject": bool(item.get("entity_is_main_subject", True)),
        "reasoning": item.get("reasoning", ""),
        "label_source": "llm_verified",
        "label_confidence": 0.85,
    }

def record_fail(r, err):
    done[r["row_index"]] = {
        "row_index": r["row_index"],
        "entity_name": r["entity_name"],
        "context_text": r["context_text"],
        "pseudo_label": r["pseudo_label"],
        "heuristic_label": r["current_label"],
        "heuristic_source": r["current_source"],
        "gold_label": r["current_label"],  # keep heuristic
        "gold_relevancy": "relevant",
        "entity_is_main_subject": True,
        "reasoning": f"LLM verify failed: {err}",
        "label_source": "llm_verify_failed",
        "label_confidence": 0.5,
    }

def flush():
    import os
    tmp = str(OUT) + ".tmp"
    with open(tmp, "w") as f:
        for idx in sorted(done.keys()):
            f.write(json.dumps(done[idx], ensure_ascii=False) + "\n")
    os.replace(tmp, OUT)
    with open(PROGRESS, "w") as f:
        json.dump({"done": len(done), "total": len(to_verify)}, f)

# Run in batches of 3
BATCH_SIZE = 3
batches = [remaining[i:i+BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
print(f"Processing {len(batches)} batches (size={BATCH_SIZE}, delay=5s)...\n")

t0 = time.time()
for bi, batch in enumerate(batches):
    batch_with_idx = [(i, r) for i, r in enumerate(batch)]
    prompt = build_batch_prompt(batch_with_idx)
    content, err = call_llm(prompt)
    arr = parse_json_array(content) if content else None

    if arr is None or len(arr) != len(batch):
        print(f"  batch {bi+1}/{len(batches)} fail ({err}); retry 1-by-1...", flush=True)
        for j, r in enumerate(batch):
            if r["row_index"] in done: continue
            single_prompt = build_batch_prompt([(0, r)])
            c2, e2 = call_llm(single_prompt, retries=5)
            a2 = parse_json_array(c2) if c2 else None
            if a2 and len(a2) == 1:
                item = a2[0]; item["id"] = 0
                record(r, item)
            else:
                record_fail(r, e2 or "parse fail")
    else:
        for j, r in enumerate(batch):
            item = arr[j] if j < len(arr) else None
            if item is None:
                record_fail(r, "missing item")
            else:
                item["id"] = j
                record(r, item)

    time.sleep(5)

    if (bi+1) % 3 == 0:
        elapsed = time.time() - t0
        rate = len(done) / elapsed if elapsed > 0 else 0
        eta = (len(remaining) - len(done)) / rate if rate > 0 else 0
        print(f"  batch {bi+1}/{len(batches)} | done={len(done)}/{len(remaining)} | "
              f"{rate:.1f} rows/s | ETA {eta:.0f}s", flush=True)
        flush()
    # also flush every batch if <30 done (early save)
    if len(done) < 30:
        flush()

flush()
print(f"\nFinished! {len(done)} rows verified -> {OUT}")
