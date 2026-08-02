#!/usr/bin/env python3
"""Verify ALL 688 unverified rows via LLM. Saves to llm_verified_labels.jsonl."""
import json, re, subprocess, time, sys, os
from pathlib import Path
from collections import Counter

HERE = Path(__file__).parent
INPUT = Path("/tmp/all_to_verify.json")
OUT = HERE / "llm_verified_labels.jsonl"

SYSTEM = """Anda adalah annotator ahli sentimen politik Indonesia.
Tentukan sentimen TERHADAP entitas (bukan YANG DIKATAKAN entitas).

- "relevant" = entitas adalah SUBJEK UTAMA (target sentimen)
- "not_relevant" = entitas hanya disebut latar (era/masa/oleh X)
- Jika entitas mengkritik sesuatu, sentimen terhadap entitas = NEUTRAL (pembicara)
- Jika entitas dipuji/dicela, sentimen = POSITIF/NEGATIF

CONTOH:
[1] entity="Rocky Gerung", context="Rocky menyebut pasal KUHP dungu."
-> gold_label="neutral", gold_relevancy="relevant"
[2] entity="Joko Widodo", context="Eks Menteri era Presiden Jokowi dituntut Rp809M."
-> gold_label="neutral", gold_relevancy="not_relevant"
[3] entity="Thomas Lembong", context="Eks Mendag Thomas Lembong divonis korupsi."
-> gold_label="negative", gold_relevancy="relevant"

Output: JSON array di ```json ... ``` block. Setiap elemen:
  id, gold_label, gold_relevancy, entity_is_main_subject, reasoning"""

def call_llm(prompt, retries=4):
    last_err = None
    for attempt in range(retries):
        try:
            proc = subprocess.run(["z-ai","chat","-p",prompt,"-s",SYSTEM],
                capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                last_err = f"exit {proc.returncode}: {proc.stderr[:100]}"
                if attempt < retries-1: time.sleep(2**attempt); continue
                return None, last_err
            m = re.search(r'\{[\s\S]*"choices"[\s\S]*\}', proc.stdout)
            if not m:
                last_err = "no JSON"; 
                if attempt < retries-1: time.sleep(2**attempt); continue
                return None, last_err
            env = json.loads(m.group(0))
            return env["choices"][0]["message"]["content"], None
        except: 
            last_err = "exception"
            if attempt < retries-1: time.sleep(2**attempt)
    return None, last_err

def parse_arr(content):
    if not content: return None
    m = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', content)
    if m: content = m.group(1)
    else:
        m = re.search(r'(\[[\s\S]*\])', content)
        if m: content = m.group(1)
    try:
        arr = json.loads(content)
        return arr if isinstance(arr, list) else None
    except: return None

def build_prompt(batch):
    lines = ["BARIS:"]
    for i, (bi, r) in enumerate(batch):
        lines.append(f'[{i}] entity="{r["entity_name"]}"')
        lines.append(f'context="{(r["context_text"] or "")[:400]}"')
        lines.append("")
    lines.append("Output HANYA JSON array di ```json ... ``` block:")
    return "\n".join(lines)

def record(r, item):
    label = item.get("gold_label", r["current_label"])
    if label not in ("positive","neutral","negative"): label = r["current_label"]
    rel = item.get("gold_relevancy", "relevant")
    if rel not in ("relevant","not_relevant"): rel = "relevant"
    done[r["row_index"]] = {
        "row_index": r["row_index"], "entity_name": r["entity_name"],
        "context_text": r["context_text"], "pseudo_label": r["pseudo_label"],
        "heuristic_label": r["current_label"], "heuristic_source": r["current_source"],
        "gold_label": label, "gold_relevancy": rel,
        "entity_is_main_subject": bool(item.get("entity_is_main_subject", True)),
        "reasoning": item.get("reasoning", ""),
        "label_source": "llm_verified", "label_confidence": 0.85,
    }

def record_fail(r, err):
    done[r["row_index"]] = {
        "row_index": r["row_index"], "entity_name": r["entity_name"],
        "context_text": r["context_text"], "pseudo_label": r["pseudo_label"],
        "heuristic_label": r["current_label"], "heuristic_source": r["current_source"],
        "gold_label": r["current_label"], "gold_relevancy": "relevant",
        "entity_is_main_subject": True,
        "reasoning": f"LLM failed: {err}",
        "label_source": "llm_verify_failed", "label_confidence": 0.5,
    }

def flush():
    tmp = str(OUT) + ".tmp"
    with open(tmp, "w") as f:
        for idx in sorted(done.keys()):
            f.write(json.dumps(done[idx], ensure_ascii=False) + "\n")
    os.replace(tmp, OUT)

# Load
to_verify = json.load(open(INPUT)) if INPUT.exists() else []
print(f"Total to verify: {len(to_verify)}")

done = {}
if OUT.exists():
    done = {r["row_index"]: r for r in [json.loads(l) for l in open(OUT) if l.strip()]}
    print(f"Resuming: {len(done)} done.")

remaining = [r for r in to_verify if r["row_index"] not in done]
print(f"Remaining: {len(remaining)}")
if not remaining:
    print("All done!"); sys.exit(0)

BATCH = 3
batches = [remaining[i:i+BATCH] for i in range(0, len(remaining), BATCH)]
print(f"Processing {len(batches)} batches (size={BATCH}, delay=5s)...\n")

t0 = time.time()
for bi, batch in enumerate(batches):
    prompt = build_prompt([(i,r) for i,r in enumerate(batch)])
    content, err = call_llm(prompt)
    arr = parse_arr(content) if content else None

    if arr is None or len(arr) != len(batch):
        for r in batch:
            if r["row_index"] in done: continue
            c2, e2 = call_llm(build_prompt([(0,r)]), retries=5)
            a2 = parse_arr(c2) if c2 else None
            if a2 and len(a2)==1: record(r, a2[0])
            else: record_fail(r, e2 or "parse fail")
    else:
        for j, r in enumerate(batch):
            item = arr[j] if j < len(arr) else None
            if item: record(r, item)
            else: record_fail(r, "missing")

    time.sleep(5)
    if (bi+1) % 3 == 0:
        elapsed = time.time()-t0
        rate = len(done)/elapsed if elapsed > 0 else 0
        eta = (len(remaining)-len(done))/rate if rate > 0 else 0
        print(f"  batch {bi+1}/{len(batches)} | done={len(done)}/{len(remaining)} | "
              f"{rate:.1f}/s | ETA {eta:.0f}s", flush=True)
        flush()

flush()
print(f"\nFinished! {len(done)} verified -> {OUT}")
