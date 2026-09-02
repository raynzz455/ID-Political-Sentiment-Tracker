#!/usr/bin/env python3
"""
export_and_label_new_data.py
============================
TAHAP 3B: Export data baru dari Supabase + label via LLM

Steps:
  1. Fetch articles that are entity_resolved + context_extracted but NOT yet labeled
  2. Export to JSONL with entity + context + article info
  3. Label each row via LLM second-pass
  4. Save to dataset_v3_raw.jsonl

Usage:
  export SUPABASE_URL=...
  export SUPABASE_SERVICE_ROLE_KEY=...
  python export_and_label_new_data.py --limit 200
"""
import os, json, re, subprocess, time, argparse, random
from pathlib import Path
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--output", default="finetuning/datasets/dataset_v3_raw.jsonl")
    args = parser.parse_args()

    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    print("="*60)
    print("TAHAP 3B: Export + Label New Data from Supabase")
    print("="*60)

    # Step 1: Fetch entity_contexts + articles + entities
    print(f"\nFetching {args.limit} contexts from Supabase...")
    res = sb.table("entity_contexts").select(
        "raw_text_id, entity_id, context_text, metadata, "
        "political_entities(canonical_name, aliases, era, party_affiliation), "
        "raw_texts(id, title, text, source_url)"
    ).limit(args.limit).execute()

    contexts = res.data or []
    print(f"Got {len(contexts)} contexts")

    # Step 2: Format to dataset rows
    rows = []
    for ctx in contexts:
        raw = ctx.get("raw_texts") or {}
        ent = ctx.get("political_entities") or {}
        if not raw or not ent:
            continue
        rows.append({
            "raw_text_id": ctx["raw_text_id"],
            "entity_id": ctx["entity_id"],
            "entity_name": ent.get("canonical_name", ""),
            "entity_aliases": ent.get("aliases", []),
            "entity_era": ent.get("era", []),
            "entity_party": ent.get("party_affiliation", ""),
            "context_text": ctx.get("context_text", ""),
            "context_metadata": ctx.get("metadata", {}),
            "article_title": raw.get("title", ""),
            "article_text": raw.get("text", ""),
            "source_url": raw.get("source_url", ""),
        })

    print(f"Formatted: {len(rows)} rows")

    # Step 3: Label via LLM
    print(f"\nLabeling via LLM second-pass...")
    SYSTEM = """Anda adalah annotator ahli sentimen politik Indonesia.
Tentukan sentimen TERHADAP entitas (bukan YANG DIKATAKAN entitas).
- "positive": entitas dipuji/didukung
- "neutral": laporan faktal
- "negative": entitas dikritik/dicela
Output: JSON dengan field gold_label, gold_relevancy, reasoning."""

    labeled = []
    stats = Counter()

    for i, row in enumerate(rows):
        entity = row["entity_name"]
        context = row["context_text"][:400]

        prompt = f"""Entitas: "{entity}"
Konteks: "{context}"
Tentukan: gold_label (positive/neutral/negative), gold_relevancy (relevant/not_relevant), reasoning"""

        try:
            proc = subprocess.run(
                ["z-ai", "chat", "-p", prompt, "-s", SYSTEM],
                capture_output=True, text=True, timeout=30
            )
            if proc.returncode == 0:
                m = re.search(r'\{[\s\S]*"choices"[\s\S]*\}', proc.stdout)
                if m:
                    env = json.loads(m.group(0))
                    content = env["choices"][0]["message"]["content"]

                    # Parse JSON from content
                    jm = re.search(r'\{[^}]+\}', content)
                    if jm:
                        try:
                            result = json.loads(jm.group(0))
                            row["gold_label"] = result.get("gold_label", "neutral")
                            row["gold_relevancy"] = result.get("gold_relevancy", "relevant")
                            row["reasoning"] = result.get("reasoning", "")
                            row["label_source"] = "llm_second_pass"
                            row["label_confidence"] = 0.85

                            # Validate
                            if row["gold_label"] not in ("positive", "neutral", "negative"):
                                row["gold_label"] = "neutral"
                            if row["gold_relevancy"] not in ("relevant", "not_relevant"):
                                row["gold_relevancy"] = "relevant"

                            labeled.append(row)
                            stats[row["gold_label"]] += 1
                            stats["total"] += 1
                        except json.JSONDecodeError:
                            row["gold_label"] = "neutral"
                            row["gold_relevancy"] = "relevant"
                            row["label_source"] = "llm_failed"
                            row["label_confidence"] = 0.5
                            labeled.append(row)
                            stats["failed"] += 1
                    else:
                        # Try to find label in text
                        for lab in ["positive", "neutral", "negative"]:
                            if lab in content.lower():
                                row["gold_label"] = lab
                                row["gold_relevancy"] = "relevant"
                                row["reasoning"] = content[:200]
                                row["label_source"] = "llm_second_pass"
                                row["label_confidence"] = 0.85
                                labeled.append(row)
                                stats[lab] += 1
                                stats["total"] += 1
                                break
                        else:
                            stats["failed"] += 1
        except Exception as e:
            stats["error"] += 1

        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(rows)}] labeled={stats['total']} failed={stats['failed']} | "
                  f"pos={stats.get('positive',0)} neu={stats.get('neutral',0)} neg={stats.get('negative',0)}", flush=True)

        time.sleep(2)  # rate limit

    # Step 4: Save
    with open(args.output, "w") as f:
        for row in labeled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"TAHAP 3B COMPLETE")
    print(f"{'='*60}")
    print(f"  Total labeled: {len(labeled)}")
    print(f"  Labels: pos={stats.get('positive',0)} neu={stats.get('neutral',0)} neg={stats.get('negative',0)}")
    print(f"  Failed: {stats.get('failed',0)}")
    print(f"  Saved to: {args.output}")

if __name__ == "__main__":
    main()
