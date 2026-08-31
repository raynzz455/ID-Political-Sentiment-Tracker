// verify_dataset_v8.mjs
// =====================
// LLM verification for dataset_v8 — target ~327 unverified/low-confidence rows.
//
// Strategi:
//   - Verifikasi semua baris yang label_source NOT IN TRUSTED_SOURCES
//     ATAU label_confidence < 0.7
//   - Batch 5, retry 1-by-1 on parse fail
//   - Resume: simpan ke llm_verified_v8.jsonl secara incremental
//
// Usage:
//   node finetuning/scripts/verify_dataset_v8.mjs --batch 5 --delay 3000

import ZAI from 'z-ai-web-dev-sdk';
import { readFileSync, writeFileSync, existsSync } from 'fs';

const INPUT  = '/home/z/my-project/finetuning/datasets/dataset_v8_merged.jsonl';
const OUT    = '/home/z/my-project/finetuning/llm_verified_v8.jsonl';
const REPORT = '/home/z/my-project/finetuning/verify_v8_report.json';

const argv = process.argv.slice(2);
const argVal = (n) => {
  const i = argv.indexOf(n);
  return i >= 0 ? argv[i + 1] : null;
};
const LIMIT = argVal('--limit') ? parseInt(argVal('--limit'), 10) : null;
const BATCH = argVal('--batch') ? parseInt(argVal('--batch'), 10) : 5;
const DELAY_MS = argVal('--delay') ? parseInt(argVal('--delay'), 10) : 3000;

const TRUSTED_SOURCES = new Set(['llm_verified', 'llm_second_pass', 'gold_human']);

const SYSTEM = `Anda adalah annotator ahli sentimen politik Indonesia.
Tentukan sentimen TERHADAP entitas (bukan YANG DIKATAKAN entitas).

Aturan:
- "relevant" = entitas adalah SUBJEK UTAMA dari konteks (target sentimen).
- "not_relevant" = entitas hanya disebut di latar (mis: "era Jokowi", "menurut X") atau hanya konteks pembuka.
- Jika entitas mengkritik / menyatakan sesuatu -> sentimen TERHADAP entitas = NEUTRAL (ia pembicara, bukan target).
- Jika entitas dipuji / didukung / diprestasikan -> POSITIF.
- Jika entitas dikritik / dicela / divonis / dituntut / dituduh -> NEGATIF.

CONTOH:
[1] entity="Rocky Gerung", context="Rocky Gerung menyebut pasal KUHP dungu dan ketinggalan zaman."
-> gold_label="neutral", gold_relevancy="relevant"
[2] entity="Joko Widodo", context="Eks Menteri di era Presiden Jokowi dituntut Rp809M atas kasus korupsi."
-> gold_label="neutral", gold_relevancy="not_relevant"
[3] entity="Thomas Lembong", context="Eks Mendag Thomas Lembong divonis bersalah atas kasus korupsi impor gula."
-> gold_label="negative", gold_relevancy="relevant"
[4] entity="Prabowo Subianto", context="Prabowo dipuji karena keputusan ekonomi yang stabil dan tegas."
-> gold_label="positive", gold_relevancy="relevant"
[5] entity="Anies Baswedan", context="Menurut Anies, kebijakan subsidi saat ini tidak efektif menekan inflasi."
-> gold_label="neutral", gold_relevancy="relevant"
[6] entity="Megawati", context="Era pemerintahan Megawati Soekarnoputri menjadi sorotan dalam debat capres."
-> gold_label="neutral", gold_relevancy="not_relevant"

Output WAJIB: JSON array di dalam \`\`\`json ... \`\`\` block. Setiap elemen:
  { "id": <index>, "gold_label": "positive"|"neutral"|"negative",
    "gold_relevancy": "relevant"|"not_relevant",
    "entity_is_main_subject": true|false,
    "reasoning": "<alasan singkat <120 char>" }`;

async function main() {
  console.log('='.repeat(70));
  console.log('LLM VERIFICATION for dataset_v8 (unverified rows)');
  console.log('='.repeat(70));

  // Load merged dataset
  const rows = readFileSync(INPUT, 'utf-8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l));
  console.log(`Loaded v8_merged: ${rows.length} rows`);

  // Filter rows needing verification
  const need = rows.filter(r => {
    const src = r.label_source || '';
    const conf = r.label_confidence || 0;
    return !TRUSTED_SOURCES.has(src) || conf < 0.7;
  });
  console.log(`Rows needing verification: ${need.length}`);

  let toVerify = need;
  if (LIMIT) {
    toVerify = need.slice(0, LIMIT);
    console.log(`Limited to: ${toVerify.length}`);
  }

  // Resume
  const done = new Map();
  if (existsSync(OUT)) {
    const lines = readFileSync(OUT, 'utf-8').trim().split('\n').filter(Boolean);
    for (const line of lines) {
      try {
        const v = JSON.parse(line);
        const key = `${v.raw_text_id}|${v.entity_name}`;
        done.set(key, v);
      } catch {}
    }
    console.log(`Resume: ${done.size} already verified.`);
  }

  const remaining = toVerify.filter(r => !done.has(`${r.raw_text_id}|${r.entity_name}`));
  console.log(`Remaining: ${remaining.length}\n`);

  if (remaining.length === 0) {
    console.log('All rows already verified. Nothing to do.');
    writeReport(done, rows);
    return;
  }

  const zai = await ZAI.create();
  console.log('ZAI SDK initialized.\n');

  const batches = [];
  for (let i = 0; i < remaining.length; i += BATCH) {
    batches.push(remaining.slice(i, i + BATCH));
  }
  console.log(`Processing ${batches.length} batches...\n`);

  const t0 = Date.now();
  let count = done.size;

  for (let bi = 0; bi < batches.length; bi++) {
    const batch = batches[bi];

    let prompt = 'BARIS:\n';
    batch.forEach((r, i) => {
      prompt += `[${i}] entity="${r.entity_name}"\n`;
      prompt += `context="${(r.context_text || '').slice(0, 380)}"\n\n`;
    });
    prompt += 'Output HANYA JSON array di ```json ... ``` block.';

    let processedAny = false;
    try {
      const completion = await zai.chat.completions.create({
        messages: [
          { role: 'assistant', content: SYSTEM },
          { role: 'user', content: prompt },
        ],
        thinking: { type: 'disabled' },
      });
      const content = completion.choices?.[0]?.message?.content || '';

      let arr = null;
      const m = content.match(/```(?:json)?\s*(\[[\s\S]*?\])\s*```/);
      let jsonStr = m ? m[1] : null;
      if (!jsonStr) {
        const m2 = content.match(/(\[[\s\S]*\])/);
        if (m2) jsonStr = m2[1];
      }
      if (jsonStr) {
        try { arr = JSON.parse(jsonStr); } catch {}
      }

      if (arr && arr.length === batch.length) {
        for (let j = 0; j < batch.length; j++) {
          const item = arr[j];
          const r = batch[j];
          let label = item.gold_label || r.gold_label;
          if (!['positive', 'neutral', 'negative'].includes(label)) label = r.gold_label;
          let rel = item.gold_relevancy || 'relevant';
          if (!['relevant', 'not_relevant'].includes(rel)) rel = 'relevant';
          const key = `${r.raw_text_id}|${r.entity_name}`;
          done.set(key, {
            raw_text_id: r.raw_text_id,
            entity_name: r.entity_name,
            context_text: r.context_text,
            pseudo_label: r.pseudo_label,
            prev_label: r.gold_label,
            prev_source: r.label_source,
            prev_confidence: r.label_confidence,
            gold_label: label,
            gold_relevancy: rel,
            entity_is_main_subject: item.entity_is_main_subject !== false,
            reasoning: (item.reasoning || '').slice(0, 200),
            label_source: 'llm_verified_v8',
            label_confidence: 0.85,
          });
          count++;
          processedAny = true;
        }
      }
    } catch (e) {
      void e;
    }

    // Retry 1-by-1 if batch failed
    if (!processedAny) {
      for (const r of batch) {
        const key = `${r.raw_text_id}|${r.entity_name}`;
        if (done.has(key)) continue;
        try {
          const sp = `BARIS:\n[0] entity="${r.entity_name}"\ncontext="${(r.context_text || '').slice(0, 400)}"\n\nOutput JSON array:`;
          const c2 = await zai.chat.completions.create({
            messages: [
              { role: 'assistant', content: SYSTEM },
              { role: 'user', content: sp },
            ],
            thinking: { type: 'disabled' },
          });
          const ct2 = c2.choices?.[0]?.message?.content || '';
          const m2 = ct2.match(/```(?:json)?\s*(\[[\s\S]*?\])\s*```/);
          const m3 = m2 ?? ct2.match(/(\[[\s\S]*\])/);
          if (m3) {
            try {
              const a2 = JSON.parse(m3[1]);
              if (a2 && a2.length >= 1) {
                const item = a2[0];
                let label = item.gold_label || r.gold_label;
                if (!['positive', 'neutral', 'negative'].includes(label)) label = r.gold_label;
                let rel = item.gold_relevancy || 'relevant';
                if (!['relevant', 'not_relevant'].includes(rel)) rel = 'relevant';
                done.set(key, {
                  raw_text_id: r.raw_text_id,
                  entity_name: r.entity_name,
                  context_text: r.context_text,
                  pseudo_label: r.pseudo_label,
                  prev_label: r.gold_label,
                  prev_source: r.label_source,
                  prev_confidence: r.label_confidence,
                  gold_label: label,
                  gold_relevancy: rel,
                  entity_is_main_subject: item.entity_is_main_subject !== false,
                  reasoning: (item.reasoning || '').slice(0, 200),
                  label_source: 'llm_verified_v8',
                  label_confidence: 0.85,
                });
                count++;
                continue;
              }
            } catch {}
          }
        } catch {}
        console.log(`  [skip] ${r.entity_name} failed, will retry next run`);
      }
    }

    // Save progress
    const lines = Array.from(done.values()).map(v => JSON.stringify(v)).join('\n');
    writeFileSync(OUT, lines + '\n');

    const elapsed = (Date.now() - t0) / 1000;
    const rate = count > 0 ? count / elapsed : 0;
    const eta = rate > 0 ? (remaining.length - (count - (done.size - count))) / rate : 0;
    if ((bi + 1) % 5 === 0 || bi === batches.length - 1) {
      console.log(`  batch ${bi + 1}/${batches.length} | verified=${count}/${remaining.length + (count - remaining.length)} | elapsed=${elapsed.toFixed(0)}s | ETA=${eta.toFixed(0)}s`);
    }

    await new Promise(r => setTimeout(r, DELAY_MS));
  }

  console.log(`\n✅ Finished. ${count} verified -> ${OUT}`);
  writeReport(done, rows);
}

function writeReport(done, rows) {
  const verifiedLabels = { positive: 0, neutral: 0, negative: 0 };
  const verifiedRel = { relevant: 0, not_relevant: 0 };
  let flipped = 0;
  for (const v of done.values()) {
    verifiedLabels[v.gold_label] = (verifiedLabels[v.gold_label] || 0) + 1;
    verifiedRel[v.gold_relevancy] = (verifiedRel[v.gold_relevancy] || 0) + 1;
    if (v.prev_label && v.prev_label !== v.gold_label) flipped++;
  }
  const total = done.size;
  const report = {
    timestamp: new Date().toISOString(),
    input: INPUT,
    output: OUT,
    total_rows_verified: total,
    label_distribution: verifiedLabels,
    relevancy_distribution: verifiedRel,
    label_flips_from_prev: flipped,
    flip_rate: total > 0 ? (flipped / total) : 0,
    dataset_v8_merged_total: rows.length,
  };
  writeFileSync(REPORT, JSON.stringify(report, null, 2));
  console.log(`\nReport saved to ${REPORT}`);
  console.log(JSON.stringify(report, null, 2));
}

main().catch(e => { console.error('Fatal:', e); process.exit(1); });
