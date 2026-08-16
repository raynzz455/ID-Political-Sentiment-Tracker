// verify_dataset_v9.mjs
// =====================
// LLM verification for dataset_v9 — targets rows in need_llm_verify_v9.json
// (92 llm_failed rows + optionally 673 low-confidence v7 rows for upgrade).
//
// Strategi:
//   - Read need_llm_verify_v9.json (list of rows needing verification)
//   - Optional --include-low-conf flag: also re-verify v7 rows with conf 0.55-0.69
//   - Batch 5, retry 1-by-1 on parse fail
//   - Resume: save to llm_verified_v9.jsonl incrementally
//
// Usage:
//   node finetuning/scripts/verify_dataset_v9.mjs --batch 5 --delay 3000
//   node finetuning/scripts/verify_dataset_v9.mjs --include-low-conf --batch 3 --delay 4000

import ZAI from 'z-ai-web-dev-sdk';
import { readFileSync, writeFileSync, existsSync } from 'fs';

const NEED_VERIFY = '/home/z/my-project/finetuning/datasets/need_llm_verify_v9.json';
const V9_MERGED   = '/home/z/my-project/finetuning/datasets/dataset_v9.jsonl';
const OUT         = '/home/z/my-project/finetuning/llm_verified_v9.jsonl';
const REPORT      = '/home/z/my-project/finetuning/verify_v9_report.json';

const argv = process.argv.slice(2);
const argVal = (n) => {
  const i = argv.indexOf(n);
  return i >= 0 ? argv[i + 1] : null;
};
const LIMIT = argVal('--limit') ? parseInt(argVal('--limit'), 10) : null;
const BATCH = argVal('--batch') ? parseInt(argVal('--batch'), 10) : 5;
const DELAY_MS = argVal('--delay') ? parseInt(argVal('--delay'), 10) : 3000;
const INCLUDE_LOW_CONF = argv.includes('--include-low-conf');

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
  console.log('LLM VERIFICATION for dataset_v9');
  console.log('='.repeat(70));

  // Load need-verify list
  const needList = JSON.parse(readFileSync(NEED_VERIFY, 'utf-8'));
  console.log(`Need-verify list: ${needList.length} rows`);

  let toVerify = needList;

  // Optionally include low-confidence v7 rows
  if (INCLUDE_LOW_CONF) {
    const v9Rows = readFileSync(V9_MERGED, 'utf-8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l));
    const lowConf = v9Rows.filter(r => {
      const c = r.label_confidence || 0;
      return c >= 0.55 && c < 0.7;
    }).map(r => ({
      raw_text_id: r.raw_text_id,
      entity_name: r.entity_name,
      context_text: r.context_text,
      pseudo_label: r.pseudo_label,
      current_label: r.gold_label,
      current_source: r.label_source,
      current_confidence: r.label_confidence,
    }));
    console.log(`Low-confidence v7 rows (conf 0.55-0.69): ${lowConf.length}`);
    toVerify = [...needList, ...lowConf];
  }

  if (LIMIT) {
    toVerify = toVerify.slice(0, LIMIT);
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
    writeReport(done);
    return;
  }

  const zai = await ZAI.create();
  console.log('ZAI SDK initialized.\n');

  // v3: retry-with-backoff wrapper for 429 rate-limit handling
  async function createWithRetry(messages, maxRetries = 8) {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        const completion = await zai.chat.completions.create({
          messages,
          thinking: { type: 'disabled' },
        });
        return completion;
      } catch (e) {
        const is429 = e.message?.includes('429') || e.message?.includes('Too many requests');
        if (!is429 || attempt === maxRetries - 1) throw e;
        const backoff = 10000 * Math.pow(1.5, attempt); // 10s, 15s, 22s, 33s, 50s, 75s, 112s
        console.log(`  [429] backoff ${backoff/1000}s (attempt ${attempt+1}/${maxRetries})...`);
        await new Promise(r => setTimeout(r, backoff));
      }
    }
  }

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
      const completion = await createWithRetry([
        { role: 'assistant', content: SYSTEM },
        { role: 'user', content: prompt },
      ]);
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
          let label = item.gold_label || r.current_label;
          if (!['positive', 'neutral', 'negative'].includes(label)) label = r.current_label;
          let rel = item.gold_relevancy || 'relevant';
          if (!['relevant', 'not_relevant'].includes(rel)) rel = 'relevant';
          const key = `${r.raw_text_id}|${r.entity_name}`;
          done.set(key, {
            raw_text_id: r.raw_text_id,
            entity_name: r.entity_name,
            context_text: r.context_text,
            pseudo_label: r.pseudo_label,
            prev_label: r.current_label,
            prev_source: r.current_source,
            prev_confidence: r.current_confidence,
            gold_label: label,
            gold_relevancy: rel,
            entity_is_main_subject: item.entity_is_main_subject !== false,
            reasoning: (item.reasoning || '').slice(0, 200),
            label_source: 'llm_verified_v9',
            label_confidence: 0.85,
          });
          count++;
          processedAny = true;
        }
      }
    } catch (e) {
      void e;
    }

    // Retry 1-by-1
    if (!processedAny) {
      for (const r of batch) {
        const key = `${r.raw_text_id}|${r.entity_name}`;
        if (done.has(key)) continue;
        try {
          const sp = `BARIS:\n[0] entity="${r.entity_name}"\ncontext="${(r.context_text || '').slice(0, 400)}"\n\nOutput JSON array:`;
          const c2 = await createWithRetry([
            { role: 'assistant', content: SYSTEM },
            { role: 'user', content: sp },
          ]);
          const ct2 = c2.choices?.[0]?.message?.content || '';
          const m2 = ct2.match(/```(?:json)?\s*(\[[\s\S]*?\])\s*```/);
          const m3 = m2 ?? ct2.match(/(\[[\s\S]*\])/);
          if (m3) {
            try {
              const a2 = JSON.parse(m3[1]);
              if (a2 && a2.length >= 1) {
                const item = a2[0];
                let label = item.gold_label || r.current_label;
                if (!['positive', 'neutral', 'negative'].includes(label)) label = r.current_label;
                let rel = item.gold_relevancy || 'relevant';
                if (!['relevant', 'not_relevant'].includes(rel)) rel = 'relevant';
                done.set(key, {
                  raw_text_id: r.raw_text_id,
                  entity_name: r.entity_name,
                  context_text: r.context_text,
                  pseudo_label: r.pseudo_label,
                  prev_label: r.current_label,
                  prev_source: r.current_source,
                  prev_confidence: r.current_confidence,
                  gold_label: label,
                  gold_relevancy: rel,
                  entity_is_main_subject: item.entity_is_main_subject !== false,
                  reasoning: (item.reasoning || '').slice(0, 200),
                  label_source: 'llm_verified_v9',
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
      console.log(`  batch ${bi + 1}/${batches.length} | verified=${count} | elapsed=${elapsed.toFixed(0)}s | ETA=${eta.toFixed(0)}s`);
    }

    await new Promise(r => setTimeout(r, DELAY_MS));
  }

  console.log(`\n✅ Finished. ${count} verified -> ${OUT}`);
  writeReport(done);
}

function writeReport(done) {
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
    output: OUT,
    total_rows_verified: total,
    label_distribution: verifiedLabels,
    relevancy_distribution: verifiedRel,
    label_flips_from_prev: flipped,
    flip_rate: total > 0 ? (flipped / total) : 0,
  };
  writeFileSync(REPORT, JSON.stringify(report, null, 2));
  console.log(`\nReport saved to ${REPORT}`);
  console.log(JSON.stringify(report, null, 2));
}

main().catch(e => { console.error('Fatal:', e); process.exit(1); });
