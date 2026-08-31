// verify_dataset_v2.mjs
// =====================
// LLM verification for low-confidence rows in dataset_v2.jsonl
//
// Strategi:
//   - Hanya verifikasi baris dengan label_confidence < 0.7 DAN label_source BUKAN
//     llm_verified / llm_second_pass / gold_human (sudah trusted).
//   - Dedupe baris hasil oversampling: hanya verifikasi row_index BASE (strip _aug_N),
//     lalu label hasilnya akan dipropagasi ke semua baris yang punya base sama saat
//     membangun dataset_v3.
//   - Pakai z-ai-web-dev-sdk langsung (bukan CLI) → lebih tahan rate-limit.
//   - Batch 5 baris per call + retry 1-by-1 jika batch gagal parse.
//   - Resume: simpan ke llm_verified_v3.jsonl secara incremental, skip row_index yang
//     sudah ada di file tsb.
//
// Usage:
//   node finetuning/scripts/verify_dataset_v2.mjs                 # verify all
//   node finetuning/scripts/verify_dataset_v2.mjs --limit 30      # verify only 30
//   node finetuning/scripts/verify_dataset_v2.mjs --batch 8       # batch size
//
// Output:
//   finetuning/llm_verified_v3.jsonl   (1 baris NDJSON per base row_index)

import ZAI from 'z-ai-web-dev-sdk';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { createHash } from 'crypto';

const INPUT  = '/home/z/my-project/finetuning/datasets/dataset_v2.jsonl';
const OUT    = '/home/z/my-project/finetuning/llm_verified_v3.jsonl';
const REPORT = '/home/z/my-project/finetuning/verify_v3_report.json';

// CLI args
const argv = process.argv.slice(2);
const argVal = (name) => {
  const i = argv.indexOf(name);
  return i >= 0 ? argv[i + 1] : null;
};
const LIMIT = argVal('--limit') ? parseInt(argVal('--limit'), 10) : null;
const BATCH = argVal('--batch') ? parseInt(argVal('--batch'), 10) : 5;
const DELAY_MS = argVal('--delay') ? parseInt(argVal('--delay'), 10) : 2500;

// Strict annotation system prompt — same proven prompt from Task 2/15
const SYSTEM = `Anda adalah annotator ahli sentimen politik Indonesia.
Tentukan sentimen TERHADAP entitas (bukan YANG DIKATAKAN entitas).

Aturan:
- "relevant" = entitas adalah SUBJEK UTAMA dari konteks (target sentimen).
- "not_relevant" = entitas hanya disebut di latar (mis: "era Jokowi", "menurut X") atau hanya konteks pembuka.
- Jika entitas mengkritik / menyatakan sesuatu → sentimen TERHADAP entitas = NEUTRAL (ia pembicara, bukan target).
- Jika entitas dipuji / didukung / diprestasikan → POSITIF.
- Jika entitas dikritik / dicela / divonis / dituntut / dituduh → NEGATIF.

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

function baseRowIdx(ri) {
  // strip "_aug_N" suffix from oversampled row_index
  const s = String(ri ?? '');
  if (s.includes('_aug_')) return s.split('_aug_')[0];
  return s;
}

async function main() {
  console.log('='.repeat(70));
  console.log('LLM VERIFICATION for dataset_v2 (low-confidence rows)');
  console.log('='.repeat(70));
  console.log(`Input:    ${INPUT}`);
  console.log(`Output:   ${OUT}`);
  console.log(`Batch:    ${BATCH} | Delay: ${DELAY_MS}ms | Limit: ${LIMIT ?? 'all'}`);

  // Load dataset_v2
  const rows = readFileSync(INPUT, 'utf-8')
    .trim().split('\n').filter(Boolean).map(l => JSON.parse(l));
  console.log(`Loaded dataset_v2: ${rows.length} rows`);

  // Filter: only rows needing verification
  const NEED_SOURCES = new Set([
    'oversampled',
    'heuristic_speaker_upgraded',
    'heuristic_default',
    'heuristic_neg_cues',
    'heuristic_pos_cues',
    'heuristic_polarity_upgraded',
  ]);
  const needAll = rows.filter(r => {
    const conf = r.label_confidence ?? 0;
    const src  = r.label_source ?? '';
    return conf < 0.7 && NEED_SOURCES.has(src);
  });
  console.log(`Rows needing verification (raw, with duplicates): ${needAll.length}`);

  // Dedupe by base row_index — only verify each unique base once
  const needByBase = new Map();
  for (const r of needAll) {
    const b = baseRowIdx(r.row_index);
    if (!needByBase.has(b)) needByBase.set(b, r);
  }
  let need = Array.from(needByBase.values());
  console.log(`Unique base rows to verify (dedup oversampled): ${need.length}`);
  if (LIMIT) {
    need = need.slice(0, LIMIT);
    console.log(`Limited to: ${need.length}`);
  }

  // Resume: load existing verified labels from OUT
  const done = new Map();
  if (existsSync(OUT)) {
    const lines = readFileSync(OUT, 'utf-8').trim().split('\n').filter(Boolean);
    for (const line of lines) {
      try {
        const v = JSON.parse(line);
        if (v.base_row_index) done.set(v.base_row_index, v);
      } catch {}
    }
    console.log(`Resume: ${done.size} already verified.`);
  }

  const remaining = need.filter(r => !done.has(baseRowIdx(r.row_index)));
  console.log(`Remaining: ${remaining.length}\n`);

  if (remaining.length === 0) {
    console.log('All rows already verified. Nothing to do.');
    writeReport(done, rows);
    return;
  }

  // Init SDK
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

    // Build prompt
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

      // Extract JSON array
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
          const bIdx = baseRowIdx(r.row_index);
          done.set(bIdx, {
            base_row_index: bIdx,
            entity_name: r.entity_name,
            context_text: r.context_text,
            pseudo_label: r.pseudo_label,
            prev_heuristic_label: r.gold_label,
            prev_source: r.label_source,
            prev_confidence: r.label_confidence,
            gold_label: label,
            gold_relevancy: rel,
            entity_is_main_subject: item.entity_is_main_subject !== false,
            reasoning: (item.reasoning || '').slice(0, 200),
            label_source: 'llm_verified',
            label_confidence: 0.85,
          });
          count++;
          processedAny = true;
        }
      }
    } catch (e) {
      // fallthrough to 1-by-1 retry
      void e;
    }

    // If batch failed or partial — retry 1-by-1
    if (!processedAny) {
      for (const r of batch) {
        const bIdx = baseRowIdx(r.row_index);
        if (done.has(bIdx)) continue;
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
                done.set(bIdx, {
                  base_row_index: bIdx,
                  entity_name: r.entity_name,
                  context_text: r.context_text,
                  pseudo_label: r.pseudo_label,
                  prev_heuristic_label: r.gold_label,
                  prev_source: r.label_source,
                  prev_confidence: r.label_confidence,
                  gold_label: label,
                  gold_relevancy: rel,
                  entity_is_main_subject: item.entity_is_main_subject !== false,
                  reasoning: (item.reasoning || '').slice(0, 200),
                  label_source: 'llm_verified',
                  label_confidence: 0.85,
                });
                count++;
                continue;
              }
            } catch {}
          }
        } catch {}
        // Mark as failed — will be retried on next run (not added to done)
        console.log(`  [skip] ${bIdx} failed, will retry next run`);
      }
    }

    // Save progress every batch
    const lines = Array.from(done.values())
      .sort((a, b) => String(a.base_row_index).localeCompare(String(b.base_row_index), undefined, { numeric: true }))
      .map(v => JSON.stringify(v)).join('\n');
    writeFileSync(OUT, lines + '\n');

    const elapsed = (Date.now() - t0) / 1000;
    const processed = count - (done.size - count > 0 ? 0 : 0);
    const rate = count > 0 ? count / elapsed : 0;
    const eta = rate > 0 ? (remaining.length - (count - (done.size - count))) / rate : 0;
    if ((bi + 1) % 5 === 0 || bi === batches.length - 1) {
      console.log(`  batch ${bi + 1}/${batches.length} | verified=${count}/${remaining.length + (count - remaining.length)} | elapsed=${elapsed.toFixed(0)}s | ETA=${eta.toFixed(0)}s`);
    }

    // delay
    await new Promise(r => setTimeout(r, DELAY_MS));
  }

  console.log(`\n✅ Finished. ${count} verified -> ${OUT}`);
  writeReport(done, rows);
}

function writeReport(done, rows) {
  // Stats
  const verifiedLabels = { positive: 0, neutral: 0, negative: 0 };
  const verifiedRel    = { relevant: 0, not_relevant: 0 };
  let flipped = 0;
  for (const v of done.values()) {
    verifiedLabels[v.gold_label] = (verifiedLabels[v.gold_label] || 0) + 1;
    verifiedRel[v.gold_relevancy] = (verifiedRel[v.gold_relevancy] || 0) + 1;
    if (v.prev_heuristic_label && v.prev_heuristic_label !== v.gold_label) flipped++;
  }
  const total = done.size;
  const report = {
    timestamp: new Date().toISOString(),
    input: INPUT,
    output: OUT,
    total_base_rows_verified: total,
    label_distribution: verifiedLabels,
    relevancy_distribution: verifiedRel,
    label_flips_from_heuristic: flipped,
    flip_rate: total > 0 ? (flipped / total) : 0,
    dataset_v2_total_rows: rows.length,
    notes: 'Verifikasi LLM baris berconfidence <0.7. Label dipropagasi ke semua baris oversampled yang punya base_row_index sama saat build_dataset_v3.py.',
  };
  writeFileSync(REPORT, JSON.stringify(report, null, 2));
  console.log(`\nReport saved to ${REPORT}`);
  console.log(JSON.stringify(report, null, 2));
}

main().catch(e => { console.error('Fatal:', e); process.exit(1); });
