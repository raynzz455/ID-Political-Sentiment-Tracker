// llm_verify_pseudo.mjs
// =====================
// LLM verification for pseudo-labeled dataset entries using z-ai-web-dev-sdk.
// Reads from need_verify_final.json (1391 pseudo-labels), verifies each against
// the entity context, and writes gold labels to llm_verified_pseudo.jsonl.
//
// Features: Resume, adaptive rate limiting, batch processing, graceful shutdown.
// Usage:  node finetuning/scripts/llm_verify_pseudo.mjs --max-seconds=540

import ZAI from 'z-ai-web-dev-sdk';
import { readFileSync, writeFileSync, existsSync } from 'fs';

const INPUT  = '/home/z/my-project/finetuning/datasets/need_verify_final.json';
const OUT    = '/home/z/my-project/finetuning/datasets/llm_verified_pseudo.jsonl';
const PROG   = '/home/z/my-project/finetuning/datasets/llm_verify_pseudo_progress.json';

const BATCH_SIZE = 8;
const BASE_DELAY_MS = 700;
const ERROR_DELAY_MS = 6000;
const MAX_RETRIES = 3;

const args = process.argv.slice(2);
const maxSecArg = args.find(a => a.startsWith('--max-seconds='));
const MAX_SECONDS = maxSecArg ? parseInt(maxSecArg.split('=')[1], 10) : 540;

const SYSTEM = `Anda adalah annotator ahli sentimen politik Indonesia.
Tentukan sentimen TERHADAP entitas (bukan YANG DIKATAKAN entitas).

Aturan:
- "relevant" = entitas adalah SUBJEK UTAMA artikel/konteks (target sentimen)
- "not_relevant" = entitas hanya disebut sebagai latar (era/masa/oleh X/konteks historis)
- Jika entitas mengkritik sesuatu, sentimen terhadap entitas = NEUTRAL (pembicara)
- Jika entitas dipuji/dicela/dituntut/divonis, sentimen = POSITIF/NEGATIF
- Jika entitas hanya disebut tanpa penilaian, sentimen = NEUTRAL
- Pelantikan/jabatan = NEUTRAL (faktual)
- Klarifikasi hoaks = NEUTRAL
- Menang pemilu = POSITIF

Label sentimen: "positive" | "neutral" | "negative"
Label relevansi: "relevant" | "not_relevant"

Output: JSON array di dalam \`\`\`json ... \`\`\` block. Setiap elemen:
{"id": <index>, "gold_label": "...", "gold_relevancy": "...", "entity_is_main_subject": true/false, "reasoning": "..."}`;

function loadDone() {
  const done = {};
  if (existsSync(OUT)) {
    const lines = readFileSync(OUT, 'utf-8').trim().split('\n').filter(Boolean);
    for (const line of lines) {
      try {
        const r = JSON.parse(line);
        done[r.raw_text_id + '||' + r.entity_name] = r;
      } catch {}
    }
  }
  return done;
}

function saveDone(done) {
  const lines = Object.values(done).map(r => JSON.stringify(r)).join('\n');
  writeFileSync(OUT, lines + '\n');
  writeFileSync(PROG, JSON.stringify({ done: Object.keys(done).length, updated_at: new Date().toISOString() }));
}

function extractJsonArray(content) {
  if (!content) return null;
  let m = content.match(/```(?:json)?\s*(\[[\s\S]*?\])\s*```/);
  if (m) { try { return JSON.parse(m[1]); } catch {} }
  m = content.match(/(\[[\s\S]*\])/);
  if (m) { try { return JSON.parse(m[1]); } catch {} }
  return null;
}

function sanitizeLabel(label, fallback) {
  const l = String(label || '').toLowerCase().trim();
  if (['positive','neutral','negative'].includes(l)) return l;
  return fallback;
}
function sanitizeRel(rel) {
  const r = String(rel || '').toLowerCase().trim();
  if (r === 'not relevant' || r === 'not_relevant' || r === 'irrelevant') return 'not_relevant';
  if (r === 'relevant') return 'relevant';
  return 'relevant';
}

async function verifyBatch(zai, batch) {
  let prompt = 'Verifikasi sentimen untuk setiap entitas berikut.\n\nBARIS:\n';
  batch.forEach((r, i) => {
    prompt += `[${i}] entity="${r.entity_name}"\ncontext="${(r.context_text || '').slice(0, 400)}"\npseudo_label="${r.pseudo_label}"\n\n`;
  });
  prompt += 'Output HANYA JSON array di dalam ```json ... ``` block.';

  const completion = await zai.chat.completions.create({
    messages: [
      { role: 'assistant', content: SYSTEM },
      { role: 'user', content: prompt }
    ],
    thinking: { type: 'disabled' }
  });
  const content = completion.choices?.[0]?.message?.content || '';
  const arr = extractJsonArray(content);
  if (!arr || !Array.isArray(arr)) throw new Error('No JSON array in response');
  return arr;
}

async function verifySingle(zai, r) {
  const prompt = 'Verifikasi sentimen untuk entitas berikut.\n\nBARIS:\n[0] entity="' + r.entity_name + '"\ncontext="' + (r.context_text || '').slice(0, 450) + '"\npseudo_label="' + r.pseudo_label + '"\n\nOutput HANYA JSON array di dalam ' + '```json ... ```' + ' block.';
  const completion = await zai.chat.completions.create({
    messages: [
      { role: 'assistant', content: SYSTEM },
      { role: 'user', content: prompt }
    ],
    thinking: { type: 'disabled' }
  });
  const content = completion.choices?.[0]?.message?.content || '';
  const arr = extractJsonArray(content);
  if (!arr || arr.length < 1) throw new Error('No JSON in single response');
  return arr[0];
}

async function main() {
  console.log('=== LLM Pseudo-Label Verification ===');
  const toVerify = JSON.parse(readFileSync(INPUT, 'utf-8'));
  console.log(`Total entries: ${toVerify.length}`);
  const done = loadDone();
  console.log(`Already verified: ${Object.keys(done).length}`);
  const remaining = toVerify.filter(r => !(r.raw_text_id + '||' + r.entity_name in done));
  console.log(`Remaining: ${remaining.length}`);
  if (remaining.length === 0) { console.log('All done!'); return; }

  const zai = await ZAI.create();
  console.log('ZAI SDK initialized.\n');

  const batches = [];
  for (let i = 0; i < remaining.length; i += BATCH_SIZE) batches.push(remaining.slice(i, i + BATCH_SIZE));
  console.log(`Processing ${batches.length} batches (size=${BATCH_SIZE}, delay=${BASE_DELAY_MS}ms)...\n`);

  const t0 = Date.now();
  let failed = 0;

  for (let bi = 0; bi < batches.length; bi++) {
    const elapsedSoFar = (Date.now() - t0) / 1000;
    if (elapsedSoFar > MAX_SECONDS) {
      console.log(`\nTime limit (${MAX_SECONDS}s) reached. Saving and exiting.`);
      saveDone(done);
      console.log(`Progress: ${Object.keys(done).length}/${toVerify.length}. Re-run to continue.`);
      return;
    }

    const batch = batches[bi];
    let batchResults = null;

    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      try {
        batchResults = await verifyBatch(zai, batch);
        if (batchResults && batchResults.length === batch.length) break;
        if (batchResults && batchResults.length > 0) break;
        batchResults = null;
      } catch (e) {
        const msg = String(e.message || e).slice(0, 100);
        if (attempt < MAX_RETRIES - 1) {
          await new Promise(r => setTimeout(r, ERROR_DELAY_MS * (attempt + 1)));
        }
      }
    }

    if (batchResults && batchResults.length === batch.length) {
      for (let j = 0; j < batch.length; j++) {
        const item = batchResults[j];
        const r = batch[j];
        const key = r.raw_text_id + '||' + r.entity_name;
        done[key] = {
          raw_text_id: r.raw_text_id, entity_name: r.entity_name,
          context_text: r.context_text, pseudo_label: r.pseudo_label,
          gold_label: sanitizeLabel(item.gold_label, r.pseudo_label),
          gold_relevancy: sanitizeRel(item.gold_relevancy),
          entity_is_main_subject: item.entity_is_main_subject !== false,
          reasoning: item.reasoning || '',
          label_source: 'llm_verified', label_confidence: 0.85,
        };
      }
    } else {
      // Fallback 1-by-1
      for (const r of batch) {
        const key = r.raw_text_id + '||' + r.entity_name;
        if (key in done) continue;
        try {
          const item = await verifySingle(zai, r);
          done[key] = {
            raw_text_id: r.raw_text_id, entity_name: r.entity_name,
            context_text: r.context_text, pseudo_label: r.pseudo_label,
            gold_label: sanitizeLabel(item.gold_label, r.pseudo_label),
            gold_relevancy: sanitizeRel(item.gold_relevancy),
            entity_is_main_subject: item.entity_is_main_subject !== false,
            reasoning: item.reasoning || '',
            label_source: 'llm_verified_single', label_confidence: 0.85,
          };
        } catch {
          done[key] = {
            raw_text_id: r.raw_text_id, entity_name: r.entity_name,
            context_text: r.context_text, pseudo_label: r.pseudo_label,
            gold_label: r.pseudo_label, gold_relevancy: 'relevant',
            entity_is_main_subject: true, reasoning: 'LLM verify failed',
            label_source: 'llm_verify_failed', label_confidence: 0.5,
          };
          failed++;
        }
      }
    }

    saveDone(done);
    const rateDone = Object.keys(done).length;
    const elapsed = (Date.now() - t0) / 1000;
    const rate = rateDone / Math.max(elapsed, 1);
    const eta = (toVerify.length - rateDone) / Math.max(rate, 0.01);
    console.log(`  batch ${bi+1}/${batches.length} | done=${rateDone}/${toVerify.length} (${(rateDone/toVerify.length*100).toFixed(1)}%) | failed=${failed} | ${rate.toFixed(1)}/s | ETA ${eta.toFixed(0)}s`);

    await new Promise(r => setTimeout(r, BASE_DELAY_MS));
  }

  saveDone(done);
  console.log(`\nCOMPLETE! Verified: ${Object.keys(done).length}/${toVerify.length}, Failed: ${failed}`);
}

main().catch(e => { console.error('Fatal:', e); process.exit(1); });
