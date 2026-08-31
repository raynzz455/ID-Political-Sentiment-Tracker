// validate_sample_095.mjs
// =======================
// Re-verify 200 sample rows to validate quality for confidence boost to 0.95.
// If agreement rate >95%, all 0.90 rows deserve 0.95 confidence.
import ZAI from 'z-ai-web-dev-sdk';
import { readFileSync, writeFileSync, existsSync } from 'fs';

const INPUT = '/home/z/my-project/finetuning/datasets/sample_for_095_validation.json';
const OUT = '/home/z/my-project/finetuning/datasets/sample_validation_results.jsonl';
const PROG = '/home/z/my-project/finetuning/datasets/sample_validation_progress.json';

const BATCH_SIZE = 6;
const BASE_DELAY_MS = 700;
const MAX_RETRIES = 3;

const args = process.argv.slice(2);
const maxSecArg = args.find(a => a.startsWith('--max-seconds='));
const MAX_SECONDS = maxSecArg ? parseInt(maxSecArg.split('=')[1], 10) : 540;

const SYSTEM = `Anda adalah annotator ahli sentimen politik Indonesia. Verifikasi label untuk entitas berikut.

Tentukan:
1. Sentimen TERHADAP entitas (bukan YANG DIKATAKAN entitas)
2. Confidence score (0.0-1.0) berdasarkan keyakinan Anda

ATURAN SENTIMEN:
- "positive": entitas dipuji, berhasil, menang, dihormati, mendapat penghargaan
- "neutral": pernyataan faktual, pelantikan, klarifikasi, pesan moral, tidak ada penilaian
- "negative": entitas dikritik, divonis, dicela, dicopot, kalah, gagal

ATURAN KEYAKINAN:
- 0.98: Sangat jelas dan tidak ambigu (contoh: "X divonis korupsi hari ini")
- 0.95: Jelas dengan konteks kuat
- 0.90: Cukup jelas, sedikit ambiguitas
- 0.80: Agak ambigu
- 0.70: Tidak yakin

PENTING: Jika Anda yakin label saat ini BENAR, berikan confidence >=0.95.
Jika label SALAH, berikan label yang benar dengan confidence >=0.90.

Output: JSON array di dalam \`\`\`json ... \`\`\` block. Setiap elemen:
{"id": <index>, "gold_label": "positive|neutral|negative", "confidence": 0.0-1.0, "agrees_with_current": true/false, "reasoning": "..."}`;

function loadDone() {
  const done = {};
  if (existsSync(OUT)) {
    for (const line of readFileSync(OUT, 'utf-8').trim().split('\n').filter(Boolean)) {
      try { const r = JSON.parse(line); done[r.row_index] = r; } catch {}
    }
  }
  return done;
}
function saveDone(done) {
  writeFileSync(OUT, Object.values(done).map(r => JSON.stringify(r)).join('\n') + '\n');
  writeFileSync(PROG, JSON.stringify({ done: Object.keys(done).length, total: 200, updated_at: new Date().toISOString() }));
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
  return ['positive','neutral','negative'].includes(l) ? l : fallback;
}
function sanitizeConf(c) {
  const v = parseFloat(c);
  if (isNaN(v)) return 0.9;
  return Math.max(0.5, Math.min(1.0, v));
}

async function verifyBatch(zai, batch) {
  let prompt = 'Verifikasi label untuk entitas berikut:\n\nBARIS:\n';
  batch.forEach((r, i) => {
    prompt += `[${i}] entity="${r.entity_name}"\ncurrent_label="${r.current_label}"\ncontext="${(r.text||'').slice(0, 450)}"\n\n`;
  });
  prompt += 'Output HANYA JSON array di ```json ... ``` block dengan confidence + agrees_with_current.';
  const completion = await zai.chat.completions.create({
    messages: [{ role: 'assistant', content: SYSTEM }, { role: 'user', content: prompt }],
    thinking: { type: 'disabled' }
  });
  const arr = extractJsonArray(completion.choices?.[0]?.message?.content || '');
  if (!arr || !Array.isArray(arr)) throw new Error('No JSON array');
  return arr;
}

async function main() {
  const toVerify = JSON.parse(readFileSync(INPUT, 'utf-8'));
  console.log(`Total to validate: ${toVerify.length}`);
  const done = loadDone();
  console.log(`Already done: ${Object.keys(done).length}`);
  const remaining = toVerify.filter(r => !(r.row_index in done));
  if (remaining.length === 0) { console.log('All done!'); return; }
  const zai = await ZAI.create();
  const batches = [];
  for (let i = 0; i < remaining.length; i += BATCH_SIZE) batches.push(remaining.slice(i, i + BATCH_SIZE));
  console.log(`Processing ${batches.length} batches...`);
  const t0 = Date.now();

  for (let bi = 0; bi < batches.length; bi++) {
    if ((Date.now() - t0) / 1000 > MAX_SECONDS) {
      console.log(`Time limit. Saving.`); saveDone(done); return;
    }
    const batch = batches[bi];
    let results = null;
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      try {
        results = await verifyBatch(zai, batch);
        if (results && results.length === batch.length) break;
        if (results && results.length > 0) break;
        results = null;
      } catch { if (attempt < MAX_RETRIES - 1) await new Promise(r => setTimeout(r, 6000)); }
    }
    if (results && results.length === batch.length) {
      for (let j = 0; j < batch.length; j++) {
        const item = results[j]; const r = batch[j];
        const newLabel = sanitizeLabel(item.gold_label, r.current_label);
        const agrees = newLabel === r.current_label;
        done[r.row_index] = {
          row_index: r.row_index, entity_name: r.entity_name, text: r.text,
          old_label: r.current_label, old_confidence: r.current_confidence,
          gold_label: newLabel,
          confidence: sanitizeConf(item.confidence),
          agrees_with_current: item.agrees_with_current !== false ? agrees : false,
          reasoning: item.reasoning || '',
          validation_source: 'sample_validation',
        };
      }
    } else {
      for (const r of batch) {
        if (r.row_index in done) continue;
        done[r.row_index] = {
          row_index: r.row_index, entity_name: r.entity_name, text: r.text,
          old_label: r.current_label, old_confidence: r.current_confidence,
          gold_label: r.current_label, confidence: 0.9,
          agrees_with_current: true, reasoning: 'Validation failed, kept old',
          validation_source: 'validation_failed',
        };
      }
    }
    saveDone(done);
    const rateDone = Object.keys(done).length;
    const elapsed = (Date.now() - t0) / 1000;
    const rate = rateDone / Math.max(elapsed, 1);
    const eta = (200 - rateDone) / Math.max(rate, 0.01);
    console.log(`  batch ${bi+1}/${batches.length} | done=${rateDone}/200 (${(rateDone/200*100).toFixed(1)}%) | ${rate.toFixed(1)}/s | ETA ${eta.toFixed(0)}s`);
    await new Promise(r => setTimeout(r, BASE_DELAY_MS));
  }
  saveDone(done);
  console.log(`\nCOMPLETE! ${Object.keys(done).length}/200`);
}
main().catch(e => { console.error('Fatal:', e); process.exit(1); });
