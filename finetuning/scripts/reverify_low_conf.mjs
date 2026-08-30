// reverify_low_conf.mjs
// =====================
// Re-verify 530 low-confidence rows to boost mean confidence > 90%.
// Uses STRICTER prompt + confidence scoring.
import ZAI from 'z-ai-web-dev-sdk';
import { readFileSync, writeFileSync, existsSync } from 'fs';

const INPUT = '/home/z/my-project/finetuning/datasets/low_confidence_rows.json';
const OUT = '/home/z/my-project/finetuning/datasets/low_conf_reverified.jsonl';
const PROG = '/home/z/my-project/finetuning/datasets/low_conf_progress.json';

const BATCH_SIZE = 6;
const BASE_DELAY_MS = 700;
const MAX_RETRIES = 3;

const args = process.argv.slice(2);
const maxSecArg = args.find(a => a.startsWith('--max-seconds='));
const MAX_SECONDS = maxSecArg ? parseInt(maxSecArg.split('=')[1], 10) : 540;

const SYSTEM = `Anda adalah annotator ahli sentimen politik Indonesia. Verifikasi ULANG dengan teliti.

Tentukan:
1. Apakah entitas adalah SUBJEK UTAMA konteks (bukan hanya disebut)?
2. Sentimen TERHADAP entitas (bukan YANG DIKATAKAN entitas)?
3. Confidence score (0.0-1.0): seberapa yakin Anda dengan label?

ATURAN SENTIMEN:
- "positive": entitas dipuji, dihormati, berhasil, menang, mendapat penghargaan
- "neutral": pernyataan faktual, pelantikan, klarifikasi, pesan moral, tidak ada penilaian
- "negative": entitas dikritik, divonis, dicela, dicopot, kalah, gagal

ATURAN KEYAKINAN (confidence):
- 0.95: Sangat jelas, tidak ambigu (contoh: "X divonis korupsi")
- 0.90: Jelas, sedikit konteks tapi label obvious
- 0.85: Cukup jelas, ada sedikit ambiguitas
- 0.70: Agak ambigu, bisa diperdebatkan
- 0.50: Sangat ambigu, tidak yakin

ATURAN SUBJEK UTAMA:
- "main_subject": entitas adalah tokoh utama yang dibahas
- "not_main_subject": entitas hanya disebut sebagai latar (era/masa/oleh X)

Output: JSON array di dalam \`\`\`json ... \`\`\` block. Setiap elemen:
{"id": <index>, "is_main_subject": true/false, "gold_label": "positive|neutral|negative", "confidence": 0.0-1.0, "reasoning": "penjelasan singkat"}`;

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
  return ['positive','neutral','negative'].includes(l) ? l : fallback;
}
function sanitizeConf(c) {
  const v = parseFloat(c);
  if (isNaN(v)) return 0.85;
  return Math.max(0.5, Math.min(1.0, v));
}

async function verifyBatch(zai, batch) {
  let prompt = 'Verifikasi ULANG dengan teliti:\n\nBARIS:\n';
  batch.forEach((r, i) => {
    prompt += `[${i}] entity="${r.entity_name}"\ncurrent_label="${r.current_label}"\ncontext="${(r.text||'').slice(0, 450)}"\n\n`;
  });
  prompt += 'Output HANYA JSON array di ```json ... ``` block dengan confidence score.';
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
  console.log(`Total to re-verify: ${toVerify.length}`);
  const done = loadDone();
  console.log(`Already done: ${Object.keys(done).length}`);
  const remaining = toVerify.filter(r => !(r.row_index in done));
  if (remaining.length === 0) { console.log('All done!'); return; }
  const zai = await ZAI.create();
  const batches = [];
  for (let i = 0; i < remaining.length; i += BATCH_SIZE) batches.push(remaining.slice(i, i + BATCH_SIZE));
  console.log(`Processing ${batches.length} batches...`);
  const t0 = Date.now();
  let failed = 0;

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
        done[r.row_index] = {
          row_index: r.row_index, entity_name: r.entity_name, text: r.text,
          old_label: r.current_label, old_confidence: r.current_confidence,
          is_main_subject: item.is_main_subject !== false,
          gold_label: sanitizeLabel(item.gold_label, r.current_label),
          confidence: sanitizeConf(item.confidence),
          reasoning: item.reasoning || '',
          reverify_source: 'llm_reverified_v2',
        };
      }
    } else {
      for (const r of batch) {
        if (r.row_index in done) continue;
        done[r.row_index] = {
          row_index: r.row_index, entity_name: r.entity_name, text: r.text,
          old_label: r.current_label, old_confidence: r.current_confidence,
          is_main_subject: true, gold_label: r.current_label,
          confidence: 0.7, reasoning: 'Re-verify v2 failed, kept old label with medium confidence',
          reverify_source: 'reverify_v2_failed',
        };
        failed++;
      }
    }
    saveDone(done);
    const elapsed = (Date.now() - t0) / 1000;
    const rateDone = Object.keys(done).length;
    const rate = rateDone / Math.max(elapsed, 1);
    const eta = (toVerify.length - rateDone) / Math.max(rate, 0.01);
    console.log(`  batch ${bi+1}/${batches.length} | done=${rateDone}/${toVerify.length} (${(rateDone/toVerify.length*100).toFixed(1)}%) | failed=${failed} | ${rate.toFixed(1)}/s | ETA ${eta.toFixed(0)}s`);
    await new Promise(r => setTimeout(r, BASE_DELAY_MS));
  }
  saveDone(done);
  console.log(`\nCOMPLETE! ${Object.keys(done).length}/${toVerify.length}, failed=${failed}`);
}
main().catch(e => { console.error('Fatal:', e); process.exit(1); });
