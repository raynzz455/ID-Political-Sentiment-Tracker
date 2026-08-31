// final_121_retry.mjs — Re-verify the 121 first-pass-only rows
import ZAI from 'z-ai-web-dev-sdk';
import { readFileSync, writeFileSync, existsSync } from 'fs';

const INPUT = '/home/z/my-project/finetuning/datasets/final_121_retry.json';
const OUT = '/home/z/my-project/finetuning/datasets/final_121_results.jsonl';

const SYSTEM = `Anda adalah annotator ahli sentimen politik Indonesia. Verifikasi label dan berikan confidence score.

ATURAN:
- "positive": dipuji, berhasil, menang, dihormati, mendapat penghargaan
- "neutral": faktual, pelantikan, klarifikasi, pesan moral, tidak ada penilaian
- "negative": dikritik, divonis, dicela, dicopot, kalah, gagal

CONFIDENCE: 0.98=sangat jelas, 0.95=jelas, 0.90=cukup jelas

Output: JSON array: [{"id": 0, "gold_label": "...", "confidence": 0.0-1.0, "reasoning": "..."}]`;

function extractJsonArray(content) {
  if (!content) return null;
  let m = content.match(/```(?:json)?\s*(\[[\s\S]*?\])\s*```/);
  if (m) { try { return JSON.parse(m[1]); } catch {} }
  m = content.match(/(\[[\s\S]*\])/);
  if (m) { try { return JSON.parse(m[1]); } catch {} }
  return null;
}

async function main() {
  const toVerify = JSON.parse(readFileSync(INPUT, 'utf-8'));
  console.log(`Retry: ${toVerify.length} rows`);
  const done = {};
  if (existsSync(OUT)) {
    for (const line of readFileSync(OUT, 'utf-8').trim().split('\n').filter(Boolean)) {
      try { const r = JSON.parse(line); done[r.row_index] = r; } catch {}
    }
  }
  const remaining = toVerify.filter(r => !(r.row_index in done));
  console.log(`Remaining: ${remaining.length}`);
  if (remaining.length === 0) { console.log('Done!'); return; }

  const zai = await ZAI.create();
  const t0 = Date.now();
  let success = 0, failed = 0;

  // 1-by-1 with generous delay (rate-limit recovery)
  for (let i = 0; i < remaining.length; i++) {
    if ((Date.now() - t0) / 1000 > 520) { console.log('Time limit.'); break; }
    const r = remaining[i];
    let ok = false;
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const prompt = `entity="${r.entity_name}"\ncurrent_label="${r.current_label}"\ncontext="${(r.text||'').slice(0, 450)}"`;
        const completion = await zai.chat.completions.create({
          messages: [{ role: 'assistant', content: SYSTEM }, { role: 'user', content: prompt }],
          thinking: { type: 'disabled' }
        });
        const arr = extractJsonArray(completion.choices?.[0]?.message?.content || '');
        if (arr && arr.length >= 1) {
          const item = arr[0];
          const label = String(item.gold_label || '').toLowerCase().trim();
          if (['positive','neutral','negative'].includes(label)) {
            const conf = Math.max(0.85, Math.min(1.0, parseFloat(item.confidence) || 0.95));
            done[r.row_index] = {
              row_index: r.row_index, entity_name: r.entity_name, text: r.text,
              old_label: r.current_label, gold_label: label,
              confidence: conf, reasoning: item.reasoning || '',
              reverify_source: 'llm_final_retry',
            };
            ok = true; success++;
            break;
          }
        }
      } catch (e) {
        const msg = String(e.message || e);
        const wait = msg.includes("429") ? 60000 : 8000;
        console.log(`  [${i+1}] ${msg.includes('429') ? '429' : 'err'} — wait ${wait/1000}s`);
        await new Promise(res => setTimeout(res, wait));
      }
    }
    if (!ok) { failed++; }
    if ((i+1) % 10 === 0 || i === remaining.length - 1) {
      writeFileSync(OUT, Object.values(done).map(r => JSON.stringify(r)).join('\n') + '\n');
      console.log(`  progress: ${i+1}/${remaining.length} (ok=${success}, fail=${failed})`);
    }
    await new Promise(res => setTimeout(res, 5000));
  }
  writeFileSync(OUT, Object.values(done).map(r => JSON.stringify(r)).join('\n') + '\n');
  console.log(`\nComplete: ${Object.keys(done).length}/${toVerify.length} (success=${success}, failed=${failed})`);
}
main().catch(e => { console.error('Fatal:', e); process.exit(1); });
