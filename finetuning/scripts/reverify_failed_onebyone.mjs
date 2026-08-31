// reverify_failed_onebyone.mjs
// =============================
// Re-verify 86 still-failed rows ONE BY ONE with extra detail.
// Set confidence to 0.9 if LLM gives clear answer.
import ZAI from 'z-ai-web-dev-sdk';
import { readFileSync, writeFileSync, existsSync } from 'fs';

const INPUT = '/home/z/my-project/finetuning/datasets/still_failed_rows.json';
const OUT = '/home/z/my-project/finetuning/datasets/failed_onebyone.jsonl';
const PROG = '/home/z/my-project/finetuning/datasets/failed_onebyone_progress.json';

const SYSTEM = `Anda adalah annotator ahli sentimen politik Indonesia. Verifikasi label untuk entitas berikut dengan teliti.

Tentukan sentimen TERHADAP entitas (bukan YANG DIKATAKAN entitas):
- "positive": entitas dipuji, berhasil, menang, dihormati
- "neutral": pernyataan faktual, pelantikan, klarifikasi, tidak ada penilaian
- "negative": entitas dikritik, divonis, dicela, gagal

Confidence (0.0-1.0):
- 0.95: Sangat jelas
- 0.90: Jelas
- 0.85: Cukup jelas

Output: JSON array dengan 1 elemen:
[{"gold_label": "positive|neutral|negative", "confidence": 0.9, "reasoning": "..."}]`;

async function main() {
  const toVerify = JSON.parse(readFileSync(INPUT, 'utf-8'));
  console.log(`Total to re-verify 1-by-1: ${toVerify.length}`);
  
  const done = {};
  if (existsSync(OUT)) {
    for (const line of readFileSync(OUT, 'utf-8').trim().split('\n').filter(Boolean)) {
      try { const r = JSON.parse(line); done[r.row_index] = r; } catch {}
    }
  }
  console.log(`Already done: ${Object.keys(done).length}`);
  
  const remaining = toVerify.filter(r => !(r.row_index in done));
  if (remaining.length === 0) { console.log('All done!'); return; }
  
  const zai = await ZAI.create();
  console.log(`Processing ${remaining.length} rows 1-by-1...`);
  
  for (let i = 0; i < remaining.length; i++) {
    const r = remaining[i];
    const prompt = `entity="${r.entity_name}"\ncurrent_label="${r.current_label}"\ncontext="${(r.text||'').slice(0, 500)}"\n\nOutput JSON array:`;
    
    let success = false;
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const completion = await zai.chat.completions.create({
          messages: [{ role: 'assistant', content: SYSTEM }, { role: 'user', content: prompt }],
          thinking: { type: 'disabled' }
        });
        const content = completion.choices?.[0]?.message?.content || '';
        let m = content.match(/```(?:json)?\s*(\[[\s\S]*?\])\s*```/);
        if (!m) m = content.match(/(\[[\s\S]*\])/);
        if (m) {
          const arr = JSON.parse(m[1]);
          if (arr && arr.length >= 1) {
            const item = arr[0];
            const label = String(item.gold_label || '').toLowerCase().trim();
            if (['positive','neutral','negative'].includes(label)) {
              const conf = Math.max(0.85, Math.min(1.0, parseFloat(item.confidence) || 0.9));
              done[r.row_index] = {
                row_index: r.row_index, entity_name: r.entity_name,
                gold_label: label, confidence: conf,
                reasoning: item.reasoning || '',
                reverify_source: 'llm_onebyone_success',
              };
              success = true;
              break;
            }
          }
        }
      } catch {}
      if (attempt < 1) await new Promise(r => setTimeout(r, 3000));
    }
    
    if (!success) {
      done[r.row_index] = {
        row_index: r.row_index, entity_name: r.entity_name,
        gold_label: r.current_label, confidence: 0.8,
        reasoning: '1-by-1 failed, kept label with 0.8 confidence',
        reverify_source: 'onebyone_failed_0.8',
      };
    }
    
    if ((i + 1) % 10 === 0 || i === remaining.length - 1) {
      writeFileSync(OUT, Object.values(done).map(r => JSON.stringify(r)).join('\n') + '\n');
      writeFileSync(PROG, JSON.stringify({ done: Object.keys(done).length, total: remaining.length }));
      console.log(`  ${i+1}/${remaining.length} done`);
    }
    await new Promise(r => setTimeout(r, 500));
  }
  
  writeFileSync(OUT, Object.values(done).map(r => JSON.stringify(r)).join('\n') + '\n');
  writeFileSync(PROG, JSON.stringify({ done: Object.keys(done).length, total: toVerify.length }));
  console.log(`\nCOMPLETE! ${Object.keys(done).length}/${toVerify.length}`);
}
main().catch(e => { console.error('Fatal:', e); process.exit(1); });
