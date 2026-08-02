// llm_verify_sdk.js — LLM verification via z-ai-web-dev-sdk directly (bypass CLI rate-limit)
import ZAI from 'z-ai-web-dev-sdk';
import { readFileSync, writeFileSync, existsSync } from 'fs';

const INPUT = '/tmp/need_llm_verify.json';
const OUT = '/home/z/my-project/finetuning/llm_verified_v2.jsonl';

const SYSTEM = `Anda adalah annotator ahli sentimen politik Indonesia.
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

Output: JSON array di \`\`\`json ... \`\`\` block. Setiap elemen:
  id, gold_label, gold_relevancy, entity_is_main_subject, reasoning`;

async function main() {
  const toVerify = JSON.parse(readFileSync(INPUT, 'utf-8'));
  console.log(`Total to verify: ${toVerify.length}`);

  // Resume
  let done = {};
  if (existsSync(OUT)) {
    const lines = readFileSync(OUT, 'utf-8').trim().split('\n').filter(Boolean);
    for (const line of lines) {
      const r = JSON.parse(line);
      done[r.row_index] = r;
    }
    console.log(`Resuming: ${Object.keys(done).length} done.`);
  }

  const remaining = toVerify.filter(r => !(r.row_index in done));
  console.log(`Remaining: ${remaining.length}`);
  if (remaining.length === 0) { console.log('All done!'); return; }

  const zai = await ZAI.create();
  console.log('ZAI SDK initialized.');

  const BATCH = 5;
  const batches = [];
  for (let i = 0; i < remaining.length; i += BATCH) {
    batches.push(remaining.slice(i, i + BATCH));
  }
  console.log(`Processing ${batches.length} batches (size=${BATCH}, delay=3s)...\n`);

  let count = Object.keys(done).length;
  const t0 = Date.now();

  for (let bi = 0; bi < batches.length; bi++) {
    const batch = batches[bi];
    let prompt = 'BARIS:\n';
    batch.forEach((r, i) => {
      prompt += `[${i}] entity="${r.entity_name}"\ncontext="${(r.context_text || '').slice(0, 350)}"\n\n`;
    });
    prompt += 'Output HANYA JSON array di ```json ... ``` block:';

    try {
      const completion = await zai.chat.completions.create({
        messages: [
          { role: 'assistant', content: SYSTEM },
          { role: 'user', content: prompt }
        ],
        thinking: { type: 'disabled' }
      });

      const content = completion.choices[0]?.message?.content || '';
      // extract JSON array
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
          if (!['positive','neutral','negative'].includes(label)) label = r.current_label;
          let rel = item.gold_relevancy || 'relevant';
          if (!['relevant','not_relevant'].includes(rel)) rel = 'relevant';
          done[r.row_index] = {
            row_index: r.row_index,
            entity_name: r.entity_name,
            context_text: r.context_text,
            pseudo_label: r.pseudo_label,
            heuristic_label: r.current_label,
            heuristic_source: r.current_source,
            gold_label: label,
            gold_relevancy: rel,
            entity_is_main_subject: item.entity_is_main_subject !== false,
            reasoning: item.reasoning || '',
            label_source: 'llm_verified',
            label_confidence: 0.85,
          };
          count++;
        }
      } else {
        // batch failed, try 1-by-1
        console.log(`  batch ${bi+1} failed (arr=${arr ? arr.length : 0}), retry 1-by-1...`);
        for (const r of batch) {
          if (r.row_index in done) continue;
          try {
            const sp = `BARIS:\n[0] entity="${r.entity_name}"\ncontext="${(r.context_text || '').slice(0, 400)}"\n\nOutput JSON array:`;
            const c2 = await zai.chat.completions.create({
              messages: [{role:'assistant',content:SYSTEM},{role:'user',content:sp}],
              thinking: { type: 'disabled' }
            });
            const ct2 = c2.choices[0]?.message?.content || '';
            const m2 = ct2.match(/```(?:json)?\s*(\[[\s\S]*?\])\s*```/) || ct2.match(/(\[[\s\S]*\])/);
            if (m2) {
              const a2 = JSON.parse(m2[1]);
              if (a2 && a2.length >= 1) {
                const item = a2[0];
                let label = item.gold_label || r.current_label;
                if (!['positive','neutral','negative'].includes(label)) label = r.current_label;
                let rel = item.gold_relevancy || 'relevant';
                if (!['relevant','not_relevant'].includes(rel)) rel = 'relevant';
                done[r.row_index] = {
                  row_index: r.row_index, entity_name: r.entity_name,
                  context_text: r.context_text, pseudo_label: r.pseudo_label,
                  heuristic_label: r.current_label, heuristic_source: r.current_source,
                  gold_label: label, gold_relevancy: rel,
                  entity_is_main_subject: item.entity_is_main_subject !== false,
                  reasoning: item.reasoning || '',
                  label_source: 'llm_verified', label_confidence: 0.85,
                };
                count++;
                continue;
              }
            }
          } catch {}
          // fail
          done[r.row_index] = {
            row_index: r.row_index, entity_name: r.entity_name,
            context_text: r.context_text, pseudo_label: r.pseudo_label,
            heuristic_label: r.current_label, heuristic_source: r.current_source,
            gold_label: r.current_label, gold_relevancy: 'relevant',
            entity_is_main_subject: true,
            reasoning: 'LLM SDK failed',
            label_source: 'llm_verify_failed', label_confidence: 0.5,
          };
          count++;
        }
      }
    } catch (e) {
      console.log(`  batch ${bi+1} error: ${e.message?.slice(0, 80)}`);
      // mark as failed
      for (const r of batch) {
        if (r.row_index in done) continue;
        done[r.row_index] = {
          row_index: r.row_index, entity_name: r.entity_name,
          context_text: r.context_text, pseudo_label: r.pseudo_label,
          heuristic_label: r.current_label, heuristic_source: r.current_source,
          gold_label: r.current_label, gold_relevancy: 'relevant',
          entity_is_main_subject: true,
          reasoning: `Error: ${e.message?.slice(0, 60)}`,
          label_source: 'llm_verify_failed', label_confidence: 0.5,
        };
        count++;
      }
    }

    // save every 5 batches
    if ((bi + 1) % 5 === 0 || bi === batches.length - 1) {
      const lines = Object.keys(done).sort((a,b)=>a-b).map(k => JSON.stringify(done[k])).join('\n');
      writeFileSync(OUT, lines + '\n');
      const elapsed = (Date.now() - t0) / 1000;
      const rate = count / elapsed;
      const eta = (remaining.length - count) / rate;
      console.log(`  batch ${bi+1}/${batches.length} | done=${count}/${remaining.length} | ${rate.toFixed(1)}/s | ETA ${eta.toFixed(0)}s`);
    }

    // delay
    await new Promise(r => setTimeout(r, 3000));
  }

  console.log(`\nFinished! ${count} verified -> ${OUT}`);
}

main().catch(e => { console.error('Fatal:', e); process.exit(1); });
