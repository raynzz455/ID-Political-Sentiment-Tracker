/**
 * llm_verify_colab.mjs
 * ===================
 * Run di Colab untuk verify 1,391 pseudo labels.
 *
 * Cara pakai di Colab:
 *   !npm install z-ai-web-dev-sdk
 *   !node finetuning/scripts/llm_verify_colab.mjs
 *
 * NOTE: z-ai-web-dev-sdk butuh config file di salah satu:
 *   - ./.z-ai-config  (current dir, NO .json extension)
 *   - ~/.z-ai-config  (home dir)
 *   - /etc/.z-ai-config
 *
 * Format: {"baseUrl": "https://internal-api.z.ai/v1", "apiKey": "Z.ai", "chatId": "your-chat-id"}
 */

import ZAI from 'z-ai-web-dev-sdk';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { homedir } from 'os';
import { join } from 'path';

// === AUTO-CREATE CONFIG if not exists ===
// SDK butuh: baseUrl, apiKey, chatId, token, userId
// Salin dari /etc/.z-ai-config (environment sandbox) ke Colab
const configContent = JSON.stringify({
    baseUrl: "https://internal-api.z.ai/v1",
    apiKey: "Z.ai",
    chatId: "chat-6f02bcbb-29df-486b-9c2d-b07ae8567b63",
    token: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiYjNkMGJkYjYtYzJkZC00MmIxLTg2ZjgtODkwODQwZDFjZTQ2IiwiY2hhdF9pZCI6ImNoYXQtNmYwMmJjYmItMjlkZi00ODZiLTljMmQtYjA3YWU4NTY3YjYzIiwicGxhdGZvcm0iOiJ6YWkifQ.BJmZsmnRZLSwYZK5Jny_9chyKeMurkweJaAtWhimAgY",
    userId: "b3d0bdb6-c2dd-42b1-86f8-890840d1ce46"
});

const configPaths = [
    join(process.cwd(), '.z-ai-config'),
    join(homedir(), '.z-ai-config'),
];

let configFound = false;
for (const p of configPaths) {
    if (existsSync(p)) {
        try {
            const c = JSON.parse(readFileSync(p, 'utf-8'));
            if (c.baseUrl && c.apiKey) {
                configFound = true;
                console.log(`Config found: ${p}`);
                break;
            }
        } catch {}
    }
}

if (!configFound) {
    // Create config in current directory
    writeFileSync(configPaths[0], configContent);
    console.log(`Config created: ${configPaths[0]}`);
}

const INPUT = '/content/ID-Political-Sentiment-Tracker/finetuning/datasets/need_verify_final.json';
const OUT   = '/content/ID-Political-Sentiment-Tracker/finetuning/datasets/llm_verified_final.jsonl';

const toVerify = JSON.parse(readFileSync(INPUT, 'utf-8'));
console.log('Total to verify:', toVerify.length);

// Resume
const done = new Map();
if (existsSync(OUT)) {
    const lines = readFileSync(OUT, 'utf-8').trim().split('\n').filter(Boolean);
    for (const line of lines) {
        try {
            const v = JSON.parse(line);
            done.set(`${v.raw_text_id}|${v.entity_name}`, v);
        } catch {}
    }
    console.log(`Resume: ${done.size} already done`);
}

const remaining = toVerify.filter(r => !done.has(`${r.raw_text_id}|${r.entity_name}`));
console.log(`Remaining: ${remaining.length}\n`);

const SYS = `Anda adalah annotator ahli sentimen politik Indonesia.
Tentukan sentimen TERHADAP entitas (bukan YANG DIKATAKAN entitas).

Aturan:
- "relevant" = entitas adalah SUBJEK UTAMA dari konteks
- "not_relevant" = entitas hanya disebut di latar
- Jika entitas mengkritik/menyatakan sesuatu -> NEUTRAL (pembicara)
- Jika entitas dipuji/didukung/diprestasikan -> POSITIF
- Jika entitas dikritik/dicela/divonis/dituntut -> NEGATIF

Output WAJIB: JSON array di dalam \`\`\`json ... \`\`\` block. Setiap elemen:
  {"id": <index>, "gold_label": "positive"|"neutral"|"negative",
   "gold_relevancy": "relevant"|"not_relevant",
   "reasoning": "<alasan singkat <120 char>"}`;

async function main() {
    const zai = await ZAI.create();
    console.log('ZAI SDK ready\n');

    const BATCH = 5;
    const DELAY = 3000;
    let count = done.size;
    const t0 = Date.now();

    for (let i = 0; i < remaining.length; i += BATCH) {
        const batch = remaining.slice(i, i + BATCH);

        let prompt = 'BARIS:\n';
        batch.forEach((r, j) => {
            prompt += `[${j}] entity="${r.entity_name}"\n`;
            prompt += `context="${(r.context_text || '').slice(0, 380)}"\n\n`;
        });
        prompt += 'Output HANYA JSON array di ```json ... ``` block.';

        let processedAny = false;
        try {
            const completion = await zai.chat.completions.create({
                messages: [
                    { role: 'assistant', content: SYS },
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
                    let label = item.gold_label || r.pseudo_label;
                    if (!['positive', 'neutral', 'negative'].includes(label)) label = r.pseudo_label;
                    let rel = item.gold_relevancy || 'relevant';
                    if (!['relevant', 'not_relevant'].includes(rel)) rel = 'relevant';
                    done.set(`${r.raw_text_id}|${r.entity_name}`, {
                        raw_text_id: r.raw_text_id,
                        entity_name: r.entity_name,
                        pseudo_label: r.pseudo_label,
                        gold_label: label,
                        gold_relevancy: rel,
                        reasoning: (item.reasoning || '').slice(0, 200),
                        label_source: 'llm_verified_final',
                        label_confidence: 0.85,
                    });
                    count++;
                    processedAny = true;
                }
            }
        } catch (e) {
            // Will retry 1-by-1 below
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
                            { role: 'assistant', content: SYS },
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
                                let label = item.gold_label || r.pseudo_label;
                                if (!['positive', 'neutral', 'negative'].includes(label)) label = r.pseudo_label;
                                let rel = item.gold_relevancy || 'relevant';
                                if (!['relevant', 'not_relevant'].includes(rel)) rel = 'relevant';
                                done.set(key, {
                                    raw_text_id: r.raw_text_id,
                                    entity_name: r.entity_name,
                                    pseudo_label: r.pseudo_label,
                                    gold_label: label,
                                    gold_relevancy: rel,
                                    reasoning: (item.reasoning || '').slice(0, 200),
                                    label_source: 'llm_verified_final',
                                    label_confidence: 0.85,
                                });
                                count++;
                                continue;
                            }
                        } catch {}
                    }
                } catch {}
                console.log(`  [skip] ${r.entity_name} failed`);
            }
        }

        // Save progress every batch
        const lines = Array.from(done.values()).map(v => JSON.stringify(v)).join('\n');
        writeFileSync(OUT, lines + '\n');

        const elapsed = (Date.now() - t0) / 1000;
        const rate = count > 0 ? count / elapsed : 0;
        const eta = rate > 0 ? (remaining.length - count) / rate : 0;
        if (count % 20 === 0 || i + BATCH >= remaining.length) {
            console.log(`[${count}/${remaining.length}] ${elapsed.toFixed(0)}s | ETA ${eta.toFixed(0)}s`);
        }

        await new Promise(r => setTimeout(r, DELAY));
    }

    console.log(`\n✅ DONE! ${count} verified -> ${OUT}`);
}

main().catch(e => { console.error('Fatal:', e); process.exit(1); });
