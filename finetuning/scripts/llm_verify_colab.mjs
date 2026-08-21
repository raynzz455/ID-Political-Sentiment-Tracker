/**
 * llm_verify_colab.mjs — NO optional chaining (compatible with old Node.js)
 * 
 * Cara pakai di Colab:
 *   !cd /content/ID-Political-Sentiment-Tracker && npm install z-ai-web-dev-sdk
 *   !node finetuning/scripts/llm_verify_colab.mjs
 */

import ZAI from 'z-ai-web-dev-sdk';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { homedir } from 'os';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

var __filename = fileURLToPath(import.meta.url);
var __dirname = dirname(__filename);
var REPO_ROOT = join(__dirname, '..', '..');

// === AUTO-CREATE CONFIG ===
var configContent = JSON.stringify({
    baseUrl: "https://internal-api.z.ai/v1",
    apiKey: "Z.ai",
    chatId: "chat-6f02bcbb-29df-486b-9c2d-b07ae8567b63",
    token: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiYjNkMGJkYjYtYzJkZC00MmIxLTg2ZjgtODkwODQwZDFjZTQ2IiwiY2hhdF9pZCI6ImNoYXQtNmYwMmJjYmItMjlkZi00ODZiLTljMmQtYjA3YWU4NTY3YjYzIiwicGxhdGZvcm0iOiJ6YWkifQ.BJmZsmnRZLSwYZK5Jny_9chyKeMurkweJaAtWhimAgY",
    userId: "b3d0bdb6-c2dd-42b1-86f8-890840d1ce46"
});

var configPaths = [join(process.cwd(), '.z-ai-config'), join(homedir(), '.z-ai-config')];
var configFound = false;
for (var i = 0; i < configPaths.length; i++) {
    if (existsSync(configPaths[i])) {
        try {
            var c = JSON.parse(readFileSync(configPaths[i], 'utf-8'));
            if (c.baseUrl && c.apiKey) { configFound = true; console.log('Config found: ' + configPaths[i]); break; }
        } catch (e) {}
    }
}
if (!configFound) { writeFileSync(configPaths[0], configContent); console.log('Config created: ' + configPaths[0]); }

// === PATHS ===
var INPUT = join(REPO_ROOT, 'finetuning', 'datasets', 'need_verify_final.json');
var OUT = join(REPO_ROOT, 'finetuning', 'datasets', 'llm_verified_final.jsonl');

var toVerify = JSON.parse(readFileSync(INPUT, 'utf-8'));
console.log('Total to verify:', toVerify.length);

// Resume
var done = new Map();
if (existsSync(OUT)) {
    var lines = readFileSync(OUT, 'utf-8').trim().split('\n').filter(Boolean);
    for (var i = 0; i < lines.length; i++) {
        try { var v = JSON.parse(lines[i]); done.set(v.raw_text_id + '|' + v.entity_name, v); } catch (e) {}
    }
    console.log('Resume:', done.size, 'already done');
}

var remaining = toVerify.filter(function(r) { return !done.has(r.raw_text_id + '|' + r.entity_name); });
console.log('Remaining:', remaining.length, '\n');

var SYS = 'Anda adalah annotator ahli sentimen politik Indonesia. Tentukan sentimen TERHADAP entitas (bukan YANG DIKATAKAN entitas). Aturan: entitas mengkritik/menyatakan -> NEUTRAL (pembicara). entitas dipuji/didukung -> POSITIF. entitas dikritik/divonis/dituntut -> NEGATIF. Output: JSON array [{"id":0,"gold_label":"positive|neutral|negative","gold_relevancy":"relevant|not_relevant","reasoning":"<120 char>"}]';

// Helper: safe access (no optional chaining)
function getContent(completion) {
    try {
        var choices = completion.choices;
        if (choices && choices.length > 0) {
            var msg = choices[0].message;
            if (msg && msg.content) return msg.content;
        }
    } catch (e) {}
    return '';
}

function extractJsonArray(content) {
    var m = content.match(/```(?:json)?\s*(\[[\s\S]*?\])\s*```/);
    var jsonStr = m ? m[1] : null;
    if (!jsonStr) {
        var m2 = content.match(/(\[[\s\S]*\])/);
        if (m2) jsonStr = m2[1];
    }
    if (jsonStr) {
        try { return JSON.parse(jsonStr); } catch (e) {}
    }
    return null;
}

async function main() {
    var zai = await ZAI.create();
    console.log('ZAI SDK ready\n');

    var BATCH = 5;
    var DELAY = 3000;
    var count = done.size;
    var t0 = Date.now();

    for (var i = 0; i < remaining.length; i += BATCH) {
        var batch = remaining.slice(i, i + BATCH);

        var prompt = 'BARIS:\n';
        for (var j = 0; j < batch.length; j++) {
            var r = batch[j];
            prompt += '[' + j + '] entity="' + r.entity_name + '"\n';
            prompt += 'context="' + (r.context_text || '').slice(0, 380) + '"\n\n';
        }
        prompt += 'Output HANYA JSON array di ```json ... ``` block.';

        var processedAny = false;
        try {
            var completion = await zai.chat.completions.create({
                messages: [
                    { role: 'assistant', content: SYS },
                    { role: 'user', content: prompt }
                ],
                thinking: { type: 'disabled' }
            });
            var content = getContent(completion);
            var arr = extractJsonArray(content);

            if (arr && arr.length === batch.length) {
                for (var j = 0; j < batch.length; j++) {
                    var item = arr[j];
                    var r = batch[j];
                    var label = item.gold_label || r.pseudo_label;
                    if (!['positive', 'neutral', 'negative'].includes(label)) label = r.pseudo_label;
                    var rel = item.gold_relevancy || 'relevant';
                    if (!['relevant', 'not_relevant'].includes(rel)) rel = 'relevant';
                    done.set(r.raw_text_id + '|' + r.entity_name, {
                        raw_text_id: r.raw_text_id,
                        entity_name: r.entity_name,
                        pseudo_label: r.pseudo_label,
                        gold_label: label,
                        gold_relevancy: rel,
                        reasoning: (item.reasoning || '').slice(0, 200),
                        label_source: 'llm_verified_final',
                        label_confidence: 0.85
                    });
                    count++;
                }
                processedAny = true;
            }
        } catch (e) {
            // Will retry 1-by-1
        }

        // Retry 1-by-1 if batch failed
        if (!processedAny) {
            for (var k = 0; k < batch.length; k++) {
                var r = batch[k];
                var key = r.raw_text_id + '|' + r.entity_name;
                if (done.has(key)) continue;
                try {
                    var sp = 'BARIS:\n[0] entity="' + r.entity_name + '"\ncontext="' + (r.context_text || '').slice(0, 400) + '"\n\nOutput JSON array:';
                    var c2 = await zai.chat.completions.create({
                        messages: [
                            { role: 'assistant', content: SYS },
                            { role: 'user', content: sp }
                        ],
                        thinking: { type: 'disabled' }
                    });
                    var ct2 = getContent(c2);
                    var a2 = extractJsonArray(ct2);
                    if (a2 && a2.length >= 1) {
                        var item = a2[0];
                        var label = item.gold_label || r.pseudo_label;
                        if (!['positive', 'neutral', 'negative'].includes(label)) label = r.pseudo_label;
                        var rel = item.gold_relevancy || 'relevant';
                        if (!['relevant', 'not_relevant'].includes(rel)) rel = 'relevant';
                        done.set(key, {
                            raw_text_id: r.raw_text_id,
                            entity_name: r.entity_name,
                            pseudo_label: r.pseudo_label,
                            gold_label: label,
                            gold_relevancy: rel,
                            reasoning: (item.reasoning || '').slice(0, 200),
                            label_source: 'llm_verified_final',
                            label_confidence: 0.85
                        });
                        count++;
                    }
                } catch (e) {}
            }
        }

        // Save progress
        var lines = [];
        done.forEach(function(v) { lines.push(JSON.stringify(v)); });
        writeFileSync(OUT, lines.join('\n') + '\n');

        var elapsed = (Date.now() - t0) / 1000;
        var rate = count > 0 ? count / elapsed : 0;
        var eta = rate > 0 ? (remaining.length - count) / rate : 0;
        if (count % 20 === 0 || i + BATCH >= remaining.length) {
            console.log('[' + count + '/' + remaining.length + '] ' + elapsed.toFixed(0) + 's | ETA ' + eta.toFixed(0) + 's');
        }

        await new Promise(function(r) { setTimeout(r, DELAY); });
    }

    console.log('\nDONE! ' + count + ' verified -> ' + OUT);
}

main().catch(function(e) { console.error('Fatal:', e); process.exit(1); });
