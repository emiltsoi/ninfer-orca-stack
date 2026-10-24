// bonsai-attractor-probe.js — does a looped assistant turn preserved in history
// re-trigger degeneration at depth? Reuses the repo haystack; injects a synthetic
// assistant message whose reasoning_content is a captured-style loop, then asks
// "continue". Same depth (~390k) where the incident occurred.

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const http = require('http');

const HOST = '127.0.0.1', PORT = 11434, MODEL = 'bonsai2-27b-heretic-vision';

const ROOTS = [
  'C:\\work\\localMachine\\ninfer-consolidated\\docs',
  'C:\\work\\localMachine\\ninfer-consolidated\\src',
  'C:\\work\\localMachine\\ninfer-consolidated\\tests',
  'C:\\work\\localMachine\\ninfer-consolidated\\include',
  'C:\\work\\localMachine\\ninfer-consolidated\\apps',
];
const EXTS = new Set(['.md', '.cpp', '.h', '.cuh', '.py', '.txt']);
const SKIP = /node_modules|\.git|vcpkg|corpora|ffmpeg/i;

function walk(dir, acc) {
  let ents;
  try { ents = fs.readdirSync(dir, { withFileTypes: true }); } catch { return acc; }
  for (const e of ents) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) { if (!SKIP.test(e.name)) walk(p, acc); }
    else if (EXTS.has(path.extname(e.name).toLowerCase())) acc.push(p);
  }
  return acc;
}
let files = [];
for (const r of ROOTS) walk(r, files);
files.sort();
const parts = [];
let total = 0;
for (const f of files) {
  let t;
  try { t = fs.readFileSync(f, 'utf8'); } catch { continue; }
  const chunk = `\n\n===== ${f} =====\n\n` + t;
  parts.push(chunk);
  total += chunk.length;
  if (total > 1500000) break;
}
const HAY = parts.join('');

// mimics the captured failure: reasoning collapsing into repeated table cells
const loop =
  'Let me verify the field table.\n\n' +
  ('| field | value | expected |\n| `remoteusermachinename` | 15 | 15 |\n' +
   '| `remoteusermachinenamefull` | 15 | 15 |\n').repeat(80) +
  '\nI still need to check the remaining columns before answering.';

function ratio(t) {
  if (!t || t.length < 200) return NaN;
  return zlib.deflateSync(Buffer.from(t)).length / t.length;
}
function maxDup(t, k = 60) {
  if (!t) return 0;
  const m = new Map();
  for (let i = 0; i + k <= t.length; i += Math.max(1, k >> 1)) {
    const s = t.slice(i, i + k);
    m.set(s, (m.get(s) || 0) + 1);
  }
  let mx = 0;
  for (const v of m.values()) if (v > mx) mx = v;
  return mx;
}
function post(body) {
  return new Promise((resolve, reject) => {
    const data = Buffer.from(JSON.stringify(body));
    const req = http.request(
      { host: HOST, port: PORT, path: '/v1/chat/completions', method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Content-Length': data.length },
        timeout: 900000 },
      (res) => {
        const chunks = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => {
          try { resolve(JSON.parse(Buffer.concat(chunks).toString())); }
          catch { reject(new Error('bad json')); }
        });
      });
    req.on('timeout', () => req.destroy(new Error('timeout')));
    req.on('error', reject);
    req.end(data);
  });
}

(async () => {
  const chars = Math.round(390000 * 3.3);
  const hayPrompt = HAY.slice(0, chars) +
    '\n\nUsing the material above, enumerate the major components of the codebase and ' +
    'describe each in one line. Then state the total count.';
  const r = await post({
    model: MODEL,
    messages: [
      { role: 'user', content: hayPrompt },
      { role: 'assistant', content: '', reasoning_content: loop },
      { role: 'user', content: 'Continue.' },
    ],
    max_tokens: 16000,
    reasoning_effort: 'xhigh',
    stream: false,
  });
  if (r.error) { console.log('ERROR', JSON.stringify(r.error).slice(0, 300)); return; }
  const msg = r.choices?.[0]?.message || {};
  const fin = r.choices?.[0]?.finish_reason;
  const u = r.usage || {};
  const cd = u.completion_tokens_details || {};
  const reasoning = msg.reasoning_content || '';
  const content = msg.content || '';
  console.log(`attractor@~390k -> prompt ${u.prompt_tokens} | fin ${fin} | ` +
    `out ${u.completion_tokens} (think ${cd.reasoning_tokens ?? '?'}) | ` +
    `ratioR ${isNaN(ratio(reasoning)) ? 'n/a' : ratio(reasoning).toFixed(3)} ` +
    `dupR ${maxDup(reasoning)} | ratioC ${isNaN(ratio(content)) ? 'n/a' : ratio(content).toFixed(3)} ` +
    `dupC ${maxDup(content)}`);
  console.log('--- reasoning tail ---');
  console.log(reasoning.slice(-600));
  console.log('--- content head ---');
  console.log(content.slice(0, 400));
})().catch((e) => { console.log('FATAL', e.message); process.exit(1); });
