// bonsai-context-probe.js — ascending-depth context probe for bonsai2-27b-heretic-vision.
// Sends needle-in-haystack prompts at increasing prompt depths, records prompt_tokens,
// finish_reason, thinking usage, needle hits, and a zlib compression ratio on the
// reasoning text (loops collapse toward ~0.02; coherent text ~0.25+).

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const http = require('http');

const HOST = '127.0.0.1';
const PORT = 11434;
const MODEL = 'bonsai2-27b-heretic-vision';
const MAX_OUT = 16000;
const CHARS_PER_TOKEN0 = 3.6; // calibrated after the first probe
const TARGETS = [32000, 150000, 262144, 330000, 400000, 470000];

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
  if (total > 2400000) break;
}
const HAY = parts.join('');
console.log(`haystack ${HAY.length} chars from ${parts.length} files`);

const NEEDLE_EARLY =
  "\n\nPROJECT MEMO EA-77A (operations, internal): the codename for the Q4 cache " +
  "migration is 'amber-tide'. Do not share outside the team.\n\n";
const NEEDLE_LATE =
  "\n\nPROJECT MEMO LB-88B (operations, internal): the deployment password for the " +
  "crimson-falcon service is 'KX-94-ZEBRA-7'. Rotate quarterly.\n\n";

const QUESTION =
  'Two project memos are embedded in the material above. ' +
  "(1) What codename does memo EA-77A assign to the Q4 cache migration? " +
  "(2) What deployment password does memo LB-88B give for the crimson-falcon service? " +
  '(3) Name two distinct components or topics that appear in this codebase material. ' +
  'Answer each briefly.';

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
          catch (e) { reject(new Error('bad json: ' + Buffer.concat(chunks).toString().slice(0, 300))); }
        });
      });
    req.on('timeout', () => req.destroy(new Error('request timeout')));
    req.on('error', reject);
    req.end(data);
  });
}

let ratioCPT = CHARS_PER_TOKEN0;

async function probe(targetTokens, tag) {
  const chars = Math.round(targetTokens * ratioCPT);
  const early = Math.round(chars * 0.08);
  const late = Math.round(chars * 0.88);
  const prompt = HAY.slice(0, early) + NEEDLE_EARLY + HAY.slice(early, late) +
                 NEEDLE_LATE + HAY.slice(late, chars);
  const t0 = Date.now();
  const r = await post({
    model: MODEL,
    messages: [{ role: 'user', content: prompt + '\n\n' + QUESTION }],
    max_tokens: MAX_OUT,
    reasoning_effort: 'xhigh',
    stream: false,
  });
  const dt = ((Date.now() - t0) / 1000).toFixed(0);
  if (r.error) { console.log(`${tag} ~${targetTokens} | ERROR ${JSON.stringify(r.error).slice(0, 200)}`); return null; }
  const msg = r.choices?.[0]?.message || {};
  const fin = r.choices?.[0]?.finish_reason;
  const u = r.usage || {};
  const cd = u.completion_tokens_details || {};
  const reasoning = msg.reasoning_content || '';
  const content = msg.content || '';
  const rr = ratio(reasoning), cr = ratio(content);
  const rd = maxDup(reasoning), cdup = maxDup(content);
  const hitA = /amber-tide/i.test(content);
  const hitB = /KX-94-ZEBRA-7/i.test(content);
  console.log(
    `${tag} target ${targetTokens} -> prompt ${u.prompt_tokens} | fin ${fin} | out ${u.completion_tokens} (think ${cd.reasoning_tokens ?? '?'}) | ` +
    `hitA ${hitA} hitB ${hitB} | ratioR ${isNaN(rr) ? 'n/a' : rr.toFixed(3)} dupR ${rd} | ratioC ${isNaN(cr) ? 'n/a' : cr.toFixed(3)} dupC ${cdup} | ${dt}s`);
  if (u.prompt_tokens) ratioCPT = chars / u.prompt_tokens; // calibrate
  return { fin, reasoning, content, prompt: u.prompt_tokens };
}

(async () => {
  for (const t of TARGETS) {
    await probe(t, `probe@${Math.round(t / 1024)}k`);
  }
  // one cheap repeat at the last two depths (prefix cached => near-free prefill) to
  // sample stochasticity where degeneration was observed historically
  for (const t of [400000, 470000]) {
    await probe(t, `repeat@${Math.round(t / 1024)}k`);
  }
  console.log('done');
})().catch((e) => { console.log('FATAL', e.message); process.exit(1); });
