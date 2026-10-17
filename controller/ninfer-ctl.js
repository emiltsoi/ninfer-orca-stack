#!/usr/bin/env node
/**
 * ninfer-ctl — Ollama-like model controller over ninfer-serve (Windows, single GPU)
 *
 * One endpoint, name any alias: the controller kills the current ninfer-serve
 * child and spawns the requested profile (ninver-serve boots in ~5s), then
 * proxies through. Child binds loopback only; the controller is the LAN face.
 *
 * Endpoints (controller port, default :11434):
 *   POST /v1/chat/completions   — proxy to loaded child (auto-swap by model name)
 *   POST /v1/messages           — same, Anthropic schema
 *   GET  /v1/models             — aliases with real context windows advertised
 *   POST /yield                 — kill child; free ALL VRAM
 *   GET  /health                — controller liveness + current alias
 *
 * Configuration (env):
 *   NINFER_HOME        — root of the ninfer checkout containing
 *                        build\apps\ninfer-serve.exe and models\*.ninfer
 *   NINFER_LOG_DIR     — where ninfer-child.log / ninfer-req.log go
 *                        (default: this script's directory)
 *   NINFER_PORT        — public listen port        (default 11434)
 *   NINFER_CHILD_PORT  — loopback child port       (default 11435)
 *
 * Profiles below are VRAM-matched (~2.2–2.5 GiB free @C=2) on an RTX 5090 32 GB
 * with ~3 GiB of desktop WDDM usage. KV is ~18.9 KiB/token in NVFP4 — retune
 * ctx to taste; keep >=2 GiB free for desktop safety.
 */
'use strict';

const http = require('http');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const PORT = Number(process.env.NINFER_PORT || 11434);
const CHILD_PORT = Number(process.env.NINFER_CHILD_PORT || 11435);
const LOG_DIR = process.env.NINFER_LOG_DIR || __dirname;
const CHILD_LOG = path.join(LOG_DIR, 'ninfer-child.log');
const REQ_LOG = path.join(LOG_DIR, 'ninfer-req.log');
const NINFER_HOME = process.env.NINFER_HOME || path.resolve(__dirname, '..', '..', 'ninfer-5090-windows');
const NINFER_SERVE = path.join(NINFER_HOME, 'build', 'apps', 'ninfer-serve.exe');
// Consolidated 3090-line build (iamwavecut/ninfer-3090 master), compiled locally for
// sm_120a on the compatibility path — the only line that loads T2G128 ternary artifacts.
// It does NOT build the FP8/NVFP4 routes, so Orca profiles keep the stock binary.
const NINFER_CONS_SERVE = process.env.NINFER_CONS_SERVE ||
  path.resolve(__dirname, '..', '..', 'ninfer-consolidated', 'build', 'apps', 'ninfer-serve.exe');
// ---- API key: unified across the local stacks (llm-ctl family) ----
// Resolution order: LLM_API_KEY env -> NINFER_API_KEY env (per-stack
// override) -> shared key file -> stack-local key file (legacy).
// Key present => bind 0.0.0.0 and require Bearer/x-api-key on every route
// except GET /health (open liveness). No key => loopback only, no auth.
const SHARED_KEY_FILE = 'C:\\work\\localMachine\\llm-api-key.txt';
const LOCAL_KEY_FILE = path.join(__dirname, 'ninfer-api-key.txt');
function readKeyFile(f) {
  try {
    const line = fs.readFileSync(f, 'utf8').split('\n')
      .map(s => s.trim()).filter(s => s && !s.startsWith('#'))[0];
    return line || null;
  } catch (e) { return null; }
}
const API_KEY = process.env.LLM_API_KEY || process.env.NINFER_API_KEY ||
  readKeyFile(SHARED_KEY_FILE) || readKeyFile(LOCAL_KEY_FILE);
const BIND = API_KEY ? '0.0.0.0' : '127.0.0.1';
function authorized(req) {
  if (!API_KEY) return true;
  const bearer = (req.headers.authorization || '').replace(/^Bearer\s+/i, '');
  const k = bearer || req.headers['x-api-key'] || '';
  return k.length === API_KEY.length &&
    crypto.timingSafeEqual(Buffer.from(k), Buffer.from(API_KEY));
}

const MODEL_DIR = path.join(NINFER_HOME, 'models');
const ART_MTP = path.join(MODEL_DIR, 'qwen3_8_27b_orca_nvfp4.ninfer');
const ART_DF2 = path.join(MODEL_DIR, 'qwen3_8_27b_orca_nvfp4_dflash2.ninfer');
const ART_BONSAI = path.join(MODEL_DIR, 'Ternary-Bonsai-2-27B-ninfer-v3.ninfer');
const ART_BONSAI_HERETIC = path.join(MODEL_DIR, 'Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer');

// ---- Profile registry: alias -> launch config ----
// { artifact, ctx, kv, extra[] } — ctx drives both --max-context and /v1/models
// advertising (agents read context_length/max_model_len to size requests).
// kv pins --kv-capacity (64-token page aligned): at concurrency 2 the pool is
// shared, so two in-flight requests must fit kv tokens combined. 'auto' would
// let the engine greedily expand the pool to ~all free VRAM (measured: 147 MiB
// free at C=2/orca — unsafe under WDDM desktop pressure).
const COMMON = ['--max-concurrency', '2', '--kv-dtype', 'nvfp4', '--cors'];
const YARN125 = ['--rope-yarn-factor', '1.25', '--rope-original-max-position', '262144'];
const MTP = ['--spec', 'mtp', '--draft-tokens', '5', '--lm-head-draft'];
const DF2 = ['--spec', 'dflash2', '--draft-tokens', '7', '--lm-head-draft'];

// Bonsai profiles run the consolidated binary (t2_g128_fp16 + hadamard_signs exist only
// there). kv-dtype differs: rk4v4-e8 is ~9.5 KiB/token, so the full 1,048,576-token
// context fits at C=2 with ~2.5-3 GiB free (measured 22:56 boot: 18.0 GiB runtime,
// 3.3 GiB free after the desktop pile was trimmed). Per the artifact card, DFlash2 takes
// no --lm-head-draft (proposal head gains nothing there). No --rope-yarn: plain RoPE
// needle-tested better than YaRN past the native 262,144 window on this model.
// --default-thinking-budget 32768 (2026-10-14): past ~350k of prompt the model can fall
// into exact-repeat reasoning loops; the cap force-closes thinking so the request still
// yields an answer instead of burning the whole max_tokens budget on a loop.
// --min-p 0.05 (2026-10-14): upstream card's thinking-mode value; the engine preset is
// card-exact otherwise. Applies to both modes (flag is not per-mode), which is a slight
// tightening over the card's non-thinking 0.0 — negligible under top-p 0.8.
const BONSAI_DF2 = ['--model-id', 'bonsai2-27b', '--spec', 'dflash2', '--draft-tokens', '7',
  '--kv-dtype', 'rk4v4-e8', '--gdn-state-fp16', '--max-concurrency', '2', '--cors',
  '--default-thinking-budget', '32768', '--min-p', '0.05'];
// Same flags, different reported id: the Heretic (abliterated) conversion is a
// separate artifact built locally from OS-Software's ternary GGUF.
const BONSAI_HERETIC_DF2 = ['--model-id', 'bonsai2-27b-heretic', '--spec', 'dflash2',
  '--draft-tokens', '7', '--kv-dtype', 'rk4v4-e8', '--gdn-state-fp16',
  '--max-concurrency', '2', '--cors', '--default-thinking-budget', '32768',
  '--min-p', '0.05'];

const MODELS = {
  'qwen-3.8-orca': {
    artifact: ART_MTP, ctx: 315000, kv: 315008,
    extra: [...COMMON, ...MTP, ...YARN125],
  },
  'qwen-3.8-orca-fast': {
    artifact: ART_DF2, ctx: 230000, kv: 230016,
    extra: [...COMMON, ...DF2],
  },
  'qwen-3.8-orca-vision': {
    artifact: ART_MTP, ctx: 288000, kv: 288000,
    extra: ['--vision', ...COMMON, ...MTP, ...YARN125],
  },
  'qwen-3.8-orca-vision-fast': {
    artifact: ART_DF2, ctx: 200000, kv: 200000,
    extra: ['--vision', ...COMMON, ...DF2],
  },
  'bonsai2-27b-vision': {
    artifact: ART_BONSAI, ctx: 1048576, kv: 1048576, serve: NINFER_CONS_SERVE,
    reportedId: 'bonsai2-27b',
    extra: ['--vision', '--vision-residency', 'overlay', '--vision-max-merged', '12288', ...BONSAI_DF2],
  },
  'bonsai2-27b-heretic-vision': {
    artifact: ART_BONSAI_HERETIC, ctx: 1048576, kv: 1048576, serve: NINFER_CONS_SERVE,
    reportedId: 'bonsai2-27b-heretic',
    extra: ['--vision', '--vision-residency', 'overlay', '--vision-max-merged', '12288', ...BONSAI_HERETIC_DF2],
  },
};

// Retired aliases -> closest new profile (keeps old clients working).
const LEGACY = {
  'qwen38-orca': 'qwen-3.8-orca',
  'qwen38-orca-yarn': 'qwen-3.8-orca',
  'qwen38-orca-vision': 'qwen-3.8-orca-vision',
};
const DEFAULT_MODEL = 'qwen-3.8-orca';

let child = null;        // current ninfer-serve child
let currentModel = null; // alias of loaded profile
let proxyTarget = null;  // child's API base
let loadingPromise = null;
let inFlight = 0;
let healthFails = 0;

// A supervisorless controller must not die on a stray socket exception.
process.on('uncaughtException', (e) => console.log('ninfer-ctl: uncaught ' + ((e && e.stack) || e)));
process.on('unhandledRejection', (e) => console.log('ninfer-ctl: unhandled rejection ' + ((e && e.stack) || e)));

// Lightweight conversation-digest logger (prefix-hash cache-miss diagnosis).
function logRequest(body, model) {
  try {
    const data = JSON.parse(body);
    const messages = Array.isArray(data.messages) ? data.messages : [];
    const hash = (x) => crypto.createHash('sha256').update(JSON.stringify(x)).digest('hex').slice(0, 16);
    const phash = (n) => `p${n}=${hash(messages.slice(0, Math.min(n, messages.length)))}`;
    const totalChars = messages.reduce((n, m) => n + (typeof m.content === 'string' ? m.content.length : JSON.stringify(m.content || '').length), 0);
    const last = messages[messages.length - 1] || {};
    const line = [
      new Date().toISOString(),
      model,
      `n=${messages.length}`,
      `chars=${totalChars}`,
      `msgHash=${hash(messages)}`,
      [1, 2, 5, 10, 25, 50, 100, 200].map(phash).join(' '),
      `last=${last.role || 'none'}/${hash(last)}`,
    ].join(' | ') + '\n';
    fs.appendFileSync(REQ_LOG, line);
  } catch (e) { /* swallow parse/log errors */ }
}

// ---- Child management ----
function killChild() {
  return new Promise((resolve) => {
    if (!child) return resolve();
    child.kill('SIGKILL');
    child.on('exit', () => { child = null; currentModel = null; resolve(); });
    setTimeout(resolve, 10000); // SIGKILL'd child can take a beat to release the port
  });
}

function waitPortFree(timeoutMs = 20000) {
  return new Promise((resolve) => {
    const t0 = Date.now();
    const tick = () => {
      const s = require('net').connect(CHILD_PORT, '127.0.0.1');
      s.once('connect', () => { s.destroy(); (Date.now() - t0 > timeoutMs) ? resolve() : setTimeout(tick, 250); });
      s.once('error', () => { resolve(); });
    };
    tick();
  });
}

function waitForHealth(url, timeoutMs = 90000) {
  return new Promise((resolve, reject) => {
    const t0 = Date.now();
    const tick = () => {
      http.get(url + '/health', { agent: false }, (res) => {
        let d = '';
        res.on('data', c => d += c);
        res.on('end', () => {
          if (d.includes('"ok"')) return resolve();
          (Date.now() - t0 > timeoutMs) ? reject(new Error('health timeout')) : setTimeout(tick, 1000);
        });
      }).on('error', () => {
        (Date.now() - t0 > timeoutMs) ? reject(new Error('health timeout')) : setTimeout(tick, 1000);
      });
    };
    tick();
  });
}

// Serialize loads: kill/reap/spawn/health-wait must never interleave.
let loadQueue = Promise.resolve();
function loadModel(alias) {
  const next = loadQueue.then(() => loadModelInner(alias));
  loadQueue = next.catch(() => {});
  return next;
}

// Kill whatever owns the child port — orphaned children spawned by a
// dead/restarted controller are unreachable via killChild() and a new spawn
// dies on bind. MUST be awaited: powershell takes ~1s to start and evaluates
// the port owner at exec time — fired async, it would kill our own
// just-bound child.
function killPortOwner() {
  return new Promise((resolve) => {
    const kp = spawn('powershell', ['-NoProfile', '-Command',
      `(Get-NetTCPConnection -LocalPort ${CHILD_PORT} -State Listen -ErrorAction SilentlyContinue).OwningProcess | ForEach-Object { try { Stop-Process -Id $_ -Force } catch {} }`],
      { stdio: 'ignore' });
    kp.on('exit', resolve);
    setTimeout(resolve, 15000);
  });
}

async function loadModelInner(alias) {
  const cfg = MODELS[alias];
  if (!cfg) throw new Error('unknown model: ' + alias);
  if (currentModel === alias && child) {
    if (loadingPromise) await loadingPromise;
    return;
  }

  await killChild();
  await killPortOwner();
  await waitPortFree();

  const serve = cfg.serve || NINFER_SERVE;
  const args = [cfg.artifact, '--host', '127.0.0.1', '--port', String(CHILD_PORT),
                '--max-context', String(cfg.ctx), '--kv-capacity', String(cfg.kv), ...cfg.extra];

  // Boot: spawn + health-wait, but reject FAST if the child exits mid-boot —
  // a startup FATAL (e.g. transient desktop VRAM pressure) must not stall the
  // serialized queue for the full 90s health timeout. Retry after a delay:
  // observed spikes clear within ~60-90s, so 3 attempts cover most cases.
  const MAX_ATTEMPTS = 3, RETRY_DELAY_MS = 15000;
  let lastErr = null;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    if (attempt > 1) {
      console.log('ninfer-ctl: ' + alias + ' boot attempt ' + attempt + ' after ' + (lastErr && lastErr.message));
      await new Promise((r) => setTimeout(r, RETRY_DELAY_MS));
      await killPortOwner();
      await waitPortFree();
    }

    const childLogFd = fs.openSync(CHILD_LOG, 'a');
    fs.writeSync(childLogFd, `\n===== spawn ${alias} (attempt ${attempt}) @ ${new Date().toISOString()} =====\n`);
    child = spawn(serve, args, { stdio: ['ignore', childLogFd, childLogFd], detached: false });
    child.on('exit', (code) => {
      try { fs.closeSync(childLogFd); } catch (e) {}
      fs.appendFileSync(CHILD_LOG, `===== exit ${alias} code ${code} @ ${new Date().toISOString()} =====\n`);
      child = null; currentModel = null;
    });
    currentModel = alias;
    proxyTarget = 'http://127.0.0.1:' + CHILD_PORT;

    const proc = child;
    loadingPromise = new Promise((resolve, reject) => {
      proc.once('exit', (code) => reject(new Error('ninfer-serve exited during boot (code ' + (code === null ? 'signal' : code) + ')')));
      proc.once('error', (e) => reject(new Error('spawn failed: ' + e.message)));
      waitForHealth(proxyTarget).then(resolve, reject);
    });
    try {
      await loadingPromise;
      return;
    } catch (e) {
      lastErr = e;
      console.log('ninfer-ctl: ' + alias + ' boot attempt ' + attempt + ' failed: ' + e.message);
    } finally {
      loadingPromise = null;
    }
  }
  throw lastErr;
}

// ---- Child health watchdog ----
function checkChildHealth() {
  return new Promise((resolve) => {
    if (!child || !proxyTarget) return resolve(false);
    const req = http.get(proxyTarget + '/health', { timeout: 10000, agent: false }, (res) => {
      res.resume(); req.destroy();
      resolve(true); // any HTTP response means alive
    });
    req.on('error', () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
  });
}

async function ensureChildHealthy() {
  if (!child || !proxyTarget || loadingPromise || inFlight > 0) return;
  const healthy = await checkChildHealth();
  healthFails = healthy ? 0 : healthFails + 1;
  if (healthFails >= 2) {
    console.log('ninfer-ctl: child unresponsive x' + healthFails + ' - reloading ' + currentModel);
    healthFails = 0;
    const alias = currentModel;
    await killChild();
    if (alias) await loadModel(alias);
  }
}

// ---- HTTP plumbing ----
// Never forward hop-by-hop headers upstream->client (connection: close RSTs on
// Windows loopback before the body flushes).
const HOP_BY_HOP = new Set(['connection', 'keep-alive', 'transfer-encoding', 'te', 'trailer', 'upgrade', 'proxy-authenticate', 'proxy-authorization']);
function stripHopHeaders(h) {
  const out = {};
  for (const k in h) if (!HOP_BY_HOP.has(k)) out[k] = h[k];
  return out;
}

function resolveAlias(raw) {
  let model = (raw || DEFAULT_MODEL).split(':')[0].replace(/:(256k|latest|q4|q8)$/, '');
  if (MODELS[model]) return model;
  if (LEGACY[model]) return LEGACY[model];
  return DEFAULT_MODEL; // single-model stack: unknown names get the base profile
}

// Two upstream-body rewrites the children need:
// - NInfer's embedded chat template only supports reasoning effort
//   none/low/medium/xhigh — it 400s on minimal/high/max. Translate here so
//   clients tuned for bigger providers keep working.
// - The consolidated server validates the request 'model' against its reported
//   id; Orca ignores it. Profiles with reportedId get body.model rewritten so
//   alias names (e.g. bonsai2-27b-vision) don't trip the check.
function normalizeUpstreamBody(body, cfg) {
  try {
    const data = JSON.parse(body);
    let changed = false;
    const REWRITE = { minimal: 'low', high: 'xhigh', max: 'xhigh' };
    if (typeof data.reasoning_effort === 'string' && REWRITE[data.reasoning_effort]) {
      data.reasoning_effort = REWRITE[data.reasoning_effort];
      changed = true;
    }
    if (cfg && cfg.reportedId && data.model !== cfg.reportedId) {
      data.model = cfg.reportedId;
      changed = true;
    }
    // vLLM-alias clash: assistant messages may carry `reasoning` and
    // `reasoning_content`; the engine 400s when both exist with different
    // values (some clients emit both and they drift). Engine prefers
    // `reasoning` — drop the legacy alias on conflict.
    let dropped = 0;
    for (const m of (data.messages || [])) {
      if (m && m.role === 'assistant' &&
          typeof m.reasoning === 'string' && typeof m.reasoning_content === 'string' &&
          m.reasoning && m.reasoning_content && m.reasoning !== m.reasoning_content) {
        delete m.reasoning_content;
        dropped++; changed = true;
      }
    }
    if (dropped) console.log('ninfer-ctl: dropped conflicting reasoning_content on ' + dropped + ' assistant message(s)');
    if (changed) return JSON.stringify(data);
  } catch (e) {}
  return body;
}

function proxy(req, res) {
  if (!proxyTarget) { res.writeHead(503); return res.end('no model loaded'); }
  inFlight++;
  res.on('close', () => { inFlight--; });
  let done = false;
  const finish = (code, msg) => {
    if (done) return;
    done = true;
    if (code) { try { res.writeHead(code); } catch (e) {} }
    if (msg) { try { res.end(msg); } catch (e) {} }
  };
  const p = http.request(proxyTarget + req.url, { method: req.method, headers: req.headers, agent: false }, (pr) => {
    done = true;
    res.writeHead(pr.statusCode, stripHopHeaders(pr.headers));
    pr.pipe(res);
    pr.on('error', () => { try { res.destroy(); } catch (e) {} });
  });
  p.on('error', (e) => { finish(502, 'proxy error: ' + e.message); });
  req.pipe(p);
  req.on('error', () => { try { p.destroy(); } catch (e) {} });
}

const server = http.createServer(async (req, res) => {
  // Open liveness only; everything else requires the key when one is set.
  if (req.method !== 'GET' || (req.url !== '/health' && req.url !== '/')) {
    if (!authorized(req)) {
      res.writeHead(401, { 'content-type': 'application/json' });
      return res.end('{"error":"unauthorized"}');
    }
  }

  if (req.method === 'POST' && (req.url.startsWith('/v1/chat/completions') || req.url.startsWith('/v1/messages'))) {
    let body = '';
    let responded = false;
    const guard = (fn) => { if (!responded) { responded = true; fn(); } };
    req.on('data', c => body += c);
    req.on('end', async () => {
      try {
        let requested = '';
        try { requested = JSON.parse(body).model || ''; } catch (e) {}
        const model = resolveAlias(requested);
        await loadModel(model);
        logRequest(body, requested || model);
        await ensureChildHealthy();
        inFlight++;
        res.on('close', () => { inFlight--; });
        const upstreamBody = normalizeUpstreamBody(body, MODELS[model]);
        const sendUpstream = (attempt) => {
          const p = http.request(proxyTarget + req.url, { method: 'POST', headers: { ...req.headers, 'content-length': Buffer.byteLength(upstreamBody), 'connection': 'close' }, agent: false }, (pr) => {
            guard(() => { res.writeHead(pr.statusCode, stripHopHeaders(pr.headers)); });
            pr.pipe(res);
            pr.on('error', () => { try { res.destroy(); } catch (e) {} });
          });
          p.on('error', (e) => {
            if (attempt < 2) return sendUpstream(attempt + 1);
            guard(() => { res.writeHead(502); res.end('proxy error: ' + e.message); });
          });
          p.end(upstreamBody);
        };
        sendUpstream(0);
      } catch (e) {
        guard(() => {
          res.writeHead(500, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ error: { message: String(e.message || e) } }));
        });
      }
    });
    req.on('error', () => {});
    return;
  }

  if (req.method === 'GET' && req.url.startsWith('/v1/models')) {
    const created = Math.floor(Date.now() / 1000);
    const data = Object.entries(MODELS).map(([a, cfg]) => ({
      id: a,
      object: 'model',
      created,
      owned_by: 'ninfer-ctl',
      // Advertise the real per-request context limit in the three common field
      // names: max_model_len (vLLM/llama.cpp), context_length (OpenRouter), and
      // context_window (DSH/LiteLLM-style harnesses).
      max_model_len: cfg.ctx,
      context_length: cfg.ctx,
      context_window: cfg.ctx,
      // Reasoning effort: the embedded Qwen template natively implements only
      // low/medium/xhigh (template default xhigh when the field is omitted);
      // 'none' is handled at protocol level and disables thinking. The
      // controller additionally translates minimal→low, high→xhigh, max→xhigh
      // so those wire values are accepted but are NOT distinct levels.
      reasoning_efforts: ['none', 'low', 'medium', 'xhigh'],
      default_reasoning_effort: 'xhigh',
      reasoning_effort_aliases: { minimal: 'low', high: 'xhigh', max: 'xhigh' },
      supported_parameters: ['reasoning_effort', 'temperature', 'top_p', 'top_k',
        'min_p', 'presence_penalty', 'frequency_penalty', 'seed', 'stop',
        'max_tokens', 'chat_template_kwargs'],
      vision: cfg.extra.includes('--vision'),
    }));
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ object: 'list', data }));
    return;
  }

  if (req.method === 'POST' && req.url.startsWith('/yield')) {
    await killChild();
    res.writeHead(200); res.end('{"yielded":true}');
    return;
  }

  if (req.method === 'POST' && req.url.startsWith('/stop')) {
    res.writeHead(200); res.end('{"stopping":true}');
    setTimeout(async () => { try { await killChild(); } catch (e) {} process.exit(0); }, 100);
    return;
  }

  if (req.method === 'GET' && (req.url === '/health' || req.url === '/')) {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ controller: 'ok', loaded: currentModel || null }));
    return;
  }

  // Fallback: transparent proxy (other /v1 routes)
  if (proxyTarget) return proxy(req, res);
  res.writeHead(404); res.end('not found');
});

server.listen(PORT, BIND, () => {
  console.log(`ninfer-ctl on ${BIND}:${PORT} — models: ${Object.keys(MODELS).join(', ')}`);
  console.log(API_KEY ? 'API key active — auth required on all routes except GET /health.'
                      : 'No API key — loopback only.');
  console.log('Ready. Auto-swaps on model name; POST /yield frees VRAM.');
});
