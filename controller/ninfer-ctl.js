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
 * Profiles below are VRAM-matched (~2.7–3.2 GiB free) on an RTX 5090 32 GB
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
const MODEL_DIR = path.join(NINFER_HOME, 'models');
const ART_MTP = path.join(MODEL_DIR, 'qwen3_8_27b_orca_nvfp4.ninfer');
const ART_DF2 = path.join(MODEL_DIR, 'qwen3_8_27b_orca_nvfp4_dflash2.ninfer');

// ---- Profile registry: alias -> launch config ----
// { artifact, ctx, extra[] } — ctx drives both --max-context and /v1/models
// advertising (agents read context_length/max_model_len to size requests).
const COMMON = ['--kv-capacity', 'auto', '--max-concurrency', '1', '--kv-dtype', 'nvfp4', '--cors'];
const YARN125 = ['--rope-yarn-factor', '1.25', '--rope-original-max-position', '262144'];
const MTP = ['--spec', 'mtp', '--draft-tokens', '5', '--lm-head-draft'];
const DF2 = ['--spec', 'dflash2', '--draft-tokens', '7', '--lm-head-draft'];

const MODELS = {
  'qwen-3.8-orca': {
    artifact: ART_MTP, ctx: 315000,
    extra: [...COMMON, ...MTP, ...YARN125],
  },
  'qwen-3.8-orca-fast': {
    artifact: ART_DF2, ctx: 230000,
    extra: [...COMMON, ...DF2],
  },
  'qwen-3.8-orca-vision': {
    artifact: ART_MTP, ctx: 288000,
    extra: ['--vision', ...COMMON, ...MTP, ...YARN125],
  },
  'qwen-3.8-orca-vision-fast': {
    artifact: ART_DF2, ctx: 200000,
    extra: ['--vision', ...COMMON, ...DF2],
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

async function loadModelInner(alias) {
  const cfg = MODELS[alias];
  if (!cfg) throw new Error('unknown model: ' + alias);
  if (currentModel === alias && child) {
    if (loadingPromise) await loadingPromise;
    return;
  }

  await killChild();
  // Orphaned children (spawned by a dead/restarted controller) still hold the
  // child port — our killChild() can't reach them and a new spawn dies on bind.
  // MUST be awaited: powershell takes ~1s to start and evaluates the port owner
  // at exec time — fired async, it would kill our own just-bound child.
  await new Promise((resolve) => {
    const kp = spawn('powershell', ['-NoProfile', '-Command',
      `(Get-NetTCPConnection -LocalPort ${CHILD_PORT} -State Listen -ErrorAction SilentlyContinue).OwningProcess | ForEach-Object { try { Stop-Process -Id $_ -Force } catch {} }`],
      { stdio: 'ignore' });
    kp.on('exit', resolve);
    setTimeout(resolve, 15000);
  });
  await waitPortFree();

  const args = [cfg.artifact, '--host', '127.0.0.1', '--port', String(CHILD_PORT),
                '--max-context', String(cfg.ctx), ...cfg.extra];

  const childLogFd = fs.openSync(CHILD_LOG, 'a');
  fs.writeSync(childLogFd, `\n===== spawn ${alias} @ ${new Date().toISOString()} =====\n`);
  child = spawn(NINFER_SERVE, args, { stdio: ['ignore', childLogFd, childLogFd], detached: false });
  child.on('exit', (code) => {
    try { fs.closeSync(childLogFd); } catch (e) {}
    fs.appendFileSync(CHILD_LOG, `===== exit ${alias} code ${code} @ ${new Date().toISOString()} =====\n`);
    child = null; currentModel = null;
  });
  currentModel = alias;
  proxyTarget = 'http://127.0.0.1:' + CHILD_PORT;

  loadingPromise = waitForHealth(proxyTarget);
  try { await loadingPromise; } finally { loadingPromise = null; }
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

// NInfer's embedded chat template only supports reasoning effort
// none/low/medium/xhigh — it 400s on minimal/high/max. Translate here so
// clients tuned for bigger providers keep working.
function normalizeReasoningEffort(body) {
  try {
    const data = JSON.parse(body);
    const REWRITE = { minimal: 'low', high: 'xhigh', max: 'xhigh' };
    if (typeof data.reasoning_effort === 'string' && REWRITE[data.reasoning_effort]) {
      data.reasoning_effort = REWRITE[data.reasoning_effort];
      return JSON.stringify(data);
    }
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
        const upstreamBody = normalizeReasoningEffort(body);
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
      // Advertise the real per-request context limit in both common fields:
      // max_model_len (vLLM/llama.cpp, matches ninfer-serve's own listing) and
      // context_length (OpenRouter convention read by many agent harnesses).
      max_model_len: cfg.ctx,
      context_length: cfg.ctx,
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

  if (req.method === 'GET' && (req.url === '/health' || req.url === '/')) {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ controller: 'ok', loaded: currentModel || null }));
    return;
  }

  // Fallback: transparent proxy (other /v1 routes)
  if (proxyTarget) return proxy(req, res);
  res.writeHead(404); res.end('not found');
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`ninfer-ctl on :${PORT} — models: ${Object.keys(MODELS).join(', ')}`);
  console.log('Ready. Auto-swaps on model name; POST /yield frees VRAM.');
});
