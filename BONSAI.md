# Bonsai Ternary Deployment (ninfer-orca-stack)

Operational reference for the Ternary Bonsai 2 27B deployment alongside the
Qwen3.8 Orca NVFP4 stack. Written 2026-09-26; reflects measured state on this
machine (RTX 5090 32 GB, CUDA 13.1, MSVC 14.38, Windows/WDDM).

## What is deployed

`ninfer-ctl.js` (controller on `127.0.0.1:11434`) serves four model families
behind per-profile engine binaries:

| Alias | Engine | Artifact | Spec | Vision | Ctx/KV |
|---|---|---|---|---|---|
| `qwen-3.8-orca-*` (existing) | stock `ninfer-serve` | `qwen3_8_27b_orca_nvfp4*.ninfer` | per profile | per profile | unchanged |
| `bonsai2-27b-vision` | consolidated | `Ternary-Bonsai-2-27B-ninfer-v3.ninfer` | DFlash2 ×7 | overlay | 1,048,576 `rk4v4-e8` |
| `bonsai2-27b-heretic-vision` | consolidated | `Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer` | DFlash2 ×7 | overlay | 1,048,576 `rk4v4-e8` |

(Text-only `bonsai2-27b` / `bonsai2-27b-heretic` aliases were removed — the
vision profiles serve text identically, so the extra aliases were pure
duplication costing media-workspace VRAM.)

Common flags for all Bonsai profiles: `--spec dflash2 --draft-tokens 7
--kv-dtype rk4v4-e8 --gdn-state-fp16 --max-concurrency 2 --cors`.
Vision adds `--vision --vision-residency overlay --vision-max-merged 12288`.
Profile definitions live at `controller/ninfer-ctl.js` (`MODELS` block,
artifact constants near the top).

## Artifacts

Located in `C:\work\localMachine\ninfer-5090-windows\models\`.

| File | Bytes | SHA-256 | Origin |
|---|---|---|---|
| `Ternary-Bonsai-2-27B-ninfer-v3.ninfer` | 9,520,051,456 | `cdc4810b…0c14a` | `WaveCut/Ternary-Bonsai-2-27B-NInfer-v3` (official; hash verified vs published sums) |
| `Ternary-Bonsai-2-27B-Heretic-ninfer.ninfer` | 9,520,051,456 | `99f94f44…1de2fd` | **built locally** from `OS-Software/…-Uncensored-Heretic-PQ2_0.gguf`; published at `emiltsoi/Ternary-Bonsai-2-27B-Uncensored-Heretic-NInfer` |

## Engine builds — two binaries, deliberate

Orca profiles run the proven stock binary in `ninfer-5090-windows`.
Bonsai requires the **consolidated ternary line** (fork of
`iamwavecut/ninfer-3090`, cloned to `C:\work\localMachine\ninfer-consolidated`):

- Binary: `ninfer-consolidated\build\apps\ninfer-serve.exe`
- Arch: `CMAKE_CUDA_ARCHITECTURES=120a`, `NINFER_SM120_NATIVE=OFF` — the compat
  path is required; it is the only path that promotes `t2_g128_fp16` weights to
  the integer-activation prefill GEMM. `NINFER_SM8X_COMPAT=1`.
- Toolchain: Ninja + CUDA 13.1 + MSVC **14.38** explicitly (`-vcvars_ver`;
  VS18 defaults to 14.5x, which nvcc rejects).
- Deps: vendored `ffmpeg/` copied from the 5090 tree; curl-for-win package;
  FFmpeg + curl DLLs must sit beside the exe (else exit 127 with no output).
- Build script: `ninfer-consolidated\build_5090.bat`.

MSVC portability fixes applied to the fork (upstream may still need them):

- `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu` — C3495:
  hoisted `constexpr` out of a nested lambda.
- `src/ops/softmax_attention/dense/causal_cache/small_t.cu` — same C3495 pattern.
- `src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu` — `__builtin_ctz`
  → `std::countr_zero`.

## Controller mechanics

- **Per-profile `serve` override**: each `MODELS` entry can name its engine
  binary; Orca stays on stock, Bonsai on consolidated.
- **`reportedId` + body rewrite**: the consolidated server validates the
  request `model` field against its own `--model-id`. The controller tracks the
  child's reported id and rewrites the proxied body's `model` field before
  forwarding, so client-facing aliases can differ from child ids. Orca's server
  ignored the field; consolidated does not.
- Children spawn on demand on :11435+, previous engine is torn down on swap
  (~3 s). Logs: `controller/ninfer-child.log`.

## Measured performance (RTX 5090)

Stock Bonsai (`bonsai2-27b`):

| Metric | Value |
|---|---|
| Boot | 2.6–3.4 s (weights 7.99 GiB @ ~8.8 GiB/s) |
| Prefill | ~1,100–1,560 tok/s |
| Decode (DFlash2) | 544–590 tok/s |
| Draft acceptance | 57–63 % |
| Vision (overlay) | correct on test image, ~330 tok/s |
| Engine runtime | 19.1 GiB at 1M KV |

Heretic (`bonsai2-27b-heretic-vision`): boots 3.9 s, prefill ~5,850–6,950
tok/s at depth (~1,000 short), sustained decode **325–327 t/s** on 3K-token
runs, draft acceptance ~25–30 %. Matched-content A/B vs stock (296–343 t/s,
25–32 %) shows **no acceptance penalty from the ablation** — the earlier
"20 % vs 60 %" reading conflated content-dependent acceptance with a model
gap. Uncensored behavior verified; vision verified (~313 t/s).

## VRAM accounting

- Bonsai profile self-reports ~19.1 GiB runtime (8 GiB weights + ~10 GiB KV
  pool at 1,048,576 `rk4v4-e8` + ~1 GiB graphs/GDN state).
- With a clean desktop, `nvidia-smi` free ≈ 2.3–3.4 GiB — same margin the
  Orca profiles run at.
- The dominant *external* consumer is the browser/WebView/desktop pile —
  measured >1 GiB, historically up to ~10 GiB with ~50 processes on the GPU.
  If boot OOMs appear, kill browsers first; then reduce `kv`/`ctx` in
  `ninfer-ctl.js` or step down to `rk2v4-e8`.
- Vision adds ~0.6 GiB of media workspace over text-only.

## Context sizing rationale

1,048,576 tokens at `rk4v4-e8` was chosen because: `rk8v4` would cap near
~660K within the same VRAM; `rk2v4-e8` reaches 1M more cheaply but costs real
quality (ppl 5.82 vs 5.65). `rk4v4-e8` is the documented sweet spot
(−2 % decode, +0.019 ppl vs `rk8v4`). Needle-retrieval is dependable to
~500K, degrades toward 1M (upstream tests: 3/3 → 2/3). Cold prefill of a very
large prompt is slow; decode slows as KV fills — mitigated by the prefix
cache + host KV tier, not eliminated.

## Measured quality delta — local perplexity benchmark (2026-09-26)

`ninfer-perplexity` on both engines, fixed corpus `ninfer-ppl-1m-v1 --quick`
(261,223 scored tokens, 124 windows, context 4096 / stride 2048). Reports under
`bonsai-src\ppl\*\report.json`.

| Artifact | Weights | KV | PPL | Note |
|---|---|---|---|---|
| Orca | NVFP4 | bf16 | 4.827 | baseline |
| Orca | NVFP4 | nvfp4 (production) | 4.782 | KV cost ~0 |
| Bonsai | ternary T2 | bf16 | 5.855 | **+21.3% vs Orca** |
| Bonsai | ternary T2 | rk4v4-e8 (production) | 5.879 | KV cost +0.4% |
| Bonsai Heretic | ternary T2 + ablation | bf16 | 5.876 | +0.35% vs stock |

Reading: the quality gap is entirely in the ternary weights — both KV codecs
are essentially free, which validates 1M ctx at `rk4v4-e8`. Per-domain the gap
concentrates in long-form English (9.31 vs 7.18) and is narrowest on code
(+13.5%). Perplexity measures distribution drift, not task accuracy — treat it
as the honest price of the compression, not a capability score.

## Heretic conversion pipeline (reproducible)

NOTE: the ~83 GB of conversion intermediates under `bonsai-src\` were cleaned
after publication (base checkpoint, MTP/DFlash2 sources, staging copies). To
reconvert, re-download: the full `Qwen/Qwen3.8-27B` checkpoint,
`ProCreations/Ternary-Bonsai-2-27B-MTP` (`model_mtp.safetensors`),
`ProCreations/Ternary-Bonsai-2-27B-DFlash2` (weights **and** `config.json`),
and the source GGUF (`OS-Software/…-Uncensored-Heretic-PQ2_0.gguf`).

Sources that were used:

- `Qwen3.8-27B\` — full official checkpoint (18 shards + index + config +
  tokenizer + `preprocessor_config.json` + `video_preprocessor_config.json`;
  provides vision tower + frontend resources for `--model`)
- `Ternary-Bonsai-2-27B-Uncensored-Heretic-PQ2_0.gguf` — ternary text weights
  (`--source ternary=`); also in `C:\Users\tsoiy\Downloads\`
- `model_mtp.safetensors` — ProCreations MTP head (`--source mtp=`)
- `Ternary-Bonsai-2-27B-DFlash2\` — ProCreations adapter (`model.safetensors`
  + `config.json`; both required — config supplies `intermediate_size: 17408`)

Converter: `C:\Python311\python.exe -m tools.convert` from the consolidated
repo; recipe `bonsai2_27b_ternary`; `--components text,vision,mtp,dflash2`;
`--proposal`; `--device cuda`. Run took 222.5 s → 1,193 objects, output size
byte-identical to the official v3 artifact.

Gotchas hit: the DFlash2 repo must include `config.json`; the Qwen dir must
include `video_preprocessor_config.json` (converter's `VISION_RESOURCES`
requires it); the GPU should be idle (`--device cuda` needs headroom — the
running engine was stopped during conversion).

Published: `https://huggingface.co/emiltsoi/Ternary-Bonsai-2-27B-Uncensored-Heretic-NInfer`
(commit `724ae2e`; card documents provenance, the ~20 % acceptance caveat, and
the no-safety-layer disclosure). First uncensored Bonsai in NInfer format.

## DSH desktop registration

`C:\Users\tsoiy\.dsh\profiles\desktop\cordis.patch.yml` →
`llm-pi-ai → providers.ninfer → models` has entries for `bonsai2-27b` and
`bonsai2-27b-vision` (ctx 1,048,576). Heretic aliases not yet added — same
shape if wanted. Patch-layer provider rows take effect on app restart.

## Deployment — which `ninfer-ctl.js` is live

Two copies exist; the **deployed** one is what `start/stop.bat` actually run:

- Canonical source: `ninfer-orca-stack\controller\ninfer-ctl.js` (edit this one)
- Deployed copy: `C:\Users\tsoiy\AppData\Local\Ollama\ninfer-ctl.js` — must be
  re-copied after every edit (backup of the pre-Bonsai version kept as
  `ninfer-ctl.js.bak` there)
- Env wrapper: `C:\Users\tsoiy\AppData\Local\Ollama\ninfer-ctl-run.bat` —
  sets `NINFER_HOME` and `NINFER_CONS_SERVE` then runs node. Required because
  the WMI spawn in `start.bat` does NOT inherit the bat's environment, and
  both paths default to `__dirname`-relative resolution that only works under
  the repo tree. Logs land in the Ollama dir (`LOG_DIR` defaults to
  `__dirname`), same as before.
- `ninfer-ctl-stop.bat` needed no changes (kills by `ninfer-ctl.js` cmdline
  match + :11434 owner fallback).
- Gotcha: `.bat` files need CRLF endings — an LF-only bat silently
  misbehaves under cmd.exe.

Verified 2026-09-26: `start.bat` → all 8 aliases served (Orca on stock
binary, Bonsai/Heretic on consolidated), `stop.bat` → full teardown, 29 GiB
VRAM freed.

## Operational notes / open items

- Bonsai template quirk: `reasoning_effort: high` → HTTP 500 upstream; the
  controller already maps `high→xhigh`, `minimal→low`, `max→xhigh`.
- Unused levers if needed: `--no-thinking` (a `bonsai2-27b-fast` alias),
  `--wddm-evictable-budget`, `--host-kv-mib`/`--disk-kv-path` (KV host/disk
  tier — the mechanism for >GPU-memory contexts and the likely 16 GB laptop
  route; untested on this rig).
- 5080 mobile 16 GB plan: same consolidated binary (Blackwell family),
  smaller `kv`/`ctx`; fallback is the llama.cpp KVMem fork if the engine
  won't run there.
- `test-img.png` in `controller/` is the red-square vision smoke image.
- Non-ASCII paths break the engine's file IO (documented upstream) — keep all
  artifact/working paths ASCII.
