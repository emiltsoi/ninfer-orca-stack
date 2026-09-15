# Retiring llama.cpp for a single-purpose NInfer rig: Orca 27B, vision, YaRN, and DFlash2 on one RTX 5090

*Lab notes, 2026-09-15. Machine: Windows 11, RTX 5090 32 GB (sm_120a), CUDA 13.1, MSVC 14.38.*

## The old setup, and why it died

The previous stack was a hand-rolled Ollama replacement: `llama-ctl.js`, a Node controller on `:11434` that faked the "one endpoint, name any model" contract on top of llama-server's one-model-per-process reality. Nine aliases pointed at GGUFs; a request for an unloaded model killed the child and spawned the right one on loopback `:11435`. `/yield` freed VRAM for ComfyUI windows.

Two things killed it. Operationally, the registry had rotted — a Downloads cleanup deleted most of the GGUFs and half the aliases pointed at ghosts. Strategically, llama.cpp decode on this card topped out around ~47–68 tok/s, and every new capability (vision via mmproj, YaRN via rope flags, MTP via `--spec-type`) meant another alias permutation to maintain.

## NInfer

[NInfer](https://github.com/headpiece747/ninfer-5090-windows) is a from-scratch C++/CUDA engine built for exactly one GPU: a 32 GB RTX 5090. The Windows fork ships prebuilt binaries and registered `.ninfer` artifacts — a single file containing weights (NVFP4), the MTP draft head, the vision tower, and metadata. No mmproj pairing, no chat-template files, no GGUF flag soup.

Published numbers claimed ~210 tok/s decode with MTP5 and up to ~356 with DFlash2 — 3–5× llama.cpp. That was worth chasing.

## Porting YaRN to Windows

One gap: the Windows fork capped `--max-context` at the model's native 262,144. The sister fork [`splickz/ninfer-yarn-nvfp4`](https://github.com/splickz/ninfer-yarn-nvfp4) had YaRN RoPE scaling — but Linux/WSL2 only.

The port turned out to be small: 3 commits, ~280 lines, 16 files. `git apply` took 15 of them cleanly; two hunks in `serve_options.cpp` collided with newer DFlash2 CLI code and got hand-applied. The only build-system wrinkle: CUDA 13.1's nvcc rejects the MSVC 14.51 toolset in VS18, so the build pins `-vctools-version=14.38`.

Verification, not vibes:

- `--max-context 393216` is *refused* without the flag (`raise --rope-yarn-factor`), accepted with `--rope-yarn-factor 1.5`.
- Needle test at ~309k tokens (past the 262,144 native wall): both needles retrieved, `sanity=True deep=True`.

Published at [`github.com/emiltsoi/ninfer-5090-windows`](https://github.com/emiltsoi/ninfer-5090-windows), commit `5bc7984`.

## Validating the Hugging Face artifacts

Three repos claim the same Orca checkpoint. Rather than trust filenames, we parsed the `.ninfer` v2 container directly — 8-byte magic, u64 length, then a JSON directory with `identity` and the object table — readable via HTTP range requests:

| Repo | Identity | Objects | Notes |
|---|---|---|---|
| `gearwave00001/...nvfp4-NInfer` | `qwen3.8-27b/nvfp4` | 1,124 | text + MTP + vision; SHA-256 verified |
| `Yuuyuuyuuyuu/...DFlash2` | `qwen3.8-27b/nvfp4` | 1,190 | **strict superset**: same 1,124 + 66 `dflash2/*` |
| `DreamFast/...-Ninfer` | `qwen3.8-27b/groupwise-int` | 1,124 | older int-groupwise recipe |

Same weights, different packaging. The Yuuyuuyuuyuu file is a free upgrade — DFlash2 unlocks with a flag, no conversion needed. Both downloads verified byte-exact against HF's LFS records (`a72935d6…`, `b57bb82f…`).

## What DFlash2 actually is

Not a model — a **block-parallel speculative drafter**: 66 extra objects forming a 5-layer mini-network (attention + causal conv + MLP + candidate selector codebooks) that proposes a whole block of candidate tokens in one masked pass. The main model still verifies and authorizes every emitted token, so output quality is preserved structurally. Flag: `--spec dflash2 --draft-tokens 7`.

## A/B: MTP vs DFlash2

Same 6 prompts, same seed, single request — the shape of the actual workload (agentic coding):

| Prompt | MTP | DFlash2 | Δ |
|---|---:|---:|---:|
| code | 159 t/s | 216 t/s | +36% |
| JSON | 247 | 324 | +31% |
| reasoning | 244 | 341 | +40% |
| story | 108 | 134 | +25% |
| summarize | 216 | 263 | +22% |
| tool-call | 179 | 259 | +45% |

Mean ~192 → ~256 tok/s (+33%). Even story — where published acceptance is only ~17% — improved, because block-parallel drafting amortizes the loss over 7-token rounds. One caveat from upstream's own benchmarks: a pathological repetition loop appeared on one long-reasoning fixture; verification guarantees correctness *per token*, not sensible trajectory.

## The VRAM ledger (all measured, vision on)

The binding constraint isn't the flag ceiling — it's 32 GB minus desktop WDDM (~3 GiB, fluctuates ±0.5–1). KV cache costs ~18.9 KiB/token in NVFP4. The engine enforces `runtime_reserve + 1 GiB ≤ free_after_weights` at startup, and *that* check is what actually gates DFlash2.

| Config | Weights | Context | Free after boot |
|---|---|---|---|
| MTP + vision | 20.0 GiB | 262,144 native | 3.19 GiB |
| MTP + vision | 20.0 GiB | 393,216 (YaRN 1.5) | 0.77 GiB — too tight |
| DFlash2 + vision | 21.7 GiB | 262,144 | **fails startup check** |
| DFlash2 + vision | 21.7 GiB | 240,000 | ~2.0 GiB — borderline |
| DFlash2 + vision | 21.7 GiB | 200,000 | **2.75 GiB** ✓ |
| MTP, no vision | 19.7 GiB | 262,144 | 3.76 GiB |

Two surprises from measuring rather than assuming:

- **Vision costs only ~0.6 GiB** physically (0.3 GiB tower + ~0.27 fixed workspace). The "media cache 1 GiB / live 2 GiB" log lines are internal caps — trimming `--media-cache-mib`/`--media-live-mib` changed the reservation by exactly 0 bytes.
- **YaRN flag vs context size**: YaRN rescales RoPE across the whole range, so for ≤262k native is strictly better. We run `--rope-yarn-factor 1.25` only to unlock headroom above native.

## The result: ninfer-ctl

Same controller pattern as llama-ctl, rebuilt for NInfer: [`ninfer-ctl.js`](controller/ninfer-ctl.js) on `:11434`, child on loopback `:11435`, serialized swaps, `/yield` to free everything, `/health`, digest request log. Boot is ~6 s, so swapping profiles is cheap.

Four VRAM-matched profiles (~2.7–3.2 GiB free each):

| Alias | Spec | Vision | Context |
|---|---|---|---|
| `qwen-3.8-orca` | MTP | – | 315k (YaRN 1.25) |
| `qwen-3.8-orca-fast` | DFlash2 | – | 230k native |
| `qwen-3.8-orca-vision` | MTP | ✓ | 288k (YaRN 1.25) |
| `qwen-3.8-orca-vision-fast` | DFlash2 | ✓ | 200k native |

Legacy `qwen38-orca*` names remap to the closest profile, and NInfer echoes whatever model string the client sent — zero client reconfiguration. All four verified live: completions on text profiles, "Red" on a red-square image for both vision profiles.

## First blood: the `reasoning_effort` 400

The first real client to hit the router — a remote coding agent — died with *"model provider failed after retries"*. The request log (`ninfer-req.log`) showed its traffic arriving fine (365-message, ~580k-char sessions); the child log held the actual verdict:

```
req#1 rejected during prepare | openai-chat stream | HTTP 400 | reasoning effort not supported | messages 21 | tools 40
```

The agent sends `reasoning_effort: "max"`. NInfer's embedded chat template only supports `none`/`low`/`medium`/`xhigh` — `minimal`/`high`/`max` are rejected at request-prepare time (`translate.cpp`). The old stack never saw this because the hand-patched `newmodel-template.jinja` silently remapped `high`/`max` → `xhigh`; NInfer's template is embedded in the artifact and can't be patched.

Fix landed in the router — same translation-layer role the patched template played: `normalizeReasoningEffort()` rewrites `minimal→low`, `high→xhigh`, `max→xhigh` on both `/v1/chat/completions` and `/v1/messages` before forwarding upstream. Verified live with the exact failing shape; no client change needed.

Moral for future migrations: the patched jinja wasn't just a template tweak — it was a **contract adapter**. Any capability the old template quietly absorbed (effort aliases, tool-call guidance) needs an explicit home in the new stack, and the router is the right place for it.

## Advertising context correctly

Agent harnesses discover a model's context window by reading `/v1/models` — an undocumented-but-load-bearing extension field. NInfer's own listing emits `max_model_len` (the vLLM/llama.cpp convention), with a hardcoded clamp of 131,072 on any id containing "vision" — wrong for our profiles on both counts.

The router now serves its own listing: each profile entry carries a single `ctx` field that drives **both** the `--max-context` launch flag and the advertised limits (can't drift apart), and the response emits it under both `max_model_len` (matches what this stack's clients already parse) and `context_length` (OpenRouter convention). Verified output:

```json
{ "id": "qwen-3.8-orca-vision-fast", "max_model_len": 200000, "context_length": 200000 }
```

Same lesson as the reasoning-effort fix: the controller isn't a dumb pipe — it's where the product's external contract gets enforced.

## Ops cheat sheet

```
start:  <ops-dir>\ninfer-ctl-start.bat   (detached)
stop:   <ops-dir>\ninfer-ctl-stop.bat    (frees all VRAM)
yield:  curl -X POST http://<host>:11434/yield                     (free VRAM, keep controller)
logs:   ninfer-ctl.log / ninfer-child.log / ninfer-req.log  (same folder)
health: GET /health → {"controller":"ok","loaded":"qwen-3.8-orca-vision-fast"}
```

Standalone single-profile launchers remain in the repo (`ninfer-start.bat`, `ninfer-start-dflash2.bat`) for pinned deployments; they conflict with the controller on `:11434` — run one or the other.

## Gotchas learned the hard way

- `ninfer-serve` startup check is stricter than steady-state: it wants reserve + 1 GiB *before* it starts. A config that ran yesterday can fail today if the desktop's using more VRAM. If boot fails, check `ninfer-serve.log` for the `available after weights` number.
- Detached `cmd /c` spawns can lag — a "failed" launch produced a zombie that bound the port a minute later. Always check the port owner, not just the PID you meant to kill.
- `python` on this box is Python 2.7. Use `py -3` (3.14) for scripts.
- NInfer rejects what llama.cpp tolerated: no `/v1/completions`, `/v1/embeddings`, JSON mode, `n>1`, logprobs, or forced `tool_choice` — and `reasoning_effort` is gated to `none`/`low`/`medium`/`xhigh` (the router rewrites `minimal`/`high`/`max`). It *does* accept `seed` and returns rich `timings` (decode t/s, draft acceptance) per response.
- `/v1/models` advertises a 131k context cap for IDs containing "vision" — cosmetic clamp in `http_server.cpp`, not a real limit.

## What's deliberately *not* done

- No authentication on `:11434` — LAN is trusted; revisit if that changes.
- No DFlash2 + YaRN combination (unvalidated upstream and in our port; also VRAM-blocked anyway).
- Old llama-ctl files and ~35 GB of GGUFs in `.ollama/imports` still on disk for rollback.

## If you're picking this up cold

1. `ninfer-ctl-start.bat`, then `curl http://<host>:11434/health`.
2. Clients point at `http://<host>:11434/v1` and name one of the four aliases.
3. To retune context: edit the `MODELS` table in `ninfer-ctl.js`, restart controller. KV slope is ~18.9 KiB/token in NVFP4; keep ≥2 GiB free for desktop safety.
4. To re-benchmark: `py -3 ab-spec-test.py <label>` in the repo dir writes `ab-<label>.json`.
