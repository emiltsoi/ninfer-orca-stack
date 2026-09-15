# ninfer-orca-stack

Ops layer for running the Orca Qwen3.8-27B NVFP4 model on a single RTX 5090 (Windows) with
[NInfer](https://github.com/headpiece747/ninfer-5090-windows) — including the YaRN-enabled
Windows fork at [emiltsoi/ninfer-5090-windows](https://github.com/emiltsoi/ninfer-5090-windows).

One LAN endpoint, four VRAM-matched profiles (~2.7–3.2 GiB free each), auto-swapped by model name:

| Alias | Spec decode | Vision | Context |
|---|---|---|---|
| `qwen-3.8-orca` | MTP | – | 315k (YaRN 1.25) |
| `qwen-3.8-orca-fast` | DFlash2 | – | 230k native |
| `qwen-3.8-orca-vision` | MTP | ✓ | 288k (YaRN 1.25) |
| `qwen-3.8-orca-vision-fast` | DFlash2 | ✓ | 200k native |

## Layout

- `controller/` — `ninfer-ctl.js` Node router (auto-swap, `/yield`, `/health`, context-aware
  `/v1/models`, `reasoning_effort` normalization) + start/stop batch scripts
- `scripts/` — standalone single-profile launchers (alternative to the controller)
- `bench/` — the A/B spec-decode suite (`ab-spec-test.py`) and measured results
- `build/` — Windows build helpers for the YaRN fork (`build-yarn.bat`, `probe-env.bat`)
- `docs/MIGRATION.md` — the full writeup: port notes, VRAM ledger, benchmarks, gotchas

## Quick start

```bat
set NINFER_HOME=C:\path\to\ninfer-5090-windows
controller\ninfer-ctl-start.bat
curl http://<your-host>:11434/health
```

Clients: OpenAI-compatible at `http://<your-host>:11434/v1` (Anthropic `/v1/messages` also proxied).
Model artifacts are not committed — download links and validation notes are in `docs/MIGRATION.md`.

Requirements: Node.js (any modern LTS), PowerShell, and a built NInfer engine at `NINFER_HOME`.
