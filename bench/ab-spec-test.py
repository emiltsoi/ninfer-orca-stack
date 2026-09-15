#!/usr/bin/env python3
# A/B spec-decoding test: same prompts+seeds against current server config.
# Usage: python ab-spec-test.py <label>   -> writes ab-<label>.json
import json, sys, time, urllib.request

URL = "http://127.0.0.1:11434/v1/chat/completions"
LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"

PROMPTS = [
    ("code", "Write a Python async function that fetches a list of URLs concurrently with a semaphore, retries with exponential backoff, and returns (url, status, body) tuples. Include type hints and a brief docstring.", 700),
    ("json", "Output ONLY a JSON array of 12 objects. Each object: {\"id\": int, \"name\": string, \"email\": string, \"role\": one of admin/editor/viewer, \"active\": bool, \"score\": float 0-100}. Use realistic names. No prose, no markdown fences.", 900),
    ("reason", "A train travels 300 km in 2 hours, then 180 km in 1.5 hours, then stops 30 minutes, then 240 km in 2 hours. What is its average speed for the whole journey including the stop? Think step by step.", 1200),
    ("story", "Write the opening two paragraphs of a noir detective story set in a flooded coastal city. Moody, first-person.", 500),
    ("summ", "Summarize in exactly 3 bullet points: The mitochondria is the powerhouse of the cell, generating most of the chemical energy needed to power biochemical reactions through oxidative phosphorylation. Energy is stored as ATP. Mitochondria have their own DNA, replicate independently, and are thought to descend from ancient bacteria engulfed by early eukaryotic cells.", 300),
    ("toolcall", "Extract a function-call spec from this request as JSON: 'Search the database for users named Alice or Alicia who logged in after 2024-01-01, limit 50, sorted by last_login desc.' Emit {\"function\": ..., \"arguments\": {...}} only.", 400),
]

def run(tag, prompt, max_tokens):
    body = json.dumps({
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "seed": 42,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    wall = time.time() - t0
    t = d.get("timings", {})
    u = d.get("usage", {})
    return {
        "tag": tag,
        "completion_tokens": u.get("completion_tokens"),
        "decode_tps": round(t.get("predicted_per_second", 0), 1),
        "prompt_tps": round(t.get("prompt_per_second", 0), 1),
        "draft_n": t.get("draft_n"),
        "draft_accepted": t.get("draft_n_accepted"),
        "wall_s": round(wall, 1),
    }

results = []
for tag, prompt, mt in PROMPTS:
    r = run(tag, prompt, mt)
    results.append(r)
    acc = "{:.0f}%".format(100.0*r["draft_accepted"]/r["draft_n"]) if r["draft_n"] else "n/a"
    print(f"{tag:9} | {r['completion_tokens']:5} tok | decode {r['decode_tps']:6} t/s | draft acc {acc} | {r['wall_s']}s", flush=True)

with open(f"ab-{LABEL}.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved ab-" + LABEL + ".json")
