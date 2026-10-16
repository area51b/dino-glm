---
title: Dino Run with Laya
emoji: 🦖
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Dino Run with Laya 🦖

A Chrome-dino-style endless runner where a **decision model** — the [`laya`](https://pypi.org/project/laya/) package — controls the dinosaur. All game arithmetic (threat detection, reaction windows, action constraints) happens in server code; the model only classifies the situation.

## How it works

- **Backend** ([main.py](main.py)): a FastAPI app that serves the game and exposes `POST /dino/act`. It builds a natural-language description of the game state, asks the `laya` model for a choice among `run` / `jump` / `duck`, and constrains the answer to physically valid actions.
- **Frontend** ([dino.html](dino.html)): a single self-contained HTML file (vanilla JS, no dependencies, no build step) with a canvas game. In **AI mode** it sends snapshots of the two nearest obstacles to the backend and applies the returned actions in the game loop.

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` and `/dino` | GET | Serve the game |
| `/dino/act` | POST | AI decision endpoint (always enabled). Body: game snapshot. Returns `{action, debug}`. |
| `/predict` | POST | Raw model access, **disabled by default**. Set the `PREDICT_TOKEN` env var to enable; callers must then send the token in an `x-api-key` header (404 when disabled, 401 on a wrong key). |

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000>.

## Deploy to Hugging Face Spaces

Create a Space with the **Docker** SDK and push this repository. The app listens on port 7860 (configured via the `app_port` front-matter above). The Dockerfile pre-downloads the model weights at build time; if the prefetch fails, weights download at first startup instead.

## Notes

- **Cold-start latency:** the model loads once at startup, and the first request after a cold start can be slow. The backend compensates for round-trip latency by extrapolating obstacle positions with the measured latency, so decisions stay correct at ~200–500 ms.
- **Local timing assist:** the checkbox in the UI skips the server round trip entirely and acts locally using the same lead times as the backend (0.30 s jump / 0.35 s duck) — useful for playing offline or comparing the model against pure timing.
- **Safety rules:** the AI never jumps mid-air and never stays ducked without a head-height bird threat; invalid model answers fall back to the highest-probability allowed action, then to `run`.
