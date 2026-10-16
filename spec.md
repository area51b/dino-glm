# Requirements: "Dino Run with Laya" — AI-Powered Runner Game

## 1. Overview

Build a Chrome-dino-style endless runner game where a **decision model** (the [`laya` package from PyPI](https://pypi.org/project/laya/)) controls the dinosaur. The project has two parts:

1. **Backend** — a FastAPI app that serves the game and exposes an AI decision endpoint backed by the `laya` model.
2. **Frontend** — a single self-contained HTML file with a canvas-based game that, in **AI mode**, sends game snapshots to the backend and applies the returned actions.

**Deployment target:** Hugging Face Spaces (Docker SDK, app port **7860**), but it must also run locally with `uvicorn main:app --host 127.0.0.1 --port 8000`.

**Tech stack:** Python 3.11, FastAPI, Pydantic v2, uvicorn, `laya` (with CPU-only PyTorch), vanilla JavaScript (no frameworks, no build step).

---

## 2. Backend (`main.py`)

### 2.1 Setup

- Create `FastAPI()` app and load the model **once at startup**: `router = laya.Router(preload=True, device="cpu")`.
- Set `os.environ["USE_TF"] = "0"` before importing/using the model to avoid TensorFlow overhead.
- Serve the game HTML at `GET /` and `GET /dino` via `FileResponse`.

### 2.2 Endpoints

**`POST /dino/act`** — the game's AI endpoint (always enabled). Accepts a snapshot (schema below), runs the decision pipeline (§2.4), returns:

```json
{
  "action": "run" | "jump" | "duck",
  "debug": {
    "state": "<natural-language state description>",
    "threat": {...} | null,
    "allowed_actions": ["run", ...],
    "model_choice": "...",
    "applied_action": "...",
    "reason": "<why this action was chosen>",
    "probabilities": {...},
    "confidence": ...
  }
}
```

**`POST /predict`** — raw model access, **disabled by default**:
- If env var `PREDICT_TOKEN` is empty/unset → return **404**.
- If set → require the token in an `x-api-key` header; compare with `secrets.compare_digest`; mismatch → **401**.
- Body: `{"state": ..., "questions": ...}` → forwards directly to `router.predict(state, questions)`.

### 2.3 Snapshot schema (Pydantic v2, strict validation)

```python
class DinoObstacle(BaseModel):
    kind: Literal["cactus", "bird"]
    variant: str                      # for birds: "duck" | "jump" | high variant
    x: float    # relative distance from dino's front edge; allow_inf_nan=False
    y: float    # ge=0
    w: float    # gt=0
    h: float    # gt=0

class DinoSnapshot(BaseModel):
    grounded: bool
    ducking: bool
    speed: float                      # gt=0
    score: float                      # ge=0
    decision_latency_ms: float        # default 0, ge=0, le=5000
    obstacles: list[DinoObstacle]     # max_length=2
```

All float fields must reject NaN/Infinity (`allow_inf_nan=False`).

### 2.4 Decision pipeline (all arithmetic in server code, NOT the model)

**Constants** (must stay in sync with frontend):
```python
JUMP_LEAD_SECONDS = 0.30
DUCK_LEAD_SECONDS = 0.35
JUMP_LATE_PIXELS = 12       # jump still useful if obstacle is at most this far past the dino
DINO_STAND_WIDTH = 44       # obstacle further behind than this has passed
ACTIONS = ("run", "jump", "duck")
```

**Step 1 — `required_action(obstacle)`:** cactus → `"jump"`; bird with variant `"duck"` → `"duck"`; bird with variant `"jump"` → `"jump"`; high bird → `"ignore"`.

**Step 2 — `find_threat(snapshot)`:** sort obstacles by `x`, take the first non-ignorable one. Compensate for network latency: compute `distance_on_arrival = obstacle.x - speed * (decision_latency_ms / 1000)`. Skip obstacles that will already be behind the dino (`distance_on_arrival + w < -DINO_STAND_WIDTH`). Classify the reaction window:
- For jumps: `distance_on_arrival < -JUMP_LATE_PIXELS` → `"missed"`; `time_on_arrival <= JUMP_LEAD_SECONDS` → `"now"`; else `"early"`.
- For ducks: reached (`distance < 0`) or `time_on_arrival <= DUCK_LEAD_SECONDS` → `"now"`; else `"early"`.

Return `{kind, variant, need, window, distance_on_arrival (rounded 1dp), time_on_arrival (rounded 3dp)}`.

**Step 3 — `allowed_actions(snapshot, threat)`:** always include `"run"`; add the threat's required action only if the dino is `grounded` **and** window is `"now"`.

**Step 4 — `describe_state(snapshot, threat)`:** build a short English paragraph for the model:
- Grounded → "The dino is on the ground." else "The dino is in the air and cannot jump or duck."
- No threat → "No obstacle needs a reaction right now."
- Threat → describe it ("a cactus" / "a head-height bird" / "a low bird"), how it must be passed, distance/time on arrival, and the reaction window in caps: "NOW, act on this obstacle immediately." / "NOT YET, ... keep running." / "MISSED, it is too late to jump, so keep running."
- **Deliberately omit the dino's current posture** (ducking state) — a small classifier tends to echo it ("ducking" → "duck") instead of reasoning about the obstacle.

**Step 5 — model call:** `router.predict({"body": state}, {"action": {"type": "choice", "instructions": ..., "criteria": {...run/jump/duck descriptions...}}})`. The instructions must encode: window NOW + must jump → jump; window NOW + must duck → duck; every other case (airborne, no threat, NOT YET, MISSED) → run.

**Step 6 — `pick_action(allowed, model_choice, probabilities)`:**
1. If the model's choice is a string in `allowed` → use it.
2. Else, if `probabilities` is a dict, pick the highest-probability action that is in `allowed`.
3. Else default to `"run"`.
Return the action plus a human-readable `reason` explaining which fallback path was taken.

---

## 3. Frontend (`dino.html`)

A single self-contained HTML file (inline CSS + JS, no dependencies).

### 3.1 Game core

- **Canvas:** 600×150 logical pixels, scaled for `devicePixelRatio`; ground line at `GROUND_Y = 128`.
- **Physics:** `JUMP_VELOCITY = -600` px/s, `GRAVITY = 2160` px/s², delta-time game loop.
- **Dino:** at `DINO_X = 44`; standing 44×47, ducking 59×25; hitbox shrunk by `HITBOX_INSET = 6` px vs. the visible sprite. Draw pixel-art-style sprites (dino run/duck/jump frames, cacti, birds) on canvas.
- **Speed ramp:** `BASE_SPEED = 240` px/s → `MAX_SPEED = 780` px/s over `RAMP_SECONDS = 120` via `SPEED_STEPS = 48` small bumps (~11.25 px/s each) instead of spikes. Score = pixels × `SCORE_PER_PIXEL = 0.05`.
- **Obstacle spawning:** gap-based; minimum extra gap shrinks 180→90 px and random gap 340→180 px as speed rises; initial spawn delay 520 px. Obstacle types: cacti (jump) and birds at three heights — low (jump), head-height (duck), high (ignore).
- **Game states:** start screen → playing → game over (collision). Persist high score in `localStorage` under key `"dino-run-high-score"`.
- **Controls:** Space/↑ = jump, ↓ = duck (hold); plus a mode `<select>` (Human / AI) and a "Local timing assist" checkbox in the UI.

### 3.2 AI mode

- Poll the backend with `POST /dino/act`, sending at most the **2 nearest upcoming obstacles** with `x` measured relative to the dino's front edge (`obstacle.x - (dino.x + dino.w)`), plus `grounded`, `ducking`, `speed`, `score`, and `decision_latency_ms` = rolling average of the last **5** measured round-trip latencies.
- Throttle: skip polling when the nearest obstacle is beyond `AI_LOOKAHEAD_DISTANCE = 650` px unless ≥ `AI_EMPTY_POLL_MS = 300` ms since the last request. Never overlap requests (`aiBusy` flag) and tag each request with an incrementing ID; discard stale responses.
- **Apply decisions in the game loop, not in the fetch callback:** on response, set a pending action; the loop applies it with these safety rules:
  - `jump` only executes if the dino is grounded now; otherwise downgrade to run with an explanatory reason (airborne at request time vs. airborne when response arrived).
  - `duck` only holds while a head-height bird is the nearest threat — a stale/wrong "duck" must never keep the dino crouched.
  - Track and display the action source (`"Laya"`, `"local timing enabled"`, `"idle"`).
- **Local timing assist:** when checked, skip the server round trip and act locally using `AI_JUMP_LEAD_SECONDS = 0.30` / `AI_DUCK_LEAD_SECONDS = 0.35` leads (these MUST match the backend constants).
- On request failure: fall back to `"run"`, show the error in the debug panel.

### 3.3 AI debug panel

A collapsible `<details>` panel showing: latest request snapshot, response, latency, decision status, applied action + application reason, and a history of the last 5 decisions.

---

## 4. Packaging & Deployment

**`requirements.txt`:**
```
--extra-index-url https://download.pytorch.org/whl/cpu
torch
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
laya
pydantic>=2
```

**`Dockerfile`:** `python:3.11-slim`; run as non-root user (uid 1000, HF Spaces requirement); install deps; **pre-download model weights at build time** with `python -c "import laya; laya.Router(preload=True, device='cpu')"` wrapped in `|| echo` so a failed prefetch doesn't break the build (weights then download at first startup); `EXPOSE 7860`; `CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]`.

**README.md:** HF Spaces front-matter (`sdk: docker`, `app_port: 7860`), description, endpoint docs, local run instructions, and notes about cold-start latency and the local timing assist.

**`.gitignore`:** `.venv/`, `__pycache__/`, etc.

---

## 5. Acceptance Criteria

1. `GET /` serves a playable game; human mode works with keyboard only.
2. AI mode: the dino successfully jumps cacti, ducks under head-height birds, ignores high birds, and survives to high scores.
3. Latency compensation works: decisions remain correct at ~200–500 ms round-trip latency (obstacle positions extrapolated by measured latency).
4. The AI never performs an invalid action (no mid-air jumps, no ducking without a head-height bird threat).
5. `/predict` returns 404 without `PREDICT_TOKEN` set, and 401 with a wrong `x-api-key` when set.
6. Malformed snapshots (NaN, negative widths, >2 obstacles, unknown kinds) are rejected with 422.
7. Container builds and runs on port 7860 as a non-root user.

