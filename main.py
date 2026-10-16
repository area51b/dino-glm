"""Dino Run with Laya - FastAPI backend.

Serves the game (dino.html) and exposes the AI decision endpoint backed by the
`laya` decision model. See spec.md for the full requirements.
"""

import os
import secrets
import threading
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Avoid TensorFlow import overhead; must be set before the model is imported/used.
os.environ["USE_TF"] = "0"

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

import laya

# ---------------------------------------------------------------------------
# Tuning constants (must stay in sync with dino.html)
# ---------------------------------------------------------------------------
JUMP_LEAD_SECONDS = 0.30
DUCK_LEAD_SECONDS = 0.35
JUMP_LATE_PIXELS = 12       # jump still useful if obstacle is at most this far past the dino
DINO_STAND_WIDTH = 44       # obstacle further behind than this has passed
ACTIONS = ("run", "jump", "duck")


# ---------------------------------------------------------------------------
# Snapshot schema (strict validation)
# ---------------------------------------------------------------------------
class DinoObstacle(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    kind: Literal["cactus", "bird"]
    variant: str                      # for birds: "duck" | "jump" | high variant
    x: float                          # relative distance from dino's front edge
    y: float = Field(ge=0)
    w: float = Field(gt=0)
    h: float = Field(gt=0)


class DinoSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    grounded: bool
    ducking: bool
    speed: float = Field(gt=0)
    score: float = Field(ge=0)
    decision_latency_ms: float = Field(default=0, ge=0, le=5000)
    obstacles: List[DinoObstacle] = Field(max_length=2)


class PredictRequest(BaseModel):
    state: Any
    questions: Dict[str, Any]


# ---------------------------------------------------------------------------
# Decision pipeline (all arithmetic lives here, NOT in the model)
# ---------------------------------------------------------------------------
def required_action(obstacle: DinoObstacle) -> str:
    """The action needed to pass an obstacle, or "ignore" if none is needed."""
    if obstacle.kind == "cactus":
        return "jump"
    if obstacle.variant == "duck":
        return "duck"
    if obstacle.variant == "jump":
        return "jump"
    return "ignore"


def find_threat(snapshot: DinoSnapshot) -> Optional[Dict[str, Any]]:
    """Pick the first non-ignorable obstacle and classify its reaction window.

    Obstacle positions are extrapolated by the measured decision latency so the
    classification describes the world as it will be when the answer arrives.
    """
    for obstacle in sorted(snapshot.obstacles, key=lambda o: o.x):
        need = required_action(obstacle)
        if need == "ignore":
            continue
        distance_on_arrival = obstacle.x - snapshot.speed * (snapshot.decision_latency_ms / 1000.0)
        if distance_on_arrival + obstacle.w < -DINO_STAND_WIDTH:
            continue  # will already be behind the dino when the decision arrives
        time_on_arrival = distance_on_arrival / snapshot.speed
        if need == "jump":
            if distance_on_arrival < -JUMP_LATE_PIXELS:
                window = "missed"
            elif time_on_arrival <= JUMP_LEAD_SECONDS:
                window = "now"
            else:
                window = "early"
        else:
            if distance_on_arrival < 0 or time_on_arrival <= DUCK_LEAD_SECONDS:
                window = "now"
            else:
                window = "early"
        return {
            "kind": obstacle.kind,
            "variant": obstacle.variant,
            "need": need,
            "window": window,
            "distance_on_arrival": round(distance_on_arrival, 1),
            "time_on_arrival": round(time_on_arrival, 3),
        }
    return None


def allowed_actions(snapshot: DinoSnapshot, threat: Optional[Dict[str, Any]]) -> List[str]:
    """Actions the dino may take right now; the model's choice is constrained to these."""
    allowed = ["run"]
    if threat is not None and snapshot.grounded and threat["window"] == "now":
        allowed.append(threat["need"])
    return allowed


def describe_state(snapshot: DinoSnapshot, threat: Optional[Dict[str, Any]]) -> str:
    """Natural-language state for the model.

    Deliberately omits the dino's current posture (ducking): a small classifier
    tends to echo it ("ducking" -> "duck") instead of reasoning about the obstacle.
    """
    if snapshot.grounded:
        parts = ["The dino is on the ground."]
    else:
        parts = ["The dino is in the air and cannot jump or duck."]
    if threat is None:
        parts.append("No obstacle needs a reaction right now.")
        return " ".join(parts)

    if threat["kind"] == "cactus":
        label, passing = "a cactus", "It must be jumped over."
    elif threat["variant"] == "duck":
        label, passing = "a head-height bird", "It must be ducked under."
    else:
        label, passing = "a low bird", "It must be jumped over."
    parts.append(
        f"Ahead is {label}. {passing} When the decision arrives it will be "
        f"{threat['distance_on_arrival']} pixels away, with about "
        f"{threat['time_on_arrival']} seconds of travel time."
    )
    if threat["window"] == "now":
        parts.append("NOW, act on this obstacle immediately.")
    elif threat["window"] == "early":
        parts.append("NOT YET, the obstacle is still too far away, keep running.")
    else:
        parts.append("MISSED, it is too late to jump, so keep running.")
    return " ".join(parts)


def pick_action(allowed: List[str], model_choice: Any, probabilities: Any) -> tuple[str, str]:
    """Constrain the model's decision to the allowed actions, with fallbacks."""
    if isinstance(model_choice, str) and model_choice in allowed:
        return model_choice, f"Model chose '{model_choice}', which is allowed."
    if isinstance(probabilities, dict) and probabilities:
        best: Optional[str] = None
        best_p = -1.0
        for action in allowed:
            p = probabilities.get(action)
            if isinstance(p, (int, float)) and p > best_p:
                best, best_p = action, float(p)
        if best is not None:
            return best, (
                f"Model choice {model_choice!r} is not allowed; fell back to the "
                f"highest-probability allowed action '{best}' (p={best_p:.3f})."
            )
    return "run", (
        f"No usable model decision ({model_choice!r} not in {list(allowed)}); defaulted to 'run'."
    )


ACTION_QUESTIONS: Dict[str, Any] = {
    "action": {
        "type": "choice",
        "instructions": (
            "Choose the dinosaur's next action. If the reaction window is NOW and the obstacle "
            "must be jumped over, choose jump. If the reaction window is NOW and the obstacle "
            "must be ducked under, choose duck. In every other case - the dino is airborne, "
            "there is no obstacle needing a reaction, the window is NOT YET, or the window is "
            "MISSED - choose run."
        ),
        "criteria": {
            "run": (
                "Keep running. No reaction is needed right now: the dino is airborne, no "
                "obstacle needs a reaction, the window is NOT YET, or the window is MISSED."
            ),
            "jump": "Jump over the obstacle. The reaction window is NOW and the obstacle must be jumped over.",
            "duck": "Duck under the obstacle. The reaction window is NOW and the obstacle must be ducked under.",
        },
    },
}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Dino Run with Laya")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default 422 body echoes the raw input, which itself fails to
    # serialize for non-JSON-compliant values (NaN/Infinity). Return a clean 422.
    return JSONResponse(status_code=422, content={"detail": "invalid game snapshot"})

# Load the model once at startup.
router = laya.Router(preload=True, device="cpu")
_predict_lock = threading.Lock()  # serialize model calls; the runtime is not thread-safe

PREDICT_TOKEN = os.environ.get("PREDICT_TOKEN", "")


@app.get("/", include_in_schema=False)
@app.get("/dino", include_in_schema=False)
def serve_game() -> FileResponse:
    return FileResponse(BASE_DIR / "dino.html")


@app.post("/dino/act")
def dino_act(snapshot: DinoSnapshot) -> Dict[str, Any]:
    threat = find_threat(snapshot)
    allowed = allowed_actions(snapshot, threat)
    state = describe_state(snapshot, threat)

    model_choice: Any = None
    probabilities: Any = None
    confidence: Any = None
    error_note = ""
    try:
        with _predict_lock:
            result = router.predict({"body": state}, ACTION_QUESTIONS)
        answer = result.get("answers", {}).get("action", {})
        model_choice = answer.get("choice")
        probabilities = answer.get("probabilities")
        confidence = answer.get("confidence")
    except Exception as exc:  # a model failure must never crash the game
        error_note = f" (model call failed: {exc})"

    action, reason = pick_action(allowed, model_choice, probabilities)
    return {
        "action": action,
        "debug": {
            "state": state,
            "threat": threat,
            "allowed_actions": allowed,
            "model_choice": model_choice,
            "applied_action": action,
            "reason": reason + error_note,
            "probabilities": probabilities if isinstance(probabilities, dict) else {},
            "confidence": confidence,
        },
    }


@app.post("/predict")
def predict(req: PredictRequest, x_api_key: Optional[str] = Header(default=None)) -> Any:
    if not PREDICT_TOKEN:
        raise HTTPException(status_code=404, detail="Not found")
    if x_api_key is None or not secrets.compare_digest(
        x_api_key.encode("utf-8"), PREDICT_TOKEN.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")
    with _predict_lock:
        return router.predict(req.state, req.questions)