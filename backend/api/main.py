"""QBoost coordination API.

Serves the dashboard, the inference endpoint used by the clinic console, the
simulation manager that powers the live federation view, and a manual
federation session API that lets real separate clinic processes join over
HTTP (see scripts/clinic_simulator.py).
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend import __version__
from backend.config import ARTIFACTS_DIR, RESULTS_DIR, ensure_dirs
from backend.data.tasks import TASKS
from backend.federated import crypto
from backend.federated.server import FederationServer, HTTPConflict
from backend.quantum.encoding import FeatureEncoder
from backend.quantum.vqc import VQC
from backend.simulation.manager import MANAGER

app = FastAPI(title="QBoost Coordination API", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_dirs()

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class SimulationStart(BaseModel):
    task: str = "diabetes"
    strategy: str = "qaoa"
    clinics: int = Field(default=4, ge=2, le=8)
    rounds: int = Field(default=5, ge=1, le=12)
    n_total: int = Field(default=1400, ge=300, le=6000)
    dp: bool = True
    poison: bool = False
    seed: int = 42
    local_epochs: int = Field(default=3, ge=1, le=6)


class InferenceRequest(BaseModel):
    task: str
    features: dict[str, float]


class FedInit(BaseModel):
    task: str
    strategy: str = "qaoa"
    seed: int = 42


class FedRegister(BaseModel):
    cid: str
    name: str
    region: str = ""
    n_samples: int = 0


class FedPayload(BaseModel):
    cid: str
    token: str


# ---------------------------------------------------------------------------
# System / catalog
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "qboost-coordination",
        "version": __version__,
        "tasks": sorted(TASKS),
    }


@app.get("/api/tasks")
def tasks():
    out = []
    for spec in TASKS.values():
        out.append(
            {
                "key": spec.key,
                "name": spec.display_name,
                "subtitle": spec.subtitle,
                "prevalence": spec.prevalence,
                "positive_label": spec.positive_label,
                "negative_label": spec.negative_label,
                "features": [
                    {
                        "name": f.name,
                        "label": f.label,
                        "unit": f.unit,
                        "lo": f.lo,
                        "hi": f.hi,
                        "direction": f.direction,
                    }
                    for f in spec.features
                ],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Model registry + inference
# ---------------------------------------------------------------------------
_model_cache: dict[str, tuple[float, dict]] = {}


def _load_artifact(name: str) -> dict:
    path = ARTIFACTS_DIR / name
    if not path.is_file():
        raise HTTPException(404, f"model '{name}' not found")
    mtime = path.stat().st_mtime
    cached = _model_cache.get(name)
    if cached and cached[0] == mtime:
        return cached[1]
    art = json.loads(path.read_text(encoding="utf-8"))
    _model_cache[name] = (mtime, art)
    return art


@app.get("/api/models")
def models():
    out = []
    for path in sorted(ARTIFACTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            art = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        out.append(
            {
                "name": path.name,
                "task": art.get("task"),
                "task_name": art.get("task_name"),
                "strategy": art.get("strategy"),
                "rounds_completed": art.get("rounds_completed"),
                "val_acc": (art.get("val_metrics") or {}).get("acc"),
                "arch": art.get("arch"),
                "privacy": art.get("privacy"),
            }
        )
    return out


@app.get("/api/models/{name}")
def model_detail(name: str):
    return _load_artifact(name)


def _latest_model_for_task(task: str) -> str:
    best, best_mtime = None, -1.0
    for path in ARTIFACTS_DIR.glob(f"{task}_*.json"):
        if path.stat().st_mtime > best_mtime:
            best, best_mtime = path.name, path.stat().st_mtime
    if best is None:
        raise HTTPException(404, f"no trained model for task '{task}' yet - launch a training run first")
    return best


@app.post("/api/inference")
def inference(req: InferenceRequest):
    art = _load_artifact(_latest_model_for_task(req.task))
    spec = TASKS[req.task]

    values = []
    for f in spec.features:
        if f.name not in req.features:
            raise HTTPException(422, f"missing feature '{f.name}'")
        v = float(req.features[f.name])
        v = min(max(v, f.lo), f.hi)
        values.append(v)
    x = np.asarray([values], dtype=np.float64)

    encoder = FeatureEncoder.from_dict(art["encoder"])
    model = VQC(n_qubits=art["arch"]["n_qubits"], n_layers=art["arch"]["n_layers"])
    model.set_param_vector(np.asarray(art["params"], dtype=np.float64))

    risk = float(model.predict_proba(encoder.transform(x))[0])
    medians = np.asarray(art["medians"], dtype=np.float64)

    contributions = []
    for i, f in enumerate(spec.features):
        x_alt = x.copy()
        x_alt[0, i] = float(medians[i])
        p_alt = float(model.predict_proba(encoder.transform(x_alt))[0])
        delta = risk - p_alt
        contributions.append(
            {
                "name": f.name,
                "label": f.label,
                "value": values[i],
                "unit": f.unit,
                "direction": f.direction,
                "impact": round(delta, 4),
            }
        )
    contributions.sort(key=lambda c: abs(c["impact"]), reverse=True)

    label = spec.positive_label if risk >= 0.5 else spec.negative_label
    return {
        "task": req.task,
        "risk": round(risk, 4),
        "label": label,
        "threshold": 0.5,
        "model": {
            "name": _latest_model_for_task(req.task),
            "strategy": art.get("strategy"),
            "rounds": art.get("rounds_completed"),
            "val_acc": (art.get("val_metrics") or {}).get("acc"),
            "arch": art.get("arch"),
        },
        "contributions": contributions,
    }


# ---------------------------------------------------------------------------
# Live simulation (dashboard)
# ---------------------------------------------------------------------------
@app.post("/api/simulation/start")
def simulation_start(req: SimulationStart):
    try:
        run_id = MANAGER.start(
            {
                "task": req.task,
                "strategy": req.strategy,
                "clinics": req.clinics,
                "rounds": req.rounds,
                "n_total": req.n_total,
                "dp": req.dp,
                "poison": req.poison,
                "seed": req.seed,
                "local_epochs": req.local_epochs,
            }
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"run_id": run_id}


@app.get("/api/simulations")
def simulations():
    return MANAGER.list()


@app.get("/api/simulation/{run_id}")
def simulation_state(run_id: str):
    state = MANAGER.get(run_id)
    if state is None:
        raise HTTPException(404, "unknown run id")
    return state


# ---------------------------------------------------------------------------
# Benchmarks (produced by scripts/benchmark.py)
# ---------------------------------------------------------------------------
@app.get("/api/benchmarks")
def benchmarks():
    path = RESULTS_DIR / "benchmarks.json"
    if not path.is_file():
        return {"available": False, "message": "run scripts/benchmark.py to generate results"}
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/benchmarks/chart")
def benchmarks_chart():
    from fastapi.responses import FileResponse

    path = RESULTS_DIR / "benchmarks.png"
    if not path.is_file():
        raise HTTPException(404, "chart not generated yet - run scripts/benchmark.py")
    return FileResponse(path, media_type="image/png")


# ---------------------------------------------------------------------------
# Manual federation sessions (real separate clinic processes over HTTP)
# ---------------------------------------------------------------------------
_fed_sessions: dict[str, FederationServer] = {}
_fed_events: dict[str, deque] = {}


@app.post("/api/federation/init")
def federation_init(req: FedInit):
    sid = f"fed-{int(time.time() * 1000) % 100000000:08d}"
    events = deque(maxlen=2000)
    server = FederationServer(
        req.task, strategy=req.strategy, seed=req.seed, event_sink=events
    )
    _fed_sessions[sid] = server
    _fed_events[sid] = events
    return {"session_id": sid, "task": req.task, "strategy": req.strategy, "round": 0}


@app.post("/api/federation/{sid}/register")
def federation_register(sid: str, req: FedRegister):
    server = _fed_sessions.get(sid)
    if server is None:
        raise HTTPException(404, "unknown session")
    creds = server.register_clinic(req.cid, req.name, req.region, req.n_samples)
    return creds


@app.get("/api/federation/{sid}/round/{round_id}/bundle")
def federation_bundle(sid: str, round_id: int):
    server = _fed_sessions.get(sid)
    if server is None:
        raise HTTPException(404, "unknown session")
    return server.round_payload(round_id)


@app.post("/api/federation/{sid}/phase1")
def federation_phase1(sid: str, req: FedPayload):
    server = _fed_sessions.get(sid)
    if server is None:
        raise HTTPException(404, "unknown session")
    try:
        return server.collect_phase1(req.cid, req.token)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/federation/{sid}/select")
def federation_select(sid: str):
    server = _fed_sessions.get(sid)
    if server is None:
        raise HTTPException(404, "unknown session")
    try:
        return server.select_cohort()
    except (HTTPConflict, RuntimeError) as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/federation/{sid}/status")
def federation_status(sid: str):
    server = _fed_sessions.get(sid)
    if server is None:
        raise HTTPException(404, "unknown session")
    return server.round_status()


@app.post("/api/federation/{sid}/phase2")
def federation_phase2(sid: str, req: FedPayload):
    server = _fed_sessions.get(sid)
    if server is None:
        raise HTTPException(404, "unknown session")
    try:
        return server.collect_phase2(req.cid, req.token)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/federation/{sid}/aggregate")
def federation_aggregate(sid: str):
    server = _fed_sessions.get(sid)
    if server is None:
        raise HTTPException(404, "unknown session")
    try:
        return server.aggregate()
    except (RuntimeError, HTTPConflict) as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/federation/{sid}/events")
def federation_events(sid: str):
    if sid not in _fed_events:
        raise HTTPException(404, "unknown session")
    return list(_fed_events[sid])


# ---------------------------------------------------------------------------
# Static frontend (production build) with SPA history fallback
# ---------------------------------------------------------------------------
_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _dist.is_dir():
    from fastapi.responses import FileResponse

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404, "not found")
        candidate = (_dist / full_path).resolve()
        if full_path and candidate.is_file() and str(candidate).startswith(str(_dist.resolve())):
            return FileResponse(candidate)
        return FileResponse(_dist / "index.html")
