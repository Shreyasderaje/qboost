"""Runs complete federated training simulations.

``SimulationManager`` drives a full federation in a background thread (used
by the dashboard's "launch training" action) while ``run_federation`` offers
the same loop synchronously for the CLI and the test suite.  Both share
``_execute`` so behaviour cannot drift between the two entry points.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import deque

from backend.config import (
    BATCH_SIZE,
    DP_CLIP,
    DP_DELTA,
    DP_SIGMA,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    NONIID_ALPHA,
    N_LAYERS,
    N_QUBITS,
    QAOA_P,
    QAOA_STEPS,
    SEED,
    ARTIFACTS_DIR,
)
from backend.data.synth import make_federated_splits
from backend.data.tasks import get_task
from backend.federated.client import ClinicNode
from backend.federated.server import FederationServer

CLINIC_PROFILES = [
    ("Phulbani CHC", "Odisha, India", "Android tablet · 4 GB"),
    ("Bastar Rural DH", "Chhattisgarh, India", "Raspberry Pi 4 · 8 GB"),
    ("Gadchiroli PHC", "Maharashtra, India", "Android tablet · 3 GB"),
    ("Barmer Mobile Unit", "Rajasthan, India", "Raspberry Pi 3 · 1 GB"),
    ("Kisumu County Clinic", "Kisumu, Kenya", "Android tablet · 4 GB"),
    ("Mekong River Post", "An Giang, Vietnam", "Raspberry Pi 4 · 4 GB"),
    ("Sylhet Tea-Garden Clinic", "Sylhet, Bangladesh", "Android tablet · 3 GB"),
    ("Mon State Outreach", "Mon, Myanmar", "Raspberry Pi 3 · 1 GB"),
]

DEFAULT_RUN_CONFIG = {
    "task": "diabetes",
    "strategy": "qaoa",
    "clinics": 4,
    "rounds": 5,
    "n_total": 1400,
    "noniid_alpha": NONIID_ALPHA,
    "dp": True,
    "dp_clip": DP_CLIP,
    "dp_sigma": DP_SIGMA,
    "dp_delta": DP_DELTA,
    "local_epochs": LOCAL_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "n_qubits": N_QUBITS,
    "n_layers": N_LAYERS,
    "qaoa_p": QAOA_P,
    "qaoa_steps": QAOA_STEPS,
    "seed": SEED,
    "poison": False,
}


class _Run:
    def __init__(self, config: dict) -> None:
        self.run_id = uuid.uuid4().hex[:10]
        self.config = dict(config)
        self.created = time.time()
        self.status = "queued"
        self.rounds: list[dict] = []
        self.events: deque = deque(maxlen=4000)
        self.result: dict | None = None
        self.error: str | None = None
        self._lock = threading.Lock()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "run_id": self.run_id,
                "status": self.status,
                "created": self.created,
                "config": self.config,
                "rounds": list(self.rounds),
                "events": list(self.events),
                "result": self.result,
                "error": self.error,
            }


def _execute(run: _Run) -> None:
    """Run a complete federation; mutates ``run`` in place."""
    cfg = run.config
    run.status = "running"
    spec = get_task(cfg["task"])

    events = run.events
    server = FederationServer(
        cfg["task"],
        strategy=cfg["strategy"],
        seed=cfg["seed"],
        n_qubits=cfg["n_qubits"],
        n_layers=cfg["n_layers"],
        dp=cfg["dp"],
        dp_clip=cfg["dp_clip"],
        dp_sigma=cfg["dp_sigma"],
        dp_delta=cfg["dp_delta"],
        qaoa_p=cfg["qaoa_p"],
        qaoa_steps=cfg["qaoa_steps"],
        event_sink=events,
    )

    splits = make_federated_splits(
        spec,
        cfg["clinics"],
        n_total=cfg["n_total"],
        alpha=cfg["noniid_alpha"],
        seed=cfg["seed"],
    )

    nodes: list[ClinicNode] = []
    for k in range(cfg["clinics"]):
        cid = f"clinic-{k + 1}"
        name, region, device = CLINIC_PROFILES[k % len(CLINIC_PROFILES)]
        creds = server.register_clinic(cid, name, region, int(len(splits["clients"][k]["y"])))
        poisoned = bool(cfg.get("poison")) and k == cfg["clinics"] - 1
        node = ClinicNode(
            cid,
            name,
            region,
            device,
            splits["clients"][k]["X"],
            splits["clients"][k]["y"],
            arch=server.arch,
            local_epochs=cfg["local_epochs"],
            batch_size=cfg["batch_size"],
            learning_rate=cfg["learning_rate"],
            dp=cfg["dp"],
            dp_clip=cfg["dp_clip"],
            dp_sigma=cfg["dp_sigma"],
            seed=cfg["seed"] * 100 + k,
            gradient_ascent=poisoned,
            fake_metrics={"val_acc": 0.93, "val_loss": 0.18} if poisoned else None,
        )
        node.set_credentials(
            creds["key"], creds["secret"], [f"clinic-{j + 1}" for j in range(cfg["clinics"])]
        )
        nodes.append(node)

    n_rounds = cfg["rounds"]
    for r in range(1, n_rounds + 1):
        payload = server.round_payload(r)
        server.begin_round(r)
        for node in nodes:
            token = node.phase1(r, payload, payload["round_seed"])
            server.collect_phase1(node.cid, token)
        server.select_cohort()
        participating = nodes if server.strategy == "fedavg" else [n for n in nodes if n.cid in server.cohort]
        for node in participating:
            token = node.phase2(r, masked=(server.strategy == "fedavg"))
            server.collect_phase2(node.cid, token)
        record = server.aggregate()
        with run._lock:
            run.rounds.append(record)

    artifact_name = f"{cfg['task']}_{cfg['strategy']}_global.json"
    artifact_path = server.save_artifact(ARTIFACTS_DIR / artifact_name)
    final = server.round_records[-1]["global"] if server.round_records else {"acc": None, "loss": None}
    run.result = {
        "artifact": artifact_name,
        "artifact_path": str(artifact_path),
        "rounds_completed": len(server.round_records),
        "final_acc": final["acc"],
        "final_loss": final["loss"],
        "privacy": server.privacy_snapshot(n_rounds),
        "excluded_total": sum(1 for rec in server.round_records for c in rec["clients"] if not c["selected"]),
    }
    run.status = "completed"


class SimulationManager:
    """Registry of background federation runs for the dashboard."""

    def __init__(self) -> None:
        self._runs: dict[str, _Run] = {}
        self._lock = threading.Lock()

    def start(self, overrides: dict | None = None) -> str:
        config = dict(DEFAULT_RUN_CONFIG)
        if overrides:
            unknown = set(overrides) - set(config)
            if unknown:
                raise ValueError(f"unknown simulation options: {sorted(unknown)}")
            config.update(overrides)
        if config["strategy"] not in ("qaoa", "fedavg"):
            raise ValueError("strategy must be 'qaoa' or 'fedavg'")
        if config["task"] not in ("tb", "malaria", "diabetes"):
            raise ValueError("unknown task")
        config["clinics"] = max(2, min(8, int(config["clinics"])))
        config["rounds"] = max(1, min(12, int(config["rounds"])))
        config["n_total"] = max(300, min(6000, int(config["n_total"])))

        run = _Run(config)
        with self._lock:
            self._runs[run.run_id] = run
        thread = threading.Thread(target=self._guarded, args=(run,), daemon=True, name=f"fed-{run.run_id}")
        thread.start()
        return run.run_id

    def _guarded(self, run: _Run) -> None:
        try:
            _execute(run)
        except Exception:
            run.error = traceback.format_exc()
            run.status = "failed"

    def get(self, run_id: str) -> dict | None:
        run = self._runs.get(run_id)
        return run.snapshot() if run else None

    def list(self) -> list[dict]:
        out = []
        for run in self._runs.values():
            snap = run.snapshot()
            out.append(
                {
                    "run_id": snap["run_id"],
                    "status": snap["status"],
                    "config": snap["config"],
                    "created": snap["created"],
                    "rounds_completed": len(snap["rounds"]),
                    "final_acc": (snap["result"] or {}).get("final_acc") if snap["result"] else None,
                }
            )
        return sorted(out, key=lambda r: r["created"], reverse=True)


MANAGER = SimulationManager()
