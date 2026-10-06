"""Federation coordination server.

Owns the global model and the round protocol.  The server fits the feature
encoder on its *own* reference distribution (never on clinic data), publishes
the model bundle each round, collects encrypted client payloads, runs the
QAOA cohort selection, aggregates, and evaluates the global model on a
held-out coordinator benchmark set.

Two aggregation strategies are supported:

``fedavg``
    Every clinic's pairwise-masked update is summed; masks cancel and only
    the mean is ever learned (Bonawitz-style secure aggregation).

``qaoa``
    Phase-1 signals (quality + sketches) are fed into a QUBO solved with
    QAOA; only the selected cohort's differentially-private updates are
    revealed and combined with quality weights.  Faulty or poisoned updates
    are excluded by their anti-correlated sketches.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from backend.config import N_LAYERS, N_QUBITS
from backend.data.synth import make_dataset
from backend.data.tasks import get_task
from backend.federated import aggregator, crypto, dp
from backend.quantum.encoding import FeatureEncoder
from backend.quantum.vqc import VQC


class HTTPConflict(Exception):
    """Round protocol guard: the operation cannot proceed yet.

    Clinic processes treat this as 'retry after polling round status'.
    """


class FederationServer:
    def __init__(
        self,
        task: str,
        *,
        strategy: str = "qaoa",
        seed: int = 42,
        n_qubits: int = N_QUBITS,
        n_layers: int = N_LAYERS,
        dp: bool = True,
        dp_clip: float = 1.5,
        dp_sigma: float = 0.9,
        dp_delta: float = 1e-5,
        qaoa_p: int = 2,
        qaoa_steps: int = 120,
        quality_weight: float = 1.0,
        coherence_weight: float = 1.0,
        min_cohort_fraction: float = 0.5,
        reference_size: int = 480,
        event_sink: deque | None = None,
    ) -> None:
        self.spec = get_task(task)
        self.task = self.spec.key
        self.strategy = strategy
        self.seed = seed
        self.dp = dp
        self.dp_clip = dp_clip
        self.dp_sigma = dp_sigma
        self.dp_delta = dp_delta
        self.qaoa_p = qaoa_p
        self.qaoa_steps = qaoa_steps
        self.quality_weight = quality_weight
        self.coherence_weight = coherence_weight
        self.min_cohort_fraction = min_cohort_fraction

        self._events = event_sink if event_sink is not None else deque(maxlen=2000)
        self._rng = np.random.default_rng(seed)

        # Coordinator-owned reference distribution (no clinic data involved).
        ref = make_dataset(self.spec, reference_size, seed=seed + 1001)
        self.encoder = FeatureEncoder(n_components=n_qubits).fit(ref["X"])
        val_n = int(0.45 * reference_size)
        self.val_X, self.val_y = ref["X"][:val_n], ref["y"][:val_n]
        self._ref_medians = np.median(ref["X"], axis=0)

        self.global_model = VQC(n_qubits=n_qubits, n_layers=n_layers, seed=seed + 5)
        self.arch = {"n_qubits": n_qubits, "n_layers": n_layers}

        self.clinics: dict[str, dict] = {}
        # Pairwise masks cancel only if both members of a pair derive the same
        # mask, so the coordinator issues ONE federation-level aggregation
        # secret to every clinic (standing in for Bonawitz-style pairwise key
        # agreement).  Envelope keys, by contrast, are unique per clinic.
        self._aggregation_secret = crypto.issue_clinic_key()
        self.summaries: list[aggregator.ClientSummary] = []
        self.cohort: list[str] = []
        self.selection: aggregator.SelectionResult | None = None
        self.deltas: dict[str, list[float]] = {}
        self.current_round = 0
        self.round_records: list[dict] = []
        self._round_t0 = 0.0

    # -- events ----------------------------------------------------------------
    def events(self) -> list[dict]:
        return list(self._events)

    def _emit(self, kind: str, message: str, **data) -> None:
        self._events.append(
            {"ts": time.time(), "round": self.current_round, "kind": kind, "message": message, "data": data}
        )

    # -- registration ------------------------------------------------------------
    def register_clinic(self, cid: str, name: str, region: str, n_samples: int) -> dict:
        key = crypto.issue_clinic_key()      # unique envelope key per clinic
        secret = self._aggregation_secret    # shared pairwise-masking secret
        self.clinics[cid] = {"cid": cid, "name": name, "region": region, "key": key, "secret": secret, "n_samples": n_samples}
        self._emit("clinic_registered", f"{name} joined the federation", cid=cid, region=region, n_samples=n_samples)
        return {"key": key, "secret": secret}

    def round_payload(self, round_id: int) -> dict:
        return {
            "round_id": round_id,
            "round_seed": int(self.seed + round_id * 7919),
            "total_samples": int(sum(c["n_samples"] for c in self.clinics.values())) or None,
            "params": [float(v) for v in self.global_model.get_param_vector()],
            "encoder": self.encoder.to_dict(),
            "arch": self.arch,
        }

    # -- round protocol -------------------------------------------------------------
    def begin_round(self, round_id: int) -> None:
        self.current_round = round_id
        self.summaries = []
        self.cohort = []
        self.deltas = {}
        self.selection = None
        self._round_t0 = time.perf_counter()
        self._emit("round_started", f"Round {round_id} opened", round_id=round_id)

    def collect_phase1(self, cid: str, token: str) -> dict:
        payload = crypto.unseal(token, self.clinics[cid]["key"])
        if payload.get("kind") != "phase1" or payload.get("round_id") != self.current_round:
            raise ValueError("stale or malformed phase-1 payload")
        self.summaries.append(
            aggregator.ClientSummary(
                cid=cid,
                name=self.clinics[cid]["name"],
                n_samples=int(payload["n_samples"]),
                val_acc=float(payload["metrics"]["val_acc"]),
                val_loss=float(payload["metrics"]["val_loss"]),
                update_norm=float(payload["update_norm"]),
                sketch=np.asarray(payload["sketch"], dtype=np.float64),
            )
        )
        name = self.clinics[cid]["name"]
        self._emit(
            "client_trained",
            f"{name} finished local training",
            cid=cid,
            val_acc=round(float(payload["metrics"]["val_acc"]), 4),
            val_loss=round(float(payload["metrics"]["val_loss"]), 4),
            n_samples=int(payload["n_samples"]),
        )
        self._emit("secure_submit", f"Encrypted update received from {name}", cid=cid)
        return {"status": "accepted", "clients": len(self.summaries)}

    def round_status(self) -> dict:
        """Machine-readable round progress for independent clinic processes."""
        return {
            "round": self.current_round,
            "registered": [c for c in self.clinics],
            "phase1_received": [s.cid for s in self.summaries],
            "cohort": list(self.cohort),
            "phase2_received": list(self.deltas),
            "rounds_completed": len(self.round_records),
        }

    def select_cohort(self) -> dict:
        if not self.summaries:
            raise RuntimeError("no phase-1 payloads collected")
        if len(self.summaries) < len(self.clinics):
            raise HTTPConflict(
                f"phase-1 incomplete: {len(self.summaries)}/{len(self.clinics)} clinic updates received"
            )
        if self.selection is not None or (self.strategy == "fedavg" and self.cohort):
            excluded = [s.cid for s in self.summaries if not s.selected]
            return {"cohort": list(self.cohort), "excluded": excluded}
        if self.strategy == "fedavg":
            for s in self.summaries:
                s.selected = True
            self.cohort = [s.cid for s in self.summaries]
            self._emit("cohort_selected", "FedAvg: all clinics included (secure sum)", cohort=self.cohort)
            return {"cohort": self.cohort}

        self._emit("selection_started", "Running QAOA cohort selection", qubits=len(self.summaries), p=self.qaoa_p)
        self.selection = aggregator.select_cohort(
            self.summaries,
            strategy="qaoa",
            qaoa_p=self.qaoa_p,
            qaoa_steps=self.qaoa_steps,
            seed=self.seed + self.current_round,
            min_cohort_fraction=self.min_cohort_fraction,
            quality_weight=self.quality_weight,
            coherence_weight=self.coherence_weight,
        )
        self.cohort = self.selection.selected
        excluded = [s.cid for s in self.summaries if not s.selected]
        msg = f"QAOA selected {len(self.cohort)}/{len(self.summaries)} clinics"
        if excluded:
            msg += f" (excluded: {', '.join(excluded)})"
        self._emit(
            "cohort_selected",
            msg,
            cohort=self.cohort,
            excluded=excluded,
            quality=self.selection.quality,
            similarity=self.selection.similarity,
            energy=self.selection.qaoa["energy"] if self.selection.qaoa else None,
        )
        return {"cohort": self.cohort, "excluded": excluded}

    def collect_phase2(self, cid: str, token: str) -> dict:
        payload = crypto.unseal(token, self.clinics[cid]["key"])
        if payload.get("kind") != "phase2" or payload.get("round_id") != self.current_round:
            raise ValueError("stale or malformed phase-2 payload")
        self.deltas[cid] = [float(v) for v in payload["delta"]]
        return {"status": "accepted", "clients": len(self.deltas)}

    # -- aggregation ---------------------------------------------------------------
    def _weighted_delta(self) -> tuple[np.ndarray, list[dict]]:
        summaries = {s.cid: s for s in self.summaries}
        contributions = []
        if self.strategy == "fedavg":
            # Masked updates were pre-scaled by n_i / N client-side, so the
            # sum of the masks cancels into the sample-weighted average.
            n_total = sum(s.n_samples for s in self.summaries) or 1
            total = np.zeros(len(next(iter(self.deltas.values()))), dtype=np.float64)
            for cid, delta in self.deltas.items():
                total += np.asarray(delta)
                contributions.append({"cid": cid, "weight": summaries[cid].n_samples / n_total})
            return total, contributions

        weights = []
        for cid in self.cohort:
            s = summaries[cid]
            weights.append(max(s.n_samples, 1) * (0.5 + s.quality))
        weights = np.asarray(weights, dtype=np.float64)
        weights /= weights.sum()

        dim = len(next(iter(self.deltas.values())))
        total = np.zeros(dim, dtype=np.float64)
        for w, cid in zip(weights, self.cohort):
            total += w * np.asarray(self.deltas[cid], dtype=np.float64)
            contributions.append({"cid": cid, "weight": float(w)})
        return total, contributions

    def aggregate(self) -> dict:
        if self.round_records and self.round_records[-1]["round"] == self.current_round:
            return self.round_records[-1]
        if self.selection is None and self.strategy == "qaoa":
            raise HTTPConflict("cohort selection has not run for this round yet")
        if len(self.deltas) != len(self.cohort):
            raise HTTPConflict(
                f"phase-2 incomplete: {len(self.deltas)}/{len(self.cohort)} cohort updates received"
            )
        combined, contributions = self._weighted_delta()
        new_params = self.global_model.get_param_vector() + combined
        self.global_model.set_param_vector(new_params)

        acc, loss = self.evaluate(self.val_X, self.val_y)
        elapsed = time.perf_counter() - self._round_t0

        record = {
            "round": self.current_round,
            "strategy": self.strategy,
            "cohort": list(self.cohort),
            "contributions": contributions,
            "clients": [
                {
                    "cid": s.cid,
                    "name": s.name,
                    "n_samples": s.n_samples,
                    "val_acc": round(s.val_acc, 4),
                    "val_loss": round(s.val_loss, 4),
                    "update_norm": round(s.update_norm, 5),
                    "quality": round(s.quality, 4),
                    "mean_similarity": round(s.mean_similarity, 4),
                    "selected": s.selected,
                }
                for s in self.summaries
            ],
            "qaoa": self.selection.qaoa if self.selection else None,
            "global": {"acc": round(acc, 4), "loss": round(loss, 4)},
            "privacy": self.privacy_snapshot(),
            "elapsed_s": round(elapsed, 2),
        }
        self.round_records.append(record)
        self._emit(
            "round_complete",
            f"Round {self.current_round} complete - global accuracy {acc:.3f}",
            acc=round(acc, 4),
            loss=round(loss, 4),
            elapsed_s=round(elapsed, 2),
        )
        return record

    # -- evaluation / persistence ----------------------------------------------------
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        angles = self.encoder.transform(X)
        p = self.global_model.predict_proba(angles)
        acc = float(((p >= 0.5).astype(int) == np.asarray(y)).mean())
        from backend.federated.client import bce_loss

        return acc, bce_loss(y, p)

    def privacy_snapshot(self, rounds: int | None = None) -> dict:
        rounds = rounds or max(1, self.current_round)
        return {
            "dp_enabled": self.dp,
            "clip_norm": self.dp_clip,
            "sigma": self.dp_sigma,
            "delta": self.dp_delta,
            "epsilon_total": dp.epsilon_bound(self.dp_sigma, rounds, self.dp_delta) if self.dp else None,
        }

    def artifact(self) -> dict:
        acc, loss = self.evaluate(self.val_X, self.val_y)
        return {
            "schema": "qboost.model/1",
            "task": self.task,
            "task_name": self.spec.display_name,
            "strategy": self.strategy,
            "arch": self.arch,
            "encoder": self.encoder.to_dict(),
            "params": [float(v) for v in self.global_model.get_param_vector()],
            "feature_names": self.spec.feature_names,
            "feature_ranges": [
                {"name": f.name, "label": f.label, "unit": f.unit, "lo": f.lo, "hi": f.hi, "direction": f.direction}
                for f in self.spec.features
            ],
            "medians": [float(v) for v in self._ref_medians],
            "labels": {"positive": self.spec.positive_label, "negative": self.spec.negative_label},
            "val_metrics": {"acc": round(acc, 4), "loss": round(loss, 4)},
            "rounds_completed": len(self.round_records),
            "history": self.round_records,
            "privacy": self.privacy_snapshot(),
        }

    def save_artifact(self, path) -> str:
        import json
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.artifact(), indent=1), encoding="utf-8")
        self._emit("model_saved", f"Global model bundle written to {path.name}", path=str(path))
        return str(path)
