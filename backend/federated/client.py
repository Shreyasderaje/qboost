"""Clinic-side federated node.

Everything in this module is designed to run on a low-power device: a small
VQC, a couple of local epochs over the clinic's own patients, then a
differentially-private, masked, encrypted update.  No patient record ever
appears in any payload.

Round protocol
--------------
phase 1  train locally -> publish (encrypted) validation metrics + a
         Johnson-Lindenstrauss sketch of the DP update
phase 2  publish the DP update itself, either pairwise-masked (FedAvg:
         only the sum is learnable) or revealed to the selected cohort
         (QAOA strategy: weighted aggregation over the cohort)
"""

from __future__ import annotations

import numpy as np
from pennylane import numpy as pnp
import pennylane as qml

from backend.federated import crypto, dp, secureagg
from backend.quantum.encoding import FeatureEncoder
from backend.quantum.vqc import VQC


def bce_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class ClinicNode:
    """One rural clinic: its local dataset, device profile and privacy stack."""

    def __init__(
        self,
        cid: str,
        name: str,
        region: str,
        device: str,
        X: np.ndarray,
        y: np.ndarray,
        *,
        arch: dict,
        local_epochs: int = 3,
        batch_size: int = 24,
        learning_rate: float = 0.06,
        dp: bool = True,
        dp_clip: float = 1.5,
        dp_sigma: float = 0.9,
        seed: int = 0,
        n_sketch: int = 32,
        gradient_ascent: bool = False,
        fake_metrics: dict | None = None,
    ) -> None:
        self.cid = cid
        self.name = name
        self.region = region
        self.device = device
        self.gradient_ascent = gradient_ascent
        self.fake_metrics = fake_metrics

        self.dp = dp
        self.dp_clip = dp_clip
        self.dp_sigma = dp_sigma
        self.n_sketch = n_sketch
        self.rng = np.random.default_rng(seed)
        self.local_epochs = local_epochs

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).ravel()

        # Stratified local holdout for honest validation signals; tiny or
        # class-skewed clinics fall back to validating on all local data.
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]
        self.rng.shuffle(idx_pos)
        self.rng.shuffle(idx_neg)
        n_val = max(2, int(0.25 * len(y)))
        half = n_val // 2
        if len(idx_pos) >= 2 and len(idx_neg) >= 2:
            val_idx = np.concatenate([idx_pos[:half], idx_neg[:half]])
        else:
            val_idx = np.arange(len(y))
        tr_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        if len(tr_idx) == 0:
            tr_idx = val_idx
        self.X_train, self.y_train = X[tr_idx], y[tr_idx]
        self.X_val, self.y_val = X[val_idx], y[val_idx]

        self.model = VQC(
            n_qubits=arch["n_qubits"],
            n_layers=arch["n_layers"],
            learning_rate=learning_rate,
            seed=seed + 17,
        )
        self.n_samples = int(len(self.y_train))

        self._key: str | None = None
        self._secret: bytes | None = None
        self._peers: list[str] = []
        self._pending_delta: np.ndarray | None = None
        self._encoder_key: str | None = None
        self._angles_train: np.ndarray | None = None
        self._angles_val: np.ndarray | None = None
        self._total_samples: int | None = None

    # -- credentials ---------------------------------------------------------
    def set_credentials(self, key: str, secret: str, peers: list[str]) -> None:
        self._key = key
        self._secret = secret.encode("utf-8")
        self._peers = peers

    # -- protocol ------------------------------------------------------------
    def load_bundle(self, bundle: dict) -> None:
        """Adopt the current global model published by the coordinator."""
        self.model.set_param_vector(np.asarray(bundle["params"], dtype=np.float64))
        self._total_samples = int(bundle.get("total_samples") or 0) or None
        enc_key = repr(sorted(bundle["encoder"].items(), key=lambda kv: kv[0]))[:256]
        if enc_key != self._encoder_key:
            encoder = FeatureEncoder.from_dict(bundle["encoder"])
            self._angles_train = encoder.transform(self.X_train)
            self._angles_val = encoder.transform(self.X_val)
            self._encoder_key = enc_key

    def phase1(self, round_id: int, bundle: dict, round_seed: int) -> str:
        """Train locally; return the sealed phase-1 payload."""
        self.load_bundle(bundle)
        params_before = self.model.get_param_vector()

        sign = -1.0 if self.gradient_ascent else 1.0
        X, y = self._angles_train, self.y_train.astype(np.float64)
        n = X.shape[0]
        batch = max(8, min(24, n))

        # The simulated attacker ascends the loss.  Bounding the excursion keeps
        # the attack well-posed on tiny variational circuits (divergent angles
        # would degrade circuit simulation for everyone, which is not the
        # behaviour being modelled) while preserving the anti-correlated update
        # direction that the coherence signal is supposed to detect.
        opt = qml.optimize.AdamOptimizer(self.model.learning_rate * sign)
        bound = 8.0
        for _ in range(self.local_epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, batch):
                idx = order[start : start + batch]
                xb, yb = X[idx], y[idx]

                def batch_loss(weights):
                    total = 0.0
                    for i in range(xb.shape[0]):
                        z = self.model._circuit(pnp.array(xb[i], requires_grad=False), weights)
                        p = 1e-7 + (1.0 - 2e-7) * (1.0 - z) / 2.0
                        total = total + yb[i] * pnp.log(p) + (1.0 - yb[i]) * pnp.log(1.0 - p)
                    return -total / xb.shape[0]

                self.model.params = opt.step(batch_loss, self.model.params)

            params = np.array(self.model.params, dtype=np.float64)
            if not np.isfinite(params).all():
                self.model.set_param_vector(params_before)
                break
            if np.abs(params).max() > bound:
                self.model.set_param_vector(np.clip(params, -bound, bound))
        params_after = self.model.get_param_vector()
        if not np.isfinite(params_after).all():
            self.model.set_param_vector(params_before)

        delta = self.model.get_param_vector() - params_before

        if self.dp:
            delta = dp.add_gaussian_noise(dp.clip_update(delta, self.dp_clip), self.dp_clip, self.dp_sigma, self.rng)

        proj = secureagg.random_projection(round_seed, delta.size, self.n_sketch)
        sketch = proj @ delta

        p_val = self.model.predict_proba(self._angles_val)
        if self.fake_metrics is not None:
            metrics = dict(self.fake_metrics)
        else:
            metrics = {
                "val_acc": float(((p_val >= 0.5).astype(int) == self.y_val).mean()),
                "val_loss": bce_loss(self.y_val, p_val),
            }

        self._pending_delta = delta
        payload = {
            "kind": "phase1",
            "cid": self.cid,
            "round_id": round_id,
            "n_samples": self.n_samples,
            "metrics": metrics,
            "update_norm": float(np.linalg.norm(delta)),
            "sketch": [round(float(v), 6) for v in sketch],
        }
        return crypto.seal(payload, self._key)

    def phase2(self, round_id: int, masked: bool) -> str:
        """Publish the DP update, pairwise-masked (FedAvg) or revealed (QAOA).

        In the masked mode the update is pre-scaled by n_i / N (N announced by
        the coordinator), so the sum of all masked updates is exactly the
        sample-weighted FedAvg average while individual updates stay hidden.
        """
        delta = self._pending_delta
        if delta is None:
            raise RuntimeError("phase2 called before phase1")
        if masked:
            scale = (self.n_samples / self._total_samples) if self._total_samples else 1.0
            delta = delta * scale
            delta = delta + secureagg.client_mask(self.cid, self._peers, round_id, delta.size, self._secret)
        payload = {
            "kind": "phase2",
            "cid": self.cid,
            "round_id": round_id,
            "masked": bool(masked),
            "delta": [float(v) for v in delta],
        }
        return crypto.seal(payload, self._key)
