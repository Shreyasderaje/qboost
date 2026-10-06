"""Variational quantum classifier used at every clinic.

The circuit is deliberately tiny (4 qubits, 2 entangling layers) so it can be
trained on a Raspberry-Pi-class device: angle embedding of the compressed
clinical features followed by a StronglyEntanglingLayers ansatz, with the
expectation value of Z on wire 0 mapped to P(class = 1).

Gradients are computed with the adjoint differentiation rule, which needs
only two circuit executions per optimisation step regardless of parameter
count - the cheapest option that stays faithful to how the model would run
on real hardware.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

_EPS = 1e-7


class VQC:
    """Binary variational quantum classifier (angle encoding + SEL ansatz)."""

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        learning_rate: float = 0.06,
        seed: int = 7,
    ) -> None:
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)

        self.params = pnp.array(
            np.random.default_rng(self.seed).uniform(
                0.0, np.pi, (self.n_layers, self.n_qubits, 3)
            ),
            requires_grad=True,
        )

        self._dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self._dev, interface="autograd", diff_method="adjoint")
        def _circuit(features, weights):
            qml.AngleEmbedding(features, wires=range(self.n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return qml.expval(qml.PauliZ(0))

        self._circuit = _circuit

    # -- forward -------------------------------------------------------------
    def _logit_z(self, features: np.ndarray) -> float:
        return float(self._circuit(pnp.array(features, requires_grad=False), self.params))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(class = 1) for each row of X."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        out = np.empty(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            z = self._logit_z(X[i])
            out[i] = (1.0 - z) / 2.0
        return np.clip(out, 0.0, 1.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(np.int64)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == np.asarray(y)).mean())

    # -- training --------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 3,
        batch_size: int = 24,
        verbose: bool = False,
    ) -> dict:
        """Minimise BCE on (X, y); returns the loss history."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n = X.shape[0]
        rng = np.random.default_rng(self.seed + 99)
        opt = qml.optimize.AdamOptimizer(self.learning_rate)
        history: list[float] = []

        for _epoch in range(int(epochs)):
            order = rng.permutation(n)
            losses = []
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                xb, yb = X[idx], y[idx]

                def batch_loss(weights):
                    total = 0.0
                    for i in range(xb.shape[0]):
                        z = self._circuit(pnp.array(xb[i], requires_grad=False), weights)
                        p = _EPS + (1.0 - 2.0 * _EPS) * (1.0 - z) / 2.0
                        total = total + yb[i] * pnp.log(p) + (1.0 - yb[i]) * pnp.log(1.0 - p)
                    return -total / xb.shape[0]

                self.params = opt.step(batch_loss, self.params)
                losses.append(float(batch_loss(self.params)))
            history.append(float(np.mean(losses)))
            if verbose:
                print(f"    epoch loss={history[-1]:.4f}")
        return {"loss_history": history}

    # -- federated parameter plumbing -----------------------------------------
    def get_param_vector(self) -> np.ndarray:
        return np.array(self.params, dtype=np.float64).ravel()

    def set_param_vector(self, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=np.float64)
        expected = self.n_layers * self.n_qubits * 3
        if vec.size != expected:
            raise ValueError(f"expected {expected} parameters, got {vec.size}")
        self.params = pnp.array(vec.reshape(self.n_layers, self.n_qubits, 3), requires_grad=True)

    # -- serialization -----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "params": np.array(self.params, dtype=np.float64).tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VQC":
        model = cls(
            n_qubits=data["n_qubits"],
            n_layers=data["n_layers"],
            learning_rate=data.get("learning_rate", 0.06),
            seed=data.get("seed", 7),
        )
        model.set_param_vector(np.asarray(data["params"], dtype=np.float64).ravel())
        return model
