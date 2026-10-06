"""Feature encoding for variational quantum classifiers.

Clinical feature vectors are compressed with PCA down to the qubit count and
then scaled into an angle window so they can be loaded onto the circuit with
RY rotations (angle encoding).  The encoder is fitted once on the coordinator
side and shipped to every clinic as part of the global model bundle, so all
sites interpret inputs identically.

Serialization is plain JSON (no pickle) so a model bundle can be inspected,
signed and diffed like any other artifact.
"""

from __future__ import annotations

import numpy as np

# Keep angles away from 0 and pi: extreme saturation makes gradients vanish.
_ANGLE_LOW = 0.15 * np.pi
_ANGLE_HIGH = 0.85 * np.pi


class FeatureEncoder:
    """PCA compression + min/max angle scaling for VQC inputs."""

    def __init__(self, n_components: int = 4) -> None:
        self.n_components = int(n_components)
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.x_min_: np.ndarray | None = None
        self.x_max_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    # -- fitting ------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "FeatureEncoder":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("FeatureEncoder expects a 2-D array")
        if X.shape[1] < self.n_components:
            raise ValueError(
                f"cannot compress {X.shape[1]} features to {self.n_components} components"
            )

        self.n_features_in_ = X.shape[1]
        self.mean_ = X.mean(axis=0)

        # Deterministic power-iteration-free PCA via SVD on the centred data.
        Xc = X - self.mean_
        # Full SVD on (n x d) with n >> d is cheap and perfectly stable.
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = vt[: self.n_components].copy()

        Z = Xc @ self.components_.T
        self.x_min_ = Z.min(axis=0)
        self.x_max_ = Z.max(axis=0)
        span = np.where(self.x_max_ - self.x_min_ < 1e-12, 1.0, self.x_max_ - self.x_min_)
        self.x_max_ = self.x_min_ + span
        return self

    # -- transform ----------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("FeatureEncoder must be fitted before transform")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} input features, got {X.shape[1]}"
            )
        Z = (X - self.mean_) @ self.components_.T
        t = (Z - self.x_min_) / (self.x_max_ - self.x_min_)
        t = np.clip(t, 0.0, 1.0)
        return _ANGLE_LOW + t * (_ANGLE_HIGH - _ANGLE_LOW)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    # -- serialization --------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "n_components": self.n_components,
            "mean": self.mean_.tolist(),
            "components": self.components_.tolist(),
            "x_min": self.x_min_.tolist(),
            "x_max": self.x_max_.tolist(),
            "n_features_in": int(self.n_features_in_),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureEncoder":
        enc = cls(n_components=data["n_components"])
        enc.mean_ = np.asarray(data["mean"], dtype=np.float64)
        enc.components_ = np.asarray(data["components"], dtype=np.float64)
        enc.x_min_ = np.asarray(data["x_min"], dtype=np.float64)
        enc.x_max_ = np.asarray(data["x_max"], dtype=np.float64)
        enc.n_features_in_ = int(data["n_features_in"])
        return enc
