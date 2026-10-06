"""Update-level differential privacy (DP-FLD, Gaussian mechanism).

Each clinic clips its parameter update to L2 norm C and adds Gaussian noise
scaled to C before anything leaves the device.  Combined with secure
aggregation this defends against both a curious coordinator and
reconstruction attacks on the published model.

The accountant below is intentionally the simple basic-composition bound
(the loosest useful estimate).  Production deployments should swap in an
RDP or moments accountant; the plumbing is identical.
"""

from __future__ import annotations

import math

import numpy as np


def clip_update(delta: np.ndarray, clip_norm: float) -> np.ndarray:
    """Scale ``delta`` down so its L2 norm is at most ``clip_norm``."""
    delta = np.asarray(delta, dtype=np.float64)
    norm = float(np.linalg.norm(delta))
    if norm <= clip_norm or norm == 0.0:
        return delta.copy()
    return delta * (clip_norm / norm)


def add_gaussian_noise(delta: np.ndarray, clip_norm: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise with total L2 magnitude ``sigma * clip_norm``."""
    d = delta.size
    per_coord = sigma * clip_norm / math.sqrt(d)
    return delta + rng.normal(0.0, per_coord, size=d)


def epsilon_bound(sigma: float, rounds: int, delta: float = 1e-5) -> float:
    """Basic-composition epsilon for ``rounds`` releases of one Gaussian
    mechanism with noise multiplier ``sigma`` (noise std = sigma * L2 sensitivity)."""
    if sigma <= 0:
        return float("inf")
    eps_one = math.sqrt(2.0 * math.log(1.25 / delta)) / sigma
    return round(eps_one * max(1, rounds), 3)
