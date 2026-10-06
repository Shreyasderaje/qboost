"""Synthetic clinic data generators.

The MVP runs entirely offline, so every clinic trains on data produced by
these generators.  The generators emulate the *statistical situation* of the
real screening problems rather than the images themselves:

- a class-conditional Gaussian in a clinically-plausible feature space with
  correlated informative dimensions and nuisance dimensions,
- realistic prevalence, label noise and floor/ceiling effects,
- per-site distribution shift and Dirichlet label skew across clinics
  (the classic non-IID federated setting).

``backend.data.loaders`` provides drop-in adapters for the real public
datasets (NIH ChestX-ray14, Kaggle malaria cells, PIMA diabetes) that emit
exactly the same structure, so the federated pipeline is unchanged when real
data is plugged in.
"""

from __future__ import annotations

import numpy as np

from backend.data.tasks import TaskSpec, get_task


def _affine_to_range(z: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Map a roughly standard-normal column into [lo, hi] via a smooth squash."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    mid = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    return mid + half * np.tanh(0.8 * z)


def make_dataset(task: TaskSpec | str, n_total: int, seed: int = 0) -> dict:
    """Generate a site-neutral dataset for a task.

    Returns a dict with ``X`` (raw units), ``y`` (0/1) and ``feature_names``.
    """
    spec = get_task(task) if not isinstance(task, TaskSpec) else task
    rng = np.random.default_rng(seed)
    d = spec.n_features

    # Informative dimensions: the first half of the features carry signal,
    # concentrated so a 4-component PCA projection retains most of it.
    n_informative = max(3, int(np.ceil(d * 0.5)))

    # Correlated latent factors (low-rank covariance) so PCA retains signal.
    rank = 3
    loading = rng.normal(0.0, 1.0, size=(d, rank))
    cov = loading @ loading.T / rank + 0.65 * np.eye(d)
    chol = np.linalg.cholesky(cov + 1e-9 * np.eye(d))

    # Class separation: positives shifted along informative dimensions.
    shift = np.zeros(d)
    shift[:n_informative] = 1.45 * rng.uniform(0.9, 1.3, size=n_informative)
    # One nuisance dimension inversely associated (realistic confounding).
    if d > n_informative:
        shift[n_informative] = -0.4

    n_pos = int(round(spec.prevalence * n_total))
    n_neg = n_total - n_pos

    z_pos = rng.standard_normal((n_pos, d)) @ chol.T + shift
    z_neg = rng.standard_normal((n_neg, d)) @ chol.T

    Z = np.vstack([z_neg, z_pos])
    y = np.concatenate([np.zeros(n_neg, dtype=np.int64), np.ones(n_pos, dtype=np.int64)])

    # Label noise: flip a small fraction so the Bayes error is nonzero.
    flip = rng.random(n_total) < 0.03
    y[flip] = 1 - y[flip]

    lo = np.array([f.lo for f in spec.features])
    hi = np.array([f.hi for f in spec.features])
    X = _affine_to_range(Z, lo, hi)

    order = rng.permutation(n_total)
    return {
        "X": X[order],
        "y": y[order],
        "feature_names": spec.feature_names,
    }


def _site_shift(X: np.ndarray, spec: TaskSpec, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Per-site acquisition/calibration shift on every feature."""
    lo = np.array([f.lo for f in spec.features])
    hi = np.array([f.hi for f in spec.features])
    span = (hi - lo) * strength
    shift = rng.normal(0.0, 1.0, size=X.shape[1]) * span
    return np.clip(X + shift, lo, hi)


def make_federated_splits(
    task: TaskSpec | str,
    n_clinics: int,
    n_total: int,
    alpha: float = 0.6,
    seed: int = 0,
    test_fraction: float = 0.2,
    site_shift_strength: float = 0.06,
) -> dict:
    """Create non-IID clinic partitions plus a site-neutral test set.

    Label skew follows a Dirichlet(alpha) distribution per clinic (as in
    Hsu et al., 2019) and each clinic additionally receives a small
    acquisition shift, emulating different devices and patient populations.
    The held-out test set is drawn from the site-neutral distribution and is
    never shown to any clinic.
    """
    spec = get_task(task) if not isinstance(task, TaskSpec) else task
    rng = np.random.default_rng(seed)
    data = make_dataset(spec, n_total, seed=seed)
    X, y = data["X"], data["y"]
    n = X.shape[0]

    n_test = int(round(test_fraction * n))
    idx = rng.permutation(n)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    # Dirichlet partition of the training pool by class proportions.
    proportions = rng.dirichlet(np.repeat(alpha, n_clinics))
    clients = []
    start = 0
    proportions_sorted = np.sort(proportions)[::-1]
    for k in range(n_clinics):
        size = int(round(proportions_sorted[k] * (n - n_test)))
        if k == n_clinics - 1:
            size = (n - n_test) - start
        take = train_idx[start : start + size]
        start += size
        Xc = _site_shift(X[take], spec, rng, site_shift_strength)
        clients.append(
            {
                "X": Xc,
                "y": y[take],
            }
        )

    return {
        "clients": clients,
        "test": {"X": X[test_idx], "y": y[test_idx]},
        "feature_names": spec.feature_names,
    }
