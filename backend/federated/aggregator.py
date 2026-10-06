"""QAOA-based aggregation cohort selection.

The coordinator never sees raw patient data, but clinics do publish two
low-dimensional signals each round:

- *quality signals*: local validation accuracy/loss and sample count
  (aggregate statistics, standard practice in federated learning),
- *coherence sketches*: a 32-dim Johnson-Lindenstrauss random projection of
  the differentially-private update, from which pairwise cosine similarity
  between update directions can be estimated.

The selection problem is encoded as a QUBO: selecting clinic i earns its
normalized quality, and selecting a *coherent pair* (i, j) earns their
similarity, while anti-correlated pairs (the signature of a poisoned or
faulty update) are penalised.  QAOA finds low-energy assignments; a
post-hoc floor guarantees a minimum cohort size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backend.quantum.qaoa import solve_qubo_qaoa


@dataclass
class ClientSummary:
    cid: str
    name: str
    n_samples: int
    val_acc: float
    val_loss: float
    update_norm: float
    sketch: np.ndarray
    selected: bool = False
    quality: float = 0.0
    mean_similarity: float = 0.0


@dataclass
class SelectionResult:
    selected: list[str]
    mask: list[bool]
    quality: list[float]
    similarity: list[list[float]]
    qaoa: dict | None = None
    qubo: dict = field(default_factory=dict)


def _normalize(values: list[float]) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-12:
        return np.ones_like(v) * 0.5
    return (v - lo) / (hi - lo)


def _cosine_matrix(sketches: list[np.ndarray]) -> np.ndarray:
    S = np.vstack([np.asarray(s, dtype=np.float64) for s in sketches])
    norms = np.linalg.norm(S, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    Sn = S / norms
    return np.clip(Sn @ Sn.T, -1.0, 1.0)


def build_aggregation_qubo(
    summaries: list[ClientSummary],
    quality_weight: float = 1.0,
    coherence_weight: float = 1.0,
    inclusion_bonus: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Assemble the QUBO  min x'Qx + c'x  over cohort indicators x.

    Self-reported quality is discounted by a *behavioral trust* factor
    derived from the sketch coherence of each client: a clinic whose update
    direction anti-correlates with the rest of the cohort (the signature of a
    poisoned or faulty update) cannot buy its way in with inflated metrics.
    """
    n = len(summaries)
    acc_n = _normalize([s.val_acc for s in summaries])
    loss_inv = _normalize([-s.val_loss for s in summaries])
    size_n = _normalize([np.log1p(s.n_samples) for s in summaries])
    sims = _cosine_matrix([s.sketch for s in summaries])

    reported = 0.55 * acc_n + 0.20 * loss_inv + 0.25 * size_n
    trust = np.clip((1.0 + (sims.sum(axis=1) - np.diag(sims)) / max(1, n - 1)) / 2.0, 0.0, 1.0)
    quality = reported * trust

    Q = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = Q[j, i] = -coherence_weight * sims[i, j] / 2.0

    c = -(quality_weight * quality + inclusion_bonus)

    for i, (s, q) in enumerate(zip(summaries, quality)):
        s.quality = float(q)
        s.mean_similarity = float(np.mean(np.delete(sims[:, i], i)))

    meta = {
        "cids": [s.cid for s in summaries],
        "names": [s.name for s in summaries],
        "quality": quality.tolist(),
        "similarity": np.round(sims, 4).tolist(),
    }
    return Q, c, meta


def _enforce_floor(mask: np.ndarray, quality: np.ndarray, floor: int) -> np.ndarray:
    """Greedily add the best-quality unselected clinics until the floor holds."""
    mask = mask.copy()
    order = np.argsort(-quality)
    for idx in order:
        if mask.sum() >= floor:
            break
        mask[idx] = True
    return mask


def select_cohort(
    summaries: list[ClientSummary],
    strategy: str = "qaoa",
    qaoa_p: int = 2,
    qaoa_steps: int = 120,
    seed: int = 7,
    min_cohort_fraction: float = 0.5,
    quality_weight: float = 1.0,
    coherence_weight: float = 1.0,
    inclusion_bonus: float = 0.15,
) -> SelectionResult:
    """Pick the aggregation cohort.

    ``strategy="fedavg"`` returns every clinic (the classical baseline);
    ``strategy="qaoa"`` solves the cohort QUBO with QAOA.
    """
    n = len(summaries)
    if strategy != "qaoa":
        return SelectionResult(
            selected=[s.cid for s in summaries],
            mask=[True] * n,
            quality=[s.quality for s in summaries],
            similarity=[],
        )

    Q, c, meta = build_aggregation_qubo(
        summaries,
        quality_weight=quality_weight,
        coherence_weight=coherence_weight,
        inclusion_bonus=inclusion_bonus,
    )
    result = solve_qubo_qaoa(Q, c, p=qaoa_p, steps=qaoa_steps, seed=seed)
    mask = result["solution"].copy()

    floor = int(np.ceil(min_cohort_fraction * n))
    mask = _enforce_floor(mask, np.asarray(meta["quality"]), floor)

    quality = np.asarray(meta["quality"], dtype=np.float64)
    for i, s in enumerate(summaries):
        s.selected = bool(mask[i])

    return SelectionResult(
        selected=[s.cid for s, m in zip(summaries, mask) if m],
        mask=[bool(m) for m in mask],
        quality=quality.tolist(),
        similarity=meta["similarity"],
        qaoa={k: v for k, v in result.items() if k != "solution"},
        qubo=meta,
    )
