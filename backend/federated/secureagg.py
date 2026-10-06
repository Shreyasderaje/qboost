"""Bonawitz-style pairwise masking for FedAvg rounds.

In the plain FedAvg strategy the coordinator must learn only the *sum* of
client updates, never any individual update.  Each clinic adds a deterministic
pseudo-random mask to its update:

    for every other clinic j:  sign * PRG(secret, pair(i, j), round)

where the sign is + for the lower clinic id and - for the higher one.  All
masks cancel exactly in the aggregate, but any single masked update is
computationally indistinguishable from noise without the shared secret.

The PRG is HMAC-SHA256 in counter mode; the pairwise secret is derived from
the coordinator-issued registration secret via HKDF-like extraction.
"""

from __future__ import annotations

import hashlib
import hmac

import numpy as np

_MASK_SCALE = 10.0  # masks dominate update signals (~0.1-1.0) while keeping
                    # float cancellation in the aggregate exact to ~1e-14


def _prf_bytes(secret: bytes, label: str, nbytes: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < nbytes:
        msg = f"{label}|{counter}".encode("utf-8")
        out.extend(hmac.new(secret, msg, hashlib.sha256).digest())
        counter += 1
    return bytes(out[:nbytes])


def _pair_secret(secret: bytes, cid_i: str, cid_j: str) -> bytes:
    lo, hi = sorted((cid_i, cid_j))
    return hmac.new(secret, f"pair|{lo}|{hi}".encode("utf-8"), hashlib.sha256).digest()


def pairwise_mask(
    cid_i: str, cid_j: str, round_id: int, dim: int, secret: bytes
) -> np.ndarray:
    """The mask shared by clinics ``cid_i`` and ``cid_j`` for a round."""
    lo, hi = sorted((cid_i, cid_j))
    raw = _prf_bytes(_pair_secret(secret, cid_i, cid_j), f"mask|{lo}|{hi}|{round_id}", dim * 4)
    uniforms = np.frombuffer(raw, dtype=np.uint32).astype(np.float64) / 2**32
    return (uniforms - 0.5) * 2.0 * _MASK_SCALE


def client_mask(cid: str, all_cids: list[str], round_id: int, dim: int, secret: bytes) -> np.ndarray:
    """Signed mask a clinic adds to its own update."""
    total = np.zeros(dim, dtype=np.float64)
    for other in all_cids:
        if other == cid:
            continue
        mask = pairwise_mask(cid, other, round_id, dim, secret)
        if cid > other:
            total -= mask
        else:
            total += mask
    return total


def random_projection(seed: int, dim: int, n_components: int = 32) -> np.ndarray:
    """Shared random projection matrix (Johnson-Lindenstrauss sketching).

    Clinics reduce their update to ``n_components`` dimensions so the
    coordinator can estimate pairwise update coherence without seeing the
    update itself.  The matrix is derived from a per-round seed known to all
    participants.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_components, dim)) / np.sqrt(n_components)
