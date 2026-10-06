"""Security primitives: envelope encryption, pairwise masking, DP."""

import numpy as np
import pytest

from backend.federated import crypto, dp, secureagg


def test_seal_unseal_roundtrip():
    key = crypto.issue_clinic_key()
    payload = {"cid": "clinic-1", "metrics": {"val_acc": 0.83}, "sketch": [0.1, -0.2]}
    token = crypto.seal(payload, key)
    assert crypto.unseal(token, key) == payload


def test_tampered_payload_rejected():
    key = crypto.issue_clinic_key()
    token = crypto.seal({"hello": "world"}, key)
    tampered = token[:-4] + ("AAAA" if token[-4:] != "AAAA" else "BBBB")
    with pytest.raises(ValueError):
        crypto.unseal(tampered, key)


def test_wrong_key_rejected():
    token = crypto.seal({"a": 1}, crypto.issue_clinic_key())
    with pytest.raises(ValueError):
        crypto.unseal(token, crypto.issue_clinic_key())


def test_pairwise_masks_cancel_in_aggregate():
    cids = ["clinic-1", "clinic-2", "clinic-3", "clinic-4", "clinic-5"]
    dim, secret, round_id = 36, b"coord-secret", 4
    rng = np.random.default_rng(0)
    deltas = {c: rng.normal(0, 0.3, dim) for c in cids}

    masked_sum = np.zeros(dim)
    for c in cids:
        masked_sum += deltas[c] + secureagg.client_mask(c, cids, round_id, dim, secret)

    plain_sum = sum(deltas.values())
    assert np.abs(masked_sum - plain_sum).max() < 1e-6


def test_single_masked_update_is_dominated_by_mask():
    cids = ["a", "b", "c"]
    dim, secret = 36, b"coord-secret"
    mask = secureagg.client_mask("a", cids, 1, dim, secret)
    assert np.abs(mask).mean() > 1.0  # update signals (~0.1) are invisible under it


def test_masks_are_deterministic_per_round():
    cids = ["a", "b"]
    m1 = secureagg.client_mask("a", cids, 7, 36, b"s")
    m2 = secureagg.client_mask("a", cids, 7, 36, b"s")
    m3 = secureagg.client_mask("a", cids, 8, 36, b"s")
    assert np.array_equal(m1, m2)
    assert not np.array_equal(m1, m3)


def test_dp_clipping_and_noise():
    rng = np.random.default_rng(0)
    v = rng.normal(0, 10, 100)  # norm ~100
    clipped = dp.clip_update(v, 1.0)
    assert np.linalg.norm(clipped) <= 1.0 + 1e-12
    small = dp.clip_update(np.full(10, 0.01), 1.0)
    assert np.allclose(small, np.full(10, 0.01))  # under threshold: untouched

    noisy = dp.add_gaussian_noise(np.zeros(400), 1.0, 0.5, rng)
    expected_norm = 0.5
    assert abs(np.linalg.norm(noisy) - expected_norm) < 0.15


def test_epsilon_grows_with_rounds_and_shrinks_with_sigma():
    assert dp.epsilon_bound(0.3, 5) > dp.epsilon_bound(0.3, 1)
    assert dp.epsilon_bound(0.3, 5) > dp.epsilon_bound(0.6, 5)
    assert dp.epsilon_bound(0.0, 5) == float("inf")
