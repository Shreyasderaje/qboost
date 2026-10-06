"""Variational quantum classifier behaviour."""

import numpy as np

from backend.quantum.encoding import FeatureEncoder
from backend.quantum.vqc import VQC


def _blobs(n=240, seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(-1.1, 0.9, (n // 2, 6))
    X1 = rng.normal(+1.1, 0.9, (n // 2, 6))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])
    idx = rng.permutation(n)
    return X[idx], y[idx]


def test_probabilities_are_valid():
    X, y = _blobs()
    enc = FeatureEncoder(n_components=4).fit(X)
    vqc = VQC(n_qubits=4, n_layers=2, seed=1)
    p = vqc.predict_proba(enc.transform(X[:20]))
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_training_learns_the_task():
    X, y = _blobs(300, seed=5)
    enc = FeatureEncoder(n_components=4).fit(X)
    A = enc.transform(X)
    vqc = VQC(n_qubits=4, n_layers=2, learning_rate=0.12, seed=1)
    before = vqc.score(A, y)
    vqc.fit(A, y, epochs=2, batch_size=24)
    after = vqc.score(A, y)
    assert after > 0.8
    assert after > before


def test_param_vector_roundtrip():
    X, y = _blobs()
    enc = FeatureEncoder(n_components=4).fit(X)
    A = enc.transform(X)
    vqc = VQC(n_qubits=4, n_layers=3, seed=2)
    vec = vqc.get_param_vector()
    assert vec.shape == (3 * 4 * 3,)
    clone = VQC.from_dict(vqc.to_dict())
    assert np.allclose(clone.get_param_vector(), vec)
    assert np.allclose(clone.predict_proba(A[:5]), vqc.predict_proba(A[:5]))


def test_encoder_roundtrip_and_shapes():
    X, _ = _blobs(120)
    enc = FeatureEncoder(n_components=4).fit(X)
    A = enc.transform(X)
    assert A.shape == (120, 4)
    assert A.min() >= 0.15 * 3.14159 - 1e-9
    clone = FeatureEncoder.from_dict(enc.to_dict())
    assert np.allclose(clone.transform(X), A)
