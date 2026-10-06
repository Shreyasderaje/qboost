"""QAOA correctness: the variational solution must match the exact optimum."""

import numpy as np
import pytest

from backend.quantum.qaoa import qubo_cost, qubo_to_ising, solve_qubo_bruteforce, solve_qubo_qaoa


def _random_qubo(n, seed):
    rng = np.random.default_rng(seed)
    Q = rng.normal(0, 1, (n, n))
    Q = (Q + Q.T) / 2
    np.fill_diagonal(Q, 0)
    c = rng.normal(0, 1, n)
    return Q, c


def test_qubo_to_ising_matches_direct_evaluation():
    Q, c = _random_qubo(5, seed=3)
    h, J, const = qubo_to_ising(Q, c)
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.integers(0, 2, 5).astype(float)
        z = 1 - 2 * x  # x_i = (1 - z_i) / 2
        ising_exact = const + float(np.dot(h, z))
        for i in range(5):
            for j in range(i + 1, 5):
                ising_exact += J[i, j] * z[i] * z[j]
        assert abs(qubo_cost(x, Q, c) - ising_exact) < 1e-9


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_qaoa_finds_optimum_on_random_qubos(seed):
    Q, c = _random_qubo(4, seed=seed)
    exact = solve_qubo_bruteforce(Q, c)
    result = solve_qubo_qaoa(Q, c, p=2, steps=150, seed=seed)
    assert result["energy"] <= exact["energy"] + 1e-6
    assert abs(result["energy"] - exact["energy"]) < 0.05 or np.array_equal(result["solution"], exact["solution"])


def test_qaoa_history_and_outputs_are_wellformed():
    Q, c = _random_qubo(4, seed=9)
    result = solve_qubo_qaoa(Q, c, p=2, steps=40, seed=0)
    assert isinstance(result["solution"], np.ndarray)
    assert result["solution"].dtype == bool
    assert len(result["history"]) == 40
    assert 0.0 <= result["prob_mass"] <= 1.0
