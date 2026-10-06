"""QAOA solver for the federated aggregation QUBO.

The coordinator turns "which clinic updates should shape the next global
model?" into a binary optimisation problem

    min_{x in {0,1}^n}  x' Q x + c' x

where x_i = 1 means clinic i joins the aggregation cohort.  The QUBO is
translated into an Ising cost Hamiltonian and relaxed with a QAOA ansatz
(p layers of cost/mixing unitaries) executed on the simulator; the classical
angles are optimised by gradient descent and the highest-probability basis
states are scored exactly to pick the final cohort.

`solve_qubo_bruteforce` provides the exact optimum, used in the test suite to
verify the QAOA relaxation and as the fallback for >12 clinics.
"""

from __future__ import annotations

import itertools

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def qubo_to_ising(Q: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Map min x'Qx + c'x over x in {0,1} to min h'z + Σ_{i<j} J_ij z_i z_j
    over z in {-1,1}, using x_i = (1 - z_i) / 2."""
    Q = np.asarray(Q, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64).ravel()
    n = c.size
    if Q.shape != (n, n):
        raise ValueError("Q must be a square matrix matching c")

    h = -(Q.sum(axis=1) + Q.sum(axis=0)) / 4.0 - c / 2.0
    J = Q / 2.0  # strict-upper coefficients: z'Jz sums each pair once
    const = Q.sum() / 4.0 + np.trace(Q) / 4.0 + c.sum() / 2.0
    return h, J, const


def qubo_cost(x: np.ndarray, Q: np.ndarray, c: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    return float(x @ Q @ x + c @ x)


def solve_qubo_bruteforce(Q: np.ndarray, c: np.ndarray) -> dict:
    """Exact minimum by enumeration (feasible up to ~20 variables)."""
    Q = np.asarray(Q, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64).ravel()
    if not (np.isfinite(Q).all() and np.isfinite(c).all()):
        raise ValueError("QUBO coefficients must be finite")
    n = c.size
    best_x, best_e = None, np.inf
    for bits in itertools.product((0, 1), repeat=n):
        x = np.array(bits, dtype=np.float64)
        e = qubo_cost(x, Q, c)
        if e < best_e:
            best_e, best_x = e, x
    return {"solution": best_x.astype(bool), "energy": best_e}


def solve_qubo_qaoa(
    Q: np.ndarray,
    c: np.ndarray,
    p: int = 2,
    steps: int = 120,
    learning_rate: float = 0.1,
    seed: int = 7,
    top_k: int = 24,
) -> dict:
    """Solve the QUBO with a p-layer QAOA on the statevector simulator.

    Returns the selected cohort (boolean mask), the exact energy of that
    cohort, the optimised expectation value and the training history for
    dashboards.
    """
    Q = np.asarray(Q, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64).ravel()
    n = c.size
    h, J, const = qubo_to_ising(Q, c)

    dev = qml.device("default.qubit", wires=n)
    wires = list(range(n))
    gamma = pnp.array(np.random.default_rng(seed).uniform(0.2, 0.8, p), requires_grad=True)
    beta = pnp.array(np.random.default_rng(seed + 1).uniform(0.2, 0.8, p), requires_grad=True)

    def ansatz(gammas, betas):
        for w in wires:
            qml.Hadamard(w)
        for layer in range(p):
            for i in range(n):
                if abs(h[i]) > 1e-12:
                    qml.RZ(2.0 * gammas[layer] * h[i], wires=i)
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(J[i, j]) > 1e-12:
                        # exp(-i*theta*Z_i Z_j) via the standard CNOT-RZ-CNOT identity
                        qml.CNOT(wires=[i, j])
                        qml.RZ(2.0 * gammas[layer] * J[i, j], wires=j)
                        qml.CNOT(wires=[i, j])
            for i in range(n):
                qml.RX(2.0 * betas[layer], wires=i)

    terms: list[tuple[float, object]] = []
    for i in range(n):
        if abs(h[i]) > 1e-12:
            terms.append((float(h[i]), qml.PauliZ(i)))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-12:
                terms.append((float(J[i, j]), qml.PauliZ(i) @ qml.PauliZ(j)))

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def expectations(gammas, betas):
        ansatz(gammas, betas)
        return [qml.expval(obs) for _, obs in terms]

    def cost_expectation(gammas, betas):
        exps = expectations(gammas, betas)
        val = const
        for (coeff, _), e in zip(terms, exps):
            val = val + coeff * e
        return val

    @qml.qnode(dev, interface=None)
    def state_probs(gammas, betas):
        ansatz(gammas, betas)
        return qml.probs(wires=wires)

    opt = qml.optimize.AdamOptimizer(learning_rate)
    history = []
    for _ in range(int(steps)):
        gamma, beta = opt.step(cost_expectation, gamma, beta)
        history.append(float(cost_expectation(gamma, beta)))

    probs = state_probs(gamma, beta)
    # PennyLane orders basis states with wire 0 as the least significant bit,
    # so bit i of the state index is exactly x_i.
    order = np.argsort(probs)[::-1][: int(top_k)]
    best_x, best_e = None, np.inf
    for idx in order:
        x = np.array([(int(idx) >> i) & 1 for i in range(n)], dtype=np.float64)
        e = qubo_cost(x, Q, c)
        if e < best_e:
            best_e, best_x = e, x

    exact = solve_qubo_bruteforce(Q, c) if n <= 14 else {"energy": None, "solution": None}
    return {
        "solution": best_x.astype(bool),
        "energy": best_e,
        "expectation": history[-1],
        "history": [round(v, 6) for v in history],
        "prob_mass": float(probs[order].sum()),
        "optimum_energy": exact["energy"],
        "optimum_solution": None if exact["solution"] is None else [bool(v) for v in exact["solution"]],
        "p": int(p),
        "steps": int(steps),
        "gammas": np.array(gamma, dtype=float).tolist(),
        "betas": np.array(beta, dtype=float).tolist(),
    }
