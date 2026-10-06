"""Central configuration for the QBoost platform.

All tunable defaults live here so that the CLI, the HTTP API and the
tests share one source of truth.
"""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
RESULTS_DIR = BASE_DIR / "docs" / "results"

# ---------------------------------------------------------------------------
# Quantum model defaults
# ---------------------------------------------------------------------------
N_QUBITS = 4
N_LAYERS = 3
LOCAL_EPOCHS = 3
BATCH_SIZE = 24
LEARNING_RATE = 0.12

# ---------------------------------------------------------------------------
# Federated defaults
# ---------------------------------------------------------------------------
N_CLINICS = 4
ROUNDS = 5
NONIID_ALPHA = 0.6

# Differential privacy (update-level DP-SGD style accounting)
DP_CLIP = 1.0
DP_SIGMA = 0.3
DP_DELTA = 1e-5

# ---------------------------------------------------------------------------
# QAOA aggregation defaults
# ---------------------------------------------------------------------------
QAOA_P = 2
QAOA_STEPS = 120
QUALITY_WEIGHT = 1.0
COHERENCE_WEIGHT = 0.5
MIN_COHORT_FRACTION = 0.5

SEED = 42


def ensure_dirs() -> None:
    """Create the output directories if they do not exist yet."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
