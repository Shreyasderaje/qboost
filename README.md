<div align="center">

# QBoost

**Quantum-enhanced federated learning for rural health diagnostics.**

Every clinic trains a 4-qubit variational circuit on its own patients.
Only noised, masked, sketch-compressed signals reach the coordinator,
where a QAOA optimizer decides which updates earn a place in the next
global model. Patient data never leaves the clinic.

[Quickstart](#quickstart) · [Architecture](#architecture) ·
[Benchmarks](#benchmarks) · [Research notes](docs/RESEARCH.md)

</div>

---

> **Research demonstration.** QBoost is a protocol prototype, not a medical
> device. The models shipped here are trained on synthetic latent features to
> validate the pipeline; nothing in this repository is certified for clinical
> use.

## Why

- **1.4 billion people** lack reliable access to specialist diagnosis, while the diseases that hit them hardest — tuberculosis (~25% of the global burden sits in India, WHO 2025), malaria (~66% of the WHO South-East Asia burden), and diabetes (~90M adults in India, IDF Atlas) — are exactly the ones AI screening handles well.
- Training those models classically means centralizing patient records: a non-starter under DPDP/GDPR-era rules and basic medical ethics.
- Rural clinics have no GPU compute, so even *running* big models is unrealistic.

QBoost resolves both constraints at once: the learning itself is quantum — small enough for a Raspberry-Pi-class device — and the aggregation protocol is mathematically incapable of exposing an individual update.

## How it works

| Step | What happens | Where |
| --- | --- | --- |
| 1 · Train locally | Each clinic runs a 4-qubit, 3-layer variational circuit (36 params, adjoint-differentiated Adam) on its own patients | clinic device |
| 2 · Seal & sketch | Updates are L2-clipped, Gaussian-noised (DP), reduced to a 32-dim Johnson–Lindenstrauss sketch, pairwise-masked (Bonawitz), and sealed in an AES envelope | clinic device |
| 3 · Aggregate with QAOA | The coordinator encodes cohort selection as a QUBO — trust-discounted quality × sketch coherence — and solves it with a 2-layer QAOA. Poisoned or diverged clinics are excluded by their anti-correlated sketches | coordinator |
| 4 · Screen at the edge | The improved global bundle installs back into the clinic PWA; nurses get a risk read with per-factor attribution | clinic device |

## Quickstart

Requires Python 3.10+ and Node 18+ (only to build the console).

```bash
# 1 · backend
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux

# 2 · train a federation from the terminal (~4 min, CPU only)
python scripts/run_federation.py --task diabetes --rounds 5

# 3 · serve the platform (API + clinic console)
python -m uvicorn backend.api.main:app --port 8000

# 4 · frontend (dev server with hot reload, http://localhost:5173)
cd frontend && npm install && npm run dev

# 5 · or a production build served by the API at http://localhost:8000
cd frontend && npm install && npm run build
```

Then open:

- **`/`** — overview
- **`/dashboard`** — launch federations, watch rounds, QAOA cohort decisions and the event stream live
- **`/diagnose`** — clinic console: screening input → risk gauge → factor attribution
- **`/network`** — clinic roster, live topology, security posture
- **`/research`** — protocol documentation and measured benchmarks

CLI options: `--task diabetes|tb|malaria`, `--strategy qaoa|fedavg`, `--poison`
(makes the last clinic a gradient-ascent attacker with fake metrics), `--no-dp`,
`--clinics`, `--rounds`, `--seed`.

Run the test suite (covers the QAOA-vs-brute-force equivalence, mask
cancellation, DP mechanics, end-to-end federations, and the API):

```bash
.venv/Scripts/python -m pytest
```

Regenerate the benchmark suite (~20 min):

```bash
python scripts/benchmark.py          # writes docs/results/benchmarks.json + chart
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full diagram and
module map. The short version:

```
clinic process                                coordinator
─────────────                                 ───────────
patient data (never leaves)
      │
VQC training (PennyLane, adjoint)             phase 1: metrics + JL sketches
      │                                   ──►       └── QUBO ──► QAOA ──► cohort
Δθ = clip ≤ C + gaussian noise (DP)           phase 2: cohort reveals DP updates
      │                                   ──►       └── quality-weighted mean
Fernet envelope + pairwise masks                        └── global model v(n+1)
```

Two aggregation strategies ship side by side:

- **QAOA** — cohort selection on sketch coherence + trust-discounted quality; only the selected cohort's DP updates are revealed; weights ∝ nᵢ·(0.5 + qualityᵢ).
- **FedAvg** — every clinic participates through a pairwise-masked secure sum; clients pre-scale by nᵢ/N so the cancelled sum is exactly the canonical sample-weighted average. The server never sees an individual update.

## Benchmarks

Full-fidelity runs of the shipped pipeline (4 clinics, Dirichlet non-IID
partitions with site shift, DP on). Regenerate with `python scripts/benchmark.py`;
the live table renders at `/research`.

| task | centralized | FedAvg | QAOA | FedAvg + poison | QAOA + poison | attacker excluded |
| --- | --- | --- | --- | --- | --- | --- |
| diabetes | 0.755 | 0.736 | **0.782** | 0.750 | **0.778** | 4/4 rounds |
| malaria | 0.880 | 0.801 | **0.861** | 0.755 | **0.857** | 4/4 rounds |
| tb | 0.865 | 0.764 | **0.857** | 0.810 | **0.861** | 4/4 rounds |

Observed behavior of the shipped code: QAOA aggregation beats FedAvg on every
task (+4.6 to +9.3 points under DP + non-IID) and comes within ~1–2 points of
centralized training *despite* differential privacy, pairwise masking and no
data leaving the clinics. Under a gradient-ascent attacker that inflates its
metrics, FedAvg loses up to 4.6 points while QAOA holds its accuracy and
excludes the attacker in every single round — its sketch anti-correlates with
honest updates, so the trust discount and coherence reward push it out of the
cohort.

## Repository layout

```
backend/
  quantum/        vqc.py · encoding.py · qaoa.py
  federated/      server.py · client.py · aggregator.py · crypto.py · dp.py · secureagg.py
  data/           tasks.py · synth.py · loaders.py (PIMA CSV + image-folder adapters)
  simulation/     manager.py (in-process federation runner)
  api/            main.py (FastAPI: inference, registry, simulation control, federation sessions)
scripts/          run_federation.py · benchmark.py · clinic_simulator.py
frontend/         React + Vite PWA (console, clinic app, network, research)
tests/            pytest suite: quantum, security, federation, API
docs/             RESEARCH.md · ARCHITECTURE.md · results/
```

## Multi-process demo (true distributed mode)

The dashboard runs every clinic in one process for convenience. To exercise the
real HTTP protocol with independent clinic processes:

```bash
# terminal 1 — coordinator
python -m uvicorn backend.api.main:app --port 8000

# terminal 2 — open a session
curl -X POST localhost:8000/api/federation/init \
     -H "content-type: application/json" -d "{\"task\":\"diabetes\",\"strategy\":\"qaoa\"}"

# terminals 3..6 — clinics (any clinic may trigger the collective steps; guards make it safe)
python scripts/clinic_simulator.py --session <session_id> --clinic clinic-1 --name "Phulbani CHC"
python scripts/clinic_simulator.py --session <session_id> --clinic clinic-2 --name "Bastar Rural DH"
```

## Scaling notes & roadmap

- Real data: swap the synthetic generator for `backend/data/loaders.py`
  (PIMA CSV, NIH ChestX-ray14, Kaggle malaria cells — pretrained CNN embeddings
  recommended for image tasks).
- Quantum hardware: the QUBO and the VQC are device-agnostic; moving to IBM
  Quantum / AWS Braket changes a device string, not the protocol.
- Deployment path: NHM / Ayushman Bharat integration, ABDM-compatible record
  pointers, on-device TFLite-class inference export.

## License

[MIT](LICENSE)
