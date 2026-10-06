# QBoost Architecture

## System overview

```
        ┌───────────────────────────── clinic process ─────────────────────┐
        │                                                                  │
        │   local patient data (never leaves)                              │
        │        │                                                         │
        │        ▼                                                         │
        │   [FeatureEncoder]──► [VQC: 4-qubit variational circuit]         │
        │                              │ local epochs (Adam, adjoint diff) │
        │                              ▼                                   │
        │                    Δθ = params_new − params_global               │
        │                              │                                   │
        │              clip ‖Δθ‖ ≤ C · gaussian noise (DP)                 │
        │                              │                                   │
        │        ┌────────────┬───────────┴──────────┐                        │
        │        ▼            ▼                       ▼                      │
        │   metrics    JL sketch (32-dim)        pairwise mask              │
        │  (acc,loss)  of Δθ                     ±PRG (Bonawitz)            │
        │        └────────────┴───────────┬──────────┘                        │
        │                                 ▼                                  │
        │                    Fernet envelope (AES-128 + HMAC)               │
        └─────────────────────────────────│─────────────────────────────────┘
                                          │ HTTPS (envelope in demo)
                                          ▼
        ┌────────────────────────── coordinator ────────────────────────────┐
        │                                                                   │
        │  collect phase-1 signals ──► QUBO  min −q̃ᵢxᵢ − κ sᵢⱼ xᵢxⱼ        │
        │                                   └──► QAOA (p=2) ──► cohort      │
        │  collect phase-2 updates (cohort only) ──► quality-weighted mean  │
        │                                                                   │
        │  evaluate on coordinator-owned benchmark ──► new global bundle    │
        └─────────────────────────────────│─────────────────────────────────┘
                                          │
                              global model bundle (JSON)
                                          ▼
        ┌────────────────────────── clinic PWA ─────────────────────────────┐
        │   React console: feature input → risk gauge + factor attribution │
        └───────────────────────────────────────────────────────────────────┘
```

## Module map

| Path | Responsibility |
| --- | --- |
| `backend/quantum/vqc.py` | Variational quantum classifier: angle embedding + StronglyEntanglingLayers, adjoint-differentiated training, JSON serialization |
| `backend/quantum/encoding.py` | PCA compression + min/max angle scaling; travels inside the model bundle |
| `backend/quantum/qaoa.py` | QUBO→Ising mapping, p-layer QAOA on `default.qubit`, exact brute-force verifier |
| `backend/federated/aggregator.py` | Cohort QUBO construction (quality × sketch-trust + coherence), QAOA selection, cohort floor |
| `backend/federated/client.py` | Clinic node: local training, stratified validation, DP clipping/noise, sketching, masking, envelope sealing |
| `backend/federated/server.py` | Round protocol, idempotent cohort selection/aggregation, weighted aggregation, artifact registry, event log |
| `backend/federated/crypto.py` | Fernet envelope seal/unseal (simulated mTLS) |
| `backend/federated/dp.py` | Update clipping, Gaussian mechanism, basic-composition ε accountant |
| `backend/federated/secureagg.py` | HMAC-based pairwise masks (cancel exactly in aggregate), JL random projections |
| `backend/data/synth.py` | Class-conditional synthetic generator with site shift + Dirichlet non-IID partitions |
| `backend/data/loaders.py` | Adapters for PIMA CSV and image folders (real-data drop-ins) |
| `backend/simulation/manager.py` | In-process federation runner (dashboard + CLI) |
| `backend/api/main.py` | FastAPI: inference, model registry, simulation control, manual federation sessions |
| `scripts/run_federation.py` | One-command federated training from the terminal |
| `scripts/benchmark.py` | Centralized / FedAvg / QAOA / poisoned comparisons → `docs/results/` |
| `scripts/clinic_simulator.py` | Independent clinic process over live HTTP (multi-process demo) |

## Round protocol

1. **Dispatch** — coordinator publishes the round bundle: global parameters, feature encoder, round seed, total sample count.
2. **Local training** — each clinic runs a few Adam epochs on its own patients, computes a stratified local validation, forms Δθ.
3. **Privacy pipeline** — clip ‖Δθ‖ ≤ C, add Gaussian noise (σ·C total), project to a 32-dim sketch via the round's shared JL matrix.
4. **Phase-1 submit** — sealed {metrics, sketch, n_samples, ‖Δθ‖}. The server learns aggregate statistics only.
5. **Cohort selection** — QAOA minimizes the QUBO over sketch coherence and trust-discounted quality; a floor of ⌈n/2⌉ clinics is enforced post-hoc. (FedAvg strategy skips selection: every clinic participates through a pairwise-masked secure sum.)
6. **Phase-2 submit** — cohort clinics reveal their DP updates; the server combines them with weights ∝ nᵢ·(0.5 + qualityᵢ).
7. **Evaluation** — the coordinator benchmarks the new global model on its own held-out reference set and logs the round record.

## Design decisions

- **Why two phase submissions?** Secure aggregation (masks that cancel only in sums) and QAOA weighted aggregation (which needs per-client signals) are in tension. Phase-1 resolves it: selection decisions use only metrics + sketches; raw DP updates are revealed only for the selected cohort, and only after noise has already been applied.
- **Why trust-discounted quality?** Self-reported validation metrics are spoofable. Discounting them by the client's sketch coherence (a behavioral signal the client cannot easily fake while running gradient ascent) makes the QUBO robust to liars.
- **Why masked FedAvg pre-scales by nᵢ/N?** The server can only sum masked updates. Publishing N in the round bundle lets each client pre-multiply by nᵢ/N, so the cancelled sum is exactly the canonical sample-weighted FedAvg average.
- **Why JSON artifacts?** Model bundles (encoder + params + metadata) stay inspectable, diffable, and signable — no pickle, no hidden state.

## Scale boundaries of the MVP

- 2–8 clinics, ≤ 12 rounds per run (simulation manager caps; HTTP sessions uncapped).
- QUBO up to ~12 variables stays comfortably in QAOA territory; beyond that `select_cohort` can fall back to the exact solver.
- Synthetic latent features by default; real datasets plug in through `backend/data/loaders.py`.
