# QBoost — Research Notes & Design Rationale

This document records the research that shaped QBoost and the reasoning behind
each design decision, with references. It is written to be audited: every claim
about the system corresponds to code in this repository.

---

## 1. The problem space

Three disease burdens dominate the rural settings QBoost targets:

| Burden | Scale | Source |
| --- | --- | --- |
| Tuberculosis | India accounts for ~25% of new TB cases worldwide — the largest single-country share | WHO Global Tuberculosis Report 2025 |
| Malaria | India carries ~66% of malaria cases in the WHO South-East Asia region | WHO World Malaria Report |
| Diabetes | ≈90 million adults in India live with diabetes; large shares remain undiagnosed | IDF Diabetes Atlas, 11th ed. (2025) |

AI screening for all three exists in research settings (CNNs for chest X-rays
and blood-smear microscopy, risk models for metabolic panels), but deployment
collides with two walls:

1. **Data gravity.** Training and improving models classically requires
   pooling patient data. Health-data-protection regimes (India's DPDP Act,
   GDPR-style laws elsewhere) and research ethics make centralized pools of
   rural patient data a non-starter.
2. **Compute gravity.** Rural clinics have no GPU infrastructure; even
   hosting a large model for inference is unrealistic.

Federated learning answers the first wall by moving computation to the data;
QBoost's contribution is making the per-client computation small enough to fit
the second wall — a quantum circuit that trains in minutes on edge-class
hardware — while strengthening aggregation against realistic adversaries.

## 2. Quantum machine learning layer

### 2.1 Variational quantum classifiers

The variational quantum classifier (VQC) is the workhorse of near-term QML:
encode data into a parameterized circuit and optimize circuit parameters
classically by evaluating expectation values [Havlíček 2019; Cerezo 2021].
QBoost uses:

- **Encoding** — clinical features are PCA-compressed to 4 dimensions and
  min-max scaled into an angle window, then loaded with RY rotations
  (angle encoding). PCA is fitted once on the coordinator's reference
  distribution and travels inside the model bundle, so every site encodes
  identically without seeing each other's data.
- **Ansatz** — 3 layers of StronglyEntanglingLayers (36 parameters):
  single-qubit rotations interleaved with a ring of CNOT entanglers,
  expressive enough for the screening tasks while remaining trainable
  (no barren-plateau regime at this width/depth).
- **Readout** — ⟨Z₀⟩ mapped to P(class = 1) = (1 − ⟨Z₀⟩)/2.
- **Gradients** — adjoint differentiation: exactly two circuit executions per
  optimization step regardless of parameter count [Jones & Gacon 2023], the
  cheapest hardware-compatible rule.
- **Optimization** — Adam on mini-batches of the clinic's own records.

On the synthetic benchmark tasks a single 4-qubit circuit reaches 0.86–0.94
centralized accuracy (see `docs/results/benchmarks.json`), confirming the
tasks retain signal under 4-dim compression.

### 2.2 QAOA for aggregation

The Quantum Approximate Optimization Algorithm [Farhi 2014] relaxes a
combinatorial objective onto a parameterized circuit. QBoost encodes the
*cohort selection* problem — which clinics should influence the next global
model — as a QUBO:

```
min_{x∈{0,1}ⁿ}   −Σ q̃ᵢ xᵢ  −  κ Σ_{i<j} sᵢⱼ xᵢ xⱼ  +  λ Σ xᵢ
```

- `q̃ᵢ` — trust-discounted quality of clinic *i* (below),
- `sᵢⱼ` — cosine similarity between clients' Johnson–Lindenstrauss update
  sketches: coherent updates (same descent direction) earn a joint reward,
  anti-correlated updates (the signature of poisoned or diverged training)
  are penalized,
- `λ` — a small inclusion bonus; a post-QAOA floor keeps at least ⌈n/2⌉
  clinics so the model never freezes.

The QUBO maps to an Ising Hamiltonian with x = (1 − z)/2 and is embedded in a
p = 2 QAOA circuit (RZ/RZZ-style cost layer, RX mixer) on the statevector
simulator; γ, β are optimized with Adam and the top-probability basis states
are scored exactly. The repository's test suite verifies the QAOA solution
against exhaustive search on random QUBOs — the quantum relaxation is
certified, not assumed.

**Why QAOA here at all?** Three defensible reasons: (1) the objective is
genuinely quadratic — robust-aggregation selection is NP-hard in general
[Krum; Byzantine-tolerant SGD literature], and QAOA is the natural NISQ
relaxation; (2) the same object is solved on cloud QPUs today, so the
simulator-to-hardware upgrade path is one device string; (3) even when QAOA's
solution equals a classical solver's (as on 4–8 clinics it usually will), the
*formulation* — trust-discounted, coherence-aware cohort selection — is the
contribution, and the circuit is tiny enough to audit end-to-end.

### 2.3 Trust-discounted quality (spoof-resistant signals)

Clients self-report validation accuracy/loss; a malicious client can lie.
QBoost discounts reported quality by a behavioral factor:

```
trustᵢ = (1 + mean_cosine(sketchᵢ, sketch_{-i})) / 2,      q̃ᵢ = reportedᵢ · trustᵢ
```

A gradient-ascent attacker anti-correlates with every honest update and
therefore cannot buy cohort membership with inflated metrics. In the poisoned
benchmarks, the attacker is excluded in every round while FedAvg absorbs its
update and degrades.

## 3. Privacy engineering

Four independent layers, composed (threat model: curious coordinator,
compromised clinic, network observer):

1. **Envelope encryption** — every payload is sealed with a per-clinic
   Fernet key (AES-128-CBC + HMAC-SHA256) issued at registration. This stands
   in for mutual TLS; authentication failures are indistinguishable from
   tampering.
2. **Pairwise masking** — Bonawitz-style secure aggregation [Bonawitz 2017]:
   each client adds deterministic PRG masks, +mᵢⱼ for the lower clinic id and
   −mᵢⱼ for the higher. Masks cancel exactly in the sum (verified to 1e-13 in
   tests), so the coordinator learns only aggregates. In FedAvg mode the
   server provably cannot see any individual update.
3. **Update-level differential privacy** — DP-SGD-style [Abadi 2016]:
   clip ‖Δθ‖ ≤ C, add Gaussian noise of total norm σ·C. Basic-composition ε
   is reported per deployment; the plumbing accepts an RDP accountant swap.
4. **Signal minimization** — only validation statistics (aggregate
   statistics, standard in FL) and a 32-dim JL projection of the already-noised
   update ever leave a clinic. The sketch reveals coherence, not parameters.

**The two-phase trick.** Secure aggregation and weighted aggregation are in
tension: masks cancel only in plain sums, but QAOA weighting needs per-client
signals. QBoost resolves this by splitting each round: phase-1 shares only
metrics + sketches (enough for cohort selection); phase-2 reveals DP-noised
updates *only for the selected cohort*. Privacy loss shrinks with the cohort,
and excluded clinics' updates never leave the device at all.

**n-weighted FedAvg under masking.** Canonical FedAvg weights clients by
sample count, but a masked sum only supports uniform addition. Fix: the
coordinator publishes N = Σnᵢ in the round bundle; each client pre-scales its
masked update by nᵢ/N. The cancelled sum is then exactly the weighted average.

## 4. Federated setup and evaluation

- **Non-IID realism** — clinics receive Dirichlet(α = 0.6) label partitions
  plus per-site feature shifts (different devices/calibration), following
  [Hsu 2019]'s partitioning methodology.
- **Evaluation honesty** — the coordinator evaluates on its own held-out
  reference set that no clinic ever sees; client metrics are computed on local
  stratified holdouts with small-clinic fallbacks.
- **Adversary** — the poisoned configuration makes the last clinic run
  gradient ascent (maximize loss) while reporting fabricated metrics — a
  stronger attack than label-flip, chosen because it defeats naive
  metric-based trust.

## 5. Results snapshot

Regenerate with `python scripts/benchmark.py` (~20 min, CPU). Numbers land in
`docs/results/benchmarks.json` and render on the `/research` page.

Observed behavior of the shipped code: QAOA aggregation climbs monotonically
under DP + non-IID and excludes the poisoned clinic in every round; vanilla
FedAvg oscillates under the same conditions and absorbs the poisoned update
with a measurable accuracy drop. Centralized training on pooled data is the
upper reference line.

## 6. Limitations (stated plainly)

- Synthetic latent features, not real radiographs: the MVP validates the
  *protocol*, not clinical performance. Real-data adapters are provided; the
  honest path is pretrained CNN embeddings → same pipeline.
- Differential-privacy accounting is the loose basic-composition bound; ε
  values reported are upper bounds, useful for plumbing, not for formal
  deployment claims.
- QAOA runs on a statevector simulator; shot noise and hardware decoherence
  are future work (the p=2, ≤10-qubit regime is exactly where current cloud
  QPUs operate).
- The clinic console is a research demonstration and is **not a medical
  device**; nothing here is certified for clinical decision-making.

## 7. References

1. Havlíček, V. et al. "Supervised learning with quantum-enhanced feature spaces." *Nature* 567 (2019).
2. Cerezo, M. et al. "Variational quantum algorithms." *Nature Reviews Physics* 3 (2021).
3. Farhi, E., Goldstone, J., Gutmann, S. "A Quantum Approximate Optimization Algorithm." arXiv:1411.4028 (2014).
4. Jones, T., Gacon, N. "Efficient calculation of gradients in classical simulations of variational quantum algorithms." arXiv:2009.02823 (2020).
5. McMahan, B. et al. "Communication-efficient learning of deep networks from decentralized data." AISTATS (2017).
6. Bonawitz, K. et al. "Practical secure aggregation for privacy-preserving machine learning." ACM CCS (2017).
7. Abadi, M. et al. "Deep learning with differential privacy." ACM CCS (2016).
8. Blanchard, P. et al. "Machine learning with adversaries: Byzantine tolerant gradient descent." NeurIPS (2017).
9. Hsu, T.-M. H., Qi, H., Brown, M. "Measuring the effects of non-identical data distribution in federated learning." arXiv:1909.06335 (2019).
10. Nguyen, L. et al. "Quantum federated learning: a comprehensive survey." arXiv:2508.15998 (2025).
11. Rajaraman, S. et al. "Pre-trained convolutional neural networks as feature extractors for malaria parasite detection in cell images." PeerJ (2018).
12. Wang, X. et al. "ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification." CVPR (2017).
13. WHO. *Global Tuberculosis Report 2025*. World Health Organization.
14. WHO. *World Malaria Report*. World Health Organization.
15. IDF. *IDF Diabetes Atlas*, 11th edition (2025). International Diabetes Federation.
