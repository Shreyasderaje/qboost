import { useEffect, useState } from "react";
import { api } from "../lib/api.js";
import { Card, SectionHead } from "../components/ui.jsx";
import { IconBook, IconChart } from "../components/icons.jsx";

const references = [
  ["Havlíček et al. (2019)", "Supervised learning with quantum-enhanced feature spaces", "Nature 567"],
  ["Cerezo et al. (2021)", "Variational quantum algorithms", "Nature Reviews Physics 3"],
  ["Farhi, Goldstone, Gutmann (2014)", "A Quantum Approximate Optimization Algorithm", "arXiv:1411.4028"],
  ["McMahan et al. (2017)", "Communication-efficient learning of deep networks from decentralized data (FedAvg)", "AISTATS"],
  ["Bonawitz et al. (2017)", "Practical secure aggregation for privacy-preserving machine learning", "ACM CCS"],
  ["Abadi et al. (2016)", "Deep learning with differential privacy (DP-SGD)", "ACM CCS"],
  ["Blanchard et al. (2017)", "Machine learning with adversaries: Byzantine tolerant gradient descent (Krum)", "NeurIPS"],
  ["Hsu, Qi, Brown (2019)", "Measuring the effects of non-identical data distribution in federated learning", "arXiv:1909.06335"],
  ["Nguyen et al. (2025)", "Quantum federated learning: a comprehensive survey", "arXiv:2508.15998"],
  ["Rajaraman et al. (2018)", "Pre-trained CNNs as feature extractors for malaria parasite detection", "PeerJ"],
  ["Wang et al. (2017)", "ChestX-ray8 / ChestX-ray14: hospital-scale chest X-ray database", "CVPR"],
  ["Johnson–Lindenstrauss (1984)", "Extensions of Lipschitz mappings into a Hilbert space", "Contemp. Math."],
];

const roadmap = [
  ["Now", "Simulated federations on synthetic latent features; full privacy stack; QAOA cohort selection verified against exact solvers."],
  ["Next", "Plug in real data: NIH ChestX-ray14 embeddings and Kaggle malaria cells through the adapter layer; private-set evaluation."],
  ["Pilot", "Deploy the clinic PWA with ASHA-worker workflows in 2–3 districts; measure screening time saved and agreement with radiologists."],
  ["Hardware", "Move VQC training and QAOA aggregation to cloud QPUs (IBM Quantum / AWS Braket) as error rates allow; keep classical fallbacks."],
];

export default function Research() {
  const [bench, setBench] = useState(null);

  useEffect(() => {
    api.benchmarks().then(setBench).catch(() => {});
  }, []);

  return (
    <div className="page">
      <div className="container">
        <div className="page-head">
          <div className="eyebrow">research</div>
          <h1 className="h2" style={{ maxWidth: "30ch" }}>
            The protocol, the math, and the measured results.
          </h1>
          <p className="lead">
            Everything QBoost runs is small enough to audit. This page documents the quantum model, the
            aggregation objective, the privacy stack, and the benchmark numbers produced by the shipped code
            (<code>scripts/benchmark.py</code>).
          </p>
        </div>

        {/* ---------------------------------------------------------- model */}
        <section className="section-tight" id="model">
          <SectionHead eyebrow="quantum model" title="A 4-qubit variational classifier per clinic.">
            Deliberately minimal: it has to train in minutes on an embedded ARM board while remaining a real
            quantum model — superposition over screening outcomes, entanglement between correlated clinical
            factors, and hardware-native gradients.
          </SectionHead>
          <div className="grid grid-2">
            <Card title="Circuit specification">
              <table className="table">
                <tbody>
                  {[
                    ["Encoding", "RY angle embedding of PCA-4 clinical features"],
                    ["Ansatz", "StronglyEntanglingLayers × 3"],
                    ["Parameters", "36 (3 layers × 4 qubits × 3 rotations)"],
                    ["Readout", "⟨Z₀⟩ → P(class) = (1 − ⟨Z₀⟩)/2"],
                    ["Gradients", "Adjoint differentiation (2 executions/step)"],
                    ["Optimizer", "Adam, lr 0.12, batch 24"],
                    ["Loss", "Binary cross-entropy"],
                    ["Simulator", "PennyLane default.qubit (statevector)"],
                  ].map(([k, v]) => (
                    <tr key={k}>
                      <td className="mono faint" style={{ width: 130 }}>{k}</td>
                      <td className="strong">{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
            <Card title="Why quantum at 4 qubits?" icon={<IconBook width={17} />}>
              <div className="stack muted" style={{ fontSize: 14, gap: 12 }}>
                <p>
                  Three reasons. <b style={{ color: "var(--text)" }}>Deployment:</b> the full training step is
                  a few hundred circuit evaluations — plausible on edge hardware and on today's free-tier cloud
                  QPUs for inference-sized workloads.
                </p>
                <p>
                  <b style={{ color: "var(--text)" }}>Inductive bias:</b> entangling ansätze express
                  high-frequency correlations between clinical factors without over-parameterizing — useful
                  when each clinic holds only a few hundred patients.
                </p>
                <p>
                  <b style={{ color: "var(--text)" }}>Forward-compatibility:</b> the aggregation QUBO is the
                  same object commercial QPUs solve; upgrading from simulator to hardware changes a device
                  string, not the protocol.
                </p>
              </div>
            </Card>
          </div>
        </section>

        {/* ---------------------------------------------------- aggregation */}
        <section className="section-tight" id="aggregation">
          <SectionHead eyebrow="aggregation" title="Cohort selection as a QUBO, solved with QAOA.">
            Rather than averaging every update, the coordinator solves a small combinatorial problem each
            round: which clinics should shape the next global model?
          </SectionHead>
          <div className="grid grid-2">
            <Card title="The objective">
              <pre className="mono" style={{ background: "var(--ink-2)", border: "1px solid var(--line)", borderRadius: 12, padding: 18, overflowX: "auto", fontSize: 12, color: "#c9d6ee", margin: 0 }}>
{`min  −Σ q̃ᵢ xᵢ  −  κ Σ sᵢⱼ xᵢ xⱼ  +  0.15 Σ xᵢ
 x∈{0,1}ⁿ

q̃ᵢ  discounted quality  = reportedᵢ × trustᵢ
trustᵢ = (1 + mean_simᵢ)/2     (sketch coherence)
sᵢⱼ  cosine of JL sketches (coherence)
xᵢ  1 if clinic i joins the cohort

floor: keep ≥ ⌈n/2⌉ clinics (post-QAOA)`}</pre>
            </Card>
            <Card title="How it runs">
              <div className="stack muted" style={{ fontSize: 14, gap: 12 }}>
                <p>
                  The QUBO maps to an Ising cost Hamiltonian via x = (1 − z)/2, embedded in a 2-layer QAOA
                  ansatz on the statevector simulator. Angles γ, β are optimized with Adam (~120 steps); the
                  highest-probability basis states are scored exactly to certify the cohort.
                </p>
                <p>
                  <b style={{ color: "var(--text)" }}>Byzantine behavior is priced in:</b> a gradient-ascent
                  attacker that inflates its reported metrics still anti-correlates with honest updates, so
                  its trust discount and the negative coherence reward push it out of the cohort. In every
                  poison test, the attacker is excluded while FedAvg absorbs its update.
                </p>
                <p className="notice">
                  Verified: the QAOA solution matches exhaustive-search optima on random QUBOs (see the test
                  suite), so the quantum relaxation is not the weak link.
                </p>
              </div>
            </Card>
          </div>
        </section>

        {/* --------------------------------------------------------- privacy */}
        <section className="section-tight" id="privacy">
          <SectionHead eyebrow="privacy engineering" title="Four layers, each meaningful alone.">
            The threat model includes a curious coordinator, a compromised clinic, and a network observer.
          </SectionHead>
          <div className="grid grid-4">
            {[
              ["Transport", "AES-128-CBC + HMAC-SHA256 envelope per clinic (Fernet), keys issued at registration. Stands in for mTLS in production."],
              ["Aggregation", "Bonawitz pairwise masks: ±PRG masks cancel exactly in the sum. In FedAvg mode the server provably learns only the weighted aggregate."],
              ["Updates", "Gaussian mechanism on L2-clipped updates (C = 1.0, σ = 0.3, δ = 1e-5). Budget reported with basic composition; swap in RDP for production."],
              ["Signals", "Only validation statistics and 32-dim JL sketches leave a clinic — hundreds of times smaller than the update itself."],
            ].map(([k, v]) => (
              <div key={k} className="card card-pad">
                <div className="eyebrow" style={{ marginBottom: 10 }}>{k}</div>
                <p className="muted" style={{ fontSize: 13.5 }}>{v}</p>
              </div>
            ))}
          </div>
        </section>

        {/* --------------------------------------------------------- results */}
        <section className="section-tight" id="results">
          <SectionHead eyebrow="results" title="Measured, not promised.">
            Full-fidelity runs of the shipped pipeline: four clinics, non-IID Dirichlet partitions with site
            shift, DP on, four rounds. "Poisoned" adds a fifth column: the last clinic runs gradient ascent
            and reports fake metrics.
          </SectionHead>
          {bench?.available === false || !bench?.tasks ? (
            <Card title="Benchmarks" icon={<IconChart width={17} />}>
              <p className="muted" style={{ fontSize: 14 }}>
                Results are generated by <code>python scripts/benchmark.py</code> (~20 minutes, CPU only).
                Run it once and this section fills with the measured table and chart.
              </p>
            </Card>
          ) : (
            <div className="stack">
              <div className="card" style={{ overflow: "auto" }}>
                <table className="table" style={{ minWidth: 640 }}>
                  <thead>
                    <tr>
                      <th>task</th>
                      <th>centralized</th>
                      <th>fedavg</th>
                      <th>qaoa</th>
                      <th>fedavg + poison</th>
                      <th>qaoa + poison</th>
                      <th>attacker excluded</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(bench.tasks).map(([k, r]) => (
                      <tr key={k}>
                        <td className="strong" style={{ textTransform: "uppercase" }}>{k}</td>
                        <td className="mono">{r.centralized.final_acc.toFixed(3)}</td>
                        <td className="mono">{r.fedavg.final_acc.toFixed(3)}</td>
                        <td className="mono" style={{ color: "var(--cyan)" }}>{r.qaoa.final_acc.toFixed(3)}</td>
                        <td className="mono" style={{ color: "var(--red)" }}>{r.fedavg_poisoned.final_acc.toFixed(3)}</td>
                        <td className="mono" style={{ color: "var(--green)" }}>{r.qaoa_poisoned.final_acc.toFixed(3)}</td>
                        <td className="mono">{r.qaoa_poisoned.excluded_total > 0 ? `yes (${r.qaoa_poisoned.excluded_total}×)` : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Card title="Global accuracy after federation" sub="dashed line: same data, centralized training">
                <img src="/api/benchmarks/chart" alt="Benchmark accuracy chart" style={{ width: "100%", borderRadius: 12, border: "1px solid var(--line)" }} />
              </Card>
              <p className="notice">
                Config: {bench.config?.n_total} samples · {bench.config?.n_clinics} clinics · {bench.config?.rounds} rounds ·
                DP(C={bench.config?.dp_clip}, σ={bench.config?.dp_sigma}) · generated {bench.generated_at}
              </p>
            </div>
          )}
        </section>

        {/* -------------------------------------------------------- roadmap */}
        <section className="section-tight">
          <SectionHead eyebrow="roadmap" title="From simulator to national infrastructure." />
          <div className="grid grid-4">
            {roadmap.map(([phase, body]) => (
              <div key={phase} className="card card-pad">
                <div className="eyebrow" style={{ marginBottom: 10 }}>{phase}</div>
                <p className="muted" style={{ fontSize: 13.5 }}>{body}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ----------------------------------------------------- references */}
        <section className="section-tight">
          <SectionHead eyebrow="references" title="Grounding literature." />
          <div className="card">
            <table className="table">
              <tbody>
                {references.map(([authors, title, venue]) => (
                  <tr key={title}>
                    <td className="mono faint" style={{ width: 230 }}>{authors}</td>
                    <td className="strong">{title}</td>
                    <td className="mono faint" style={{ textAlign: "right" }}>{venue}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
}
