import { Link } from "react-router-dom";
import { CircuitHero } from "../components/CircuitHero.jsx";
import { SectionHead } from "../components/ui.jsx";
import {
  IconArrow,
  IconAtom,
  IconBolt,
  IconChart,
  IconClinic,
  IconCpu,
  IconLock,
  IconPulse,
  IconShield,
  IconStethoscope,
  IconWifiOff,
} from "../components/icons.jsx";

const burden = [
  {
    value: "~25%",
    label: "of the world's new TB cases occur in India — the largest single-country share (WHO Global TB Report 2025)",
  },
  {
    value: "66%",
    label: "of malaria cases in the WHO South-East Asia region are in India (World Malaria Report)",
  },
  {
    value: "≈90M",
    label: "adults in India live with diabetes — among the largest national burdens worldwide (IDF Diabetes Atlas)",
  },
];

const steps = [
  {
    icon: <IconCpu />,
    num: "01 · train locally",
    title: "A tiny quantum model learns on-device",
    body: "Each clinic trains a 4-qubit variational circuit on its own patients. A Raspberry-Pi-class device runs the full forward and adjoint backward pass — no GPU, no cloud, no data export.",
  },
  {
    icon: <IconLock />,
    num: "02 · seal and sketch",
    title: "Updates leave the clinic unreadable",
    body: "Updates are clipped and noised with a Gaussian DP mechanism, reduced to a 32-dim Johnson–Lindenstrauss sketch, masked pairwise à la Bonawitz, and sealed in an AES envelope. Nothing human-readable crosses the wire.",
  },
  {
    icon: <IconAtom />,
    num: "03 · aggregate with QAOA",
    title: "A quantum optimizer picks the cohort",
    body: "The coordinator encodes cohort selection as a QUBO — quality signals plus update coherence — and relaxes it with a 2-layer QAOA. Poisoned or drifting clinics are excluded by their anti-correlated sketches.",
  },
  {
    icon: <IconStethoscope />,
    num: "04 · screen at the edge",
    title: "The global model returns to the clinic",
    body: "The improved bundle installs back onto clinic devices as an installable PWA. A nurse inputs observations, gets a calibrated risk read with contributing factors, and the raw record never moved.",
  },
];

const diseases = [
  {
    name: "Tuberculosis",
    tag: "chest X-ray screening",
    accent: "linear-gradient(120deg, rgba(34,211,238,0.16), rgba(34,211,238,0.02))",
    body: "Ten radiograph-derived indices — cavitation, infiltrate density, effusion and more — reduce to a 4-dim latent that a variational circuit classifies with 0.86 federated accuracy in our benchmark.",
    chips: ["cavity", "infiltrate", "effusion", "nodules", "hilar", "asymmetry"],
  },
  {
    name: "Malaria",
    tag: "blood smear analysis",
    accent: "linear-gradient(120deg, rgba(52,211,153,0.14), rgba(52,211,153,0.02))",
    body: "Parasite density, ring-form morphology and stain deviation form the smear fingerprint. Screened against the federated model in under a second on a tablet.",
    chips: ["parasite density", "ring forms", "chromatin", "stain", "irregularity"],
  },
  {
    name: "Diabetes risk",
    tag: "metabolic risk scoring",
    accent: "linear-gradient(120deg, rgba(167,139,250,0.15), rgba(167,139,250,0.02))",
    body: "A PIMA-schema metabolic panel scored by the shared global model — the strongest non-IID test, where QAOA cohort selection clearly outperforms naive averaging.",
    chips: ["glucose", "BMI", "insulin", "BP", "age", "pedigree"],
  },
];

export default function Home() {
  return (
    <>
      {/* ------------------------------------------------------------- hero */}
      <section className="hero">
        <div className="container hero-grid">
          <div className="hero-copy reveal">
            <div className="eyebrow">quantum · federated · privacy-first</div>
            <h1 className="h1">
              Specialist-grade diagnostics for the <span className="grad-text">last mile</span>.
            </h1>
            <p className="lead" style={{ fontSize: 18 }}>
              QBoost trains one shared diagnostic model across rural clinics — with quantum circuits small
              enough to run on a tablet and an aggregation protocol mathematically incapable of seeing a
              single patient record.
            </p>
            <div className="hero-cta">
              <Link to="/dashboard" className="btn btn-primary btn-lg">
                Launch the console <IconArrow width={17} />
              </Link>
              <Link to="/research" className="btn btn-ghost btn-lg">
                Read the research
              </Link>
            </div>
            <div className="hero-facts">
              <div>
                <b>4</b>
                <span>qubits per clinic circuit</span>
              </div>
              <div>
                <b>0</b>
                <span>patient records transmitted</span>
              </div>
              <div>
                <b>ε-DP</b>
                <span>gaussian-mechanism updates</span>
              </div>
              <div>
                <b>QAOA</b>
                <span>cohort aggregation</span>
              </div>
            </div>
          </div>
          <div className="reveal" style={{ animationDelay: "0.15s" }}>
            <CircuitHero />
          </div>
        </div>
      </section>

      {/* ----------------------------------------------------------- burden */}
      <section className="section-tight">
        <div className="container">
          <div className="grid grid-3">
            {burden.map((b) => (
              <div key={b.value} className="stat">
                <div className="stat-value grad-text">{b.value}</div>
                <div className="stat-label">{b.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------ story */}
      <section className="section">
        <div className="container">
          <SectionHead eyebrow="the problem" title="The people who need diagnostic AI most can't run it.">
            Training a diagnostic model traditionally means pooling patient data into one place — a
            non-starter under India's DPDP Act, GDPR-style regimes, and basic medical ethics. And rural
            clinics have no compute to train or even host models. QBoost resolves both constraints at once.
          </SectionHead>
          <div className="grid grid-2">
            {steps.map((s) => (
              <div key={s.num} className="card card-pad step">
                <div className="row" style={{ gap: 14 }}>
                  <div className="step-icon">{s.icon}</div>
                  <div className="step-num">{s.num}</div>
                </div>
                <h3 className="h3">{s.title}</h3>
                <p className="muted">{s.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* --------------------------------------------------------- diseases */}
      <section className="section" style={{ paddingTop: 30 }}>
        <div className="container">
          <SectionHead eyebrow="screening coverage" title="Three disease burdens, one protocol.">
            Every workload is a binary screening task encoded by the same feature contract, so clinics join
            the federation once and receive all three models as they mature.
          </SectionHead>
          <div className="grid grid-3">
            {diseases.map((d) => (
              <div key={d.name} className="card card-pad disease-card">
                <div className="cover" style={{ background: d.accent }}>
                  <svg viewBox="0 0 400 118" style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
                    <polyline
                      points="0,90 40,84 80,70 120,74 160,55 200,60 240,42 280,48 320,30 360,34 400,18"
                      fill="none"
                      stroke="rgba(232,237,247,0.35)"
                      strokeWidth="1.6"
                    />
                    <polyline
                      points="0,104 50,100 100,92 150,94 200,80 250,84 300,70 350,74 400,62"
                      fill="none"
                      stroke="rgba(232,237,247,0.14)"
                      strokeWidth="1.2"
                    />
                  </svg>
                  <span className="chip chip-cyan" style={{ position: "absolute", top: 12, left: 12 }}>
                    {d.tag}
                  </span>
                </div>
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <h3 className="h3">{d.name}</h3>
                  <IconPulse width={18} style={{ color: "var(--muted)" }} />
                </div>
                <p className="muted" style={{ fontSize: 14 }}>
                  {d.body}
                </p>
                <div className="feature-chips">
                  {d.chips.map((c) => (
                    <span key={c} className="chip">
                      {c}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ---------------------------------------------------------- privacy */}
      <section className="section" style={{ paddingTop: 30 }}>
        <div className="container">
          <div className="grid grid-2" style={{ alignItems: "stretch" }}>
            <div className="card card-pad" style={{ display: "grid", gap: 18, alignContent: "start" }}>
              <div className="eyebrow">privacy architecture</div>
              <h2 className="h2" style={{ fontSize: 30 }}>
                Four independent layers. <span className="grad-text">No single point of trust.</span>
              </h2>
              <p className="muted">
                An adversary controlling the coordinator still cannot reconstruct an individual update: the
                masks hide it, the sketch truncates it, the noise calibrates it, and the envelope
                authenticates it.
              </p>
              <Link to="/research#privacy" className="btn btn-ghost" style={{ justifySelf: "start" }}>
                How the layers compose <IconArrow width={16} />
              </Link>
            </div>
            <div className="stack">
              {[
                [<IconLock key="i" />, "Envelope encryption", "Every payload sealed with a per-clinic AES-128 key issued at registration; tampering fails authentication."],
                [<IconShield key="i" />, "Pairwise masking", "Bonawitz-style masks cancel exactly in the sum — the coordinator learns aggregates, never individuals."],
                [<IconWifiOff key="i" />, "Differential privacy", "Clipped, Gaussian-noised updates with published (ε, δ) accounting per deployment."],
                [<IconAtom key="i" />, "QAOA cohort selection", "Coherence-aware quantum aggregation drops poisoned or faulty updates before they touch the model."],
              ].map(([icon, title, body]) => (
                <div key={title} className="card card-pad" style={{ display: "flex", gap: 16, alignItems: "flex-start", padding: 18 }}>
                  <div className="step-icon" style={{ width: 38, height: 38 }}>
                    {icon}
                  </div>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: 14.5 }}>{title}</div>
                    <div className="muted" style={{ fontSize: 13 }}>
                      {body}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* -------------------------------------------------------------- cta */}
      <section className="section-tight">
        <div className="container">
          <div className="cta-band">
            <div className="eyebrow" style={{ justifyContent: "center" }}>
              <IconBolt width={15} /> run it now
            </div>
            <h2 className="h2" style={{ maxWidth: "24ch" }}>
              Watch a federation train live — then screen a patient.
            </h2>
            <p className="lead" style={{ textAlign: "center" }}>
              The console simulates every clinic in one machine: local quantum training, sealed updates,
              QAOA cohort selection and global evaluation, round by round.
            </p>
            <div className="hero-cta" style={{ justifyContent: "center" }}>
              <Link to="/dashboard" className="btn btn-primary btn-lg">
                Open the training console <IconChart width={17} />
              </Link>
              <Link to="/diagnose" className="btn btn-ghost btn-lg">
                Try the clinic console <IconClinic width={17} />
              </Link>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
