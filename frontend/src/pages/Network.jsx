import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api.js";
import { Card } from "../components/ui.jsx";
import { IconClinic, IconLock, IconShield, IconWifiOff, IconAtom } from "../components/icons.jsx";

const DEVICE_PROFILE = {
  "clinic-1": "Android tablet · 4 GB",
  "clinic-2": "Raspberry Pi 4 · 8 GB",
  "clinic-3": "Android tablet · 3 GB",
  "clinic-4": "Raspberry Pi 3 · 1 GB",
  "clinic-5": "Android tablet · 4 GB",
  "clinic-6": "Raspberry Pi 4 · 4 GB",
  "clinic-7": "Android tablet · 3 GB",
  "clinic-8": "Raspberry Pi 3 · 1 GB",
};

function Topology({ clients }) {
  const n = Math.max(clients.length, 1);
  const W = 560;
  const H = 300;
  const cx = W / 2;
  const cy = H / 2;
  const radius = 108;
  const pos = (i) => {
    const a = (i / n) * 2 * Math.PI - Math.PI / 2;
    return [cx + radius * Math.cos(a), cy + radius * Math.sin(a)];
  };
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }}>
      <circle cx={cx} cy={cy} r="200" fill="none" stroke="rgba(154,173,210,0.10)" strokeDasharray="2 6" />
      {clients.map((c, i) => {
        const [x, y] = pos(i);
        const active = c.selected;
        return (
          <g key={c.cid}>
            <line
              x1={cx}
              y1={cy}
              x2={x}
              y2={y}
              stroke={active ? "rgba(34,211,238,0.5)" : "rgba(248,113,113,0.35)"}
              strokeWidth="1.2"
              className="flowline"
              strokeDasharray="4 6"
            />
            <rect x={x - 52} y={y - 16} width="104" height="32" rx="9" fill="#0c1220" stroke={active ? "rgba(34,211,238,0.5)" : "rgba(248,113,113,0.4)"} />
            <text x={x} y={y - 1} textAnchor="middle" fontSize="9.5" fill="#e8edf7" fontFamily="Inter, sans-serif">
              {c.name.length > 17 ? c.name.slice(0, 16) + "…" : c.name}
            </text>
            <text x={x} y={y + 10} textAnchor="middle" fontSize="8.5" fill="#64748b" fontFamily="JetBrains Mono, monospace">
              {active ? "cohort · in" : "excluded"}
            </text>
          </g>
        );
      })}
      <circle cx={cx} cy={cy} r="30" fill="#0c1220" stroke="rgba(167,139,250,0.6)" strokeWidth="1.4" />
      <text x={cx} y={cy - 3} textAnchor="middle" fontSize="9" fill="#cfc2fd" fontFamily="JetBrains Mono, monospace">
        QAOA
      </text>
      <text x={cx} y={cy + 9} textAnchor="middle" fontSize="9" fill="#cfc2fd" fontFamily="JetBrains Mono, monospace">
        coord
      </text>
    </svg>
  );
}

export default function Network() {
  const [runs, setRuns] = useState([]);
  const [selected, setSelected] = useState(null);
  const [state, setState] = useState(null);

  useEffect(() => {
    api.simulations().then((rs) => {
      const completed = rs.filter((r) => r.status === "completed");
      setRuns(completed);
      if (completed.length) setSelected(completed[0].run_id);
    });
  }, []);

  useEffect(() => {
    if (!selected) return;
    api.simulation(selected).then(setState).catch(() => {});
  }, [selected]);

  const clients = useMemo(() => state?.rounds?.[state.rounds.length - 1]?.clients ?? [], [state]);
  const privacy = state?.result?.privacy;

  const security = [
    [<IconLock key="a" />, "Envelope encryption", "Per-clinic Fernet keys (AES-128-CBC + HMAC-SHA256) issued at registration; every payload is sealed and authenticated end-to-end."],
    [<IconShield key="b" />, "Pairwise masking", "Bonawitz-style masks cancel exactly across the cohort — the coordinator computes aggregates without ever seeing an individual update."],
    [<IconWifiOff key="c" />, "Differential privacy", privacy ? `Gaussian mechanism on clipped updates (C=1.0, σ=0.3, δ=1e-5). Basic-composition budget for the last run: ε ≈ ${privacy.epsilon_total ?? "—"}.` : "Gaussian mechanism on clipped updates; budget published per deployment."],
    [<IconAtom key="d" />, "Sketch-based selection", "32-dim Johnson–Lindenstrauss projections let QAOA judge update coherence without exposing parameters."],
  ];

  return (
    <div className="page">
      <div className="container">
        <div className="page-head">
          <div className="eyebrow">clinic network</div>
          <h1 className="h2" style={{ maxWidth: "30ch" }}>
            Eight districts. One model. Zero centralized records.
          </h1>
          <p className="lead">
            Each site below is a federated participant with its own data distribution — different devices,
            different case mixes, different calibration. Pick a completed run to inspect its network state.
          </p>
          {runs.length > 0 && (
            <div className="field" style={{ maxWidth: 380 }}>
              <label>Completed run</label>
              <select value={selected ?? ""} onChange={(e) => setSelected(e.target.value)}>
                {runs.map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.run_id} · {r.config.task} · {r.config.strategy.toUpperCase()} · acc {r.final_acc?.toFixed?.(3)}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        <div className="grid" style={{ gridTemplateColumns: "1fr 1.05fr", alignItems: "start" }}>
          <Card title="Live topology" sub={state ? `run ${state.run_id} · last round cohort` : "launch a run in the console to populate"} icon={<IconAtom width={17} />}>
            <Topology clients={clients} />
          </Card>

          <Card title="Security posture" icon={<IconShield width={17} />} sub="four independent layers on every payload">
            <div className="stack" style={{ gap: 16 }}>
              {security.map(([icon, title, body]) => (
                <div key={title} style={{ display: "flex", gap: 14 }}>
                  <div className="step-icon" style={{ width: 38, height: 38, flexShrink: 0 }}>
                    {icon}
                  </div>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: 14 }}>{title}</div>
                    <div className="muted" style={{ fontSize: 13 }}>{body}</div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        <div className="grid grid-4" style={{ marginTop: 18 }}>
          {clients.length
            ? clients.map((c) => (
                <div key={c.cid} className="card card-pad" style={{ padding: 20 }}>
                  <div className="spread" style={{ marginBottom: 10 }}>
                    <IconClinic width={18} style={{ color: c.selected ? "var(--cyan)" : "var(--faint)" }} />
                    <span className={`chip ${c.selected ? "chip-green" : "chip-red"}`}>{c.selected ? "cohort" : "excluded"}</span>
                  </div>
                  <div style={{ fontWeight: 600, fontSize: 14.5 }}>{c.name}</div>
                  <div className="mono faint" style={{ fontSize: 10.5, margin: "2px 0 12px" }}>
                    {DEVICE_PROFILE[c.cid] ?? "clinic device"}
                  </div>
                  <div className="stack" style={{ gap: 7, fontSize: 12.5 }}>
                    <div className="spread">
                      <span className="faint">patients contributed</span>
                      <span className="mono">{c.n_samples}</span>
                    </div>
                    <div className="spread">
                      <span className="faint">local val acc</span>
                      <span className="mono">{c.val_acc.toFixed(3)}</span>
                    </div>
                    <div className="spread">
                      <span className="faint">coherence</span>
                      <span className="mono" style={{ color: c.mean_similarity < 0 ? "var(--red)" : "var(--green)" }}>
                        {c.mean_similarity >= 0 ? "+" : ""}
                        {c.mean_similarity.toFixed(2)}
                      </span>
                    </div>
                    <div className="spread">
                      <span className="faint">quality</span>
                      <span className="mono">{c.quality.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              ))
            : [
                "Phulbani CHC · Odisha",
                "Bastar Rural DH · Chhattisgarh",
                "Gadchiroli PHC · Maharashtra",
                "Barmer Mobile Unit · Rajasthan",
              ].map((name) => (
                <div key={name} className="card card-pad" style={{ padding: 20 }}>
                  <div className="spread" style={{ marginBottom: 10 }}>
                    <IconClinic width={18} style={{ color: "var(--faint)" }} />
                    <span className="chip">standby</span>
                  </div>
                  <div style={{ fontWeight: 600, fontSize: 14.5 }}>{name.split(" · ")[0]}</div>
                  <div className="mono faint" style={{ fontSize: 10.5, margin: "2px 0 12px" }}>{name.split(" · ")[1]}, India</div>
                  <p className="muted" style={{ fontSize: 12.5 }}>Waiting for the first federation run. Launch one in the console to bring this clinic online.</p>
                </div>
              ))}
        </div>
      </div>
    </div>
  );
}
