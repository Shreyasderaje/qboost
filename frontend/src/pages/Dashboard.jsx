import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api.js";
import { Card, LineChart, SimMatrix, Sparkline } from "../components/ui.jsx";
import { IconAtom, IconChart, IconCpu, IconShield, IconWarn } from "../components/icons.jsx";

const PHASES = ["dispatch", "local training", "qaoa selection", "aggregation", "evaluation"];

function phaseFromEvents(events, roundsDone, running) {
  if (!running) return { active: null, index: -1 };
  const last = events[events.length - 1];
  if (!last) return { active: "dispatch", index: 0 };
  const k = last.kind;
  if (k === "round_started") return { active: "local training", index: 1 };
  if (k === "client_trained" || k === "secure_submit") return { active: "local training", index: 1 };
  if (k === "selection_started") return { active: "qaoa selection", index: 2 };
  if (k === "cohort_selected") return { active: "aggregation", index: 3 };
  if (k === "round_complete") return { active: "evaluation", index: 4 };
  return { active: "local training", index: 1 };
}

export default function Dashboard() {
  const [tasks, setTasks] = useState([]);
  const [models, setModels] = useState([]);
  const [config, setConfig] = useState({
    task: "diabetes",
    strategy: "qaoa",
    clinics: 4,
    rounds: 5,
    dp: true,
    poison: false,
    seed: 42,
    local_epochs: 3,
  });
  const [runId, setRunId] = useState(null);
  const [state, setState] = useState(null);
  const [error, setError] = useState(null);
  const [launching, setLaunching] = useState(false);
  const feedRef = useRef(null);

  useEffect(() => {
    api.tasks().then(setTasks).catch(() => {});
    api.models().then(setModels).catch(() => {});
    // Re-attach to the most recent run so navigation doesn't lose the view.
    api
      .simulations()
      .then((runs) => {
        if (runs.length) setRunId(runs[0].run_id);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!runId) return;
    let alive = true;
    const poll = () =>
      api
        .simulation(runId)
        .then((s) => {
          if (!alive) return;
          setState(s);
          if (s.status === "failed") setError("run failed — see event log");
        })
        .catch(() => {});
    poll();
    const id = setInterval(poll, 1600);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [runId]);

  useEffect(() => {
    if (feedRef.current) feedRef.current.scrollTop = feedRef.current.scrollHeight;
  }, [state?.events?.length]);

  const launch = useCallback(async () => {
    setLaunching(true);
    setError(null);
    try {
      const { run_id } = await api.startSimulation(config);
      setRunId(run_id);
      setState(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLaunching(false);
    }
  }, [config]);

  const running = state && (state.status === "running" || state.status === "queued");
  const rounds = state?.rounds ?? [];
  const latest = rounds[rounds.length - 1];
  const accSeries = useMemo(() => [{ name: "global acc", color: "#22d3ee", values: rounds.map((r) => r.global.acc) }], [rounds]);
  const { active, index } = phaseFromEvents(state?.events ?? [], rounds.length, running);

  const clientRows = latest?.clients ?? [];
  const taskLabel = (k) => tasks.find((t) => t.key === k)?.name ?? k;

  const rangeProps = (key, min, max) => ({
    type: "range",
    min,
    max,
    value: config[key],
    style: { "--fill": `${((config[key] - min) / (max - min)) * 100}%` },
    onChange: (e) => setConfig((c) => ({ ...c, [key]: +e.target.value })),
  });

  return (
    <div className="page">
      <div className="container">
        <div className="page-head">
          <div className="eyebrow">federation console</div>
          <h1 className="h2" style={{ maxWidth: "30ch" }}>
            Coordinate the federation. Watch every round land.
          </h1>
          <p className="lead">
            Each clinic trains its quantum circuit locally, seals its update, and the coordinator runs QAOA
            cohort selection over the sketches. Everything below is the real protocol — simulated in one
            process for the demo.
          </p>
        </div>

        <div className="grid" style={{ gridTemplateColumns: "340px 1fr", alignItems: "start" }}>
          {/* ---------------------------------------------------- config */}
          <div className="stack">
            <Card title="Round configuration" icon={<IconCpu width={17} />}>
              <div className="stack" style={{ gap: 18 }}>
                <div className="field">
                  <label>Screening task</label>
                  <select value={config.task} onChange={(e) => setConfig((c) => ({ ...c, task: e.target.value }))}>
                    {["diabetes", "tb", "malaria"].map((k) => (
                      <option key={k} value={k}>
                        {taskLabel(k)}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="field">
                  <label>Aggregation strategy</label>
                  <div className="seg">
                    <button
                      className={config.strategy === "qaoa" ? "active" : ""}
                      onClick={() => setConfig((c) => ({ ...c, strategy: "qaoa" }))}
                    >
                      QAOA
                    </button>
                    <button
                      className={config.strategy === "fedavg" ? "active" : ""}
                      onClick={() => setConfig((c) => ({ ...c, strategy: "fedavg" }))}
                    >
                      FedAvg
                    </button>
                  </div>
                  <span className="notice">
                    {config.strategy === "qaoa"
                      ? "cohort selection on sketches; quality-weighted"
                      : "all clinics, pairwise-masked secure sum"}
                  </span>
                </div>

                <div className="field">
                  <label>
                    Clinics <span className="mono">{config.clinics}</span>
                  </label>
                  <input {...rangeProps("clinics", 2, 8)} />
                </div>
                <div className="field">
                  <label>
                    Rounds <span className="mono">{config.rounds}</span>
                  </label>
                  <input {...rangeProps("rounds", 1, 10)} />
                </div>
                <div className="field">
                  <label>
                    Local epochs <span className="mono">{config.local_epochs}</span>
                  </label>
                  <input {...rangeProps("local_epochs", 1, 5)} />
                </div>

                <div className="toggle">
                  <span style={{ fontSize: 13.5, color: "var(--muted)" }}>Differential privacy</span>
                  <label className="switch">
                    <input type="checkbox" checked={config.dp} onChange={(e) => setConfig((c) => ({ ...c, dp: e.target.checked }))} />
                    <span />
                  </label>
                </div>

                <div className="toggle">
                  <span style={{ fontSize: 13.5, color: "var(--muted)" }}>
                    Poison last clinic
                    <span className="notice" style={{ display: "block" }}>
                      gradient-ascent attacker, inflated metrics
                    </span>
                  </span>
                  <label className="switch">
                    <input type="checkbox" checked={config.poison} onChange={(e) => setConfig((c) => ({ ...c, poison: e.target.checked }))} />
                    <span />
                  </label>
                </div>

                <div className="field">
                  <label>Seed</label>
                  <input
                    type="number"
                    value={config.seed}
                    onChange={(e) => setConfig((c) => ({ ...c, seed: +e.target.value }))}
                  />
                </div>

                <button className="btn btn-primary btn-lg" onClick={launch} disabled={launching || running}>
                  {running ? "federation running…" : launching ? "starting…" : "Launch federation"}
                </button>
                {error && (
                  <div className="banner">
                    <IconWarn width={16} />
                    <span>{error}</span>
                  </div>
                )}
                <span className="notice">a typical run takes 3–6 minutes on CPU</span>
              </div>
            </Card>

            <Card title="Privacy posture" icon={<IconShield width={17} />}>
              <div className="stack" style={{ fontSize: 13, color: "var(--muted)" }}>
                {[
                  ["Envelope", "AES-128-CBC + HMAC (Fernet)"],
                  ["Masks", "pairwise, Bonawitz-style"],
                  ["DP", config.dp ? "clipped + gaussian (C=1.0, σ=0.3)" : "disabled"],
                  ["Sketch", "32-dim Johnson–Lindenstrauss"],
                ].map(([k, v]) => (
                  <div key={k} className="spread">
                    <span className="mono faint">{k}</span>
                    <span style={{ textAlign: "right" }}>{v}</span>
                  </div>
                ))}
                <div className="spread" style={{ borderTop: "1px solid var(--line)", paddingTop: 10 }}>
                  <span className="mono faint">ε (total, basic composition)</span>
                  <span className="mono">{state?.result?.privacy?.epsilon_total ?? (config.dp ? "—" : "∞")}</span>
                </div>
              </div>
            </Card>
          </div>

          {/* ------------------------------------------------------ live view */}
          <div className="stack">
            <Card
              title="Global model accuracy"
              icon={<IconChart width={17} />}
              sub={state ? `run ${state.run_id} · ${state.config.strategy.toUpperCase()} · ${taskLabel(state.config.task)}` : "launch a run to stream rounds"}
              right={
                <span className="chip chip-cyan">
                  <span className={`dot ${running ? "" : "off"}`} />
                  {state ? state.status : "idle"}
                </span>
              }
            >
              <div className="stack" style={{ gap: 16 }}>
                <div className="phasebar">
                  {PHASES.map((p, i) => (
                    <span key={p} className={`phase ${i < index ? "done" : ""} ${i === index ? "active" : ""}`}>
                      {p}
                    </span>
                  ))}
                </div>
                <LineChart series={accSeries} xCount={(state?.config?.rounds ?? 5) + 1} yMin={0.4} yMax={1} />
                <div className="chart-legend">
                  <span>
                    <i style={{ background: "#22d3ee" }} /> global accuracy on coordinator benchmark
                  </span>
                  <span className="mono faint">
                    {latest
                      ? `round ${latest.round}: acc ${latest.global.acc.toFixed(3)} · loss ${latest.global.loss.toFixed(3)} · ${latest.elapsed_s}s`
                      : "waiting for round 1"}
                  </span>
                </div>
              </div>
            </Card>

            {latest?.qaoa && (
              <div className="grid grid-3">
                <Card title="QAOA energy" icon={<IconAtom width={17} />} sub={`p=${latest.qaoa.p} · ${latest.qaoa.steps} steps`}>
                  <Sparkline values={latest.qaoa.history} />
                  <div className="spread mono" style={{ marginTop: 10 }}>
                    <span className="faint">final ⟨E⟩</span>
                    <span>{latest.qaoa.expectation?.toFixed(3)}</span>
                  </div>
                  <div className="spread mono">
                    <span className="faint">cohort energy</span>
                    <span style={{ color: "#8df0cd" }}>{latest.qaoa.energy?.toFixed(3)}</span>
                  </div>
                </Card>
                <Card title="Cohort decision" icon={<IconAtom width={17} />} sub="selected clinics per round">
                  <div className="cohort-bits">
                    {clientRows.map((c) => (
                      <span key={c.cid} className={`bit ${c.selected ? "on" : "excluded"}`}>
                        {c.selected ? "1" : "0"} · {c.name.split(" ")[0]}
                      </span>
                    ))}
                  </div>
                  <div className="spread mono" style={{ marginTop: 12 }}>
                    <span className="faint">γ, β</span>
                    <span className="faint" style={{ textAlign: "right", maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis" }}>
                      {latest.qaoa.gammas?.map((g) => g.toFixed(2)).join(", ")} | {latest.qaoa.betas?.map((g) => g.toFixed(2)).join(", ")}
                    </span>
                  </div>
                </Card>
                <Card title="Update coherence" icon={<IconAtom width={17} />} sub="cosine of JL sketches">
                  <SimMatrix names={clientRows.map((c) => c.name)} matrix={latest.qubo?.similarity ?? clientRows.map(() => clientRows.map(() => 0))} />
                </Card>
              </div>
            )}

            <div className="grid grid-2">
              <Card title="Clinic updates" sub="per-round signals (aggregate statistics only)">
                {clientRows.length ? (
                  <table className="table">
                    <thead>
                      <tr>
                        <th>clinic</th>
                        <th>patients</th>
                        <th>val acc</th>
                        <th>sim</th>
                        <th>cohort</th>
                      </tr>
                    </thead>
                    <tbody>
                      {clientRows.map((c) => (
                        <tr key={c.cid}>
                          <td className="strong">{c.name}</td>
                          <td className="mono">{c.n_samples}</td>
                          <td className="mono">{c.val_acc.toFixed(3)}</td>
                          <td className="mono" style={{ color: c.mean_similarity < 0 ? "var(--red)" : "var(--muted)" }}>
                            {c.mean_similarity >= 0 ? "+" : ""}
                            {c.mean_similarity.toFixed(2)}
                          </td>
                          <td>{c.selected ? <span className="chip chip-green">in</span> : <span className="chip chip-red">out</span>}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <p className="muted" style={{ fontSize: 13.5 }}>
                    Clinic signals appear here after each round: validation accuracy (a statistic, not data),
                    update norm, and sketch coherence.
                  </p>
                )}
              </Card>

              <Card title="Event stream" sub="protocol log">
                <div className="feed" ref={feedRef}>
                  {(state?.events ?? []).slice(-80).map((e, i) => (
                    <div key={i} className="feed-item">
                      <span className="feed-time">{new Date(e.ts * 1000).toLocaleTimeString([], { hour12: false })}</span>
                      <span
                        className="feed-kind"
                        style={{
                          color:
                            e.kind === "cohort_selected" || e.kind === "round_complete"
                              ? "#8df0cd"
                              : e.kind === "secure_submit"
                              ? "#cfc2fd"
                              : "var(--faint)",
                        }}
                      >
                        {e.kind.replace(/_/g, " ")}
                      </span>
                      <span className="feed-msg">{e.message}</span>
                    </div>
                  ))}
                  {!state?.events?.length && <p className="muted" style={{ fontSize: 13.5 }}>Idle. Launch a federation to open the log.</p>}
                </div>
              </Card>
            </div>

            <Card
              title="Model registry"
              sub="trained global bundles, ready for clinic inference"
              right={
                <Link to="/diagnose" className="btn btn-ghost btn-sm">
                  Open clinic console
                </Link>
              }
            >
              {models.length ? (
                <table className="table">
                  <thead>
                    <tr>
                      <th>bundle</th>
                      <th>task</th>
                      <th>strategy</th>
                      <th>rounds</th>
                      <th>val acc</th>
                    </tr>
                  </thead>
                  <tbody>
                    {models.map((m) => (
                      <tr key={m.name}>
                        <td className="mono strong">{m.name}</td>
                        <td>{m.task_name}</td>
                        <td>
                          <span className={`chip ${m.strategy === "qaoa" ? "chip-cyan" : ""}`}>{m.strategy?.toUpperCase()}</span>
                        </td>
                        <td className="mono">{m.rounds_completed}</td>
                        <td className="mono" style={{ color: "var(--green)" }}>
                          {m.val_acc != null ? m.val_acc.toFixed(3) : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="muted" style={{ fontSize: 13.5 }}>
                  No bundles yet — launch your first federation above.
                </p>
              )}
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
