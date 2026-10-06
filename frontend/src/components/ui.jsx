/* Shared UI primitives and hand-rolled SVG charts. */

export function SectionHead({ eyebrow, title, children }) {
  return (
    <div className="section-head">
      <div className="eyebrow">{eyebrow}</div>
      <h2 className="h2">{title}</h2>
      {children && <p className="lead">{children}</p>}
    </div>
  );
}

export function Card({ title, icon, sub, right, children, className = "", pad = true }) {
  return (
    <div className={`card ${className}`}>
      {(title || right) && (
        <div className="spread" style={{ padding: "18px 22px 0" }}>
          <div>
            <div className="card-title">
              {icon}
              {title}
            </div>
            {sub && <div className="card-sub">{sub}</div>}
          </div>
          {right}
        </div>
      )}
      <div style={pad ? { padding: "18px 22px 22px" } : {}}>{children}</div>
    </div>
  );
}

/* ------------------------------------------------------------ LineChart -- */
export function LineChart({ series, height = 220, yMin = 0.4, yMax = 1, xCount, markers }) {
  const W = 640;
  const H = height;
  const padL = 38;
  const padR = 14;
  const padT = 12;
  const padB = 26;
  const iw = W - padL - padR;
  const ih = H - padT - padB;

  const n = xCount || Math.max(...series.map((s) => s.values.length), 2);
  const x = (i) => padL + (n <= 1 ? iw / 2 : (i / (n - 1)) * iw);
  const y = (v) => padT + (1 - (Math.min(Math.max(v, yMin), yMax) - yMin) / (yMax - yMin)) * ih;

  const ticks = [yMin, (yMin + yMax) / 2, yMax];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }}>
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={padL} x2={W - padR} y1={y(t)} y2={y(t)} stroke="rgba(154,173,210,0.10)" />
          <text x={padL - 7} y={y(t) + 3.5} textAnchor="end" fontSize="10" fill="#64748b" fontFamily="JetBrains Mono, monospace">
            {t.toFixed(1)}
          </text>
        </g>
      ))}
      {markers?.map((m, i) => (
        <line key={`m${i}`} x1={x(m.x)} x2={x(m.x)} y1={padT} y2={padT + ih} stroke="rgba(248,113,113,0.35)" strokeDasharray="3 4" />
      ))}
      {series.map((s, si) => {
        if (!s.values.length) return null;
        const pts = s.values.map((v, i) => `${x(i)},${y(v)}`).join(" ");
        const area = `${padL},${padT + ih} ${pts} ${x(s.values.length - 1)},${padT + ih}`;
        return (
          <g key={si}>
            {si === 0 && <polygon points={area} fill={s.color} opacity="0.08" />}
            <polyline points={pts} fill="none" stroke={s.color} strokeWidth="2.2" strokeLinejoin="round" strokeLinecap="round" />
            {s.values.map((v, i) => (
              <circle key={i} cx={x(i)} cy={y(v)} r="3" fill="#0b1120" stroke={s.color} strokeWidth="1.8" />
            ))}
          </g>
        );
      })}
      {Array.from({ length: n }, (_, i) => (
        <text key={i} x={x(i)} y={H - 8} textAnchor="middle" fontSize="10" fill="#64748b" fontFamily="JetBrains Mono, monospace">
          {i === 0 ? "init" : `r${i}`}
        </text>
      ))}
    </svg>
  );
}

/* ------------------------------------------------------------- Sparkline -- */
export function Sparkline({ values, color = "#a78bfa", height = 46 }) {
  if (!values || values.length < 2) return <div className="notice">not enough data</div>;
  const W = 300;
  const H = height;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * W},${H - 4 - ((v - min) / span) * (H - 10)}`);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }}>
      <polyline points={pts.join(" ")} fill="none" stroke={color} strokeWidth="1.8" strokeLinejoin="round" />
      <circle cx={pts[pts.length - 1].split(",")[0]} cy={pts[pts.length - 1].split(",")[1]} r="2.6" fill={color} />
    </svg>
  );
}

/* ---------------------------------------------------------------- Gauge -- */
export function Gauge({ value, color }) {
  const r = 78;
  const circ = Math.PI * r; // half circle
  const v = Math.min(Math.max(value, 0), 1);
  return (
    <svg viewBox="0 0 220 132" style={{ width: 240, height: "auto" }}>
      <path d={`M 20 110 A ${r} ${r} 0 0 1 200 110`} fill="none" stroke="#141d33" strokeWidth="14" strokeLinecap="round" />
      <path
        d={`M 20 110 A ${r} ${r} 0 0 1 200 110`}
        fill="none"
        stroke={color}
        strokeWidth="14"
        strokeLinecap="round"
        strokeDasharray={`${v * circ} ${circ}`}
        style={{ transition: "stroke-dasharray 0.9s cubic-bezier(0.2,0.8,0.2,1), stroke 0.4s" }}
      />
      {[0, 0.25, 0.5, 0.75, 1].map((t) => {
        const a = Math.PI * (1 - t);
        const x1 = 110 + (r - 11) * Math.cos(a);
        const y1 = 110 - (r - 11) * Math.sin(a);
        const x2 = 110 + (r + 11) * Math.cos(a);
        const y2 = 110 - (r + 11) * Math.sin(a);
        return <line key={t} x1={x1} y1={y1} x2={x2} y2={y2} stroke="rgba(154,173,210,0.3)" strokeWidth="1.4" />;
      })}
      <text x="110" y="96" textAnchor="middle" fontFamily="Space Grotesk, sans-serif" fontWeight="700" fontSize="34" fill="#e8edf7">
        {(v * 100).toFixed(0)}
        <tspan fontSize="16" fill="#97a3bb">%</tspan>
      </text>
    </svg>
  );
}

/* ----------------------------------------------------------- SimMatrix --- */
export function SimMatrix({ names, matrix }) {
  if (!matrix || !matrix.length) return null;
  const color = (v) => {
    const t = Math.max(-1, Math.min(1, v));
    if (t >= 0) return `rgba(34, 211, 238, ${0.1 + 0.75 * t})`;
    return `rgba(248, 113, 113, ${0.1 + 0.6 * -t})`;
  };
  return (
    <div className="matrix" style={{ gridTemplateColumns: `72px repeat(${matrix.length}, 1fr)` }}>
      <div />
      {names.map((n, i) => (
        <div key={i} className="mono faint" style={{ textAlign: "center", fontSize: 10 }}>
          C{i + 1}
        </div>
      ))}
      {matrix.map((row, i) => (
        <div key={i} style={{ contents: "" , display: "contents" }}>
          <div className="mono faint" style={{ fontSize: 10, display: "flex", alignItems: "center" }}>C{i + 1}</div>
          {row.map((v, j) => (
            <div key={j} className="matrix-cell" style={{ background: i === j ? "rgba(148,163,184,0.10)" : color(v) }}>
              {i === j ? "—" : v.toFixed(2)}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
