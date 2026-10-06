import { useEffect, useState } from "react";

/* Animated variational circuit: 4 qubits, angle-encoding block, entangling
   layers, measurement. The round counter and energy readout tick live. */
export function CircuitHero() {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 1600);
    return () => clearInterval(id);
  }, []);

  const round = (tick % 5) + 1;
  const energy = (-2.1 - 0.18 * (tick % 5)).toFixed(2);
  const gates = ["RY", "H", "CZ", "RY", "CZ", "RY", "M"];

  return (
    <div className="circuit-panel">
      <div className="float-chip chip chip-cyan" style={{ top: 18, right: 18 }}>
        <span className="dot" /> round {round}/5
      </div>
      <div className="float-chip chip chip-violet" style={{ bottom: 60, left: -8, animationDelay: "1.2s" }}>
        ⟨Z₀⟩ {(0.42 + 0.11 * (tick % 4)).toFixed(2)}
      </div>

      <svg viewBox="0 0 560 250">
        {[0, 1, 2, 3].map((q) => (
          <g key={q}>
            <text className="qubit-label" x="14" y={44 + q * 54}>
              q{q}
            </text>
            <line className="wire" x1="38" y1={40 + q * 54} x2="540" y2={40 + q * 54} />
          </g>
        ))}

        {/* angle encoding */}
        {[0, 1, 2, 3].map((q) => (
          <g key={`enc${q}`}>
            <rect className="gate" x={62 + q * 6} y={40 + q * 54 - 15} width="30" height="30" rx="7" />
            <text x={77 + q * 6} y={40 + q * 54 + 4} textAnchor="middle" fontSize="10.5" fill="#9beefc" fontFamily="JetBrains Mono, monospace">
              RY
            </text>
          </g>
        ))}

        {/* entanglement ladder */}
        {[0, 1, 2].map((q) => (
          <g key={`ent${q}`} opacity="0.85">
            <line x1="210" y1={40 + q * 54} x2="210" y2={94 + q * 54} stroke="rgba(167,139,250,0.7)" strokeWidth="1.3" />
            <circle cx="210" cy={40 + q * 54} r="4.5" fill="#a78bfa" />
            <circle cx="210" cy={94 + q * 54} r="7" fill="none" stroke="#a78bfa" strokeWidth="1.6" />
          </g>
        ))}

        {/* second rotation layer */}
        {[0, 1, 2, 3].map((q) => (
          <g key={`ry2${q}`}>
            <rect className="gate" x="300" y={40 + q * 54 - 15} width="30" height="30" rx="7" />
            <text x="315" y={40 + q * 54 + 4} textAnchor="middle" fontSize="10.5" fill="#9beefc" fontFamily="JetBrains Mono, monospace">
              RY
            </text>
          </g>
        ))}

        {/* second entangle, crossing */}
        {[0, 2].map((q) => (
          <g key={`ent2${q}`} opacity="0.85">
            <line x1="380" y1={40 + q * 54} x2="380" y2={148 + q * 54} stroke="rgba(167,139,250,0.7)" strokeWidth="1.3" />
            <circle cx="380" cy={40 + q * 54} r="4.5" fill="#a78bfa" />
            <circle cx="380" cy={148 + q * 54} r="7" fill="none" stroke="#a78bfa" strokeWidth="1.6" />
          </g>
        ))}

        {/* measurements */}
        {[0, 1, 2, 3].map((q) => (
          <g key={`m${q}`}>
            <rect className={q === 0 ? "gate" : "gate gate-v"} x="470" y={40 + q * 54 - 15} width="30" height="30" rx="7" />
            <path
              d={`M 477 ${40 + q * 54 - 6} Q 485 ${40 + q * 54 + 10} 493 ${40 + q * 54 - 6}`}
              fill="none"
              stroke={q === 0 ? "#22d3ee" : "#a78bfa"}
              strokeWidth="1.4"
            />
            <line x1="483" y1={40 + q * 54 + 3} x2="489" y2={40 + q * 54 - 3} stroke={q === 0 ? "#22d3ee" : "#a78bfa"} strokeWidth="1.4" />
          </g>
        ))}

        {/* readout pulse */}
        <line className="flowline" x1="500" y1="40" x2="540" y2="40" stroke="#22d3ee" strokeWidth="1.6" />
      </svg>

      <div className="circuit-meta">
        <span>strongly-entangling ansatz · 4 qubits · 36 params</span>
        <span style={{ color: "#8df0cd" }}>E={energy}</span>
      </div>
    </div>
  );
}
