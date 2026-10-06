/* Hand-drawn 24px stroke icon set - consistent weight across the app. */

const base = {
  width: 20,
  height: 20,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round",
  strokeLinejoin: "round",
};

export const IconShield = (p) => (
  <svg {...base} {...p}>
    <path d="M12 3l7 3v5c0 4.5-3 8.4-7 10-4-1.6-7-5.5-7-10V6l7-3z" />
    <path d="M9.2 12.2l2 2 3.6-4" />
  </svg>
);

export const IconAtom = (p) => (
  <svg {...base} {...p}>
    <circle cx="12" cy="12" r="1.7" fill="currentColor" stroke="none" />
    <ellipse cx="12" cy="12" rx="9" ry="3.8" />
    <ellipse cx="12" cy="12" rx="9" ry="3.8" transform="rotate(60 12 12)" />
    <ellipse cx="12" cy="12" rx="9" ry="3.8" transform="rotate(120 12 12)" />
  </svg>
);

export const IconNetwork = (p) => (
  <svg {...base} {...p}>
    <circle cx="12" cy="5" r="2.2" />
    <circle cx="5" cy="18" r="2.2" />
    <circle cx="19" cy="18" r="2.2" />
    <path d="M11 7l-4.5 9M13 7l4.5 9M7.2 18h9.6" />
  </svg>
);

export const IconPulse = (p) => (
  <svg {...base} {...p}>
    <path d="M3 12h4l2.2-5.5 3.6 11L15 12h6" />
  </svg>
);

export const IconLock = (p) => (
  <svg {...base} {...p}>
    <rect x="5" y="10.5" width="14" height="9.5" rx="2.2" />
    <path d="M8 10.5V7.5a4 4 0 018 0v3" />
    <circle cx="12" cy="15.2" r="1.4" fill="currentColor" stroke="none" />
  </svg>
);

export const IconCpu = (p) => (
  <svg {...base} {...p}>
    <rect x="6" y="6" width="12" height="12" rx="2.4" />
    <rect x="9.5" y="9.5" width="5" height="5" rx="1" />
    <path d="M9 3v3M15 3v3M9 18v3M15 18v3M3 9h3M3 15h3M18 9h3M18 15h3" />
  </svg>
);

export const IconStethoscope = (p) => (
  <svg {...base} {...p}>
    <path d="M5 3v6a5 5 0 0010 0V3" />
    <path d="M5 3h2.4M12.6 3H15" />
    <path d="M10 14v2.5a4.5 4.5 0 009 0V13" />
    <circle cx="19" cy="10.6" r="2.2" />
  </svg>
);

export const IconChart = (p) => (
  <svg {...base} {...p}>
    <path d="M4 20V4" />
    <path d="M4 20h16" />
    <path d="M8 15l3.5-4 3 2.6L19 8" />
  </svg>
);

export const IconClinic = (p) => (
  <svg {...base} {...p}>
    <path d="M4 20v-8l8-6 8 6v8" />
    <path d="M10 20v-5h4v5" />
    <path d="M12 8.6v3M10.5 10.1h3" />
  </svg>
);

export const IconArrow = (p) => (
  <svg {...base} {...p}>
    <path d="M5 12h14M13 6l6 6-6 6" />
  </svg>
);

export const IconCheck = (p) => (
  <svg {...base} {...p}>
    <path d="M4.5 12.5l5 5L19.5 7" />
  </svg>
);

export const IconBolt = (p) => (
  <svg {...base} {...p}>
    <path d="M13 2L5 13.5h6L10 22l8-11.5h-6L13 2z" />
  </svg>
);

export const IconDatabase = (p) => (
  <svg {...base} {...p}>
    <ellipse cx="12" cy="5.5" rx="7.5" ry="2.8" />
    <path d="M4.5 5.5v13c0 1.55 3.36 2.8 7.5 2.8s7.5-1.25 7.5-2.8v-13" />
    <path d="M4.5 12c0 1.55 3.36 2.8 7.5 2.8s7.5-1.25 7.5-2.8" />
  </svg>
);

export const IconWifiOff = (p) => (
  <svg {...base} {...p}>
    <path d="M3 3l18 18" />
    <path d="M8.5 12.5a7 7 0 017 0M5 9.5a12 12 0 0114 0" />
    <circle cx="12" cy="18" r="1" fill="currentColor" stroke="none" />
  </svg>
);

export const IconBook = (p) => (
  <svg {...base} {...p}>
    <path d="M4 5.5A2.5 2.5 0 016.5 3H20v15.5H6.5A2.5 2.5 0 004 21V5.5z" />
    <path d="M4 18.5A2.5 2.5 0 016.5 16H20" />
  </svg>
);

export const IconWarn = (p) => (
  <svg {...base} {...p}>
    <path d="M12 3.5l9 16H3l9-16z" />
    <path d="M12 10v4.5" />
    <circle cx="12" cy="17.2" r="0.9" fill="currentColor" stroke="none" />
  </svg>
);

export const Logo = ({ size = 30 }) => (
  <svg width={size} height={size} viewBox="0 0 64 64" fill="none">
    <defs>
      <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stopColor="#22d3ee" />
        <stop offset="1" stopColor="#8b5cf6" />
      </linearGradient>
    </defs>
    <rect width="64" height="64" rx="15" fill="rgba(148,163,184,0.07)" stroke="rgba(154,173,210,0.25)" />
    <circle cx="32" cy="32" r="5.5" fill="url(#lg)" />
    <ellipse cx="32" cy="32" rx="19" ry="8.5" stroke="url(#lg)" strokeWidth="2.4" transform="rotate(-28 32 32)" />
    <ellipse cx="32" cy="32" rx="19" ry="8.5" stroke="#22d3ee" strokeOpacity="0.4" strokeWidth="1.5" transform="rotate(32 32 32)" />
    <circle cx="48" cy="23" r="2.4" fill="#22d3ee" />
    <circle cx="16" cy="42" r="2.4" fill="#8b5cf6" />
  </svg>
);
