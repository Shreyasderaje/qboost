import { useEffect, useState } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";
import { Logo } from "./icons.jsx";
import { api } from "../lib/api.js";

export function NavBar() {
  const [health, setHealth] = useState(null);
  const { pathname } = useLocation();

  useEffect(() => {
    let alive = true;
    api
      .health()
      .then((h) => alive && setHealth(h))
      .catch(() => alive && setHealth(false));
    return () => {
      alive = false;
    };
  }, [pathname]);

  const links = [
    ["/", "Overview"],
    ["/dashboard", "Console"],
    ["/diagnose", "Diagnose"],
    ["/network", "Clinics"],
    ["/research", "Research"],
  ];

  return (
    <header className="nav">
      <div className="container nav-inner">
        <Link to="/" className="brand">
          <Logo />
          <span>
            QBoost
            <small>quantum federated diagnostics</small>
          </span>
        </Link>
        <nav className="nav-links">
          {links.map(([to, label]) => (
            <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => (isActive ? "active" : "")}>
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="nav-right">
          <span className="nav-status" title={health ? "coordination API reachable" : "coordination API offline"}>
            <span className={`dot ${health ? "" : "off"}`} />
            {health ? `api · v${health.version}` : "api offline"}
          </span>
          <Link to="/dashboard" className="btn btn-primary btn-sm">
            Launch console
          </Link>
        </div>
      </div>
    </header>
  );
}

export function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer-grid">
          <div>
            <div className="brand" style={{ marginBottom: 12 }}>
              <Logo size={26} />
              <span>QBoost</span>
            </div>
            <p className="muted" style={{ fontSize: 13.5, maxWidth: "34ch" }}>
              Specialist-level diagnostic screening for the last mile, trained federatively with quantum
              optimization. Patient records never leave the clinic.
            </p>
          </div>
          <div>
            <h5>Platform</h5>
            <Link to="/dashboard">Training console</Link>
            <Link to="/diagnose">Clinic console</Link>
            <Link to="/network">Clinic network</Link>
            <Link to="/research">Research & benchmarks</Link>
          </div>
          <div>
            <h5>Protocol</h5>
            <Link to="/research#model">Variational quantum model</Link>
            <Link to="/research#aggregation">QAOA aggregation</Link>
            <Link to="/research#privacy">Privacy engineering</Link>
            <Link to="/research#results">Benchmarks</Link>
          </div>
          <div>
            <h5>Grounding</h5>
            <a href="https://www.who.int/teams/global-tuberculosis-programme/tb-reports" target="_blank" rel="noreferrer">
              WHO Global TB Report
            </a>
            <a href="https://www.who.int/teams/global-malaria-programme/reports/world-malaria-report-2023" target="_blank" rel="noreferrer">
              World Malaria Report
            </a>
            <a href="https://diabetesatlas.org/" target="_blank" rel="noreferrer">
              IDF Diabetes Atlas
            </a>
            <a href="https://nihcc.app.box.com/v/ChestXray-NIHCC" target="_blank" rel="noreferrer">
              NIH ChestX-ray14
            </a>
          </div>
        </div>
        <div className="footer-bottom">
          <span>© 2026 QBoost · open research platform</span>
          <span>
            Research demonstration — models are screening aids, not medical devices, and are not certified for
            clinical use.
          </span>
        </div>
      </div>
    </footer>
  );
}
