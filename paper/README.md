# QBoost — Research Paper

IEEE-conference-style paper describing the QBoost system.

- `main.tex` — LaTeX source (IEEEtran, compiled with Tectonic)
- `figures/` — vector PDF figures, generated from real run data
- `make_figures.py` — regenerates every figure from `docs/results/benchmarks.json`
  and `artifacts/*_qaoa_global.json`
- `main.pdf` — compiled paper (4 pages)

## Rebuild

```bash
# 1. regenerate figures (after re-running scripts/benchmark.py)
python paper/make_figures.py

# 2. compile (any XeTeX-based engine works; Tectonic recommended)
tectonic main.tex
```

Before submission, replace the placeholder author block in `main.tex` with
your name and affiliation. An optional, commented-out AI-assistance
disclosure is included near the end of `main.tex` — most journals and
conferences require such disclosure when generative AI assisted drafting;
uncomment and adapt it to the venue's policy.
