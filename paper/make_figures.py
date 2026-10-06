"""Generate print-quality figures for the QBoost paper from real run data.

Reads docs/results/benchmarks.json and artifacts/*_qaoa_global.json (produced
by scripts/benchmark.py and scripts/run_federation.py) and writes vector PDF
figures into paper/figures/.  Re-run after regenerating benchmarks:

    python paper/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# Print-friendly, colourblind-safe palette (restrained, no rainbow).
C_QAOA = "#0F766E"      # deep teal
C_FED = "#B91C1C"       # muted red
C_POISON_FED = "#F87171"
C_POISON_QAOA = "#5EEAD4"
C_CENTRAL = "#64748B"   # slate

plt.rcParams.update({
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,   # embed TrueType, keeps text selectable
})

bench = json.loads((ROOT / "docs" / "results" / "benchmarks.json").read_text(encoding="utf-8"))
tasks = bench["tasks"]
TASK_LABEL = {"diabetes": "Diabetes risk", "malaria": "Malaria", "tb": "Tuberculosis"}
TASK_ORDER = ["diabetes", "malaria", "tb"]


def style_axis(ax):
    ax.set_ylim(0.55, 0.95)
    ax.set_xticks(range(0, 5), ["init", "r1", "r2", "r3", "r4"])


# ---------------------------------------------------------------- trajectories
fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.55), sharey=True)
for ax, key in zip(axes, TASK_ORDER):
    fed = tasks[key]["fedavg"]["accs"]
    qaoa = tasks[key]["qaoa"]["accs"]
    ax.plot(range(len(qaoa)), qaoa, color=C_QAOA, lw=1.6, marker="o", ms=3, label="QAOA")
    ax.plot(range(len(fed)), fed, color=C_FED, lw=1.6, marker="s", ms=3, label="FedAvg")
    ax.set_title(TASK_LABEL[key], pad=2)
    ax.set_xlabel("Round", labelpad=1)
    style_axis(ax)
axes[0].set_ylabel("Global accuracy")
axes[-1].legend(frameon=False, loc="lower right", handlelength=1.6)
fig.tight_layout()
fig.savefig(OUT / "fig_traj.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ robustness
CONFIGS = [
    ("fedavg", "FedAvg", C_FED),
    ("qaoa", "QAOA", C_QAOA),
    ("fedavg_poisoned", "FedAvg + poison", C_POISON_FED),
    ("qaoa_poisoned", "QAOA + poison", "#0D9488"),
]
fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.55), sharey=True)
width = 0.62
for ax, key in zip(axes, TASK_ORDER):
    res = tasks[key]
    central = res["centralized"]["final_acc"]
    vals = [res[c]["final_acc"] for c, _, _ in CONFIGS]
    xs = range(len(CONFIGS))
    ax.axhline(central, color=C_CENTRAL, lw=1.0, ls="--", zorder=1)
    bars = ax.bar(xs, vals, width, color=[c for _, _, c in CONFIGS], zorder=2)

    ax.set_title(TASK_LABEL[key])
    ax.set_xticks(list(xs), ["FedAvg", "QAOA", "FedAvg+P", "QAOA+P"],
                  fontsize=6.5, rotation=22, ha="right", rotation_mode="anchor")
    ax.set_ylim(0.55, 0.95)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Final accuracy")
axes[-1].text(0.98, central + 0.008, "centralized", transform=axes[-1].get_yaxis_transform(),
              fontsize=6.5, color=C_CENTRAL, ha="right")
fig.tight_layout()
fig.savefig(OUT / "fig_robust.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------- QAOA energy convergence
bundle = json.loads((ROOT / "artifacts" / "diabetes_qaoa_global.json").read_text(encoding="utf-8"))
rounds = [r for r in bundle["history"] if r.get("qaoa")]
round_idx = len(rounds) - 1
hist = rounds[round_idx]["qaoa"]["history"]
energy = rounds[round_idx]["qaoa"]["energy"]

fig, ax = plt.subplots(figsize=(3.4, 2.2))
ax.plot(range(len(hist)), hist, color=C_QAOA, lw=1.5)
ax.axhline(energy, color="#B45309", lw=1.0, ls="--")
ax.annotate(f"cohort energy {energy:.2f}", xy=(len(hist) * 0.42, energy),
            xytext=(len(hist) * 0.30, energy + (max(hist) - min(hist)) * 0.28),
            fontsize=7, color="#B45309",
            arrowprops=dict(arrowstyle="-", color="#B45309", lw=0.7))
ax.set_xlabel("Adam step")
ax.set_ylabel(r"Circuit expectation $\langle H_c\rangle$")
ax.set_title(f"QAOA optimization, round {round_idx + 1} (diabetes)", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig_qaoa.pdf", bbox_inches="tight")
plt.close(fig)

print("figures written to", OUT)
