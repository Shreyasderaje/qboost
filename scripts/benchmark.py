"""Benchmark suite: centralized vs FedAvg vs QAOA aggregation, plus robustness
against a poisoned clinic.

Writes docs/results/benchmarks.json and a summary chart.  Run headless:

    python scripts/benchmark.py                 # all three tasks
    python scripts/benchmark.py --task diabetes # one task
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import RESULTS_DIR  # noqa: E402
from backend.data.synth import make_dataset  # noqa: E402
from backend.data.tasks import TASKS  # noqa: E402
from backend.quantum.encoding import FeatureEncoder  # noqa: E402
from backend.quantum.vqc import VQC  # noqa: E402
from backend.simulation.manager import DEFAULT_RUN_CONFIG, _Run, _execute  # noqa: E402

N_TOTAL = 1000
ROUNDS = 4
EPOCHS = 2


def run_fl(config_overrides: dict) -> dict:
    cfg = dict(DEFAULT_RUN_CONFIG)
    cfg.update(
        {
            "rounds": ROUNDS,
            "n_total": N_TOTAL,
            "local_epochs": EPOCHS,
        }
    )
    cfg.update(config_overrides)
    run = _Run(cfg)
    _execute(run)
    if run.status == "failed":
        raise RuntimeError(run.error)
    return {
        "accs": [rec["global"]["acc"] for rec in run.rounds],
        "final_acc": run.result["final_acc"],
        "final_loss": run.result["final_loss"],
        "excluded_total": run.result["excluded_total"],
        "elapsed_s": sum(rec["elapsed_s"] for rec in run.rounds),
        "privacy": run.result["privacy"],
    }


def run_central(task: str) -> dict:
    spec = TASKS[task]
    data = make_dataset(spec, N_TOTAL, seed=99)
    n_test = int(0.2 * N_TOTAL)
    enc = FeatureEncoder(n_components=4).fit(data["X"][:-n_test])
    A = enc.transform(data["X"][:-n_test])
    At = enc.transform(data["X"][-n_test:])
    model = VQC(n_qubits=4, n_layers=3, learning_rate=0.12, seed=7)
    hist = model.fit(A, data["y"][:-n_test], epochs=EPOCHS * ROUNDS, batch_size=24)
    acc = model.score(At, data["y"][-n_test:])
    return {"final_acc": round(acc, 4), "loss_history": [round(v, 4) for v in hist["loss_history"]]}


def benchmark_task(task: str) -> dict:
    print(f"\n=== {task} ===", flush=True)
    out: dict = {"n_total": N_TOTAL, "rounds": ROUNDS, "local_epochs": EPOCHS}

    t0 = time.perf_counter()
    out["centralized"] = run_central(task)
    print(f"  centralized      acc={out['centralized']['final_acc']:.3f}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    t0 = time.perf_counter()
    out["fedavg"] = run_fl({"task": task, "strategy": "fedavg", "seed": 42})
    print(f"  fedavg           acc={out['fedavg']['final_acc']:.3f}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    t0 = time.perf_counter()
    out["qaoa"] = run_fl({"task": task, "strategy": "qaoa", "seed": 42})
    print(f"  qaoa             acc={out['qaoa']['final_acc']:.3f}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    t0 = time.perf_counter()
    out["fedavg_poisoned"] = run_fl({"task": task, "strategy": "fedavg", "seed": 42, "poison": True})
    print(f"  fedavg+poison    acc={out['fedavg_poisoned']['final_acc']:.3f}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    t0 = time.perf_counter()
    out["qaoa_poisoned"] = run_fl({"task": task, "strategy": "qaoa", "seed": 42, "poison": True})
    print(f"  qaoa+poison      acc={out['qaoa_poisoned']['final_acc']:.3f}  "
          f"excluded={out['qaoa_poisoned']['excluded_total']}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    return out


def make_chart(results: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = list(results["tasks"].keys())
    configs = ["fedavg", "qaoa", "fedavg_poisoned", "qaoa_poisoned"]
    labels = ["FedAvg", "QAOA", "FedAvg\n+poison", "QAOA\n+poison"]
    colors = ["#64748b", "#22d3ee", "#f87171", "#a78bfa"]

    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.6), facecolor="#0b1120")
    if len(tasks) == 1:
        axes = [axes]
    width = 0.55
    xs = np.arange(len(configs))
    for ax, task in zip(axes, tasks):
        res = results["tasks"][task]
        vals = [res[c]["final_acc"] for c in configs]
        central = res["centralized"]["final_acc"]
        ax.set_facecolor("#0b1120")
        bars = ax.bar(xs, vals, width, color=colors, edgecolor="none")
        ax.axhline(central, color="#fbbf24", lw=1.2, ls="--", label="centralized")
        ax.axhline(0.5, color="#334155", lw=0.8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=9, color="#e2e8f0")
        ax.set_ylim(0.4, 1.0)
        ax.set_title(task.upper(), color="#e2e8f0", fontsize=11)
        ax.set_xticks(xs, labels, fontsize=8, color="#94a3b8")
        ax.tick_params(colors="#64748b")
        for spine in ax.spines.values():
            spine.set_color("#1e293b")
    axes[0].legend(frameon=False, labelcolor="#94a3b8", fontsize=8)
    fig.suptitle("QBoost global-model accuracy after federation", color="#e2e8f0", fontsize=12)
    fig.tight_layout()
    path = RESULTS_DIR / "benchmarks.png"
    fig.savefig(path, dpi=150, facecolor="#0b1120")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), default=None)
    args = parser.parse_args()

    task_keys = [args.task] if args.task else sorted(TASKS)
    results = {
        "schema": "qboost.benchmarks/1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"n_total": N_TOTAL, "rounds": ROUNDS, "local_epochs": EPOCHS, "dp": True,
                    "dp_clip": 1.0, "dp_sigma": 0.3, "n_clinics": 4},
        "tasks": {},
    }
    for key in task_keys:
        results["tasks"][key] = benchmark_task(key)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "benchmarks.json"
    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    chart = make_chart(results)
    print(f"\nresults  -> {out_path}")
    print(f"chart    -> {chart}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
