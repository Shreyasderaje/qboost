"""Launch a federated training run from the command line.

Examples
--------
python scripts/run_federation.py --task diabetes --rounds 5
python scripts/run_federation.py --task tb --strategy fedavg --no-dp
python scripts/run_federation.py --task malaria --poison   # robustness demo
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import ARTIFACTS_DIR  # noqa: E402
from backend.data.tasks import TASKS  # noqa: E402
from backend.simulation.manager import DEFAULT_RUN_CONFIG, _Run, _execute  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a QBoost federated training simulation")
    parser.add_argument("--task", default="diabetes", choices=sorted(TASKS))
    parser.add_argument("--strategy", default="qaoa", choices=["qaoa", "fedavg"])
    parser.add_argument("--clinics", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=DEFAULT_RUN_CONFIG["local_epochs"])
    parser.add_argument("--n-total", type=int, default=1400)
    parser.add_argument("--dp", dest="dp", action="store_true", default=True)
    parser.add_argument("--no-dp", dest="dp", action="store_false")
    parser.add_argument("--poison", action="store_true", help="make the last clinic malicious")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = dict(DEFAULT_RUN_CONFIG)
    config.update(
        {
            "task": args.task,
            "strategy": args.strategy,
            "clinics": args.clinics,
            "rounds": args.rounds,
            "local_epochs": args.epochs,
            "n_total": args.n_total,
            "dp": args.dp,
            "poison": args.poison,
            "seed": args.seed,
        }
    )

    print()
    print(f"  QBoost federation  ·  task={args.task}  strategy={args.strategy}  "
          f"clinics={args.clinics}  rounds={args.rounds}  dp={'on' if args.dp else 'off'}")
    print("  " + "-" * 78)

    run = _Run(config)
    t0 = time.perf_counter()
    _execute(run)

    if run.status == "failed":
        print(run.error)
        return 1

    for rec in run.rounds:
        cohort = f"{len(rec['cohort'])}/{len(rec['clients'])}"
        qaoa_note = ""
        if rec.get("qaoa"):
            qaoa_note = f"  qaoa energy {rec['qaoa']['energy']:+.3f}"
        print(
            f"  round {rec['round']}  cohort {cohort}  "
            f"global acc {rec['global']['acc']:.3f}  loss {rec['global']['loss']:.3f}"
            f"{qaoa_note}"
        )
        for c in rec["clients"]:
            mark = "in " if c["selected"] else "OUT"
            print(
                f"    · {c['name']:<26} n={c['n_samples']:<4} "
                f"acc {c['val_acc']:.3f}  sim {c['mean_similarity']:+.2f}  [{mark}]"
            )

    print("  " + "-" * 78)
    res = run.result
    print(
        f"  final accuracy {res['final_acc']:.3f}  ·  privacy ε≈{res['privacy']['epsilon_total']}  ·  "
        f"excluded updates {res['excluded_total']}  ·  {time.perf_counter() - t0:.1f}s"
    )
    print(f"  model bundle  →  {Path(res['artifact_path'])}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
