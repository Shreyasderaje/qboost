"""Run one clinic as an independent process against a live coordination API.

This exercises the true distributed deployment path: the clinic process owns
its local dataset, trains the VQC on-device, seals every payload, and speaks
to the coordinator only through HTTP.  Any participating clinic may trigger
the collective round steps (selection, aggregation); the coordinator guards
make those calls safe no matter which process fires them first.

Usage
-----
# terminal 1 - coordinator
uvicorn backend.api.main:app --port 8000

# terminal 2 - open a federation session
curl -X POST localhost:8000/api/federation/init \
     -H 'content-type: application/json' -d '{"task":"diabetes","strategy":"qaoa"}'

# terminal 3..n - one process per clinic
python scripts/clinic_simulator.py --session <session_id> --clinic clinic-1 \
       --name "Phulbani CHC" --region "Odisha, India" --rounds 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.data.synth import make_dataset  # noqa: E402
from backend.data.tasks import get_task  # noqa: E402
from backend.federated.client import ClinicNode  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a simulated clinic against the coordination API")
    parser.add_argument("--session", required=True)
    parser.add_argument("--clinic", default="clinic-1")
    parser.add_argument("--name", default="Phulbani CHC")
    parser.add_argument("--region", default="Odisha, India")
    parser.add_argument("--task", default="diabetes")
    parser.add_argument("--samples", type=int, default=280)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--base", default="http://127.0.0.1:8000/api")
    args = parser.parse_args()

    spec = get_task(args.task)
    api = httpx.Client(base_url=args.base, timeout=120.0)

    health = api.get("/health").json()
    print(f"coordinator {health['service']} v{health['version']} is {health['status']}")

    creds = api.post(
        f"/federation/{args.session}/register",
        json={"cid": args.clinic, "name": args.name, "region": args.region, "n_samples": args.samples},
    ).json()
    print(f"registered as {args.clinic} ({args.name}) - envelope key issued")

    data = make_dataset(spec, args.samples, seed=abs(hash(args.clinic)) % 10000)
    node = ClinicNode(
        args.clinic,
        args.name,
        args.region,
        "Android tablet · 4 GB",
        data["X"],
        data["y"],
        arch={"n_qubits": 4, "n_layers": 3},
        seed=abs(hash(args.clinic)) % 1000,
    )
    node.set_credentials(creds["key"], creds["secret"], [])

    def wait_until(predicate, description, timeout=900.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            status = api.get(f"/federation/{args.session}/status").json()
            if predicate(status):
                return status
            time.sleep(1.0)
        raise TimeoutError(f"timed out waiting for {description}")

    for r in range(1, args.rounds + 1):
        # Wait until every registered clinic is on this round.
        wait_until(lambda s: s["round"] == r and s["rounds_completed"] == r - 1, f"round {r} to open")

        bundle = api.get(f"/federation/{args.session}/round/{r}/bundle").json()
        node._total_samples = bundle.get("total_samples")
        token1 = node.phase1(r, bundle, bundle["round_seed"])
        api.post(f"/federation/{args.session}/phase1", json={"cid": args.clinic, "token": token1})
        print(f"round {r}: local training done - metrics + sketch sealed and submitted")

        # Any clinic may fire the collective selection; guards make this safe.
        while True:
            status = wait_until(
                lambda s: len(s["phase1_received"]) == len(s["registered"]), "all phase-1 payloads"
            )
            resp = api.post(f"/federation/{args.session}/select")
            if resp.status_code == 200:
                break
            time.sleep(1.0)
        cohort = resp.json()["cohort"]

        if args.clinic in cohort:
            token2 = node.phase2(r, masked=(bundle.get("strategy") == "fedavg"))
            api.post(f"/federation/{args.session}/phase2", json={"cid": args.clinic, "token": token2})
            print(f"round {r}: selected by QAOA ({len(cohort)}/{len(status['registered'])}) - DP update revealed")
        else:
            print(f"round {r}: excluded by QAOA - update stays on device")

        while True:
            resp = api.post(f"/federation/{args.session}/aggregate")
            if resp.status_code == 200:
                record = resp.json()
                break
            time.sleep(1.0)
        print(f"round {r}: aggregation complete - global acc {record['global']['acc']:.3f}")

    print("clinic run complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
