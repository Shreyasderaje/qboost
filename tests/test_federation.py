"""End-to-end federation runs (small, fast configurations)."""

import json

import numpy as np

from backend.config import ARTIFACTS_DIR
from backend.data.synth import make_federated_splits
from backend.data.tasks import get_task
from backend.federated.client import ClinicNode
from backend.federated.server import FederationServer


def _tiny_federation(strategy="qaoa", rounds=2, poison=False):
    spec = get_task("diabetes")
    splits = make_federated_splits(spec, n_clinics=3, n_total=540, alpha=0.8, seed=11)
    events = []
    server = FederationServer(
        "diabetes",
        strategy=strategy,
        seed=11,
        qaoa_steps=40,
        event_sink=events,
    )
    nodes = []
    for k in range(3):
        cid = f"clinic-{k + 1}"
        creds = server.register_clinic(cid, f"Test Clinic {k+1}", "Test Region", len(splits["clients"][k]["y"]))
        node = ClinicNode(
            cid,
            f"Test Clinic {k+1}",
            "Test Region",
            "test device",
            splits["clients"][k]["X"],
            splits["clients"][k]["y"],
            arch=server.arch,
            local_epochs=2,
            batch_size=16,
            seed=100 + k,
            gradient_ascent=(poison and k == 2),
            fake_metrics={"val_acc": 0.95, "val_loss": 0.1} if (poison and k == 2) else None,
        )
        node.set_credentials(creds["key"], creds["secret"], ["clinic-1", "clinic-2", "clinic-3"])
        nodes.append(node)

    for r in range(1, rounds + 1):
        payload = server.round_payload(r)
        server.begin_round(r)
        for node in nodes:
            server.collect_phase1(node.cid, node.phase1(r, payload, payload["round_seed"]))
        server.select_cohort()
        participating = nodes if strategy == "fedavg" else [n for n in nodes if n.cid in server.cohort]
        for node in participating:
            server.collect_phase2(node.cid, node.phase2(r, masked=(strategy == "fedavg")))
        server.aggregate()
    return server, nodes


def test_qaoa_federation_improves_and_saves_artifact():
    server, _ = _tiny_federation(rounds=3)
    accs = [rec["global"]["acc"] for rec in server.round_records]
    assert len(accs) == 3
    assert accs[-1] > 0.6
    assert server.round_records[-1]["global"]["acc"] >= server.round_records[0]["global"]["acc"] - 0.05

    path = ARTIFACTS_DIR / "test_diabetes_global.json"
    server.save_artifact(path)
    art = json.loads(path.read_text(encoding="utf-8"))
    assert art["schema"] == "qboost.model/1"
    assert art["task"] == "diabetes"
    assert len(art["params"]) == 3 * 4 * 3
    assert set(art["encoder"]) == {"n_components", "mean", "components", "x_min", "x_max", "n_features_in"}
    path.unlink()


def test_fedavg_masked_sum_matches_plain_average():
    server, nodes = _tiny_federation(strategy="fedavg", rounds=2)
    rec = server.round_records[0]
    assert set(rec["cohort"]) == {"clinic-1", "clinic-2", "clinic-3"}
    # Weights must follow sample counts (n-weighted FedAvg).
    n_total = sum(c["n_samples"] for c in rec["clients"])
    for c in rec["contributions"]:
        own = next(x for x in rec["clients"] if x["cid"] == c["cid"])
        assert abs(c["weight"] - own["n_samples"] / n_total) < 1e-9
    assert server.round_records[-1]["global"]["acc"] > 0.52


def test_qaoa_excludes_poisoned_clinic():
    server, _ = _tiny_federation(rounds=3, poison=True)
    last = server.round_records[-1]
    poison_row = next(c for c in last["clients"] if c["cid"] == "clinic-3")
    assert not poison_row["selected"], "poisoned clinic must be excluded by QAOA"
    assert last["global"]["acc"] > 0.6


def test_fedavg_with_poison_degrades():
    _, _ = _tiny_federation(strategy="fedavg", rounds=3, poison=True)
    server_poisoned, _ = _tiny_federation(strategy="fedavg", rounds=3, poison=True)
    server_clean, _ = _tiny_federation(strategy="fedavg", rounds=3, poison=False)
    assert (
        server_poisoned.round_records[-1]["global"]["acc"]
        < server_clean.round_records[-1]["global"]["acc"] + 0.12
    )


def test_round_status_and_events():
    server, _ = _tiny_federation(rounds=1)
    status = server.round_status()
    assert status["rounds_completed"] == 1
    kinds = {e["kind"] for e in server.events()}
    assert {"clinic_registered", "round_started", "client_trained", "cohort_selected", "round_complete"} <= kinds
