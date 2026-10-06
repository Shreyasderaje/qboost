"""HTTP API smoke tests (fastapi TestClient)."""

import time

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "diabetes" in body["tasks"]


def test_task_catalog(client):
    r = client.get("/api/tasks")
    assert r.status_code == 200
    tasks = {t["key"]: t for t in r.json()}
    assert set(tasks) == {"tb", "malaria", "diabetes"}
    glucose = next(f for f in tasks["diabetes"]["features"] if f["name"] == "glucose")
    assert glucose["lo"] == 60.0 and glucose["hi"] == 300.0


def test_simulation_run_and_inference(client):
    r = client.post(
        "/api/simulation/start",
        json={"task": "diabetes", "rounds": 2, "clinics": 2, "n_total": 360, "local_epochs": 1,
              "seed": 3},
    )
    assert r.status_code == 200
    run_id = r.json()["run_id"]

    deadline = time.time() + 420
    state = None
    while time.time() < deadline:
        state = client.get(f"/api/simulation/{run_id}").json()
        if state["status"] in ("completed", "failed"):
            break
        time.sleep(2.0)
    assert state is not None and state["status"] == "completed", state.get("error")
    assert len(state["rounds"]) == 2

    # The run produced an artifact; inference must now work end to end.
    models = client.get("/api/models").json()
    assert any(m["task"] == "diabetes" for m in models)

    r = client.post(
        "/api/inference",
        json={
            "task": "diabetes",
            "features": {"glucose": 185, "bmi": 38.2, "age": 54, "insulin": 210,
                          "blood_pressure": 92, "pregnancies": 4, "skinfold": 39, "pedigree": 1.2},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk"] <= 1.0
    assert body["label"]
    assert len(body["contributions"]) == 8


def test_inference_input_validation(client):
    r = client.post("/api/inference", json={"task": "diabetes", "features": {"glucose": 100}})
    assert r.status_code == 422


def test_federation_session_protocol(client):
    sid = client.post(
        "/api/federation/init", json={"task": "malaria", "strategy": "qaoa"}
    ).json()["session_id"]
    assert sid

    # Selecting with no clinics registered fails cleanly.
    r = client.post(f"/api/federation/{sid}/select")
    assert r.status_code in (400, 409, 500)
