import json

import pytest
from fastapi.testclient import TestClient

from backend import store
from backend.app import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path)
    return TestClient(app)


def test_end_to_end_persistence(client):
    d = client.post("/api/datasets", json={"scenario": "C", "n": 400}).json()
    uid = d["id"]
    assert client.get("/api/datasets").json()[0]["id"] == uid
    assert client.get(f"/api/datasets/{uid}/csv").status_code == 200
    assert len(client.get(f"/api/datasets/{uid}?zone=Zone%20A").json()["rows"]) > 0
    e = client.post("/api/train", json={"dataset_id": uid}).json()
    eid = e["id"]
    assert "geographic_zone" not in e["result"]["features"]
    assert client.get(f"/api/experiments/{eid}").json() == e
    assert store.get(eid, "experiment")["dataset_id"] == uid
    tid = e["result"]["predictions"][0]["transaction_id"]
    explanation = client.get(f"/api/experiments/{eid}/explain/{tid}")
    assert explanation.status_code == 200
    s = client.post(
        f"/api/experiments/{eid}/sensitivity",
        json={"transaction_id": tid, "geographic_zone": "Zone A"},
    ).json()
    assert s["delta"] == 0
    m = client.post(
        f"/api/experiments/{eid}/mitigate", json={"method": "operational_only"}
    ).json()
    assert m["parent_id"] == eid
    assert m["result"]["split"] == e["result"]["split"]
    assert m["result"]["observed_audit"] == e["result"]["observed_audit"]
    assert (
        client.post(
            "/api/reviews",
            json={
                "experiment_id": eid,
                "owner": "Auditor",
                "status": "Investigating",
                "note": "Review proxy correlation and omitted controls.",
            },
        ).status_code
        == 200
    )
    report = client.get(f"/api/experiments/{eid}/report")
    assert json.loads(report.text)["reviews"][0]["owner"] == "Auditor"
    assert client.get(f"/api/experiments/{eid}/audit").status_code == 200
    assert len(client.get("/api/experiments").json()) == 2
    summary = client.get(f"/api/experiments/{eid}/summary")
    assert summary.status_code == 200
    assert "Charged prices" in summary.text
    assert "Auditor" in summary.text


def test_api_validation(client):
    assert client.post("/api/datasets", json={"n": 1}).status_code == 422
    assert client.post("/api/datasets", json={"scenario": "Z"}).status_code == 422
    assert client.post("/api/train", json={"dataset_id": "notfound"}).status_code == 404
    assert (
        client.post(
            "/api/train", json={"dataset_id": "x", "features": "audit_group"}
        ).status_code
        == 422
    )
    assert client.get("/api/datasets/..%2Fsecret/csv").status_code == 404
    assert client.get("/api/datasets/nope?limit=99999").status_code == 422
    assert (
        client.post(
            "/api/reviews",
            json={
                "experiment_id": "nope",
                "owner": "A",
                "status": "Investigating",
                "note": "a valid long note",
            },
        ).status_code
        == 404
    )


def test_import_roundtrip(client):
    d = client.post("/api/datasets", json={"n": 400}).json()
    csv = client.get(f"/api/datasets/{d['id']}/csv").text
    imported = client.post("/api/datasets/import", json={"csv": csv})
    assert imported.status_code == 200
    e = client.post(
        "/api/train",
        json={"dataset_id": imported.json()["id"], "features": "operational"},
    )
    assert e.status_code == 200


def test_bad_csv_schema_and_numeric_values(client):
    assert (
        client.post("/api/datasets/import", json={"csv": "a,b\n1,2"}).status_code == 422
    )
    d = client.post("/api/datasets", json={"n": 400}).json()
    frame = store.frame(d["id"])
    frame["distance_km"] = "invalid"
    assert (
        client.post(
            "/api/datasets/import", json={"csv": frame.to_csv(index=False)}
        ).status_code
        == 422
    )
