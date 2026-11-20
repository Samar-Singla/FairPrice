import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import core, inspection, store
from backend.app import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path)
    return TestClient(app)


def create_run(client, scenario="B", model="linear", features="unrestricted"):
    d = client.post("/api/datasets", json={"n": 600, "scenario": scenario}).json()
    response = client.post(
        "/api/train", json={"dataset_id": d["id"], "model": model, "features": features}
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("scenario", ["A", "B", "C"])
@pytest.mark.parametrize("model", ["linear", "forest"])
def test_every_scenario_and_estimator(client, scenario, model):
    e = create_run(client, scenario, model)
    uid = e["id"]
    lineage = client.get(f"/api/experiments/{uid}/lineage").json()
    assert lineage["external_data_fetched"] is False
    assert lineage["source_type"] == "Synthetic simulator"
    assert lineage["features"] == e["result"]["features"]
    response = client.post(
        f"/api/experiments/{uid}/predict", json=inspection.TripInput().model_dump()
    )
    assert response.status_code == 200
    pred = response.json()
    assert np.isfinite(pred["predicted_price"])
    assert pred["experiment_id"] == uid
    v = client.post(f"/api/experiments/{uid}/verify", json={})
    assert v.status_code == 200, v.text
    assert v.json()["passed"], v.json()["checks"]
    assert len(v.json()["checks"]) == 10
    assert (
        client.get(f"/api/experiments/{uid}/verification").json()["id"]
        == v.json()["id"]
    )
    assert client.get(f"/api/experiments/{uid}/report").status_code == 200
    assert client.get(f"/api/experiments/{uid}/summary").status_code == 200


@pytest.mark.parametrize(
    "method", ["remove_geography", "remove_proxy", "operational_only", "reweight"]
)
def test_every_intervention(client, method):
    e = create_run(client, "C")
    m = client.post(f"/api/experiments/{e['id']}/mitigate", json={"method": method})
    assert m.status_code == 200, m.text
    child = m.json()
    assert child["parent_id"] == e["id"]
    assert child["result"]["observed_audit"] == e["result"]["observed_audit"]
    v = client.post(f"/api/experiments/{child['id']}/verify", json={}).json()
    assert v["passed"], v["checks"]
    assert len(v["checks"]) == 11
    if method == "remove_geography":
        assert child["result"]["features"] == e["result"]["features"]
        assert child["result"]["predictions"] == e["result"]["predictions"]


def test_new_prediction_matches_saved_estimator(client):
    e = create_run(client)
    bundle = store.model(e["id"])
    row = bundle["test"].iloc[0]
    body = inspection.TripInput(
        **{k: row[k] for k in inspection.TripInput.model_fields}
    ).model_dump()
    response = client.post(f"/api/experiments/{e['id']}/predict", json=body).json()
    assert abs(response["predicted_price"] - row.predicted_price) < 1e-8
    swap = {**body, "geographic_zone": "Zone B"}
    direct = client.post(f"/api/experiments/{e['id']}/predict", json=swap).json()
    sensitivity = client.post(
        f"/api/experiments/{e['id']}/sensitivity",
        json={"transaction_id": row.transaction_id, "geographic_zone": "Zone B"},
    ).json()
    assert abs(direct["predicted_price"] - sensitivity["after"]) < 1e-8


def test_verification_detects_artifact_tampering(client):
    e = create_run(client)
    file = store.ROOT / f"{e['dataset_id']}.csv"
    file.write_text(file.read_text() + "\n")
    v = client.post(f"/api/experiments/{e['id']}/verify", json={}).json()
    assert not v["passed"]
    assert v["checks"][0]["status"] == "fail"


def test_failed_artifact_write_does_not_publish_metadata(client):
    def broken_writer(path):
        path.write_text("partial")
        raise OSError("simulated disk failure")

    with pytest.raises(OSError):
        store.save("dataset", {"config": {}}, writer=broken_writer, suffix=".csv")
    assert store.listing("dataset") == []
    assert not list(store.ROOT.glob("*.csv"))
    assert not list(store.ROOT.glob(".*.tmp"))


@pytest.mark.parametrize(
    "body",
    [
        {"time_of_day": 24},
        {"distance_km": -2},
        {"pickup_density": 2},
        {"audit_group": "G1"},
        {"geographic_zone": "Unknown"},
    ],
)
def test_prediction_input_validation(client, body):
    # Validation happens before any nonexistent model is loaded.
    assert client.post("/api/experiments/missing/predict", json=body).status_code == 422


def test_pagination_import_and_health(client):
    e = create_run(client)
    uid = e["dataset_id"]
    a = client.get(f"/api/datasets/{uid}?limit=30").json()
    b = client.get(f"/api/datasets/{uid}?limit=30&offset=30").json()
    assert not {r["transaction_id"] for r in a["rows"]} & {
        r["transaction_id"] for r in b["rows"]
    }
    csv = client.get(f"/api/datasets/{uid}/csv").text
    imported = client.post("/api/datasets/import", json={"csv": csv}).json()
    e2 = client.post("/api/train", json={"dataset_id": imported["id"]}).json()
    lineage = client.get(f"/api/experiments/{e2['id']}/lineage").json()
    assert lineage["source_type"] == "Imported CSV"
    assert lineage["formula"] is None
    assert client.post(f"/api/experiments/{e2['id']}/verify", json={}).json()["passed"]
    assert client.get("/api/health").json()["database"] == "connected"
    assert client.get("/api/system").json()["dataset_count"] == 2


def test_unrepresented_zone_missing_and_invalid_inputs(client):
    df = core.generate(core.GenerateConfig(n=600))
    df["geographic_zone"] = "Zone A"
    df.loc[:5, "pickup_density"] = np.nan
    imported = client.post(
        "/api/datasets/import", json={"csv": df.to_csv(index=False)}
    ).json()
    response = client.post("/api/train", json={"dataset_id": imported["id"]})
    assert response.status_code == 200, response.text
    e = response.json()
    assert e["result"]["observed_audit"]["groups"] == []
    assert e["result"]["matched"]["n"] == 0
    assert client.get(f"/api/experiments/{e['id']}/summary").status_code == 200
    assert (
        client.post(
            "/api/datasets/import", json={"csv": "transaction_id,timestamp\na,b"}
        ).status_code
        == 422
    )


def test_constant_outcome_does_not_fabricate_inference():
    df = core.generate(core.GenerateConfig(n=600))
    df["observed_price"] = 10.0
    assert core.adjusted(df, "observed_price")["groups"] == []


def test_concurrent_database_initialization(client):
    from concurrent.futures import ThreadPoolExecutor

    def read_status(_):
        return client.get("/api/system").status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(read_status, range(16))) == [200] * 16


@pytest.mark.parametrize("suffix", [".csv", ".joblib"])
def test_missing_artifact_returns_a_failed_check(client, suffix):
    e = create_run(client)
    uid = e["dataset_id"] if suffix == ".csv" else e["id"]
    (store.ROOT / f"{uid}{suffix}").unlink()
    response = client.post(f"/api/experiments/{e['id']}/verify", json={})
    assert response.status_code == 200
    assert not response.json()["passed"]
    assert "missing" in response.json()["checks"][0]["detail"]
