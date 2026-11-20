import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import core, neural, store
from backend.app import app


def test_battery_simulation_preserves_original_and_has_known_effect():
    a = core.generate(core.GenerateConfig(n=600, battery_bias=0))
    b = core.generate(core.GenerateConfig(n=600, battery_bias=12))
    expected = 12 * np.maximum(0, (30 - a.battery_percent) / 30) ** 2
    np.testing.assert_allclose(
        b.observed_price - a.observed_price, expected, atol=1e-12
    )
    assert expected.max() > 5
    assert (expected[a.battery_percent >= 30] == 0).all()


@pytest.mark.parametrize("scenario", ["A", "B", "C"])
def test_network_forward_trace_checkpoint_and_leakage(scenario):
    df = core.generate(core.GenerateConfig(n=600, scenario=scenario, battery_bias=8))
    pipe, r, tr, te = core.train(
        df, core.TrainConfig(model="neural", features="proxy", include_battery=True), 42
    )
    trace = neural.trace(pipe, te[pipe.feature_names_in_].iloc[:1])
    assert trace["prediction"] == pytest.approx(
        pipe.predict(te[pipe.feature_names_in_].iloc[:1])[0], abs=1e-10
    )
    info = r["neural_training"]
    assert info["fit_n"] + info["validation_n"] == len(tr)
    assert info["fit_end"] <= info["validation_start"] < r["split"]["test_start"]
    assert (
        info["history"][info["best_epoch"] - 1]["validation_mae"]
        <= min(x["validation_mae"] for x in info["history"]) + 0.0001
    )
    assert len(info["benchmark"]) == 3
    assert np.isfinite(r["mae"])
    # Altering only final test labels cannot change learned preprocessing, weights or checkpoint.
    changed = df.copy()
    changed.loc[changed.index >= len(tr), "observed_price"] += 1000
    p2, r2, _, _ = core.train(
        changed,
        core.TrainConfig(model="neural", features="proxy", include_battery=True),
        42,
    )
    for x, y in zip(pipe["model"].coefs_, p2["model"].coefs_):
        np.testing.assert_array_equal(x, y)
    assert info["history"] == r2["neural_training"]["history"]


def test_neural_api_battery_removal_and_integrity(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path)
    client = TestClient(app)
    d = client.post("/api/datasets", json={"n": 900, "battery_bias": 15}).json()
    response = client.post(
        "/api/train",
        json={"dataset_id": d["id"], "model": "neural", "include_battery": True},
    )
    assert response.status_code == 200, response.text
    e = response.json()
    pred = client.post(f"/api/experiments/{e['id']}/predict", json={}).json()
    assert pred["neural_trace"]["prediction"] == pytest.approx(pred["predicted_price"])
    replay = client.post(f"/api/experiments/{e['id']}/neural/replay", json={}).json()
    frames = replay["frames"]
    selected = next(f for f in frames if f["epoch"] == replay["best_epoch"])
    assert selected["trace"]["prediction"] == pytest.approx(pred["predicted_price"])
    assert len(selected["surface"]) == 165
    assert frames[0]["trace"]["weights"] != frames[-1]["trace"]["weights"]
    changed = client.post(
        f"/api/experiments/{e['id']}/neural/replay", json={"distance_km": 20}
    ).json()
    assert (
        changed["frames"][0]["trace"]["prediction"] != frames[0]["trace"]["prediction"]
    )
    assert changed["frames"][0]["trace"]["weights"] == frames[0]["trace"]["weights"]
    assert all(
        replay["range"][0] <= v <= replay["range"][1]
        for f in frames
        for v in f["surface"]
    )
    assert pred["battery_used"]
    assert np.ptp([x["price"] for x in pred["battery_sweep"]]) > 0.01
    assert client.post(f"/api/experiments/{e['id']}/verify", json={}).json()["passed"]
    for method in ["remove_battery", "reweight"]:
        response = client.post(
            f"/api/experiments/{e['id']}/mitigate", json={"method": method}
        )
        assert response.status_code == 200, response.text
        child = response.json()
        p = client.post(f"/api/experiments/{child['id']}/predict", json={}).json()
        if method == "remove_battery":
            assert not p["battery_used"]
            assert np.ptp([x["price"] for x in p["battery_sweep"]]) == 0
        assert child["result"]["observed_audit"] == e["result"]["observed_audit"]
    assert (
        client.post(
            f"/api/experiments/{e['id']}/predict", json={"battery_percent": 101}
        ).status_code
        == 422
    )


def test_missing_battery_is_not_invented():
    df = core.generate(core.GenerateConfig(n=300)).drop(columns="battery_percent")
    with pytest.raises(ValueError, match="not invented"):
        core.train(df, core.TrainConfig(model="neural", include_battery=True))
