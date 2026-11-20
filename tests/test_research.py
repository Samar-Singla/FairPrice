import numpy as np
import pandas as pd
import pytest

from backend.core import (
    OPS,
    GenerateConfig,
    TrainConfig,
    adjusted,
    explain,
    generate,
    matched,
    train,
    validate,
)


def test_reproducible_and_bias_truth():
    a = generate(GenerateConfig(scenario="A"))
    pd.testing.assert_frame_equal(a, generate(GenerateConfig(scenario="A")))
    b = generate(GenerateConfig(scenario="B", bias=4))
    c = generate(GenerateConfig(scenario="C", bias=4))
    np.testing.assert_allclose(
        b.observed_price - a.observed_price,
        4 * (a.geographic_zone == "Zone B"),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        c.observed_price - a.observed_price, 4 * a.pickup_density, atol=1e-12
    )
    assert (pd.to_datetime(a.timestamp).dt.hour == a.time_of_day).all()


@pytest.mark.parametrize("scenario,expected", [("A", 0), ("B", 4), ("C", 2)])
def test_multiseed_adjusted_gap(scenario, expected):
    gaps, covered = [], []
    for seed in range(8):
        d = generate(GenerateConfig(scenario=scenario, seed=seed, n=2000))
        row = adjusted(d, "observed_price")["groups"][0]
        gaps.append(row["adjusted_gap"])
        covered.append(row["low"] <= expected <= row["high"])
        assert row["p_holm"] >= row["p"]
    assert abs(np.mean(gaps) - expected) < 0.15
    assert sum(covered) >= 5
    if scenario != "A":
        assert min(gaps) > 1.5


def test_adjustment_removes_operational_imbalance():
    d = generate(GenerateConfig(scenario="A", n=5000, noise=0.05))
    g = adjusted(d, "observed_price")["groups"][0]
    assert abs(g["raw_gap"]) > 0.5
    assert abs(g["adjusted_gap"]) < 0.02
    assert g["low"] < g["high"]


@pytest.fixture(scope="module")
def fitted():
    df = generate(GenerateConfig(scenario="C", n=1200))
    return df, train(df, TrainConfig(features="proxy"))


def test_residuals_split_and_inputs(fitted):
    _df, (pipe, r, tr, te) = fitted
    np.testing.assert_allclose(te.residual, te.observed_price - te.predicted_price)
    assert set(tr.transaction_id).isdisjoint(te.transaction_id)
    assert tr.timestamp.max() <= te.timestamp.min()
    assert set(pipe.feature_names_in_) == set(OPS + ["pickup_density"])
    assert not {
        "audit_group",
        "observed_price",
        "planted_premium",
        "ground_truth_fair_price",
    } & set(pipe.feature_names_in_)
    assert r["split"]["id_overlap"] == 0


def test_proxy_and_mitigation(fitted):
    df, (_pipe, r, _tr, te) = fitted
    assert r["sensitivity"]["zone_B_minus_A"] == 0
    assert r["sensitivity"]["density_08_minus_02"] > 1.5
    assert r["predicted_audit"]["groups"][0]["adjusted_gap"] > 1.5
    _, m, _, test_m = train(df, TrainConfig(features="operational"))
    assert abs(m["predicted_audit"]["groups"][0]["adjusted_gap"]) < 1e-8
    assert m["sensitivity"]["density_08_minus_02"] == 0
    assert r["observed_audit"] == m["observed_audit"]
    assert te.transaction_id.tolist() == test_m.transaction_id.tolist()
    assert m["mae"] > r["mae"]


def test_matching_no_reuse_and_calipers(fitted):
    _, (_, _, _, te) = fitted
    m = matched(te)
    assert m["n"] > 20
    assert 0 < m["coverage"] <= 1
    pairs = m["pairs"]
    assert len({p["reference_id"] for p in pairs}) == len(pairs)
    assert len({p["target_id"] for p in pairs}) == len(pairs)
    assert all(p["distance"] <= 1.25 for p in pairs)
    assert all(
        abs(p["target_price"] - p["reference_price"] - p["gap"]) < 1e-10 for p in pairs
    )


@pytest.mark.parametrize("model", ["linear", "forest"])
def test_explanation_additivity(model):
    d = generate(GenerateConfig(n=400))
    p, r, tr, te = train(d, TrainConfig(model=model))
    e = explain(p, tr, te.iloc[:1])
    assert (
        abs(e["base"] + sum(c["value"] for c in e["contributions"]) - e["prediction"])
        < 1e-8
    )
    assert r["mae"] < 5


def test_validation_missing_and_insufficient():
    d = generate(GenerateConfig(n=400))
    d.loc[0, "distance_km"] = np.nan
    assert validate(d)["missing_features"] == 1
    _p, r, _tr, _te = train(d, TrainConfig())
    assert np.isfinite(r["mae"])
    assert adjusted(d.iloc[:10], "observed_price")["groups"] == []
    d.loc[1, "transaction_id"] = d.loc[0, "transaction_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate(d)


def test_reweighting_runs():
    d = generate(GenerateConfig(n=400))
    _, r, _, _ = train(d, TrainConfig(reweight=True))
    assert np.isfinite(r["mae"])


def test_rank_deficiency_is_reported():
    d = generate(GenerateConfig(n=400))
    d["demand_level"] = 1.0
    result = adjusted(d, "observed_price")
    assert result["groups"] == []
    assert "Non-identifiable" in result["warning"]


def test_missing_training_predictor_is_rejected():
    d = generate(GenerateConfig(n=400))
    d.loc[:299, "distance_km"] = np.nan
    with pytest.raises(ValueError, match="training period"):
        train(d, TrainConfig())
