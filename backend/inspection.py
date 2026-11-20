"""Recompute saved-run evidence and expose transparent data/model lineage."""

import hashlib
from datetime import UTC, datetime
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from . import core, neural, store


class TripInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    distance_km: float = Field(6, ge=0.1, le=100)
    duration_min: float = Field(18, ge=1, le=360)
    demand_level: float = Field(1.2, ge=0.1, le=4)
    availability: float = Field(0.6, ge=0, le=1)
    operating_cost: float = Field(3.5, ge=0, le=100)
    time_of_day: int = Field(12, ge=0, le=23)
    pickup_density: float = Field(0.3, ge=0, le=1)
    battery_percent: float = Field(50, ge=0, le=100)
    geographic_zone: Literal["Zone A", "Zone B", "Zone C"] = "Zone A"


def predict(uid, body):
    bundle = store.model(uid)
    e = store.get(uid, "experiment")
    data = body.model_dump()
    data["hour_sin"] = np.sin(data["time_of_day"] * np.pi / 12)
    data["hour_cos"] = np.cos(data["time_of_day"] * np.pi / 12)
    row = pd.DataFrame([data])
    p, tr = bundle["pipeline"], bundle["train"]
    names = p.feature_names_in_.tolist()
    warnings = []
    for name in names:
        if name == "geographic_zone":
            if data[name] not in set(tr[name]):
                warnings.append("This zone was not represented in training.")
        elif not tr[name].min() <= data[name] <= tr[name].max():
            warnings.append(f"{name} is outside the observed training range.")
    sweep = pd.concat(
        [row.assign(battery_percent=b) for b in range(0, 101, 5)], ignore_index=True
    )
    prices = p.predict(sweep[names])
    return {
        "neural_trace": neural.trace(p, row[names]),
        "battery_sweep": [
            {"battery": b, "price": float(v)} for b, v in zip(range(0, 101, 5), prices)
        ],
        "battery_used": "battery_percent" in names,
        "experiment_id": uid,
        "predicted_price": float(p.predict(row[names])[0]),
        "units": "CU",
        "model": e["result"]["config"]["model"],
        "used_features": names,
        "source": "Imported CSV"
        if e["dataset_config"]["scenario"] == "Imported"
        else "Locally generated synthetic transactions",
        "warnings": warnings,
        "scope": "A prediction from the selected saved model, not a real-world fare quote. No actual charge or accuracy can be known for this new trip without its observed outcome.",
    }


def lineage(uid):
    e = store.get(uid, "experiment")
    d = store.get(e["dataset_id"], "dataset")
    r = e["result"]
    imported = d["config"]["scenario"] == "Imported"
    return {
        "experiment_id": uid,
        "dataset_id": d["id"],
        "dataset_created": d["created"],
        "trained_at": e["created"],
        "source_type": "Imported CSV" if imported else "Synthetic simulator",
        "source_description": d["validation"]["provenance"],
        "external_data_fetched": False,
        "config": d["config"],
        "features": r["features"],
        "target": "observed_price",
        "split": r["split"],
        "model": r["config"],
        "neural_training": r.get("neural_training"),
        "battery_assumption": f"Synthetic battery values; planted low-battery premium strength {d['config'].get('battery_bias', 0)} CU. Battery collected from real passengers: no."
        if not imported
        else "Imported battery values, if present, are user-supplied and not independently verified.",
        "reproducibility": r.get("reproducibility", {}),
        "parent_id": e["parent_id"],
        "pipeline": [
            "Validate dataset",
            "Chronological 75/25 split",
            "Fit preprocessing on training rows",
            "Fit model to observed_price",
            "Predict held-out rows",
            "Independently audit outcomes",
        ],
        "excluded_inputs": [
            "audit_group",
            "transaction_id",
            "ground_truth_fair_price",
            "planted_premium",
            "observed_price (target only)",
        ],
        "formula": None
        if imported
        else "4 + 1.2 × distance + 0.18 × duration + 2 × demand − 2 × availability + 1.3 × cost + 0.7 × sin(hour) + 0.3 × cos(hour)",
        "limitations": "No external platform data have been fetched. CSV uploads are user-supplied; this application does not verify their original provenance. Synthetic results do not establish performance on real-world fares.",
    }


def verify(uid):
    checks = []

    def add(name, passed, detail):
        checks.append(
            {"check": name, "status": "pass" if passed else "fail", "detail": detail}
        )

    e = store.get(uid, "experiment")
    r = e["result"]
    try:
        df, bundle = store.frame(e["dataset_id"]), store.model(uid)
        add(
            "Saved artifacts",
            True,
            "Dataset and estimator loaded successfully; recorded file hashes checked where available.",
        )
    except (ValueError, OSError) as exc:
        add("Saved artifacts", False, str(exc))
        return store.save(
            "verification",
            {
                "experiment_id": uid,
                "checks": checks,
                "passed": False,
                "scope": "Saved-run integrity, not a full application test.",
            },
        )
    tr, te, model = bundle["train"], bundle["test"], bundle["pipeline"]
    names = list(model.feature_names_in_)
    forbidden = {
        "audit_group",
        "transaction_id",
        "ground_truth_fair_price",
        "planted_premium",
        "observed_price",
    }
    add(
        "Pricing feature contract",
        not bool(set(names) & forbidden),
        "Audit-only fields and simulator truth are excluded from pricing inputs.",
    )
    add(
        "Scenario C policy",
        e["dataset_config"]["scenario"] != "C" or "geographic_zone" not in names,
        "Explicit geography is excluded from proxy-leakage pricing models.",
    )
    add(
        "Disjoint chronological split",
        not bool(set(tr.transaction_id) & set(te.transaction_id))
        and tr.timestamp.max() <= te.timestamp.min()
        and len(tr) + len(te) == len(df),
        f"{len(tr)} training / {len(te)} test rows; timestamps and transaction IDs checked.",
    )
    fresh = model.predict(te[names])
    saved = pd.DataFrame(r["predictions"])
    same_ids = list(saved.transaction_id) == list(te.transaction_id)
    add(
        "Prediction replay",
        same_ids and np.allclose(fresh, saved.predicted_price, atol=1e-8),
        f"Replayed all {len(te)} held-out rows with the saved fitted model.",
    )
    residual = te.observed_price.to_numpy() - fresh
    add(
        "Residual identity",
        np.allclose(residual, saved.residual, atol=1e-8),
        "Observed minus predicted equals each saved residual.",
    )
    mae, rmse = float(np.abs(residual).mean()), float(np.sqrt(np.mean(residual**2)))
    add(
        "Error metric replay",
        abs(mae - r["mae"]) < 1e-8 and abs(rmse - r["rmse"]) < 1e-8,
        f"Recomputed MAE {mae:.6f} and RMSE {rmse:.6f} CU.",
    )
    fresh_audit = core.adjusted(te, "observed_price")
    a, b = fresh_audit["groups"], r["observed_audit"]["groups"]
    add(
        "Independent audit replay",
        len(a) == len(b)
        and all(
            abs(x["adjusted_gap"] - y["adjusted_gap"]) < 1e-8 for x, y in zip(a, b)
        ),
        "Re-estimated geographic coefficients from the saved held-out charges and operational controls.",
    )
    explanation = core.explain(model, tr, te.iloc[:1])
    total = explanation["base"] + sum(v["value"] for v in explanation["contributions"])
    add(
        "Explanation additivity",
        abs(total - explanation["prediction"]) < 1e-8,
        "Background plus feature contributions equals the actual prediction for one held-out row.",
    )
    rep = r.get("reproducibility", {})
    if rep.get("dataset_sha256"):
        digest = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
        add(
            "Dataset fingerprint",
            digest == rep["dataset_sha256"],
            "Canonical dataset SHA-256 matches the input recorded during training.",
        )
    else:
        checks.append(
            {
                "check": "Dataset fingerprint",
                "status": "unavailable",
                "detail": "Legacy run predates dataset fingerprints. Retrain to record this check.",
            }
        )
    if e.get("parent_id"):
        parent = store.get(e["parent_id"], "experiment")
        parent_rows = parent["result"]["predictions"]
        add(
            "Mitigation comparison",
            e["dataset_id"] == parent["dataset_id"]
            and [p["transaction_id"] for p in parent_rows] == list(te.transaction_id)
            and np.allclose(
                [p["observed_price"] for p in parent_rows], te.observed_price
            ),
            "Parent and child use identical held-out transactions and unchanged observed charges.",
        )
    return store.save(
        "verification",
        {
            "experiment_id": uid,
            "checked_at": datetime.now(UTC).isoformat(),
            "checks": checks,
            "passed": all(c["status"] == "pass" for c in checks),
            "scope": "Recomputes saved-run integrity and numerical consistency. Does not prove scientific validity or test every browser workflow. Use the full verification script for application/API coverage.",
        },
    )


def replay(uid, body):
    bundle = store.model(uid)
    pipe = bundle["pipeline"]
    checkpoints = getattr(pipe["model"], "replay_checkpoints_", [])
    if not checkpoints:
        return {
            "frames": [],
            "message": "Train a new neural model to record epoch checkpoints. Existing models remain usable for predictions.",
        }
    data = body.model_dump()
    data["hour_sin"] = np.sin(data["time_of_day"] * np.pi / 12)
    data["hour_cos"] = np.cos(data["time_of_day"] * np.pi / 12)
    row = pd.DataFrame([data])[pipe.feature_names_in_]
    # Fixed domain and shared color range allow honest comparisons between epochs.
    grid = pd.DataFrame(
        [
            {**data, "distance_km": d, "battery_percent": b}
            for b in range(0, 101, 10)
            for d in range(1, 31, 2)
        ]
    )
    x = pipe["preprocess"].transform(grid[pipe.feature_names_in_])
    frames = []
    info = store.get(uid, "experiment")["result"]["neural_training"]
    for c in checkpoints:
        a = x.copy()
        for i, (w, bias) in enumerate(zip(c["weights"], c["biases"])):
            a = a @ w + bias
            if i < len(c["weights"]) - 1:
                a = np.maximum(a, 0)
        frames.append(
            {
                "epoch": c["epoch"],
                "trace": neural.trace(pipe, row, c),
                "surface": a[:, 0].tolist(),
                **info["history"][c["epoch"] - 1],
            }
        )
    values = [v for f in frames for v in f["surface"]]
    return {
        "frames": frames,
        "best_epoch": info["best_epoch"],
        "range": [min(values), max(values)],
        "distances": list(range(1, 31, 2)),
        "batteries": list(range(0, 101, 10)),
        "scope": "Recorded fitting checkpoints. Fixed connectivity; learned weights and biases change across epochs. Surface is model response, not confidence bounds. Grid points can lie outside training support.",
    }
