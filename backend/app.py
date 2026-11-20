import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import core, inspection, store

app = FastAPI(title="FairPrice AI", version=core.VERSION)

logger = logging.getLogger("fairprice")


@app.middleware("http")
async def request_metadata(request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    response.headers["Server-Timing"] = (
        f"app;dur={(time.perf_counter() - started) * 1000:.1f}"
    )
    return response


@app.exception_handler(KeyError)
async def not_found(request, exc):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=404, content={"detail": "Record not found"})


@app.exception_handler(ValueError)
async def invalid(request, exc):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/api/health")
def health():
    try:
        with store.connection() as db:
            db.execute("SELECT 1").fetchone()
        return {
            "status": "ok",
            "version": core.VERSION,
            "database": "connected",
            "demo": os.environ.get("FAIRPRICE_DEMO") == "1",
        }
    except Exception:
        logger.exception("Database health check failed")
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "database": "unavailable"},
        )


@app.get("/api/system")
def system_info():
    return {
        "version": core.VERSION,
        "dataset_count": len(store.listing("dataset")),
        "experiment_count": len(store.listing("experiment")),
        "review_count": len(store.listing("review")),
        "execution": "Python backend / CPU",
        "storage": "SQLite + local CSV and model artifacts",
        "external_data_fetched": False,
        "sources": ["Synthetic simulator", "User-imported CSV"],
        "deployment": "Mac-hosted Cloudflare demonstration"
        if os.environ.get("FAIRPRICE_DEMO") == "1"
        else "Local research workspace",
    }


@app.post("/api/datasets")
def generate(config: core.GenerateConfig):
    df = core.generate(config)
    item = store.save(
        "dataset",
        {
            "config": config.model_dump(),
            "validation": core.validate(df),
            "methodology_version": core.VERSION,
        },
        writer=lambda path: df.to_csv(path, index=False),
        suffix=".csv",
    )
    return item


class ImportBody(BaseModel):
    csv: str = Field(max_length=8_000_000)


@app.post("/api/datasets/import")
def import_data(body: ImportBody):
    try:
        df = pd.read_csv(io.StringIO(body.csv), dtype={"transaction_id": str})
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(
            "CSV is empty or malformed. Check the header and quoting."
        ) from exc
    required = {
        "transaction_id",
        "timestamp",
        "geographic_zone",
        "time_of_day",
        "observed_price",
        "pickup_density",
        *core.OPS[:-2],
    }
    if required - set(df):
        raise ValueError(f"Missing CSV columns: {sorted(required - set(df))}")
    if not 300 <= len(df) <= 15000:
        raise ValueError("CSV must have 300–15000 rows")
    if (
        not pd.api.types.is_numeric_dtype(df.time_of_day)
        or not df.time_of_day.between(0, 23).all()
    ):
        raise ValueError("time_of_day must be in [0,23]")
    import numpy as np

    df["hour_sin"] = np.sin(df.time_of_day * np.pi / 12)
    df["hour_cos"] = np.cos(df.time_of_day * np.pi / 12)
    if df.timestamp.isna().any():
        raise ValueError("Timestamps cannot be missing")
    df["timestamp"] = pd.to_datetime(df.timestamp, errors="raise", utc=True).astype(str)
    quality = core.validate(df)
    item = store.save(
        "dataset",
        {
            "config": {"scenario": "Imported", "n": len(df), "seed": 42},
            "validation": {
                **quality,
                "provenance": "User CSV. Provenance and control sufficiency require human review.",
            },
            "methodology_version": core.VERSION,
        },
        writer=lambda path: df.to_csv(path, index=False),
        suffix=".csv",
    )
    return item


@app.get("/api/datasets")
def datasets():
    return store.listing("dataset")


@app.get("/api/datasets/{uid}")
def dataset(
    uid: str,
    zone: str | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
):
    item, df = store.get(uid, "dataset"), store.frame(uid)
    if zone:
        df = df[df.geographic_zone == zone]
    summary = (
        df.groupby("geographic_zone")
        .agg(
            n=("observed_price", "size"),
            price=("observed_price", "mean"),
            distance=("distance_km", "mean"),
            density=("pickup_density", "mean"),
        )
        .reset_index()
        .to_dict("records")
    )
    # JSON round-trip converts missing cells to null.
    return {
        **item,
        "total": len(df),
        "rows": json.loads(df.iloc[offset : offset + limit].to_json(orient="records")),
        "summary": json.loads(pd.DataFrame(summary).to_json(orient="records")),
    }


@app.get("/api/datasets/{uid}/csv")
def csv_export(uid: str):
    df = store.frame(uid)
    return Response(
        df.to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="fairprice-data.csv"'},
    )


class TrainingRequest(core.TrainConfig):
    dataset_id: str


def execute(dataset_id, config, parent=None, intervention=None):
    dataset = store.get(dataset_id, "dataset")
    # Scenario C never admits explicit geography, including unrestricted requests.
    if dataset["config"]["scenario"] == "C" and config.features in [
        "unrestricted",
        "geographic",
    ]:
        config = config.model_copy(
            update={
                "features": "proxy"
                if config.features == "unrestricted"
                else "operational"
            }
        )
    df = store.frame(dataset_id)
    core.validate(df)
    pipe, result, tr, te = core.train(df, config, dataset["config"]["seed"])
    item = store.save(
        "experiment",
        {
            "dataset_id": dataset_id,
            "dataset_config": dataset["config"],
            "parent_id": parent,
            "intervention": intervention,
            "result": result,
        },
        writer=lambda path: joblib.dump(
            {"pipeline": pipe, "train": tr, "test": te}, path
        ),
        suffix=".joblib",
    )
    return item


@app.post("/api/train")
def train(body: TrainingRequest):
    return execute(
        body.dataset_id, core.TrainConfig(**body.model_dump(exclude={"dataset_id"}))
    )


@app.get("/api/experiments")
def experiments():
    return [
        {k: v for k, v in e.items() if k != "result"}
        | {
            "config": e["result"]["config"],
            "mae": e["result"]["mae"],
            "rmse": e["result"]["rmse"],
            "predicted_audit": e["result"]["predicted_audit"],
        }
        for e in store.listing("experiment")
    ]


@app.get("/api/experiments/{uid}")
def experiment(uid: str):
    return store.get(uid, "experiment")


@app.get("/api/experiments/{uid}/lineage")
def lineage(uid: str):
    return inspection.lineage(uid)


@app.post("/api/experiments/{uid}/predict")
def predict(uid: str, body: inspection.TripInput):
    return inspection.predict(uid, body)


@app.post("/api/experiments/{uid}/neural/replay")
def neural_replay(uid: str, body: inspection.TripInput):
    return inspection.replay(uid, body)


@app.post("/api/experiments/{uid}/verify")
def verify(uid: str):
    return inspection.verify(uid)


@app.get("/api/experiments/{uid}/verification")
def verification(uid: str):
    store.get(uid, "experiment")
    return next(
        (v for v in store.listing("verification") if v["experiment_id"] == uid), None
    )


@app.get("/api/experiments/{uid}/audit")
def audit(uid: str):
    return {
        k: v
        for k, v in store.get(uid, "experiment")["result"].items()
        if k
        in [
            "observed_audit",
            "predicted_audit",
            "residual_audit",
            "matched",
            "groups",
            "sensitivity",
        ]
    }


@app.get("/api/experiments/{uid}/explain/{tid}")
def explanation(uid: str, tid: str):
    bundle = store.model(uid)
    row = bundle["test"][bundle["test"].transaction_id == tid]
    if row.empty:
        raise HTTPException(404, "Held-out transaction not found")
    return {
        **core.explain(bundle["pipeline"], bundle["train"], row),
        "transaction": json.loads(row.to_json(orient="records"))[0],
    }


class Intervention(BaseModel):
    transaction_id: str
    geographic_zone: Literal["Zone A", "Zone B", "Zone C"] | None = None
    pickup_density: float | None = Field(None, ge=0, le=1)
    distance_km: float | None = Field(None, ge=0.1, le=100)
    demand_level: float | None = Field(None, ge=0.1, le=4)


@app.post("/api/experiments/{uid}/sensitivity")
def sensitivity(uid: str, body: Intervention):
    bundle = store.model(uid)
    row = bundle["test"][bundle["test"].transaction_id == body.transaction_id].copy()
    if row.empty:
        raise HTTPException(404, "Held-out transaction not found")
    pipe = bundle["pipeline"]
    before = float(pipe.predict(row[pipe.feature_names_in_])[0])
    for key, value in body.model_dump(
        exclude={"transaction_id"}, exclude_none=True
    ).items():
        row[key] = value
    after = float(pipe.predict(row[pipe.feature_names_in_])[0])
    return {
        "before": before,
        "after": after,
        "delta": after - before,
        "warning": "Fixed-input sensitivity only. Modified combinations may be off-support and are not causal interventions.",
    }


class Mitigation(BaseModel):
    method: Literal[
        "remove_geography",
        "remove_proxy",
        "operational_only",
        "reweight",
        "remove_battery",
    ]


@app.post("/api/experiments/{uid}/mitigate")
def mitigate(uid: str, body: Mitigation):
    e = store.get(uid, "experiment")
    c = core.TrainConfig(**e["result"]["config"])
    if body.method == "remove_geography":
        c.features = (
            "proxy" if c.features in ["proxy", "unrestricted"] else "operational"
        )
    elif body.method == "remove_proxy":
        c.features = (
            "geographic"
            if c.features in ["geographic", "unrestricted"]
            else "operational"
        )
    elif body.method == "remove_battery":
        c.include_battery = False
    elif body.method == "operational_only":
        c.features = "operational"
        c.include_battery = False
    else:
        c.reweight = True
    return execute(e["dataset_id"], c, uid, body.method)


class Review(BaseModel):
    experiment_id: str
    owner: str = Field(min_length=1, max_length=100)
    status: Literal["Investigating", "Explained", "Mitigate", "Accepted limitation"]
    note: str = Field(min_length=10, max_length=5000)


@app.post("/api/reviews")
def review(body: Review):
    store.get(body.experiment_id, "experiment")
    return store.save("review", body.model_dump())


@app.get("/api/reviews")
def reviews():
    return store.listing("review")


@app.get("/api/experiments/{uid}/report")
def report(uid: str):
    e = store.get(uid, "experiment")
    e["reviews"] = [r for r in store.listing("review") if r["experiment_id"] == uid]
    e["lineage"] = inspection.lineage(uid)
    e["verification"] = next(
        (v for v in store.listing("verification") if v["experiment_id"] == uid), None
    )
    e["limitations"] = [
        "Synthetic research prototype; no causal or legal conclusions.",
        "Model attributions describe predictions, not social causes.",
        "Repeated exploration inflates false positives beyond within-outcome Holm correction.",
        "Known-zone temporal holdout; no unseen-zone validation.",
        "Reweighting equalizes representation, not outcomes; biased labels remain biased.",
    ]
    return Response(
        json.dumps(e, indent=2),
        media_type="application/json",
        headers={
            "Content-Disposition": 'attachment; filename="fairprice-research-report.json"'
        },
    )


@app.get("/api/experiments/{uid}/summary")
def summary(uid: str):
    e = store.get(uid, "experiment")
    r = e["result"]
    lines = [
        "# FairPrice AI — Research Summary",
        "",
        f"Run: {uid}",
        f"Created: {e['created']}",
        f"Dataset: {e['dataset_id']}",
        f"Scenario: {e['dataset_config']['scenario']}; seed: {e['dataset_config']['seed']}",
        f"Model: {r['config']['model']}; feature policy: {r['config']['features']}",
        f"Methodology: {r['methodology_version']}",
        "",
        "## Predictive utility",
        "",
        f"MAE: {r['mae']:.4f} CU; RMSE: {r['rmse']:.4f} CU. Test rows: {r['split']['test_n']}.",
        f"Split: {r['split']['method']}",
        "",
        "## Independent fairness findings",
        "",
    ]
    for key, title in [
        ("observed_audit", "Charged prices"),
        ("predicted_audit", "Model predictions"),
        ("residual_audit", "Prediction errors"),
    ]:
        lines += [
            f"### {title}",
            "",
            "| Contrast | Raw gap | Adjusted gap | 95% CI | Holm p |",
            "|---|---:|---:|---|---:|",
        ]
        for g in r[key]["groups"]:
            lines.append(
                f"| {g['zone']} vs A | {g['raw_gap']:.3f} | {g['adjusted_gap']:.3f} | [{g['low']:.3f}, {g['high']:.3f}] | {g['p_holm']:.3g} |"
            )
        lines += ["", r[key]["warning"], ""]
    lines += [
        "## Model sensitivity",
        "",
        f"Mean Zone B minus A intervention: {r['sensitivity']['zone_B_minus_A']:.3f} CU.",
        f"Mean pickup-density 0.8 minus 0.2 intervention: {r['sensitivity']['density_08_minus_02']:.3f} CU.",
        r["sensitivity"]["label"],
        "",
        "## Matching",
        "",
        f"Pairs: {r['matched']['n']}; target coverage: {r['matched']['coverage']:.1%}.",
        r["matched"].get("method", ""),
        "",
        "## Intervention lineage",
        "",
        f"Parent run: {e['parent_id'] or 'None'}. Intervention: {e['intervention'] or 'Baseline'}.",
        "",
        "## Human review",
        "",
    ]
    reviews = [v for v in store.listing("review") if v["experiment_id"] == uid]
    for v in reviews:
        lines += [f"- {v['created']} — {v['owner']}: {v['status']}", f"  {v['note']}"]
    if not reviews:
        lines.append("No human review has been recorded for this run.")
    verified = next(
        (v for v in store.listing("verification") if v["experiment_id"] == uid), None
    )
    lines += ["", "## Software verification", ""]
    if verified:
        lines.append(f"Checked: {verified['created']}. Passed: {verified['passed']}.")
        lines.extend(
            f"- {c['check']}: {c['status']} — {c['detail']}" for c in verified["checks"]
        )
        lines.append(verified["scope"])
    else:
        lines.append(
            "No saved-run verification has been recorded yet. Use System checks in the application."
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "Synthetic research prototype. Statistical disparities do not establish discriminatory intent, illegality or causality. HC3 intervals assume independent rows and an adequate adjustment model. Holm correction is within each outcome only. Matching and feature interventions can lack overlap. Audit confidence intervals omit model-training uncertainty. Original charged prices are unchanged by mitigation.",
        "",
        "Full machine-readable evidence is available from the JSON report endpoint.",
    ]
    return Response(
        "\n".join(lines),
        media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="fairprice-summary.md"'},
    )


DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
