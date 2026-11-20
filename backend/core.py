"""Synthetic research engine. No sensitive attributes or simulator truth enter estimators."""

import hashlib
from importlib.metadata import version
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel, Field
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests

from . import neural

VERSION = "1.0.0"
ZONES = ["Zone A", "Zone B", "Zone C"]
OPS = [
    "distance_km",
    "duration_min",
    "demand_level",
    "availability",
    "operating_cost",
    "hour_sin",
    "hour_cos",
]


class GenerateConfig(BaseModel):
    scenario: Literal["A", "B", "C"] = "B"
    n: int = Field(2400, ge=300, le=15000)
    seed: int = Field(42, ge=0, le=2147483647)
    noise: float = Field(1, ge=0.05, le=10)
    bias: float = Field(4, ge=0, le=20)
    battery_bias: float = Field(0, ge=0, le=20)
    target_zone: Literal["Zone A", "Zone B", "Zone C"] = "Zone B"


class TrainConfig(BaseModel):
    model: Literal["linear", "forest", "neural"] = "linear"
    features: Literal["operational", "proxy", "geographic", "unrestricted"] = (
        "unrestricted"
    )
    reweight: bool = False
    include_battery: bool = False


def generate(c: GenerateConfig):
    r = np.random.default_rng(c.seed)
    z = r.choice(ZONES, c.n, p=[0.4, 0.35, 0.25])
    target = (z == c.target_zone).astype(float)
    hour = r.integers(0, 24, c.n)
    distance = r.gamma(3, 2, c.n) + 0.6 * target
    duration = distance * 2 + r.uniform(2, 9, c.n)
    demand = r.uniform(0.2, 2, c.n) + 0.15 * target
    supply = r.uniform(0.2, 1, c.n)
    cost = 1 + distance * 0.22 + r.uniform(0, 2, c.n)
    density = np.clip(r.normal(0.25 + 0.5 * target, 0.13, c.n), 0, 1)
    sine, cosine = np.sin(hour * np.pi / 12), np.cos(hour * np.pi / 12)
    fair = (
        4
        + 1.2 * distance
        + 0.18 * duration
        + 2 * demand
        - 2 * supply
        + 1.3 * cost
        + 0.7 * sine
        + 0.3 * cosine
    )
    premium = (
        c.bias * target
        if c.scenario == "B"
        else c.bias * density
        if c.scenario == "C"
        else np.zeros(c.n)
    )
    df = pd.DataFrame(
        {
            "transaction_id": [f"T{i:06d}" for i in range(c.n)],
            "timestamp": pd.date_range("2026-01-01", periods=c.n, freq="10min").astype(
                str
            ),
            "geographic_zone": z,
            "audit_group": np.where(target, "G2", "G1"),
            "distance_km": distance,
            "duration_min": duration,
            "demand_level": demand,
            "availability": supply,
            "time_of_day": hour,
            "operating_cost": cost,
            "hour_sin": sine,
            "hour_cos": cosine,
            "pickup_density": density,
            "ground_truth_fair_price": fair,
            "planted_premium": premium,
            "observed_price": fair + premium + r.normal(0, c.noise, c.n),
        }
    )
    # Calendar hour is consistent with the transaction timestamp, with random dates/order.
    df["timestamp"] = (
        pd.Timestamp("2026-01-01")
        + pd.to_timedelta(r.integers(0, 60, c.n), unit="D")
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(r.integers(0, 3600, c.n), unit="s")
    ).astype(str)
    # Draw after the original generator to preserve all existing seeded scenarios.
    df["battery_percent"] = r.uniform(1, 100, c.n)
    battery_premium = (
        c.battery_bias * np.maximum(0, (30 - df.battery_percent) / 30) ** 2
    )
    df["planted_premium"] += battery_premium
    df["observed_price"] += battery_premium
    return df.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)


def validate(df):
    required = [
        "transaction_id",
        "timestamp",
        "geographic_zone",
        "observed_price",
        *OPS,
        "pickup_density",
    ]
    missing = sorted(set(required) - set(df))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.transaction_id.isna().any() or df.timestamp.isna().any():
        raise ValueError("Transaction IDs and timestamps cannot be missing")
    for col in [*OPS, "pickup_density", "observed_price"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} must be numeric")
    if df.transaction_id.duplicated().any():
        raise ValueError("Duplicate transaction IDs")
    if not df.geographic_zone.isin(ZONES).all():
        raise ValueError("Unknown geographic zone")
    if df.observed_price.isna().any() or not np.isfinite(df.observed_price).all():
        raise ValueError("Observed prices must be finite")
    for col in OPS + ["pickup_density"]:
        if np.isinf(df[col]).any():
            raise ValueError(f"Infinite values in {col}")
        if df[col].notna().sum() == 0:
            raise ValueError(f"Entirely missing feature: {col}")
    for col in [
        "distance_km",
        "duration_min",
        "operating_cost",
        "demand_level",
        "availability",
    ]:
        if (df[col].dropna() < 0).any():
            raise ValueError(f"Column {col} cannot be negative")
    if not df.pickup_density.dropna().between(0, 1).all():
        raise ValueError("Pickup density must be in [0,1]")
    return {
        "rows": len(df),
        "duplicates": 0,
        "missing_features": int(df[OPS + ["pickup_density"]].isna().sum().sum()),
        "negative_observed_prices": int((df.observed_price < 0).sum()),
        "outlier_policy": "No automatic deletion. Extreme values retained for review; simulator Gaussian noise can produce negative prices.",
        "handling": "Predictor medians learned on training rows only. Audit uses complete operational rows.",
        "provenance": "Synthetic; arbitrary currency units (CU).",
    }


def feature_names(mode):
    return (
        OPS
        + (["pickup_density"] if mode in ["proxy", "unrestricted"] else [])
        + (["geographic_zone"] if mode in ["geographic", "unrestricted"] else [])
    )


def adjusted(df, outcome):
    clean = df.dropna(subset=OPS + [outcome]).copy()
    groups = clean.geographic_zone.value_counts()
    if len(clean) < 50 or any(groups.get(z, 0) < 15 for z in ZONES):
        return {
            "groups": [],
            "warning": "Insufficient complete rows: require 15 per zone and 50 total.",
            "n": len(clean),
        }
    if np.ptp(clean[outcome].to_numpy()) < 1e-12:
        return {
            "groups": [],
            "n": len(clean),
            "warning": "Outcome is constant; there is no variation for meaningful regression inference.",
        }
    X = pd.concat(
        [
            clean[OPS],
            pd.get_dummies(clean.geographic_zone, drop_first=True, dtype=float),
        ],
        axis=1,
    )
    design = sm.add_constant(X, has_constant="add")
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return {
            "groups": [],
            "n": len(clean),
            "warning": "Non-identifiable adjustment: operational controls are constant or collinear with geography. Review the design before interpreting gaps.",
        }
    fit = sm.OLS(clean[outcome].astype(float), design).fit(cov_type="HC3")
    ci = fit.conf_int()
    rows = []
    reference = clean.loc[clean.geographic_zone == "Zone A", outcome].mean()
    for z in ZONES[1:]:
        raw = clean.loc[clean.geographic_zone == z, outcome].mean() - reference
        rows.append(
            {
                "zone": z,
                "reference": "Zone A",
                "n": int(groups[z]),
                "raw_gap": float(raw),
                "adjusted_gap": float(fit.params[z]),
                "low": float(ci.loc[z, 0]),
                "high": float(ci.loc[z, 1]),
                "p": float(fit.pvalues[z]),
                "operational_component": float(raw - fit.params[z]),
            }
        )
    ps = multipletests([v["p"] for v in rows], method="holm")[1]
    for row, p in zip(rows, ps):
        row["p_holm"] = float(p)
        row["review"] = bool(p < 0.05)
    return {
        "groups": rows,
        "n": len(clean),
        "dropped": len(df) - len(clean),
        "r_squared": float(fit.rsquared),
        "method": "OLS with operational controls; HC3 robust 95% pointwise CI; Holm-adjusted p-values within outcome (2 contrasts).",
        "warning": "Association, not causation. Assumes independent transactions, adequate overlap and specified conditional mean. CIs do not include training uncertainty.",
    }


def matched(df, target="Zone B", tolerance=1.25):
    a = df[df.geographic_zone == "Zone A"].dropna(subset=OPS)
    b = df[df.geographic_zone == target].dropna(subset=OPS)
    scale = df[OPS].std().replace(0, 1)
    if len(a) == 0 or len(b) == 0:
        return {
            "pairs": [],
            "n": 0,
            "coverage": 0,
            "warning": "No reference or target observations",
        }
    distances = cdist(b[OPS] / scale, a[OPS] / scale, metric="chebyshev")
    used, pairs = set(), []
    for i in np.argsort(distances.min(axis=1)):
        for j in np.argsort(distances[i]):
            if distances[i, j] > tolerance:
                break
            if j not in used:
                used.add(j)
                ar, br = a.iloc[j], b.iloc[i]
                pairs.append(
                    {
                        "reference_id": ar.transaction_id,
                        "target_id": br.transaction_id,
                        "distance": float(distances[i, j]),
                        "reference_price": float(ar.observed_price),
                        "target_price": float(br.observed_price),
                        "gap": float(br.observed_price - ar.observed_price),
                        "reference_conditions": {k: float(ar[k]) for k in OPS},
                        "target_conditions": {k: float(br[k]) for k in OPS},
                    }
                )
                break
    gaps = np.array([p["gap"] for p in pairs])
    ci = (
        stats.t.interval(0.95, len(gaps) - 1, loc=gaps.mean(), scale=stats.sem(gaps))
        if len(gaps) > 2 and gaps.std() > 0
        else (None, None)
    )
    return {
        "pairs": pairs[:100],
        "n": len(pairs),
        "coverage": len(pairs) / len(b),
        "gap": float(gaps.mean()) if len(gaps) else None,
        "low": ci[0],
        "high": ci[1],
        "tolerance": tolerance,
        "method": "Greedy nearest matching without replacement. Maximum absolute difference in every operational feature <= 1.25 held-out SD. Pair t interval is exploratory; imperfect matching may leave confounding.",
        "warning": "Low coverage; selected pairs may not represent the target zone."
        if len(pairs) / len(b) < 0.5
        else None,
    }


def train(df, c: TrainConfig, seed=42):
    cut = int(len(df) * 0.75)
    ordered = df.sort_values(["timestamp", "transaction_id"])
    tr, te = ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()
    names = feature_names(c.features) + (
        ["battery_percent"] if c.include_battery else []
    )
    if c.include_battery and (
        "battery_percent" not in df
        or not pd.api.types.is_numeric_dtype(df.battery_percent)
        or not df.battery_percent.between(0, 100).all()
    ):
        raise ValueError(
            "Battery experiment requires measured or explicitly simulated battery_percent in [0,100]; missing values are not invented."
        )
    num = [x for x in names if x != "geographic_zone"]
    if tr[num].notna().sum().eq(0).any():
        raise ValueError(
            "A selected predictor is entirely missing in the training period"
        )
    numeric = (
        Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        if c.model == "neural"
        else SimpleImputer(strategy="median")
    )
    transforms = [("numeric", numeric, num)]
    if "geographic_zone" in names:
        transforms.append(
            (
                "zone",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="first",
                    categories=[ZONES],
                ),
                ["geographic_zone"],
            )
        )
    model = (
        LinearRegression()
        if c.model == "linear"
        else RandomForestRegressor(
            n_estimators=80,
            min_samples_leaf=8,
            max_depth=14,
            random_state=seed,
            n_jobs=2,
        )
    )
    pipe = Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(transforms, verbose_feature_names_out=False),
            ),
            ("model", model),
        ]
    )
    weights = (
        len(tr) / (3 * tr.geographic_zone.map(tr.geographic_zone.value_counts()))
        if c.reweight
        else np.ones(len(tr))
    )
    neural_training = None
    if c.model == "neural":
        model, neural_training = neural.fit_network(
            pipe["preprocess"], tr, names, weights, seed
        )
        pipe.steps[-1] = ("model", model)
    else:
        pipe.fit(tr[names], tr.observed_price, model__sample_weight=weights)
    te["predicted_price"] = pipe.predict(te[names])
    te["residual"] = te.observed_price - te.predicted_price
    groups = []
    for z, g in te.groupby("geographic_zone"):
        groups.append(
            {
                "zone": z,
                "n": len(g),
                "observed": float(g.observed_price.mean()),
                "predicted": float(g.predicted_price.mean()),
                "residual": float(g.residual.mean()),
                "mae": float(g.residual.abs().mean()),
                "rmse": float(np.sqrt(np.mean(g.residual**2))),
            }
        )
    importance = permutation_importance(
        pipe,
        te[names],
        te.observed_price,
        n_repeats=3,
        random_state=seed,
        scoring="neg_mean_absolute_error",
    )
    fi = [
        {"feature": n, "importance": float(v), "std": float(s)}
        for n, v, s in zip(
            names, importance.importances_mean, importance.importances_std
        )
    ]
    encoded = pipe["preprocess"].get_feature_names_out().tolist()
    coefficients = (
        [
            {"feature": n, "value": float(v)}
            for n, v in zip(encoded, pipe["model"].coef_)
        ]
        if c.model == "linear"
        else []
    )
    # Fixed input intervention distribution, on exactly the same held-out transactions.
    swapped = te.copy()
    swapped["geographic_zone"] = "Zone B"
    base = te.copy()
    base["geographic_zone"] = "Zone A"
    zone_delta = pipe.predict(swapped[names]) - pipe.predict(base[names])
    proxy_hi, proxy_lo = te.copy(), te.copy()
    proxy_hi["pickup_density"], proxy_lo["pickup_density"] = 0.8, 0.2
    proxy_delta = pipe.predict(proxy_hi[names]) - pipe.predict(proxy_lo[names])
    periods = []
    for i, part in enumerate(np.array_split(np.arange(len(te)), 4)):
        g = te.iloc[part]
        periods.append(
            {
                "period": f"Test quarter {i + 1}",
                "n": len(g),
                "mae": float(g.residual.abs().mean()),
                "rmse": float(np.sqrt(np.mean(g.residual**2))),
                "predicted_audit": adjusted(g, "predicted_price"),
            }
        )
    if neural_training:
        fitting = tr.iloc[: neural_training["fit_n"]]
        xp = pipe["preprocess"].transform(fitting[names])
        test_x = pipe["preprocess"].transform(te[names])
        linear = LinearRegression().fit(
            xp,
            fitting.observed_price,
            sample_weight=np.asarray(weights)[: len(fitting)],
        )
        candidates = {
            "Mean baseline": np.full(
                len(te),
                np.average(
                    fitting.observed_price, weights=np.asarray(weights)[: len(fitting)]
                ),
            ),
            "Linear regression": linear.predict(test_x),
            "Neural network": te.predicted_price.to_numpy(),
        }
        neural_training["benchmark"] = [
            {
                "model": name,
                "mae": float(np.abs(te.observed_price - pred).mean()),
                "rmse": float(np.sqrt(np.mean((te.observed_price - pred) ** 2))),
            }
            for name, pred in candidates.items()
        ]
        neural_training["benchmark_scope"] = (
            "Same fitting rows, preprocessing, weights and final test rows. Neural checkpoint selected on separate validation rows. Baselines are diagnostics, not an automatic model selection rule."
        )
    result = {
        "neural_training": neural_training,
        "config": c.model_dump(),
        "features": names,
        "methodology_version": VERSION,
        "reproducibility": {
            "seed": seed,
            "estimator_parameters": model.get_params(),
            "libraries": {
                name: version(name)
                for name in ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels"]
            },
            "dataset_sha256": hashlib.sha256(
                df.to_csv(index=False).encode()
            ).hexdigest(),
            "test_ids_sha256": hashlib.sha256(
                "|".join(te.transaction_id.astype(str)).encode()
            ).hexdigest(),
        },
        "mae": float(te.residual.abs().mean()),
        "rmse": float(np.sqrt(np.mean(te.residual**2))),
        "groups": groups,
        "observed_audit": adjusted(te, "observed_price"),
        "predicted_audit": adjusted(te, "predicted_price"),
        "residual_audit": adjusted(te, "residual"),
        "matched": matched(te),
        "importance": fi,
        "coefficients": coefficients,
        "sensitivity": {
            "zone_B_minus_A": float(zone_delta.mean()),
            "density_08_minus_02": float(proxy_delta.mean()),
            "label": "Model sensitivity; fixed-input interventions may be off-support, not causal counterfactuals. Responses for zones absent from training are not learned geographic effects.",
        },
        "stability": periods,
        "predictions": te[
            [
                "transaction_id",
                "geographic_zone",
                "observed_price",
                "predicted_price",
                "residual",
            ]
        ].to_dict("records"),
        "split": {
            "method": "Chronological 75/25; no fitting on test rows",
            "train_n": len(tr),
            "test_n": len(te),
            "train_end": str(tr.timestamp.max()),
            "test_start": str(te.timestamp.min()),
            "id_overlap": len(set(tr.transaction_id) & set(te.transaction_id)),
            "unseen_test_zones": sorted(
                set(te.geographic_zone) - set(tr.geographic_zone)
            ),
            "geographic_scope": "Known-zone future transactions. Shared zones are intentional; this does not test generalization to unseen geography.",
        },
    }
    return pipe, result, tr, te


def explain(pipe, tr, row):
    names = pipe.feature_names_in_.tolist()
    prep = pipe["preprocess"]
    encoded = prep.get_feature_names_out().tolist()
    x = prep.transform(row[names])[0]
    background = prep.transform(tr[names])
    model = pipe["model"]
    if hasattr(model, "coef_"):
        ref = background.mean(axis=0)
        values = (x - ref) * model.coef_
        base = float(model.intercept_ + ref @ model.coef_)
        method = "Exact additive linear attribution relative to training feature means (interventional linear SHAP convention). Correlated inputs are not causally separable."
    else:
        # Deterministic order-averaged interventional Shapley approximation, 64 permutations.
        rng = np.random.default_rng(19)
        refs = background[rng.choice(len(background), size=16, replace=False)]
        base = float(model.predict(refs).mean())
        values = np.zeros(len(x))
        for _ in range(64):
            order = rng.permutation(len(x))
            states = [refs.copy()]
            for j in order:
                state = states[-1].copy()
                state[:, j] = x[j]
                states.append(state)
            predictions = (
                model.predict(np.concatenate(states))
                .reshape(len(x) + 1, len(refs))
                .mean(axis=1)
            )
            for k, j in enumerate(order):
                values[j] += (predictions[k + 1] - predictions[k]) / 64
        method = "Monte Carlo interventional Shapley approximation: 64 orders, 16 training backgrounds, fixed seed. Encoded features; correlations broken. Attribution is not causal proof."
    return {
        "base": base,
        "prediction": float(pipe.predict(row[names])[0]),
        "contributions": [
            {"feature": n, "value": float(v)} for n, v in zip(encoded, values)
        ],
        "method": method,
    }
