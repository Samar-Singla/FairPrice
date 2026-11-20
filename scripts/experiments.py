"""Reproduce the three research scenarios across independent random seeds."""

import json
from pathlib import Path

import numpy as np
from scipy import stats

from backend.core import GenerateConfig, TrainConfig, generate, train


def main():
    rows = []
    for scenario in ["A", "B", "C"]:
        for seed in range(10):
            d = generate(GenerateConfig(scenario=scenario, seed=seed, n=2400))
            _, r, _, _ = train(
                d,
                TrainConfig(features="proxy" if scenario == "C" else "unrestricted"),
                seed,
            )
            _, m, _, _ = train(d, TrainConfig(features="operational"), seed)
            observed = r["observed_audit"]["groups"][0]
            rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "observed_gap": observed["adjusted_gap"],
                    "low": observed["low"],
                    "high": observed["high"],
                    "p_holm": observed["p_holm"],
                    "predicted_gap": r["predicted_audit"]["groups"][0]["adjusted_gap"],
                    "mitigated_prediction_gap": m["predicted_audit"]["groups"][0][
                        "adjusted_gap"
                    ],
                    "mae": r["mae"],
                    "mitigated_mae": m["mae"],
                    "matched_coverage": r["matched"]["coverage"],
                }
            )
        subset = [r for r in rows if r["scenario"] == scenario]
        print(
            scenario,
            "mean observed gap",
            np.mean([r["observed_gap"] for r in subset]),
            flush=True,
        )
    summary = []
    for scenario in ["A", "B", "C"]:
        sub = [r for r in rows if r["scenario"] == scenario]
        gaps = np.array([r["observed_gap"] for r in sub])
        ci = stats.t.interval(
            0.95, len(gaps) - 1, loc=gaps.mean(), scale=stats.sem(gaps)
        )
        summary.append(
            {
                "scenario": scenario,
                "runs": len(sub),
                "mean_adjusted_gap": float(gaps.mean()),
                "seed_mean_ci": list(ci),
                "significant_runs": sum(r["p_holm"] < 0.05 for r in sub),
                "mean_mae": float(np.mean([r["mae"] for r in sub])),
                "mean_mitigated_mae": float(np.mean([r["mitigated_mae"] for r in sub])),
            }
        )
    Path("docs/experiment-results.json").write_text(
        json.dumps(
            {
                "method": "10 independent seeds, 2400 rows, 600-row chronological test, noise SD 1, bias 4, target B; seed-mean t CI is not a per-run interval.",
                "summary": summary,
                "runs": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
