"""Create real saved demo runs; skips scenarios already populated by this script."""

from backend import core, store
from backend.app import execute, generate


def main():
    for scenario in ["A", "C", "B"]:
        existing = [
            d
            for d in store.listing("dataset")
            if d["config"].get("seed") == 7 and d["config"].get("scenario") == scenario
        ]
        if existing:
            continue
        d = generate(core.GenerateConfig(scenario=scenario, seed=7))
        policy = "proxy" if scenario == "C" else "unrestricted"
        forest = execute(d["id"], core.TrainConfig(model="forest", features=policy))
        baseline = execute(d["id"], core.TrainConfig(features=policy))
        execute(
            d["id"],
            core.TrainConfig(features="operational"),
            baseline["id"],
            "operational_only",
        )
        print(
            scenario,
            "linear MAE",
            baseline["result"]["mae"],
            "forest MAE",
            forest["result"]["mae"],
            flush=True,
        )


if __name__ == "__main__":
    main()
