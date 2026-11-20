# Where the data and predictions come from

**No Uber, Ola, taxi company, OpenAI, or external pricing dataset has been fetched.** FairPrice currently uses locally generated synthetic transactions or a CSV explicitly imported by a user. This follows the original proposal's simulation-first experiment.

## What happens when you generate data

A seeded NumPy random generator produces trip distance and duration, time, demand, supply availability, verified operating cost, synthetic zone, and correlated pickup density. A documented formula creates the legitimate price. Gaussian noise is added. Scenario B adds a chosen geographic premium; Scenario C adds a pickup-density effect. In A, neither unjustified effect is planted.

These are artificial labels designed to test whether an audit can recover a known mechanism. The data are not collected observations of a platform and cannot validate real-world fare accuracy. The fair-price truth and planted-premium fields are visible for checking the simulator, but never enter the pricing model or independent audit.

## What happens when you train

1. The Python FastAPI backend loads the saved CSV.
2. It sorts by timestamp and holds out the last 25% of transactions.
3. Imputation and categorical encoding are fitted on the first 75% only.
4. A real scikit-learn estimator learns to predict `observed_price` from its allowed features.
5. It predicts the held-out 25% and computes MAE, RMSE, group errors, and independent fairness diagnostics.
6. SQLite stores metadata/results; a server-owned joblib artifact stores the fitted preprocessing, model, and split frames.

The models are **linear regression** and **random forest**. They run on the local machine's CPU. They are not a large language model, and prediction does not call an OpenAI API or require a GPU. Data generation and model fitting happen when you explicitly request them; changing pages does not retrain.

## What happens when you predict a new transaction

The browser sends numerical inputs to `/api/experiments/{run_id}/predict`. The backend loads that saved estimator, engineers cyclic hour features and returns `model.predict(...)`. It does not look up a market price or rerun the simulator to produce the answer. Features excluded by the selected policy have no influence. Values outside training ranges are flagged.

The returned CU value is a **synthetic currency-unit prediction**, not a real fare in INR or USD. A new trip has no observed outcome, so its individual accuracy cannot be measured yet. The displayed MAE/RMSE are measured on the saved held-out test set.

## How to inspect this in the interface

- The persistent strip above each page names the data source and active model.
- **System checks** shows the source, formula, feature allowlist, excluded fields, split boundaries, model/library metadata and dataset fingerprint.
- **Verify active run** replays the saved predictions and metrics on the backend.
- **Prediction explorer** provides both new-trip inference and explanations for existing held-out transactions.

## What a real-data extension would require

A suitable licensed dataset with observed prices and defensible operational controls, documented provenance, missingness analysis, applicable privacy review, and validation under realistic time/geographic shifts. Geographic identifiers alone do not provide socioeconomic or protected-group labels. Merely uploading a real CSV does not make the scientific conclusions valid.
