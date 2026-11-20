# How to check FairPrice locally

## One-command verification

From the project directory:

```bash
./scripts/verify.sh
```

The command fails immediately if a check fails. It runs Python lint, statistical and API workflow tests, TypeScript checking, a production frontend build, and Python source compilation. Tests use temporary directories: they do not overwrite your saved research data. An automated pass does not replace checking the actual browser; use the table below.

## In-app verification

Open **System checks**, then **Verify active run**. The Python backend loads the persisted model and dataset and computes the checks again. The result is stored with its timestamp and experiment ID, survives a page refresh, and is included in the JSON and Markdown exports.

Checks cover artifact loading/integrity, feature exclusion, Scenario C geography policy, temporal split/ID overlap, every held-out prediction, residual identities, MAE/RMSE, independent observed-price audit coefficients, individual explanation additivity, dataset fingerprint, and parent/child held-out-data consistency when applicable.

A missing file or failed comparison is reported as a failure. Old runs that predate dataset fingerprints are marked **unavailable**, not quietly passed. A successful check verifies numerical consistency, not real-world validity, discrimination, or every browser control.

## Browser feature checklist

| Feature | Action | Expected evidence |
|---|---|---|
| Backend connection | Open the app | Header says **Backend connected** after `/api/health` confirms the database is reachable. It polls every 30 seconds. |
| Generation | Data & scenarios: generate A, B, C | A new dataset ID and matching scenario/seed/count appear. Preview and export show actual rows. |
| Validation | Try fewer than 300 rows, or an invalid CSV | A readable error appears; no invalid dataset is saved. |
| Filtering | Choose Zone B in the transaction explorer | Only B rows appear; matching count updates. |
| Pagination | Click Next / Previous | The shown range changes and transaction IDs change. |
| CSV export/import | Export CSV and import that file | A distinct imported dataset is created; source changes to **User-imported CSV**. |
| Training | Model training: train linear, then forest | New run IDs appear; actual MAE/RMSE, predictions, errors and feature lists appear. |
| Scenario C policy | Train C with unrestricted features | Saved effective policy is **proxy**; geography is excluded. |
| Active model selection | Choose another run in the top selector | Active run and analytical values change together. Explanations/predictions reset. |
| Dataset switching | Choose another dataset | Its most recent saved model is restored, or the app shows no trained model if none exists. No previous dataset's results remain attached. |
| Observed-price audit | Fairness audit | Raw/adjusted contrasts, 95% intervals, Holm p-values, sample sizes and warnings appear. |
| Residual audit | Inspect group errors | Residual is observed minus predicted; it is separate from charged-price disparity. |
| Matched pairs | Open matched-condition inspection | Pair IDs, calipers, coverage and actual operational conditions appear. |
| Temporal stability | Inspect four test-period slices | Smaller slices may report insufficient data rather than invented estimates. |
| Geographic analysis | Inspect schematic zones | Synthetic group prices and contrasts appear; no real geographic boundaries are claimed. |
| New-trip prediction | Prediction explorer: calculate a price, change distance and calculate again | Backend returns a fresh prediction from the selected saved estimator. The prior result clears when inputs change. |
| Individual explanation | Select a held-out transaction | Additive contributions and a saved-model prediction appear. Linear coefficients or approximate forest Shapley explanations are labeled accurately. |
| Sensitivity | Change proxy/zone and recalculate | A real model response is returned; unused features have no meaningful effect. |
| Mitigation | Apply each of four strategies | A child run is saved; parent/test rows are unchanged and before/after outcomes are shown. A strategy need not always improve every metric. |
| History comparison | Choose a saved comparison run | Metrics appear side by side; different datasets are explicitly distinguished from controlled comparisons. |
| Reports | Export report / download evidence | Markdown summary and full JSON evidence download. Actual review and verification records are included. |
| Governance | Save a named review with evidence | Timestamped review appears in the decision trail and relevant report. |
| Persistence | Reload the page/restart `./run.sh` | Saved datasets, models, reviews and verification records remain available. |
| Responsive layout | Use a narrow browser window | Navigation remains accessible, tables scroll within their containers, forms reflow. |

## Expected research behavior

Use 2,400 rows, noise 1, bias 4, seed 42 and target Zone B for a familiar demonstration.

- **A:** adjusted observed gap should be near zero on average across seeds; individual false positives are possible.
- **B:** adjusted observed B–A gap should be near +4 CU.
- **C:** adjusted observed B–A gap should be near +2 CU because the density mean difference is approximately 0.5. Removing geography alone does not remove density.
- **Operational-only linear mitigation:** predicted adjusted zone gaps are near zero under this audit's same operational linear basis. That algebraic result is not real-world fairness proof.
- **Forest mitigation:** residual conditional gaps can remain because the model is nonlinear, the audit is linear and samples are finite. Investigate them rather than forcing them away.

Run `.venv/bin/python -m scripts.experiments` for the 10-seed-per-scenario study. It writes actual results to `docs/experiment-results.json`.

## Neural network and battery extension

- Follow [NEURAL-AND-BATTERY.md](NEURAL-AND-BATTERY.md) for the saved example, model design and limitations.
- Model training: select Neural network; verify fitting/validation curves, recorded optimization loss and baseline comparison appear after training.
- Prediction explorer: changing battery/distance updates network activations and output automatically; previous results are explicitly marked while recalculating. Click neurons; play/pause the layer walkthrough.
- Battery reading: explicit one-shot button labels the browser reading or unsupported fallback. Manual edits return the source label to simulation.
- Battery exclusion: a run without the battery feature has a perfectly flat battery sweep. Remove battery intervention preserves observed charges and held-out IDs.
- Imported/legacy data without battery must fail clearly if battery inclusion is requested.

## Epoch replay and public demonstration

1. Open Prediction explorer with the newly seeded neural model, or train a new Neural network model.
2. Scroll to “See the network learn.” Press “Play training”; epoch, weights, biases, neuron activations and response map should change.
3. Pause and scrub the epoch slider. Click a neuron to inspect activation and bias; hover an edge to inspect its weight.
4. Click “Selected model.” Its trip price must match normal inference above for the same inputs.
5. Change distance or battery. Replay recomputes for the new trip; same-epoch weights stay fixed.
6. The response map holds other inputs constant and shares one color range across the timeline. It is not an uncertainty interval.
7. Visit the public Worker URL in DEPLOYMENT-NOTES.md. The shared synthetic demo is separate from your local workspace. Keeping the Mac awake is required.
