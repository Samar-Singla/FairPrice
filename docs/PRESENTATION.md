# A five-minute demonstration

**Framing:** “FairPrice is a reproducible research prototype for investigating geographic price disparities under comparable operational conditions. It measures statistical signals; it does not automatically classify discriminatory intent.”

1. **Overview:** identify the active dataset, scenario, seed, estimator and held-out sample size. Say explicitly that the demonstration uses synthetic data.
2. **System checks:** show the data origin and split diagram. Run verification and explain that the saved model's predictions and metrics are recomputed.
3. **Scenario B audit:** with bias 4, show an adjusted gap near 4 CU alongside the raw gap and confidence interval. Explain why raw differences can reflect operational conditions.
4. **Prediction explorer:** calculate a new transaction; change an operational input and recompute. Then select a held-out row and inspect its feature contributions.
5. **Interventions:** remove geography/proxy inputs and show the actual accuracy/disparity trade-off. Explain that observed charges stay unchanged.
6. **Scenario C:** show that deleting explicit geography is insufficient when density encodes geography.
7. **Review and export:** record the next investigation step and export the evidence.

Do not claim platform data were fetched, synthetic CU predictions are real fares, SHAP is causal evidence, insignificant gaps prove fairness, or the local prototype is already a production deployment.
