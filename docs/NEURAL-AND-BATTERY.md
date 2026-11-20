# Neural network and passenger battery experiment

CU means arbitrary synthetic currency units, not rupees or dollars. No real fare or passenger battery dataset has been imported.

## Try the saved example

Open http://127.0.0.1:8000/ and select dataset `9906e3` / model `4b9391a8`.

1. Model training shows the saved architecture, fitting/validation curves, optimization loss, checkpoint and same-test-set benchmarks.
2. Prediction explorer shows the exact network forward pass above the trip form. Move Explore battery or Explore distance: after a short debounce, the Python server recomputes activations and price from saved weights. Click neurons for their values and biases. Play/Pause highlights layer order; it is a pedagogical replay, not live training.
3. The battery sensitivity curve evaluates the same trip at 21 battery levels. It is not a chart of actual passenger charges.
4. Interventions → Remove battery percentage retrains a child model on unchanged recorded prices and held-out IDs. Its battery curve must be exactly flat.
5. System checks → Verify active run replays saved results. Export report includes the network training record.

## Train your own example

Data & scenarios → choose A, 2,400 rows, seed 29, noise 1 and hypothetical low-battery premium 12 CU → Generate. A/B/C describe the geographic mechanism. The battery premium is a separate explicit overlay; A with battery enabled is not an entirely unbiased world.

Model training → Neural network → Operational only → Battery research feature: Include for hypothesis testing → Train & audit.

Generator overlay: `battery_bias × max(0, (30 - battery_percent) / 30)^2`. Battery is independently sampled from a uniform distribution between 1 and 100 after the original seeded generation. This intentionally nonlinear label mechanism is a hypothetical experiment, not evidence of platform behavior. Strength 0 disables the overlay. Battery is excluded from pricing by default. Old datasets without battery cannot train a battery-enabled model; values are never invented on import.

## Model and validation

Dense MLP with 16 and 8 hidden ReLU units, linear output, Adam, learning rate .003, L2 alpha .01 and seed-controlled initialization. Numerical features are median-imputed and standardized using fitting rows only; geography, when selected, is one-hot encoded. CPU training uses scikit-learn MLPRegressor.partial_fit with sample weights.

The chronological development period is the first 75% of rows. Its first 80% (60% overall) fits gradients and preprocessing; the remaining 20% (15% overall) selects a checkpoint. The final 25% remains held out. Up to 400 epochs, patience 35 and minimum improvement .0001 CU. The best checkpoint is saved without refitting on validation data. Each epoch records fitting MAE, validation MAE and the actual regularized optimization loss. No test labels participate in checkpoint selection.

The built-in mean and linear baselines use the same fitting rows, weights, preprocessing and final test rows. For the saved single-seed nonlinear synthetic example, holdout MAE is mean 5.611, linear 1.579, neural .927 CU. These numbers demonstrate this constructed case only, not superiority across datasets or real platforms. The earlier 30-scenario benchmark covers the earlier models, not this new neural addition. A CNN is not used: these tabular inputs have no justified image/grid convolution structure.

The visualizer evaluates `a_next = ReLU(a @ W + b)` for hidden layers and identity for the output. It returns actual saved weights, biases, preactivations and activations. The automated tests independently compare its output with sklearn prediction and confirm that changing only test labels cannot change learned weights/history.

## Battery measurement

The default input is explicitly labeled manual simulation. “Read this device battery once” calls navigator.getBattery only on click, where the browser supports it. It reads the device opening the page (on this Mac, the Mac), not another passenger's phone. Browser readings can be rounded or default values. Unsupported/failed calls preserve the manual value and display an explanation. No background polling or battery-level subscription is installed. The selected number goes to the local inference endpoint; it is not persisted as passenger telemetry.

A real passenger study would require a supported client on each participant's phone, clear consent and provenance, and timestamped observed quotes paired with operational conditions. The public trip datasets discussed do not supply phone battery. Battery variation here establishes model sensitivity, not a causal effect on real fares.

References: [scikit-learn MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html), [MDN Battery API support](https://developer.mozilla.org/en-US/docs/Web/API/Navigator/getBattery).

## Recorded epoch replay (17 November 2026)

New neural training saves epoch 1, every tenth epoch and the validation-selected checkpoint in the model artifact. Prediction explorer exposes a timeline and play/pause controls. Each frame uses that checkpoint's actual weights and biases with the same fitting-only preprocessor. Changing trip inputs recomputes activations while checkpoint weights stay fixed. Changing epochs changes learned weights, biases and outputs; the dense network's connectivity remains fixed.

A distance (1–29 km) × battery (0–100%) response map varies both inputs while holding the remaining trip inputs fixed. The color bounds are shared across all frames for that request. These are model responses, not confidence intervals or classification boundaries; some grid values may be outside training support. A selected-epoch button returns to the deployed model's checkpoint, whose prediction is checked against normal inference in API tests. Older saved models without checkpoint history display a retraining instruction. This is a dense ANN; no CNN is claimed or fabricated.
