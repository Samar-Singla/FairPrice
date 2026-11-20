# Verification record

Executed locally on macOS ARM64 with Python 3.12 and Node 22.

## Local release check — 16 November 2026

- `./scripts/verify.sh`: 42 tests passed. Ruff, Python compilation, TypeScript and the production build passed.
- Coverage includes all three scenarios, both estimators, four mitigation strategies, CSV validation, new-trip inference, explanations, exports, artifact corruption/missing files, atomic storage rollback and concurrent database initialization.
- Fixed a database-lock race during simultaneous initial requests; the concurrency regression test passes.
- Browser: generated scenario C with 600 rows and seed 17; filtered/paginated rows; trained a forest; predicted new trips; inspected a held-out explanation and density sensitivity; retrained an operational-only intervention; opened audit/geographic pages; compared saved runs; saved an explicitly labeled QA review.
- New-trip predictions changed from 20.72 CU to 26.30 CU when distance changed from 6 to 10 km with the remaining inputs fixed.
- After a backend restart and page reload, the saved dataset/model restored. All 11 active-run integrity checks passed, replaying 150 held-out predictions and the saved audit. Parent/child observed charges and held-out IDs were unchanged.
- Evidence downloaded through the browser and parsed as valid JSON, including lineage and verification results.
- Redesigned interface inspected at 390×844 and 1440×1000. Screenshots: `redesigned-mobile.jpg` and `redesigned-dashboard.jpg`.
- Local health endpoint reports database connected. No deployment performed.

## Initial research checks

- 18 automated tests passed, with no warnings after installing the Starlette-supported httpx2 test transport.
- Ruff checks passed; Python bytecode compilation passed.
- TypeScript checks and Vite production build passed. Chart code is split into its own bundle.
- Thirty independent scenario/seed combinations completed; each also trained an operational-only mitigation (60 total model fits). Full output: experiment-results.json.
- Real saved demonstration datasets for A, B, C were generated with seed 7. Each includes a forest, linear baseline, and linear operational-only mitigation.
- Browser workflow: generate 2,400 B transactions, train a model, inspect an individual explanation, change geography, retrain mitigation, save an explicitly labeled QA review, reload saved parent comparison, open geography and audit pages.
- Browser observation: changing the selected example from Zone C to B changed prediction from 20.02 to 24.03 CU. Mitigation raised MAE from 0.823 to 1.825 CU while the observed adjusted gap stayed 3.914 CU.
- Browser console had no warnings/errors in the final build check.
- Responsive layout checked at desktop and narrow viewport sizes. Local API health returned status ok.

These are software and synthetic-method checks, not empirical validation of fairness on a real platform. The QA review is demonstration data, not a human determination of acceptability.

## Neural extension — 16 November 2026

- 48 tests passed, including exact neural forward-pass parity, chronological validation/checkpoint selection, no test-label influence on learned weights, battery-overlay identity, missing-battery rejection, neural reweight/removal APIs and saved-run integrity.
- Neural example: 2,400 synthetic rows, seed 29, scenario A geographic mechanism plus 12 CU battery-overlay strength. Same-fit/test benchmark MAE: mean 5.611, linear 1.579, neural .927 CU.
- Browser displayed saved learning curves and baseline metrics. Changing manual battery from 50% to 0% changed the same-trip output from 20.03 to 30.01 CU and updated the traced output activation. The zero-percent input is correctly flagged outside the generated training range.
- Explicit browser battery read succeeded and was labeled browser-reported device battery. It was not stored as passenger telemetry.
- These are constructed-data checks. No real-world battery-price relationship has been established.

### 17 November 2026 — open white interface, epoch replay, Cloudflare demo

- Removed grey fills from quick actions, scenario selectors, data headings, badges, zone tiles, pipeline steps and explanatory blocks; forms use white fields with fine outlines.
- Added stored epoch checkpoints, interactive weight/activation replay, selected-checkpoint control and distance/battery response maps with a common timeline color range.
- 48 backend tests passed, including checkpoint-to-inference equality, weight changes across epochs, activation changes for new inputs and surface bounds. TypeScript/Vite production build and Ruff passed.
- Public Worker: https://fairprice-demo.chatbot-sockscarving.workers.dev
- Verified public root, health, prediction, replay and saved-run verification. Demo experiment `9bd0fdc9f48345b3914c64a13008b784`: 42 recorded frames; default trip 19.75197772520907 CU; selected checkpoint agrees with inference; numerical verification passed.
- Public shared synthetic workspace is isolated in `demo-data/`; local workspace remains in `data/`. Both use the actual FastAPI/sklearn backend on the Mac. No new paid service enabled.
- Browser preview unavailable via CUA; no claim of screenshot-based visual QA. See DEPLOYMENT-NOTES.md for runtime limits and restart instructions.
