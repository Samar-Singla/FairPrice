import { useEffect, useState } from "react";
import {
  Check,
  RefreshCw,
  ArrowRight,
  FileText,
  CircleAlert,
} from "lucide-react";
import { api } from "./api";
import { NeuralTrace, BatteryCurve, TrainingReplay } from "./NeuralLab";
type Data = Record<string, any>;

export function PredictTrip({ runId }: { runId: string }) {
  const [input, setInput] = useState({
    battery_percent: 50,
    distance_km: 6,
    duration_min: 18,
    demand_level: 1.2,
    availability: 0.6,
    operating_cost: 3.5,
    time_of_day: 12,
    pickup_density: 0.3,
    geographic_zone: "Zone A",
  });
  const [result, setResult] = useState<Data | null>(null),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const [batterySource, setBatterySource] = useState(
      "Manual simulation — not measured",
    ),
    [revision, setRevision] = useState(0);
  useEffect(() => {
    let current = true;
    setError("");
    setBusy(true);
    const timer = setTimeout(() => {
      api(`/experiments/${runId}/predict`, input)
        .then((r) => {
          if (current) setResult(r);
        })
        .catch((e) => {
          if (current) {
            setError(e.message);
            setResult(null);
          }
        })
        .finally(() => {
          if (current) setBusy(false);
        });
    }, 350);
    return () => {
      current = false;
      clearTimeout(timer);
    };
  }, [input, runId, revision]);
  const fields = [
    ["battery_percent", "Phone battery · % (hypothesis input)", 0, 100, 1],
    ["distance_km", "Distance · km", 0.1, 100, 0.1],
    ["duration_min", "Duration · min", 1, 360, 1],
    ["demand_level", "Demand index", 0.1, 4, 0.1],
    ["availability", "Supply availability", 0, 1, 0.01],
    ["operating_cost", "Operating cost · CU", 0, 100, 0.1],
    ["time_of_day", "Hour of day · 0–23", 0, 23, 1],
    ["pickup_density", "Pickup density", 0, 1, 0.01],
  ] as const;
  return (
    <>
      {busy && (
        <p role="status" className="notice">
          Recomputing from current inputs… Previous visualization remains
          visible until the response arrives.
        </p>
      )}
      {result?.neural_trace && (
        <NeuralTrace
          trace={result.neural_trace}
          controls={
            <div className="form-grid neural-controls">
              <label>
                Explore battery · {input.battery_percent}%
                <input
                  aria-label="Explore battery"
                  type="range"
                  min="0"
                  max="100"
                  value={input.battery_percent}
                  onChange={(e) => {
                    setInput({
                      ...input,
                      battery_percent: Number(e.target.value),
                    });
                    setBatterySource("Manual simulation — not measured");
                  }}
                />
              </label>
              <label>
                Explore distance · {input.distance_km} km
                <input
                  aria-label="Explore distance"
                  type="range"
                  min="0.1"
                  max="30"
                  step="0.1"
                  value={input.distance_km}
                  onChange={(e) =>
                    setInput({ ...input, distance_km: Number(e.target.value) })
                  }
                />
              </label>
            </div>
          }
        />
      )}
      <section className="panel predict-panel">
        <div className="panel-title">
          <div>
            <h3>Predict a new transaction</h3>
            <p>
              Runs the selected saved estimator on your inputs. No observed
              price is required.
            </p>
          </div>
          <span className="badge">LIVE INFERENCE</span>
        </div>
        <div className="prediction-layout">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              setRevision((v) => v + 1);
            }}
          >
            <div className="form-grid">
              {fields.map(([key, label, min, max, step]) => (
                <label key={key}>
                  {label}
                  <input
                    required
                    type="number"
                    min={min}
                    max={max}
                    step={step}
                    value={input[key]}
                    onChange={(e) => {
                      setInput({ ...input, [key]: Number(e.target.value) });
                      if (key === "battery_percent")
                        setBatterySource("Manual simulation — not measured");
                    }}
                  />
                </label>
              ))}
              <label>
                Geographic zone
                <select
                  value={input.geographic_zone}
                  onChange={(e) => {
                    setInput({ ...input, geographic_zone: e.target.value });
                  }}
                >
                  {["Zone A", "Zone B", "Zone C"].map((z) => (
                    <option key={z}>{z}</option>
                  ))}
                </select>
              </label>
            </div>
            <p className="footnote">
              Battery source: {batterySource}. Changing inputs automatically
              recomputes the prediction.
            </p>
            <button
              type="button"
              className="secondary"
              onClick={async () => {
                const nav = navigator as Navigator & {
                  getBattery?: () => Promise<{ level: number }>;
                };
                if (!nav.getBattery) {
                  setBatterySource(
                    "Device reading unavailable in this browser; manual simulation remains active",
                  );
                  return;
                }
                try {
                  const b = await nav.getBattery();
                  if (!Number.isFinite(b.level) || b.level < 0 || b.level > 1)
                    throw new Error("Invalid reading");
                  setInput((v) => ({
                    ...v,
                    battery_percent: Math.round(b.level * 100),
                  }));
                  setBatterySource(
                    "Browser-reported battery of this device, read once; may be rounded or a desktop default",
                  );
                } catch {
                  setBatterySource(
                    "Device reading failed; manual simulation remains active",
                  );
                }
              }}
            >
              Read this device battery once
            </button>
            <p className="footnote">
              Only on your click. This reads the device running this page, not
              another passenger’s phone. The value is sent to this application’s
              prediction API, not saved as passenger telemetry. Browser support
              varies.
            </p>
            <button disabled={busy} type="submit">
              {busy ? "Calculating…" : "Calculate price"}{" "}
              <ArrowRight size={15} />
            </button>
          </form>
          <div className="prediction-output">
            <span className="eyebrow">MODEL OUTPUT</span>
            <strong>
              {result ? Number(result.predicted_price).toFixed(2) : "—"}{" "}
              <small>CU</small>
            </strong>
            <p>
              {result
                ? `${result.model === "neural" ? "Neural network" : result.model === "forest" ? "Random forest" : "Linear regression"} · ${result.source}`
                : "Enter operational conditions and calculate a prediction."}
            </p>
            {result && (
              <>
                <p>{result.scope}</p>
                <div className="tag-list">
                  {result.used_features.map((f: string) => (
                    <span key={f}>{f}</span>
                  ))}
                </div>
                {result.warnings.map((w: string) => (
                  <p className="warning-text" key={w}>
                    <CircleAlert size={14} /> {w}
                  </p>
                ))}
              </>
            )}
            <p className="footnote">
              Only the active model’s selected features affect the output. CU
              means synthetic currency units, not a quoted fare in rupees or
              dollars.
            </p>
          </div>
        </div>
        {error && (
          <div role="alert" className="notice error">
            {error}
          </div>
        )}
      </section>
      {result && <BatteryCurve result={result} />}
      {result?.neural_trace && <TrainingReplay runId={runId} input={input} />}
    </>
  );
}

export function SystemChecks({ runId }: { runId?: string }) {
  const [system, setSystem] = useState<Data | null>(null),
    [lineage, setLineage] = useState<Data | null>(null),
    [verification, setVerification] = useState<Data | null>(null),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  useEffect(() => {
    let valid = true;
    setError("");
    setLineage(null);
    setVerification(null);
    Promise.all([
      api("/system"),
      ...(runId
        ? [
            api(`/experiments/${runId}/lineage`),
            api(`/experiments/${runId}/verification`),
          ]
        : []),
    ])
      .then(([s, l, v]) => {
        if (valid) {
          setSystem(s);
          setLineage(l || null);
          setVerification(v || null);
        }
      })
      .catch((e) => {
        if (valid) setError(e.message);
      });
    return () => {
      valid = false;
    };
  }, [runId]);
  return (
    <>
      <div className="system-intro">
        <div>
          <span className="eyebrow">REPRODUCIBILITY & OPERATIONS</span>
          <h2>Inspect the evidence behind a run.</h2>
          <p>
            These checks recompute results on the Python server. A passing
            integrity check confirms consistency—not scientific validity.
          </p>
        </div>
        <button
          disabled={!runId || busy}
          onClick={async () => {
            setBusy(true);
            setError("");
            try {
              setVerification(await api(`/experiments/${runId}/verify`, {}));
            } catch (e) {
              setError(e instanceof Error ? e.message : String(e));
            } finally {
              setBusy(false);
            }
          }}
        >
          <RefreshCw size={15} className={busy ? "spin" : ""} />
          {busy ? "Verifying saved run…" : "Verify active run"}
        </button>
      </div>
      {error && (
        <div className="notice error" role="alert">
          {error}
        </div>
      )}
      {system && (
        <div className="metrics">
          <div className="metric">
            <span>Backend</span>
            <strong>Python</strong>
            <small>{system.execution}</small>
          </div>
          <div className="metric">
            <span>Saved datasets</span>
            <strong>{system.dataset_count}</strong>
            <small>Generated or imported</small>
          </div>
          <div className="metric">
            <span>Trained runs</span>
            <strong>{system.experiment_count}</strong>
            <small>Persistent model artifacts</small>
          </div>
          <div className="metric">
            <span>Review records</span>
            <strong>{system.review_count}</strong>
            <small>Append-only decision history</small>
          </div>
        </div>
      )}
      <section className="panel">
        <div className="panel-title">
          <div>
            <h3>Run integrity checks</h3>
            <p>
              {verification
                ? `Last checked ${new Date(verification.created).toLocaleString()}`
                : "Run verification to inspect artifacts, predictions, errors, explanations and training/test separation."}
            </p>
          </div>
          {verification && (
            <span
              className={
                "verification-state " + (verification.passed ? "pass" : "fail")
              }
            >
              {verification.passed ? (
                <Check size={15} />
              ) : (
                <CircleAlert size={15} />
              )}{" "}
              {verification.passed ? "All checks passed" : "Review required"}
            </span>
          )}
        </div>
        {verification ? (
          <>
            <div className="check-list">
              {verification.checks.map((c: Data) => (
                <div key={c.check}>
                  <span className={"check-state " + c.status}>
                    {c.status === "pass" ? (
                      <Check size={16} />
                    ) : (
                      <CircleAlert size={16} />
                    )}
                  </span>
                  <div>
                    <b>{c.check}</b>
                    <p>{c.detail}</p>
                  </div>
                  <span className="check-label">{c.status}</span>
                </div>
              ))}
            </div>
            <p className="footnote">{verification.scope}</p>
          </>
        ) : (
          <div className="empty">
            {runId
              ? "Select “Verify active run” to run actual numerical checks."
              : "Choose or train a model first."}
          </div>
        )}
      </section>
      {lineage && (
        <>
          <section className="panel">
            <div className="panel-title">
              <div>
                <h3>Where the training data came from</h3>
                <p>
                  Source information for the active dataset and saved model.
                </p>
              </div>
              <span className="badge">{lineage.source_type}</span>
            </div>
            <div className="lineage-grid">
              <div>
                <span>Source</span>
                <strong>{lineage.source_type}</strong>
                <p>{lineage.source_description}</p>
              </div>
              <div>
                <span>External data fetched</span>
                <strong>No</strong>
                <p>
                  There is no Uber, Ola, or live pricing feed connected to this
                  application.
                </p>
              </div>
              <div>
                <span>Prediction target</span>
                <strong>{lineage.target}</strong>
                <p>
                  The estimator learns the observed transaction price in the
                  chosen dataset.
                </p>
              </div>
            </div>
            {lineage.formula && (
              <div className="formula">
                <span>
                  Legitimate synthetic price, before noise or a planted premium
                </span>
                <code>{lineage.formula}</code>
                <p>
                  A: operational price + noise. B: adds the configured direct
                  zone premium. C: adds a pickup-density effect; explicit
                  geography is excluded from pricing inputs.
                </p>
              </div>
            )}
            <p className="footnote">{lineage.battery_assumption}</p>
            <p className="footnote">{lineage.limitations}</p>
          </section>
          <section className="panel">
            <div className="panel-title">
              <div>
                <h3>Training & prediction pipeline</h3>
                <p>
                  Train/test separation is fixed for all runs on a given
                  dataset.
                </p>
              </div>
            </div>
            <ol className="pipeline-list">
              {lineage.pipeline.map((s: string, i: number) => (
                <li key={s}>
                  <span>{String(i + 1).padStart(2, "0")}</span>
                  {s}
                </li>
              ))}
            </ol>
            <div className="split-track">
              <div>Training · {lineage.split.train_n} rows · 75%</div>
              <div>Testing · {lineage.split.test_n} rows · 25%</div>
            </div>
            <p className="footnote">
              Training ends {lineage.split.train_end}; testing starts{" "}
              {lineage.split.test_start}. {lineage.split.geographic_scope}
            </p>
          </section>
          <section className="panel">
            <div className="panel-title">
              <div>
                <h3>Reproducibility record</h3>
                <p>
                  Actual saved identifiers, configuration and software versions.
                </p>
              </div>
              <a
                className="button secondary"
                href={`/api/experiments/${runId}/report`}
              >
                <FileText size={14} /> Download evidence
              </a>
            </div>
            <dl className="metadata">
              <dt>Experiment</dt>
              <dd>{lineage.experiment_id}</dd>
              <dt>Dataset</dt>
              <dd>{lineage.dataset_id}</dd>
              <dt>Trained at</dt>
              <dd>{new Date(lineage.trained_at).toLocaleString()}</dd>
              <dt>Random seed</dt>
              <dd>{lineage.config.seed}</dd>
              <dt>Dataset SHA-256</dt>
              <dd>
                {lineage.reproducibility.dataset_sha256 ||
                  "Legacy run; retrain to record a fingerprint."}
              </dd>
              <dt>Pricing features</dt>
              <dd>{lineage.features.join(", ")}</dd>
              <dt>Excluded fields</dt>
              <dd>{lineage.excluded_inputs.join(", ")}</dd>
              <dt>Libraries</dt>
              <dd>
                {Object.entries(lineage.reproducibility.libraries || {})
                  .map(([k, v]) => `${k} ${v}`)
                  .join(" · ")}
              </dd>
            </dl>
          </section>
        </>
      )}
      <section className="panel">
        <div className="panel-title">
          <div>
            <h3>Check the complete application</h3>
            <p>
              The numerical check above is scoped to the active run. The command
              below also tests API workflows, imports, exports and statistical
              behavior.
            </p>
          </div>
        </div>
        <pre className="command">./scripts/verify.sh</pre>
        <p>
          For a presentation: generate a scenario → train a model → inspect the
          audit → predict a new transaction → apply a mitigation → verify the
          run → export a report.
        </p>
        <p className="footnote">
          A full feature checklist and expected outcomes are provided in
          docs/ACCEPTANCE.md. The public demo runs through Cloudflare while the host Mac is online.
        </p>
      </section>
    </>
  );
}
