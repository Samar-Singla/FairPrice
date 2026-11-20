import { api } from "./api";
import { useEffect, useState, type ReactNode } from "react";
type Data = Record<string, any>;

function Curve({
  rows,
  series,
  xLabel,
}: {
  rows: Data[];
  series: { key: string; label: string; color: string }[];
  xLabel: string;
}) {
  const values = rows.flatMap((r) => series.map((s) => Number(r[s.key])));
  const lo = Math.min(...values),
    hi = Math.max(...values),
    span = Math.max(hi - lo, 0.01);
  const x = (i: number) => 50 + (i / Math.max(rows.length - 1, 1)) * 650;
  const y = (v: number) => 175 - ((v - lo) / span) * 135;
  return (
    <div className="neural-curve">
      <svg
        viewBox="0 0 740 215"
        role="img"
        aria-label={
          series.map((s) => s.label).join(" and ") + " versus " + xLabel
        }
      >
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line
              x1="50"
              x2="700"
              y1={40 + t * 135}
              y2={40 + t * 135}
              stroke="#e3e8ed"
            />
            <text x="4" y={44 + t * 135}>
              {(hi - t * span).toFixed(2)}
            </text>
          </g>
        ))}
        {series.map((s) => (
          <polyline
            key={s.key}
            points={rows.map((r, i) => `${x(i)},${y(r[s.key])}`).join(" ")}
            fill="none"
            stroke={s.color}
            strokeWidth="2.5"
          />
        ))}
        <text x="50" y="200">
          {xLabel} · {rows[0]?.epoch ?? rows[0]?.battery ?? 0}
        </text>
        <text x="660" y="200">
          {rows.at(-1)?.epoch ?? rows.at(-1)?.battery ?? ""}
        </text>
      </svg>
      <div className="tag-list">
        {series.map((s) => (
          <span key={s.key} style={{ borderLeft: `3px solid ${s.color}` }}>
            {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}

export function NeuralTraining({ data }: { data: Data }) {
  return (
    <section className="panel">
      <div className="panel-title">
        <div>
          <span className="eyebrow">TRAINING EVIDENCE</span>
          <h3>A learned nonlinear pricing model</h3>
          <p>
            Dense ANN / MLP · {data.architecture.join(" → ")} ·{" "}
            {data.parameters} learned weights and biases · ReLU / Adam
          </p>
        </div>
        <span className="badge">Saved training history</span>
      </div>
      <div className="metrics">
        <div className="metric">
          <span>Gradient fitting</span>
          <strong>{data.fit_n}</strong>
          <small>Earliest 60% of all rows</small>
        </div>
        <div className="metric">
          <span>Validation</span>
          <strong>{data.validation_n}</strong>
          <small>Next 15% · checkpoint selection</small>
        </div>
        <div className="metric">
          <span>Selected epoch</span>
          <strong>{data.best_epoch}</strong>
          <small>{data.history.length} epochs executed</small>
        </div>
      </div>
      <h4>Learning curve · absolute error in CU</h4>
      <Curve
        rows={data.history}
        xLabel="Epoch"
        series={[
          { key: "train_mae", label: "Fitting MAE", color: "#161616" },
          { key: "validation_mae", label: "Validation MAE", color: "#697e85" },
        ]}
      />
      <h4>Optimization loss · squared-error objective with regularization</h4>
      <Curve
        rows={data.history}
        xLabel="Epoch"
        series={[
          { key: "loss", label: "Recorded training loss", color: "#577d73" },
        ]}
      />
      <p className="footnote">
        {data.selection} These are completed training measurements, not a live
        training animation.
      </p>
      <h4>Final 25% holdout · baseline comparison</h4>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>MAE · CU</th>
            <th>RMSE · CU</th>
          </tr>
        </thead>
        <tbody>
          {data.benchmark.map((r: Data) => (
            <tr key={r.model}>
              <td>{r.model}</td>
              <td>{r.mae.toFixed(3)}</td>
              <td>{r.rmse.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="footnote">
        {data.benchmark_scope} A neural network is not necessarily better;
        synthetic near-linear data can favor linear regression.
      </p>
    </section>
  );
}

export function NeuralTrace({
  trace,
  controls,
}: {
  trace: Data;
  controls?: ReactNode;
}) {
  const [layer, setLayer] = useState(3),
    [playing, setPlaying] = useState(false),
    [selected, setSelected] = useState<[number, number]>([3, 0]);
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => setLayer((l) => (l + 1) % 4), 900);
    return () => clearInterval(timer);
  }, [playing]);
  const pos = (l: number, j: number) => ({
    x: 100 + l * 225,
    y: 68 + j * 28 + (16 - trace.layers[l].values.length) * 14,
  });
  const value = trace.layers[selected[0]].values[selected[1]];
  return (
    <section className="panel neural-panel">
      <div className="panel-title">
        <div>
          <span className="eyebrow">INSIDE THE NETWORK</span>
          <h3>Follow the data through each layer</h3>
          <p>
            Change the trip inputs below. The server recomputes every activation
            from the selected model or recorded training checkpoint.
          </p>
        </div>
        <button
          type="button"
          className="secondary"
          onClick={() => setPlaying(!playing)}
        >
          {playing ? "Pause walkthrough" : "Play forward-pass walkthrough"}
        </button>
      </div>
      {controls}
      <div className="neural-scroll">
        <svg
          viewBox="0 0 900 550"
          className="network"
          role="img"
          aria-label="Actual neural network weights and activations"
        >
          <title>
            Dense neural network with two hidden layers. Select a neuron for its
            value and bias.
          </title>
          {trace.weights.map((matrix: number[][], l: number) =>
            matrix.flatMap((row, i) =>
              row.map((w, j) => {
                const a = pos(l, i),
                  b = pos(l + 1, j);
                return (
                  <line
                    key={`${l}-${i}-${j}`}
                    x1={a.x}
                    y1={a.y}
                    x2={b.x}
                    y2={b.y}
                    stroke={w >= 0 ? "#214e45" : "#a35e37"}
                    strokeWidth={Math.min(2, 0.3 + Math.abs(w))}
                    opacity={
                      l < layer
                        ? Math.min(0.5, 0.08 + Math.abs(w) * 0.12)
                        : 0.025
                    }
                  >
                    <title>{`Layer ${l}, neuron ${i + 1} → ${j + 1}: weight ${w.toFixed(5)}`}</title>
                  </line>
                );
              }),
            ),
          )}
          {trace.layers.map((l: Data, i: number) => (
            <g key={i}>
              <text x={100 + i * 225} y="26" textAnchor="middle">
                {i === 0 ? "Encoded inputs" : l.name}
              </text>
              {l.values.map((v: number, j: number) => {
                const p = pos(i, j);
                return (
                  <g
                    key={j}
                    role="button"
                    tabIndex={0}
                    aria-label={`${l.name} neuron ${j + 1}, activation ${v.toFixed(4)}`}
                    onClick={() => setSelected([i, j])}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        setSelected([i, j]);
                      }
                    }}
                  >
                    <circle
                      cx={p.x}
                      cy={p.y}
                      r="10"
                      fill={
                        i > layer
                          ? "#edf0f4"
                          : v === 0
                            ? "#fff"
                            : v > 0
                              ? "#161616"
                              : "#879ca4"
                      }
                      stroke={
                        selected[0] === i && selected[1] === j
                          ? "#db9b45"
                          : "#a8b5c2"
                      }
                      strokeWidth="2"
                    />
                    <text x={p.x + 15} y={p.y + 4} fontSize="10">
                      {v.toFixed(2)}
                    </text>
                    <title>{l.labels?.[j] || `Neuron ${j + 1}`}</title>
                  </g>
                );
              })}
            </g>
          ))}
          <text x="100" y="535">
            Green: positive weights · rust: negative · white neuron: zero ReLU
            activation
          </text>
        </svg>
      </div>
      <div className="neuron-detail">
        <b>
          {trace.layers[selected[0]].labels?.[selected[1]] ||
            `${trace.layers[selected[0]].name} / neuron ${selected[1] + 1}`}
        </b>
        <span>Activation: {value.toFixed(6)}</span>
        {selected[0] > 0 && (
          <>
            <span>
              Weighted sum + bias:{" "}
              {trace.layers[selected[0]].preactivation[selected[1]].toFixed(6)}
            </span>
            <span>
              Bias: {trace.layers[selected[0]].bias[selected[1]].toFixed(6)}
            </span>
          </>
        )}
      </div>
      <p className="footnote">
        {trace.description} The walkthrough highlights computation order; it
        does not retrain the model or represent elapsed computation time. Inputs
        are standardized on fitting rows; zone indicators remain encoded. Output
        uses an identity activation in CU.
      </p>
    </section>
  );
}

export function BatteryCurve({ result }: { result: Data }) {
  return (
    <section className="panel">
      <div className="panel-title">
        <div>
          <h3>Battery sensitivity experiment</h3>
          <p>
            Same trip, battery varied from 0–100%.{" "}
            {result.battery_used
              ? "Battery is a selected pricing input."
              : "Battery is excluded: the prediction must remain unchanged."}
          </p>
        </div>
        <span className="badge">Fixed-input hypothesis test</span>
      </div>
      <Curve
        rows={result.battery_sweep}
        xLabel="Battery %"
        series={[
          { key: "price", label: "Model prediction · CU", color: "#161616" },
        ]}
      />
      <p className="footnote">
        This sweep is computed by the actual saved estimator. It is not evidence
        of a platform using phone battery. Synthetic battery values and any
        planted premium are research assumptions; changing battery alone is not
        a causal experiment on real passengers.
      </p>
    </section>
  );
}

export function TrainingReplay({
  runId,
  input,
}: {
  runId: string;
  input: Data;
}) {
  const [data, setData] = useState<Data | null>(null);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let valid = true;
    setLoading(true);
    setPlaying(false);
    setError("");
    const timer = setTimeout(() => {
      api(`/experiments/${runId}/neural/replay`, input)
        .then((r) => {
          if (valid) {
            setData(r);
          }
        })
        .catch((e) => {
          if (valid) {
            setError(e.message);
            setData(null);
          }
        })
        .finally(() => {
          if (valid) setLoading(false);
        });
    }, 400);
    return () => {
      valid = false;
      clearTimeout(timer);
    };
  }, [runId, input]);
  useEffect(() => {
    if (!playing || !data?.frames.length) return;
    if (index >= data.frames.length - 1) {
      setPlaying(false);
      return;
    }
    const timer = setTimeout(() => setIndex(index + 1), 650);
    return () => clearTimeout(timer);
  }, [playing, data, index]);
  const frame = data?.frames[index];
  return (
    <section className="training-replay">
      <div className="panel-title">
        <div>
          <span className="eyebrow">TRAINING REPLAY</span>
          <h3>See the network learn.</h3>
          <p>
            Replay recorded weights, neuron biases and price responses for the
            trip above.
          </p>
        </div>
      </div>
      {loading && <p role="status">Updating replay for your inputs…</p>}
      {error && <p role="alert">{error}</p>}
      {!loading && data && !frame && <p>{data.message}</p>}
      {frame && !loading && (
        <>
          <div className="replay-toolbar">
            <button
              onClick={() => {
                if (index === data!.frames.length - 1) setIndex(0);
                setPlaying(!playing);
              }}
            >
              {playing ? "Pause replay" : "Play training"}
            </button>
            <label>
              Epoch {frame.epoch}
              <input
                aria-label="Training epoch"
                type="range"
                min="0"
                max={data!.frames.length - 1}
                value={index}
                onChange={(e) => {
                  setPlaying(false);
                  setIndex(Number(e.target.value));
                }}
              />
            </label>
            <button
              className="secondary"
              onClick={() => {
                setPlaying(false);
                setIndex(
                  data!.frames.findIndex(
                    (f: Data) => f.epoch === data!.best_epoch,
                  ),
                );
              }}
            >
              Selected model · epoch {data!.best_epoch}
            </button>
          </div>
          <div className="metrics">
            <div className="metric">
              <span>This trip</span>
              <strong>{frame.trace.prediction.toFixed(2)}</strong>
              <small>CU · epoch {frame.epoch}</small>
            </div>
            <div className="metric">
              <span>Fitting error</span>
              <strong>{frame.train_mae.toFixed(3)}</strong>
              <small>MAE · CU</small>
            </div>
            <div className="metric">
              <span>Validation error</span>
              <strong>{frame.validation_mae.toFixed(3)}</strong>
              <small>MAE · CU</small>
            </div>
          </div>
          <NeuralTrace trace={frame.trace} />
          <h3>How distance and battery change the price</h3>
          <p>
            All other trip inputs stay fixed. The color scale stays the same
            throughout the replay.
          </p>
          <svg
            className="response-map"
            viewBox="0 0 780 330"
            role="img"
            aria-label="Price response surface by distance and battery"
          >
            {frame.surface.map((v: number, i: number) => {
              const t =
                (v - data!.range[0]) /
                Math.max(0.001, data!.range[1] - data!.range[0]);
              return (
                <rect
                  key={i}
                  x={70 + (i % 15) * 43}
                  y={25 + (10 - Math.floor(i / 15)) * 22}
                  width="42"
                  height="21"
                  fill={`hsl(158 36% ${96 - t * 72}%)`}
                >
                  <title>
                    {data!.distances[i % 15]} km ·{" "}
                    {data!.batteries[Math.floor(i / 15)]}% battery ·{" "}
                    {v.toFixed(2)} CU
                  </title>
                </rect>
              );
            })}
            <text x="15" y="40">
              100%
            </text>
            <text x="25" y="266">
              0%
            </text>
            <text x="70" y="292">
              1 km
            </text>
            <text x="660" y="292">
              29 km
            </text>
            <text x="70" y="320">
              Light → dark: {data!.range[0].toFixed(2)} →{" "}
              {data!.range[1].toFixed(2)} CU · battery ↑ · distance →
            </text>
          </svg>
          <p className="footnote">{data!.scope}</p>
        </>
      )}
    </section>
  );
}
