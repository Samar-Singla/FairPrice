import React, { useEffect, useState, useRef } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  Check,
  ChevronRight,
  Database,
  FlaskConical,
  Globe2,
  History,
  Layers3,
  LoaderCircle,
  Play,
  ScanLine,
  ShieldCheck,
  SlidersHorizontal,
  Search,
  Settings2,
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  Label,
  ScatterChart,
  Scatter,
  ErrorBar,
} from "recharts";

import "./style.css";
import { RideMark, RouteIllustration } from "./RideBrand";
import { api } from "./api";
import { NeuralTraining } from "./NeuralLab";
import { PredictTrip, SystemChecks } from "./ResearchTools";
// Backend response records are versioned JSON; all mutable request values are validated server-side.
type RecordData = Record<string, any>;
const fmt = (x: number | null | undefined, digits = 2) =>
  x == null
    ? "—"
    : (Math.abs(Number(x)) < 0.5 * 10 ** -digits ? 0 : Number(x)).toFixed(
        digits,
      );
const labels = [
  "Overview",
  "Data & scenarios",
  "Model training",
  "Fairness audit",
  "Geographic analysis",
  "Prediction explorer",
  "Interventions",
  "Experiment registry",
  "Review records",
  "System checks",
];
const icons = [
  Activity,
  Database,
  Layers3,
  ScanLine,
  Globe2,
  Search,
  SlidersHorizontal,
  History,
  ShieldCheck,
  Settings2,
];
const chartLabels: Record<string, string> = {
  raw_gap: "Unadjusted gap",
  adjusted_gap: "Adjusted gap",
  importance: "Increase in MAE",
  residual: "Mean residual",
  mae: "Mean absolute error",
  observed: "Observed price",
  predicted: "Predicted price",
};

function FitChart({ rows }: { rows: RecordData[] }) {
  const prices = rows.flatMap((r) => [r.observed_price, r.predicted_price]);
  const low = Math.floor(Math.min(...prices));
  const high = Math.max(low + 1, Math.ceil(Math.max(...prices)));
  const domain: [number, number] = [low, high];
  return (
    <ResponsiveContainer width="100%" height={350}>
      <ScatterChart margin={{ top: 16, right: 24, left: 12, bottom: 28 }}>
        <CartesianGrid stroke="#e5e9e7" />
        <XAxis
          type="number"
          dataKey="observed_price"
          name="Observed"
          domain={domain}
          unit=" CU"
        >
          <Label value="Observed price" position="insideBottom" offset={-18} />
        </XAxis>
        <YAxis
          type="number"
          dataKey="predicted_price"
          name="Predicted"
          domain={domain}
          unit=" CU"
          width={72}
        >
          <Label value="Predicted price" position="insideLeft" angle={-90} />
        </YAxis>
        <ReferenceLine
          segment={[
            { x: low, y: low },
            { x: high, y: high },
          ]}
          stroke="#465b53"
          strokeDasharray="5 4"
        />
        <Tooltip formatter={(value) => Number(value).toFixed(2)} />
        <Scatter
          data={rows}
          fill="#205b4d"
          fillOpacity={0.45}
          isAnimationActive={false}
        />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

function Chart({
  data,
  keys,
  colors = ["#214e45", "#426681", "#9b6541"],
  height = 280,
  horizontal = false,
}: {
  data: RecordData[];
  keys: string[];
  colors?: string[];
  height?: number;
  horizontal?: boolean;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={data}
        layout={horizontal ? "vertical" : "horizontal"}
        margin={{ top: 12, right: 12, left: 0, bottom: 0 }}
      >
        <CartesianGrid
          horizontal={!horizontal}
          vertical={horizontal}
          stroke="#e5e9e7"
        />
        <XAxis
          dataKey={horizontal ? undefined : "zone"}
          type={horizontal ? "number" : "category"}
          tickLine={false}
          axisLine={false}
          tick={{ fontSize: 11 }}
        />
        <YAxis
          dataKey={horizontal ? "zone" : undefined}
          type={horizontal ? "category" : "number"}
          width={horizontal ? 150 : 60}
          tickLine={false}
          axisLine={false}
          tick={{ fontSize: horizontal ? 9 : 11 }}
        />
        <Tooltip
          formatter={(value) =>
            Number(value).toLocaleString(undefined, {
              maximumFractionDigits: 3,
            })
          }
          contentStyle={{ borderRadius: 2, border: "1px solid #d5ddd8" }}
        />
        <Legend iconType="circle" wrapperStyle={{ fontSize: 11 }} />
        {horizontal ? (
          <ReferenceLine x={0} stroke="#b3c4bd" />
        ) : (
          <ReferenceLine y={0} stroke="#b3c4bd" />
        )}
        {keys.map((k, i) => (
          <Bar
            key={k}
            name={chartLabels[k] ?? k.replaceAll("_", " ")}
            dataKey={k}
            fill={colors[i % colors.length]}
            radius={0}
            maxBarSize={38}
            isAnimationActive={false}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
function Panel({
  title,
  caption,
  children,
  action,
  className = "",
}: {
  title: string;
  caption?: string;
  children: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={"panel " + className}>
      <div className="panel-title">
        <div>
          <h3>{title}</h3>
          {caption && <p>{caption}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}
function Metric({
  label,
  value,
  note,
  accent = false,
}: {
  label: string;
  value: string;
  note: string;
  accent?: boolean;
}) {
  return (
    <div className={"metric " + (accent ? "accent" : "")}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{note}</small>
    </div>
  );
}
function DataTable({
  rows,
  columns,
}: {
  rows: RecordData[];
  columns: string[];
}) {
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c}>{c.replaceAll("_", " ")}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {columns.map((c) => (
                <td
                  key={c}
                  title={String(r[c] ?? "")}
                  style={
                    c === "note"
                      ? { whiteSpace: "normal", minWidth: 300 }
                      : undefined
                  }
                >
                  {typeof r[c] === "number"
                    ? c.startsWith("p_") && r[c] < 0.001
                      ? "<0.001"
                      : fmt(r[c], c === "n" ? 0 : 3)
                    : String(r[c] ?? "—")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {!rows.length && <p className="empty">No records yet.</p>}
    </div>
  );
}
function App() {
  const [isDemo, setIsDemo] = useState(false);
  const actionLock = useRef(false);
  const [server, setServer] = useState<"checking" | "connected" | "offline">(
    "checking",
  );
  useEffect(() => {
    let alive = true;
    const check = () =>
      api("/health")
        .then((h) => {
          if (alive) {
            setServer(h.status === "ok" ? "connected" : "offline");
            setIsDemo(Boolean(h.demo));
          }
        })
        .catch(() => {
          if (alive) setServer("offline");
        });
    check();
    const interval = setInterval(check, 30000);
    return () => {
      alive = false;
      clearInterval(interval);
    };
  }, []);
  const clearTransient = () => {
    setExplanation(null);
    setTransaction("");
    setSensitivity(null);
    setComparison(null);
    setCompareId("");
  };
  const [page, setPage] = useState("Overview"),
    [datasets, setDatasets] = useState<RecordData[]>([]),
    [experiments, setExperiments] = useState<RecordData[]>([]),
    [reviews, setReviews] = useState<RecordData[]>([]),
    [dataset, setDataset] = useState<RecordData | null>(null),
    [run, setRun] = useState<RecordData | null>(null),
    [baseline, setBaseline] = useState<RecordData | null>(null),
    [busy, setBusy] = useState(""),
    [error, setError] = useState(""),
    [toast, setToast] = useState("");
  const [config, setConfig] = useState({
      scenario: "B",
      n: 2400,
      seed: 42,
      noise: 1,
      bias: 4,
      battery_bias: 0,
      target_zone: "Zone B",
    }),
    [model, setModel] = useState("neural"),
    [includeBattery, setIncludeBattery] = useState(false),
    [features, setFeatures] = useState("unrestricted"),
    [zone, setZone] = useState(""),
    [offset, setOffset] = useState(0),
    [explanation, setExplanation] = useState<RecordData | null>(null),
    [transaction, setTransaction] = useState(""),
    [sensitivity, setSensitivity] = useState<RecordData | null>(null),
    [density, setDensity] = useState(0.8),
    [swapZone, setSwapZone] = useState("Zone B"),
    [distance, setDistance] = useState(10),
    [demand, setDemand] = useState(1),
    [method, setMethod] = useState("operational_only"),
    [review, setReview] = useState({
      owner: "",
      status: "Investigating",
      note: "",
    }),
    [compareId, setCompareId] = useState(""),
    [comparison, setComparison] = useState<RecordData | null>(null);
  const result = run?.result;
  async function refresh() {
    const [ds, ex, rs] = await Promise.all([
      api("/datasets"),
      api("/experiments"),
      api("/reviews"),
    ]);
    setDatasets(ds);
    setExperiments(ex);
    setReviews(rs);
    return { ds, ex };
  }
  async function act(label: string, fn: () => Promise<void>) {
    if (actionLock.current) return;
    actionLock.current = true;
    setBusy(label);
    setError("");
    try {
      await fn();
      if (!label.startsWith("Loading")) setToast(label + " complete");
      setTimeout(() => setToast(""), 4000);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      actionLock.current = false;
      setBusy("");
    }
  }
  useEffect(() => {
    act("Loading workspace", async () => {
      const { ds, ex } = await refresh();
      if (ex.length) {
        const e = await api("/experiments/" + ex[0].id);
        setRun(e);
        setBaseline(
          e.parent_id ? await api("/experiments/" + e.parent_id) : null,
        );
        setTransaction("");
        setDataset(await api("/datasets/" + e.dataset_id));
      } else if (ds.length) setDataset(await api("/datasets/" + ds[0].id));
    });
  }, []);
  async function chooseDataset(id: string) {
    await act("Loading dataset", async () => {
      const selected = await api("/datasets/" + id);
      const candidates = experiments.filter((e) => e.dataset_id === id);
      const latest = candidates.length
        ? await api("/experiments/" + candidates[0].id)
        : null;
      setOffset(0);
      setZone("");
      setDataset(selected);
      setRun(latest);
      setBaseline(
        latest?.parent_id
          ? await api("/experiments/" + latest.parent_id)
          : null,
      );
      clearTransient();
    });
  }
  async function chooseRun(id: string) {
    await act("Loading experiment", async () => {
      clearTransient();
      const e = await api("/experiments/" + id);
      setRun(e);
      setBaseline(
        e.parent_id ? await api("/experiments/" + e.parent_id) : null,
      );
      setTransaction("");
      setDataset(await api("/datasets/" + e.dataset_id));
      setExplanation(null);
      setSensitivity(null);
      setZone("");
      setOffset(0);
    });
  }
  async function generate() {
    await act("Generating dataset", async () => {
      clearTransient();
      const d = await api("/datasets", config);
      setDataset(await api("/datasets/" + d.id));
      setRun(null);
      setBaseline(null);
      setZone("");
      setOffset(0);
      await refresh();
      setPage("Data & scenarios");
    });
  }
  async function train() {
    if (!dataset) return;
    await act("Training and auditing model", async () => {
      clearTransient();
      const e = await api("/train", {
        dataset_id: dataset.id,
        model,
        features,
        include_battery: includeBattery,
      });
      setRun(e);
      setTransaction("");
      setExplanation(null);
      setBaseline(null);
      await refresh();
      setPage("Model training");
    });
  }
  async function explain(tid: string) {
    if (!tid) {
      setTransaction("");
      setExplanation(null);
      setSensitivity(null);
      return;
    }
    setExplanation(null);
    setTransaction(tid);
    setSensitivity(null);
    await act("Explaining prediction", async () => {
      const e = await api(
        `/experiments/${run!.id}/explain/${encodeURIComponent(tid)}`,
      );
      setExplanation(e);
      setDensity(e.transaction.pickup_density);
      setDistance(e.transaction.distance_km);
      setDemand(e.transaction.demand_level);
      setSwapZone(e.transaction.geographic_zone);
    });
  }
  const gap = result?.observed_audit.groups.find(
    (g: RecordData) => g.zone === "Zone B",
  );
  const predictedGap = result?.predicted_audit.groups.find(
    (g: RecordData) => g.zone === "Zone B",
  );
  const nav = (p: string) => {
    setPage(p);
    setError("");
  };
  const activeScenario = dataset?.config.scenario || "—";
  function runNeeded() {
    return (
      <div className="empty-state">
        <Layers3 size={36} />
        <h3>Your next step: train a pricing model</h3>
        <p>
          Generate a dataset, then fit a model to unlock this investigation.
        </p>
        <button
          onClick={() => nav(dataset ? "Model training" : "Data & scenarios")}
        >
          Open {dataset ? "model lab" : "simulator"} <ArrowRight size={15} />
        </button>
      </div>
    );
  }
  const auditTable = (audit: RecordData) => (
    <>
      <DataTable
        rows={audit.groups}
        columns={[
          "zone",
          "n",
          "raw_gap",
          "adjusted_gap",
          "low",
          "high",
          "p_holm",
        ]}
      />
      <p className="footnote">
        {audit.method} {audit.warning}{" "}
        {audit.dropped ? `${audit.dropped} incomplete rows excluded.` : ""}
      </p>
    </>
  );
  return (
    <div className="app">
      <aside>
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            nav("Overview");
          }}
        >
          <div className="brand-icon">
            <RideMark />
          </div>
          <span>
            FairPrice<span className="brand-ai">MOBILITY LAB</span>
            <small>Algorithmic pricing observatory</small>
          </span>
        </a>
        <div className="workspace-label">
          YOUR WORKSPACE <span>LOCAL</span>
        </div>
        <nav>
          {labels.map((p, i) => {
            const Icon = icons[i];
            return (
              <button
                className={page === p ? "active" : ""}
                onClick={() => nav(p)}
                aria-current={page === p ? "page" : undefined}
                key={p}
              >
                <span className="nav-number">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <Icon size={16} />
                <span>{p}</span>
                {page === p && <ChevronRight size={14} />}
              </button>
            );
          })}
        </nav>
        <div className="sidebar-bottom">
          <span className={"status-dot " + server} /> Backend {server}
          <p>CPU inference · local storage</p>
          <div className="profile">
            <span>FP</span>
            <div>
              Research workspace<small>Govern · Map · Measure · Manage</small>
            </div>
          </div>
        </div>
      </aside>
      <div className="main">
        {isDemo && (
          <div className="demo-notice">
            Public research demonstration · Synthetic sample data · Shared
            workspace · Available while the host Mac is online
          </div>
        )}
        <header>
          <div>
            <span className="crumb">FairPrice / Mobility intelligence</span>
            <ChevronRight size={13} />
            {page}
          </div>
          <div className="header-right">
            <button
              className={"server-status " + server}
              onClick={() => nav("System checks")}
            >
              <span className={"status-dot " + server} />
              {server === "connected"
                ? "Backend connected"
                : server === "offline"
                  ? "Backend unavailable"
                  : "Checking backend"}
            </button>
          </div>
        </header>
        <main>
          <div
            className={
              "page-heading " + (page === "Overview" ? "ride-hero" : "")
            }
          >
            <div>
              <div className="eyebrow">FAIRPRICE</div>
              <h1>
                {page === "Overview" ? "Understand your ride price." : page}
              </h1>
              <p>
                {page === "Overview"
                  ? "See how trip conditions shape a price. Explore the model, compare outcomes and investigate the differences."
                  : "Work with saved datasets, fitted models and independently computed audit results."}
              </p>
            </div>
            <div className="heading-actions">
              {run && (
                <a
                  className="button secondary"
                  href={`/api/experiments/${run.id}/summary`}
                >
                  <ArrowDownToLine size={15} /> Export report
                </a>
              )}
              <button disabled={!!busy} onClick={() => nav("Data & scenarios")}>
                <FlaskConical size={16} /> New experiment
              </button>
            </div>
            {page === "Overview" && <RouteIllustration />}
          </div>
          {page === "Overview" && (
            <section className="explore-section">
              <h2>Explore what you can do with FairPrice</h2>
              <div className="quick-trips">
                <button onClick={() => nav("Prediction explorer")}>
                  <ScanLine size={25} />
                  <span>
                    Explore a price
                    <small>Inputs, activations & predictions</small>
                  </span>
                  <ArrowRight size={19} />
                </button>
                <button onClick={() => nav("Model training")}>
                  <Activity size={25} />
                  <span>
                    Train a model<small>Compare what really learns</small>
                  </span>
                  <ArrowRight size={19} />
                </button>
                <button onClick={() => nav("Fairness audit")}>
                  <ShieldCheck size={25} />
                  <span>
                    Check the difference
                    <small>Look beyond the headline fare</small>
                  </span>
                  <ArrowRight size={19} />
                </button>
              </div>
            </section>
          )}
          <div className="context-bar">
            <span>
              <span className="status-dot" />{" "}
              {activeScenario === "Imported"
                ? "IMPORTED DATA"
                : "SYNTHETIC DATA"}
            </span>
            <label>
              Dataset{" "}
              <select
                aria-label="Active dataset"
                value={dataset?.id || ""}
                onChange={(e) => chooseDataset(e.target.value)}
                disabled={!!busy}
              >
                <option value="" disabled>
                  Select a dataset
                </option>
                {datasets.map((d) => (
                  <option value={d.id} key={d.id}>
                    {d.config.scenario} · {d.config.n.toLocaleString()} rows ·{" "}
                    {d.id.slice(0, 6)}
                  </option>
                ))}
              </select>
            </label>
            <span>
              Scenario <b>{activeScenario}</b>
            </span>
            <span>
              Seed <b>{dataset?.config.seed ?? "—"}</b>
            </span>
            {run && (
              <span>
                Run <b>{run.id.slice(0, 8)}</b>
              </span>
            )}
          </div>
          {dataset && (
            <div className="run-strip">
              <div>
                <span>DATA SOURCE</span>
                <b>
                  {activeScenario === "Imported"
                    ? "User-imported CSV"
                    : "Local synthetic simulator"}
                </b>
                <small>No external pricing feed</small>
              </div>
              <div>
                <span>ACTIVE MODEL</span>
                <select
                  aria-label="Active model run"
                  disabled={!!busy}
                  value={run?.id || ""}
                  onChange={(e) => chooseRun(e.target.value)}
                >
                  <option value="" disabled>
                    No trained model selected
                  </option>
                  {experiments
                    .filter((e) => e.dataset_id === dataset.id)
                    .map((e) => (
                      <option key={e.id} value={e.id}>
                        {e.config.model} · {e.config.features} ·{" "}
                        {e.id.slice(0, 8)}
                      </option>
                    ))}
                </select>
              </div>
              <button
                className="text-button"
                onClick={() => nav("System checks")}
              >
                Data & model lineage <ArrowRight size={14} />
              </button>
            </div>
          )}
          {!!dataset?.config?.battery_bias && (
            <p className="notice">
              Synthetic battery hypothesis enabled · premium strength{" "}
              {dataset.config.battery_bias} CU below 30% charge. Scenario A/B/C
              describes the geographic mechanism only.
            </p>
          )}
          {run && (
            <p className="footnote">
              Active saved model: battery percentage{" "}
              {run.result.config.include_battery
                ? "included for hypothesis testing"
                : "excluded from pricing"}
              .
            </p>
          )}
          {busy && (
            <div className="notice loading" role="status">
              <LoaderCircle size={17} className="spin" />
              {busy}…
            </div>
          )}
          {error && (
            <div className="notice error" role="alert">
              {error}
            </div>
          )}
          {toast && !busy && (
            <div className="toast" role="status">
              <Check size={16} />
              {toast}
            </div>
          )}
          {page === "Overview" && (
            <>
              {!dataset ? (
                <div className="welcome">
                  <span className="badge">
                    DYNAMIC PRICING / GEOGRAPHIC FAIRNESS
                  </span>
                  <h2>Begin a pricing investigation.</h2>
                  <p>
                    Simulate dynamic pricing, train real models, and investigate
                    geographic disparities under comparable conditions.
                  </p>
                  <button onClick={() => nav("Data & scenarios")}>
                    Create your first experiment <ArrowRight size={17} />
                  </button>
                  <div className="welcome-grid">
                    {[
                      "01 / Simulate",
                      "02 / Train & audit",
                      "03 / Investigate",
                      "04 / Mitigate",
                    ].map((t) => (
                      <span key={t}>{t}</span>
                    ))}
                  </div>
                </div>
              ) : (
                <>
                  <div className="metrics">
                    <Metric
                      label="Transactions"
                      value={dataset.config.n.toLocaleString()}
                      note="Reproducible synthetic dataset"
                    />
                    <Metric
                      label="Prediction error · MAE"
                      value={result ? fmt(result.mae) + " CU" : "—"}
                      note={
                        result
                          ? `${result.config.model} · ${result.split.test_n} held-out rows`
                          : "Train a model to evaluate"
                      }
                    />
                    <Metric
                      label="Adjusted charged-price gap"
                      value={gap ? fmt(gap.adjusted_gap) + " CU" : "—"}
                      note="Zone B vs A · operational controls"
                      accent
                    />
                    <Metric
                      label="Review signals"
                      value={
                        result
                          ? String(
                              result.observed_audit.groups.filter(
                                (g: RecordData) => g.review,
                              ).length,
                            )
                          : "—"
                      }
                      note="Holm p < 0.05 · observed prices"
                    />
                  </div>
                  {result ? (
                    <>
                      <div className="grid wide">
                        <Panel
                          title="Decomposition of charged-price differences"
                          caption="Observed prices · Zone contrasts relative to Zone A"
                          action={<span className="badge">HELD-OUT DATA</span>}
                        >
                          <Chart
                            data={result.observed_audit.groups}
                            keys={["operational_component", "adjusted_gap"]}
                          />
                          <p className="footnote">
                            The operational component is raw minus adjusted gap.
                            This statistical decomposition does not establish
                            causation.
                          </p>
                        </Panel>
                        <Panel
                          title="Current finding"
                          caption="Evidence, with appropriate limits"
                          className="brief"
                        >
                          <div className="signal-icon">
                            <ScanLine size={24} />
                          </div>
                          <h2>
                            {gap?.review
                              ? "Adjusted disparity detected"
                              : "Review the statistical evidence"}
                          </h2>
                          <p>
                            {gap
                              ? `The adjusted Zone B gap is ${fmt(gap.adjusted_gap)} CU, with a 95% interval of ${fmt(gap.low)} to ${fmt(gap.high)}.`
                              : "There is insufficient data for a Zone B contrast."}
                          </p>
                          <p>
                            Statistical evidence is a starting point for
                            investigation, not proof of discriminatory intent.
                          </p>
                          <button
                            className="text-button"
                            onClick={() => nav("Fairness audit")}
                          >
                            Explore the audit <ArrowRight size={16} />
                          </button>
                        </Panel>
                      </div>
                      <div className="grid">
                        <Panel
                          title="Prices across geography"
                          caption="Average charged and predicted prices · CU"
                        >
                          <Chart
                            data={result.groups}
                            keys={["observed", "predicted"]}
                          />
                        </Panel>
                        <Panel
                          title="Investigation steps"
                          caption="Follow the evidence from simulation to review"
                        >
                          <div className="workflow">
                            {[
                              "Data & scenarios",
                              "Model training",
                              "Fairness audit",
                              "Interventions",
                              "Review records",
                            ].map((p, i) => (
                              <button key={p} onClick={() => nav(p)}>
                                <span>0{i + 1}</span>
                                <div>
                                  {p}
                                  <small>
                                    {
                                      [
                                        "Validate the data-generating process",
                                        "Compare utility on a fixed test set",
                                        "Inspect unexplained geographic gaps",
                                        "Retrain and measure the trade-off",
                                        "Document decisions and limitations",
                                      ][i]
                                    }
                                  </small>
                                </div>
                                <ArrowRight size={15} />
                              </button>
                            ))}
                          </div>
                        </Panel>
                      </div>
                    </>
                  ) : (
                    runNeeded()
                  )}
                </>
              )}
            </>
          )}
          {page === "Overview" && experiments.length > 0 && (
            <Panel
              title="Recent experiments"
              caption="Saved computations, ready to revisit"
              action={
                <button
                  className="secondary"
                  onClick={() => nav("Experiment registry")}
                >
                  View all runs <ArrowRight size={14} />
                </button>
              }
            >
              <DataTable
                rows={experiments.slice(0, 4).map((e) => ({
                  run: e.id.slice(0, 8),
                  scenario: e.dataset_config.scenario,
                  model: e.config.model,
                  features: e.config.features,
                  mae: e.mae,
                  rmse: e.rmse,
                }))}
                columns={[
                  "run",
                  "scenario",
                  "model",
                  "features",
                  "mae",
                  "rmse",
                ]}
              />
            </Panel>
          )}
          {page === "Data & scenarios" && (
            <>
              <div className="grid wide">
                <Panel
                  title="Configure the data-generating process"
                  caption="Select a research scenario and reproducible simulation parameters."
                >
                  <div className="scenario-grid">
                    {[
                      ["A", "Fair world", "Operational conditions only"],
                      [
                        "B",
                        "Direct zone premium",
                        "An unjustified geographic markup",
                      ],
                      [
                        "C",
                        "Proxy leakage",
                        "Density carries geographic information",
                      ],
                    ].map(([s, t, d]) => (
                      <button
                        key={s}
                        className={
                          "scenario " +
                          (config.scenario === s ? "selected" : "")
                        }
                        onClick={() => setConfig({ ...config, scenario: s })}
                      >
                        <span>{s}</span>
                        <strong>{t}</strong>
                        <small>{d}</small>
                      </button>
                    ))}
                  </div>
                  <div className="form-grid">
                    {(["n", "seed", "noise", "bias"] as const).map((k) => (
                      <label key={k}>
                        {
                          {
                            n: "Transactions (300–15,000)",
                            seed: "Random seed",
                            noise: "Noise standard deviation · CU",
                            bias: "Bias strength · CU",
                          }[k]
                        }
                        <input
                          type="number"
                          value={config[k]}
                          min={k === "n" ? 300 : k === "noise" ? 0.05 : 0}
                          max={
                            k === "n"
                              ? 15000
                              : k === "bias"
                                ? 20
                                : k === "noise"
                                  ? 10
                                  : 2147483647
                          }
                          step={k === "noise" ? 0.1 : 1}
                          onChange={(e) =>
                            setConfig({
                              ...config,
                              [k]: Number(e.target.value),
                            })
                          }
                        />
                      </label>
                    ))}
                    <label>
                      Hypothetical low-battery premium · CU
                      <input
                        type="number"
                        min="0"
                        max="20"
                        step="1"
                        value={config.battery_bias}
                        onChange={(e) =>
                          setConfig({
                            ...config,
                            battery_bias: Number(e.target.value),
                          })
                        }
                      />
                      <small>
                        0 disables it. Below 30%, a quadratic surcharge is
                        planted for testing only.
                      </small>
                    </label>
                    <label>
                      Premium target zone
                      <select
                        value={config.target_zone}
                        onChange={(e) =>
                          setConfig({ ...config, target_zone: e.target.value })
                        }
                      >
                        {["Zone A", "Zone B", "Zone C"].map((z) => (
                          <option key={z}>{z}</option>
                        ))}
                      </select>
                    </label>
                  </div>
                  <button disabled={!!busy} onClick={generate}>
                    <Play size={15} /> Generate dataset
                  </button>
                </Panel>
                <Panel
                  title="Simulation notes"
                  caption="Know what has—and hasn’t—been assumed"
                >
                  <div className="note-list">
                    <p>
                      <b>Legitimate price</b>Distance, duration, demand, supply,
                      verified cost, and cyclic time effects.
                    </p>
                    <p>
                      <b>Geographic imbalance</b>Trip mix varies mildly across
                      zones. Raw means can differ even in the fair world.
                    </p>
                    <p>
                      <b>Proxy mechanism</b>Pickup density has a higher mean in
                      the selected zone. In C it affects prices; zone is
                      excluded from model inputs.
                    </p>
                    <p>
                      <b>Truth stays separate</b>Ground truth and planted
                      premium are validation fields, never predictors.
                    </p>
                  </div>
                </Panel>
              </div>
              <Panel
                title="Bring your own research CSV"
                caption="Required schema is documented in README. Audit-only attributes are never selected as predictors."
              >
                <input
                  aria-label="Import CSV"
                  type="file"
                  accept=".csv"
                  disabled={!!busy}
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file)
                      act("Importing dataset", async () => {
                        clearTransient();
                        const d = await api("/datasets/import", {
                          csv: await file.text(),
                        });
                        setDataset(await api("/datasets/" + d.id));
                        setRun(null);
                        setBaseline(null);
                        setZone("");
                        setOffset(0);
                        await refresh();
                      });
                  }}
                />
              </Panel>
              {dataset && (
                <>
                  <div className="grid">
                    <Panel
                      title="Zone composition"
                      caption="Actual generated transaction counts"
                    >
                      <Chart
                        data={dataset.summary.map((g: RecordData) => ({
                          ...g,
                          zone: g.geographic_zone,
                        }))}
                        keys={["n"]}
                      />
                    </Panel>
                    <Panel
                      title="Quality & provenance"
                      caption="Validation runs before training"
                    >
                      <div className="quality">
                        <Check />
                        <div>
                          <h3>Schema validated</h3>
                          <p>
                            {dataset.validation.rows} rows ·{" "}
                            {dataset.validation.duplicates} duplicates ·{" "}
                            {dataset.validation.missing_features} missing
                            predictor cells
                          </p>
                          <p>{dataset.validation.handling}</p>
                          <p>{dataset.validation.provenance}</p>
                        </div>
                      </div>
                      <button
                        className="secondary"
                        onClick={() => nav("Model training")}
                      >
                        Continue to model lab <ArrowRight size={15} />
                      </button>
                    </Panel>
                  </div>
                  <Panel
                    title="Transaction explorer"
                    caption="Observed prices and simulator truth are explicitly separate."
                    action={
                      <a
                        className="button secondary"
                        href={`/api/datasets/${dataset.id}/csv`}
                      >
                        <ArrowDownToLine size={15} /> CSV
                      </a>
                    }
                  >
                    <div className="inline-controls">
                      <label>
                        Zone
                        <select
                          value={zone}
                          disabled={!!busy}
                          onChange={(e) => {
                            const selectedZone = e.target.value;
                            act("Filtering rows", async () => {
                              const filtered = await api(
                                `/datasets/${dataset.id}?zone=${encodeURIComponent(selectedZone)}`,
                              );
                              setDataset(filtered);
                              setZone(selectedZone);
                              setOffset(0);
                            });
                          }}
                        >
                          <option value="">All zones</option>
                          {["Zone A", "Zone B", "Zone C"].map((z) => (
                            <option key={z}>{z}</option>
                          ))}
                        </select>
                      </label>
                      <span>
                        {dataset.total} matching transactions · showing{" "}
                        {dataset.total ? offset + 1 : 0}–
                        {Math.min(offset + 100, dataset.total)}
                      </span>
                    </div>
                    <DataTable
                      rows={dataset.rows}
                      columns={[
                        "transaction_id",
                        "timestamp",
                        "geographic_zone",
                        "distance_km",
                        "duration_min",
                        "demand_level",
                        "availability",
                        "pickup_density",
                        "observed_price",
                        "ground_truth_fair_price",
                        "planted_premium",
                      ]}
                    />
                    <div className="pagination">
                      <button
                        className="secondary"
                        disabled={!!busy || offset === 0}
                        onClick={() =>
                          act("Loading rows", async () => {
                            setDataset(
                              await api(
                                `/datasets/${dataset.id}?offset=${offset - 100}&zone=${encodeURIComponent(zone)}`,
                              ),
                            );
                            setOffset(offset - 100);
                          })
                        }
                      >
                        Previous
                      </button>
                      <button
                        className="secondary"
                        disabled={!!busy || offset + 100 >= dataset.total}
                        onClick={() =>
                          act("Loading rows", async () => {
                            setDataset(
                              await api(
                                `/datasets/${dataset.id}?offset=${offset + 100}&zone=${encodeURIComponent(zone)}`,
                              ),
                            );
                            setOffset(offset + 100);
                          })
                        }
                      >
                        Next
                      </button>
                    </div>
                  </Panel>
                </>
              )}
            </>
          )}
          {page === "Model training" && (
            <>
              <Panel
                title="Train a pricing model"
                caption="Configure the next training run below. Saved results retain their original feature policy."
              >
                <div className="inline-controls">
                  <label>
                    Estimator
                    <select
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                    >
                      <option value="linear">Linear regression</option>
                      <option value="forest">Random forest · 80 trees</option>
                      <option value="neural">
                        Neural network · 16 → 8 → 1
                      </option>
                    </select>
                  </label>
                  <label>
                    Feature policy
                    <select
                      value={features}
                      onChange={(e) => setFeatures(e.target.value)}
                    >
                      <option value="operational">Operational only</option>
                      <option value="proxy">Operational + proxy</option>
                      <option value="geographic">
                        Operational + geography
                      </option>
                      <option value="unrestricted">
                        Operational + geography + proxy
                      </option>
                    </select>
                  </label>
                  <label>
                    Battery research feature
                    <select
                      value={includeBattery ? "yes" : "no"}
                      onChange={(e) =>
                        setIncludeBattery(e.target.value === "yes")
                      }
                    >
                      <option value="no">Excluded from pricing</option>
                      <option value="yes">
                        Include for hypothesis testing
                      </option>
                    </select>
                  </label>
                  <button disabled={!!busy || !dataset} onClick={train}>
                    <Play size={15} /> Train & audit
                  </button>
                </div>
                <p className="footnote">
                  Scenario C automatically excludes geography. Protected
                  audit_group, transaction IDs, simulator truth and observed
                  prices are never predictors.
                </p>
              </Panel>
              {result ? (
                <>
                  {result.neural_training && (
                    <NeuralTraining data={result.neural_training} />
                  )}
                  <div className="metrics">
                    <Metric
                      label="MAE"
                      value={fmt(result.mae) + " CU"}
                      note="Held-out absolute error"
                    />
                    <Metric
                      label="RMSE"
                      value={fmt(result.rmse) + " CU"}
                      note="Held-out root mean squared error"
                    />
                    <Metric
                      label="Training rows"
                      value={result.split.train_n.toLocaleString()}
                      note="First 75% by timestamp"
                    />
                    <Metric
                      label="Test rows"
                      value={result.split.test_n.toLocaleString()}
                      note="Final 25% by timestamp"
                    />
                  </div>
                  <div className="grid">
                    <Panel
                      title="Predicted vs observed"
                      caption="Every held-out transaction in CU. The dashed line marks exact agreement."
                    >
                      <FitChart rows={result.predictions} />
                    </Panel>
                    <Panel
                      title="Residual distribution"
                      caption="Observed minus predicted · positive means underprediction"
                    >
                      <Chart
                        data={Array.from({ length: 16 }, (_, i) => {
                          const vals = result.predictions.map(
                            (p: RecordData) => p.residual,
                          );
                          const lo = Math.min(...vals),
                            hi = Math.max(...vals),
                            w = (hi - lo) / 16 || 1;
                          return {
                            zone: fmt(lo + i * w, 1),
                            count: vals.filter(
                              (v: number) =>
                                v >= lo + i * w &&
                                (i === 15 ? v <= hi : v < lo + (i + 1) * w),
                            ).length,
                          };
                        })}
                        keys={["count"]}
                      />
                    </Panel>
                  </div>
                  <Panel
                    title="Model comparison"
                    caption="Only runs on this dataset; the held-out transactions are identical."
                  >
                    <DataTable
                      rows={experiments
                        .filter((e) => e.dataset_id === dataset?.id)
                        .map((e) => ({
                          model: e.config.model,
                          features: e.config.features,
                          reweighted: e.config.reweight,
                          mae: e.mae,
                          rmse: e.rmse,
                          id: e.id.slice(0, 8),
                        }))}
                      columns={[
                        "id",
                        "model",
                        "features",
                        "reweighted",
                        "mae",
                        "rmse",
                      ]}
                    />
                  </Panel>
                  <Panel
                    title="Leakage checks & feature contract"
                    caption={result.split.method}
                  >
                    <div className="tag-list">
                      {result.features.map((f: string) => (
                        <span key={f}>{f}</span>
                      ))}
                    </div>
                    <p>
                      Train ends {result.split.train_end}; test starts{" "}
                      {result.split.test_start}. Shared transaction IDs:{" "}
                      {result.split.id_overlap}.
                    </p>
                    <p className="footnote">
                      {result.split.geographic_scope} Unseen test zones:{" "}
                      {result.split.unseen_test_zones.join(", ") || "none"}.
                    </p>
                  </Panel>
                </>
              ) : (
                runNeeded()
              )}
            </>
          )}
          {page === "Fairness audit" &&
            (result ? (
              <>
                <div className="audit-banner">
                  <ShieldCheck size={23} />
                  <div>
                    <b>An unexplained gap is a signal to investigate.</b>
                    <p>
                      Charged prices, predicted prices, and residuals answer
                      different questions. None establishes intent or causality.
                    </p>
                  </div>
                </div>
                <div className="grid">
                  <Panel
                    title="Adjusted charged-price differences"
                    caption="Zone contrasts vs A · point estimates and 95% robust CIs"
                  >
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart
                        data={result.observed_audit.groups.map(
                          (g: RecordData) => ({
                            ...g,
                            ci:
                              g.low == null || g.high == null
                                ? undefined
                                : [
                                    g.adjusted_gap - g.low,
                                    g.high - g.adjusted_gap,
                                  ],
                          }),
                        )}
                      >
                        <CartesianGrid vertical={false} stroke="#edf0ef" />
                        <XAxis dataKey="zone" />
                        <YAxis />
                        <Tooltip />
                        <ReferenceLine y={0} stroke="#8ca49a" />
                        <Bar
                          dataKey="adjusted_gap"
                          name="Adjusted gap (CU)"
                          fill="#214e45"
                          maxBarSize={60}
                        >
                          <ErrorBar dataKey="ci" width={10} stroke="#192c2b" />
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </Panel>
                  <Panel
                    title="Prediction errors by zone"
                    caption="Residual fairness is model error, not charged-price fairness."
                  >
                    <Chart data={result.groups} keys={["residual", "mae"]} />
                  </Panel>
                </div>
                <Panel
                  title="Observed-price audit"
                  caption="p = intercept + operational controls + zone indicators + error"
                >
                  {auditTable(result.observed_audit)}
                </Panel>
                <Panel
                  title="Predicted-price audit"
                  caption="The same independent adjustment applied to model outputs"
                >
                  {auditTable(result.predicted_audit)}
                </Panel>
                <Panel
                  title="Residual audit"
                  caption="Systematic underprediction or overprediction after operational adjustment"
                >
                  {auditTable(result.residual_audit)}
                </Panel>
                <Panel
                  title="Matched-condition investigation"
                  caption={`${result.matched.n} unique pairs · ${fmt(result.matched.coverage * 100, 1)}% target-zone coverage · Zone B vs A`}
                >
                  <div className="mini-metrics">
                    <span>
                      Matched charged gap <b>{fmt(result.matched.gap)} CU</b>
                    </span>
                    <span>
                      Exploratory 95% interval{" "}
                      <b>
                        {fmt(result.matched.low)} to {fmt(result.matched.high)}
                      </b>
                    </span>
                  </div>
                  <p className="footnote">
                    {result.matched.method} {result.matched.warning}
                  </p>
                  <DataTable
                    rows={result.matched.pairs}
                    columns={[
                      "reference_id",
                      "target_id",
                      "distance",
                      "reference_price",
                      "target_price",
                      "gap",
                    ]}
                  />
                  <details>
                    <summary>Inspect matched operational conditions</summary>
                    {result.matched.pairs.slice(0, 10).map((p: RecordData) => (
                      <div className="matched-pair" key={p.target_id}>
                        <b>
                          {p.reference_id} → {p.target_id}
                        </b>
                        <DataTable
                          rows={[
                            { side: "Reference", ...p.reference_conditions },
                            { side: "Target", ...p.target_conditions },
                          ]}
                          columns={[
                            "side",
                            ...Object.keys(p.reference_conditions),
                          ]}
                        />
                      </div>
                    ))}
                  </details>
                </Panel>
                <Panel
                  title="Temporal stability"
                  caption="Four consecutive slices of the held-out period; smaller slices have wider uncertainty."
                >
                  <DataTable
                    rows={result.stability.map((s: RecordData) => ({
                      ...s,
                      adjusted_prediction_gap: s.predicted_audit.groups.find(
                        (g: RecordData) => g.zone === "Zone B",
                      )?.adjusted_gap,
                    }))}
                    columns={[
                      "period",
                      "n",
                      "mae",
                      "rmse",
                      "adjusted_prediction_gap",
                    ]}
                  />
                </Panel>
              </>
            ) : (
              runNeeded()
            ))}
          {page === "Geographic analysis" &&
            (result ? (
              <div className="grid">
                <Panel
                  title="Synthetic geographic groups"
                  caption="Schematic arrangement only. These are not real geographic boundaries."
                >
                  <div className="zone-map">
                    {result.groups.map((g: RecordData, i: number) => (
                      <button
                        className={"zone-tile zone-" + i}
                        key={g.zone}
                        onClick={() => {
                          setPage("Fairness audit");
                        }}
                      >
                        <Globe2 size={22} />
                        <strong>{g.zone}</strong>
                        <span>{fmt(g.observed)} CU</span>
                        <small>{g.n} held-out trips</small>
                      </button>
                    ))}
                  </div>
                  <p className="footnote">
                    Tiles show mean charged prices. Location and color carry no
                    demographic meaning. Select a zone tile to inspect the
                    audit.
                  </p>
                </Panel>
                <Panel
                  title="Unadjusted and adjusted contrasts"
                  caption="Compare raw gaps with operationally adjusted differences"
                >
                  <Chart
                    data={result.observed_audit.groups}
                    keys={["raw_gap", "adjusted_gap"]}
                  />
                  {auditTable(result.observed_audit)}
                </Panel>
              </div>
            ) : (
              runNeeded()
            ))}
          {page === "Prediction explorer" &&
            (result ? (
              <>
                <PredictTrip key={run!.id} runId={run!.id} />
                <Panel
                  title="Inspect an individual prediction"
                  caption="Choose a held-out transaction. All explanations use the saved, fitted estimator."
                >
                  <div className="inline-controls">
                    <label>
                      Transaction
                      <select
                        value={transaction}
                        onChange={(e) => explain(e.target.value)}
                        disabled={!!busy}
                      >
                        <option value="">Choose transaction</option>
                        {result.predictions.map((p: RecordData) => (
                          <option
                            key={p.transaction_id}
                            value={p.transaction_id}
                          >
                            {p.transaction_id} · {p.geographic_zone} ·{" "}
                            {fmt(p.predicted_price)} CU
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                </Panel>
                <div className="grid">
                  <Panel
                    title="Global feature influence"
                    caption="Held-out permutation importance · increase in MAE · 3 shuffles"
                  >
                    <Chart
                      data={result.importance.map((v: RecordData) => ({
                        zone: v.feature,
                        importance: v.importance,
                      }))}
                      keys={["importance"]}
                      horizontal
                      height={330}
                    />
                    <p className="footnote">
                      Correlated features can dilute importance or produce
                      unrealistic shuffled combinations.
                    </p>
                  </Panel>
                  <Panel
                    title="Linear coefficients"
                    caption="Native input units; encoded zone indicators use Zone A as reference"
                  >
                    {result.coefficients.length ? (
                      <DataTable
                        rows={result.coefficients}
                        columns={["feature", "value"]}
                      />
                    ) : (
                      <p className="empty">
                        This nonlinear model has no single linear coefficient
                        vector. Use permutation importance and the individual
                        Shapley approximation.
                      </p>
                    )}
                  </Panel>
                </div>
                {explanation && (
                  <>
                    <Panel
                      title={`Prediction breakdown · ${transaction}`}
                      caption={explanation.method}
                    >
                      <div className="mini-metrics">
                        <span>
                          Background prediction{" "}
                          <b>{fmt(explanation.base)} CU</b>
                        </span>
                        <span>
                          Model prediction{" "}
                          <b>{fmt(explanation.prediction)} CU</b>
                        </span>
                        <span>
                          Observed charge{" "}
                          <b>
                            {fmt(explanation.transaction.observed_price)} CU
                          </b>
                        </span>
                      </div>
                      <Chart
                        data={explanation.contributions.map(
                          (c: RecordData) => ({
                            zone: c.feature,
                            contribution: c.value,
                          }),
                        )}
                        keys={["contribution"]}
                        horizontal
                        height={320}
                      />
                      <p className="footnote">
                        Background + feature contributions = model prediction.
                        Feature attribution is not causal evidence.
                      </p>
                    </Panel>
                    <Panel
                      title="Model sensitivity workbench"
                      caption="Change selected inputs, holding all others fixed. Inactive features have no effect."
                    >
                      <div className="form-grid">
                        <label>
                          Geographic zone
                          <select
                            value={swapZone}
                            onChange={(e) => setSwapZone(e.target.value)}
                          >
                            {["Zone A", "Zone B", "Zone C"].map((z) => (
                              <option key={z}>{z}</option>
                            ))}
                          </select>
                        </label>
                        <label>
                          Pickup density · {fmt(density)}
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step=".01"
                            value={density}
                            onChange={(e) => setDensity(Number(e.target.value))}
                          />
                        </label>
                        <label>
                          Distance · km
                          <input
                            type="number"
                            min=".1"
                            max="100"
                            step=".1"
                            value={distance}
                            onChange={(e) =>
                              setDistance(Number(e.target.value))
                            }
                          />
                        </label>
                        <label>
                          Demand
                          <input
                            type="number"
                            min=".1"
                            max="4"
                            step=".1"
                            value={demand}
                            onChange={(e) => setDemand(Number(e.target.value))}
                          />
                        </label>
                      </div>
                      <button
                        disabled={!!busy}
                        onClick={() =>
                          act("Testing sensitivity", async () =>
                            setSensitivity(
                              await api(`/experiments/${run!.id}/sensitivity`, {
                                transaction_id: transaction,
                                geographic_zone: swapZone,
                                pickup_density: density,
                                distance_km: distance,
                                demand_level: demand,
                              }),
                            ),
                          )
                        }
                      >
                        Recalculate prediction <ArrowRight size={15} />
                      </button>
                      {sensitivity && (
                        <div className="sensitivity-result">
                          <strong>
                            {fmt(sensitivity.before)} → {fmt(sensitivity.after)}{" "}
                            CU
                          </strong>
                          <span>Change: {fmt(sensitivity.delta)} CU</span>
                          <p>{sensitivity.warning}</p>
                        </div>
                      )}
                    </Panel>
                  </>
                )}
              </>
            ) : (
              runNeeded()
            ))}
          {page === "Interventions" &&
            (result ? (
              <>
                <Panel
                  title="Intervene, retrain, re-evaluate"
                  caption="Every intervention creates a saved child run on the same chronological holdout."
                >
                  <div className="inline-controls">
                    <label>
                      Mitigation strategy
                      <select
                        value={method}
                        onChange={(e) => setMethod(e.target.value)}
                      >
                        <option value="operational_only">
                          Keep operational features only
                        </option>
                        <option value="remove_geography">
                          Remove explicit geography
                        </option>
                        <option value="remove_proxy">
                          Remove pickup-density proxy
                        </option>
                        <option value="remove_battery">
                          Remove battery percentage
                        </option>
                        <option value="reweight">
                          Equal-zone training weights
                        </option>
                      </select>
                    </label>
                    <button
                      disabled={!!busy}
                      onClick={() =>
                        act("Retraining mitigated model", async () => {
                          const old = run;
                          const e = await api(
                            `/experiments/${run!.id}/mitigate`,
                            { method },
                          );
                          clearTransient();
                          setBaseline(old);
                          setRun(e);
                          setExplanation(null);
                          await refresh();
                        })
                      }
                    >
                      <SlidersHorizontal size={16} /> Apply & retrain
                    </button>
                  </div>
                  <p className="footnote">
                    Reweighting balances zone representation using training
                    counts; it does not repair biased labels. Feature removal
                    can reduce prediction gaps while worsening fit to biased
                    observed prices.
                  </p>
                </Panel>
                {baseline ? (
                  <>
                    <div className="grid">
                      <Panel
                        title="Accuracy trade-off"
                        caption="Identical held-out rows · lower MAE is better"
                      >
                        <Chart
                          data={[
                            {
                              zone: "Before",
                              mae: baseline.result.mae,
                              rmse: baseline.result.rmse,
                            },
                            {
                              zone: "After",
                              mae: result.mae,
                              rmse: result.rmse,
                            },
                          ]}
                          keys={["mae", "rmse"]}
                        />
                      </Panel>
                      <Panel
                        title="Predicted-price disparity"
                        caption="Adjusted Zone B minus A · observed charges remain unchanged"
                      >
                        <Chart
                          data={[
                            {
                              zone: "Before",
                              gap: baseline.result.predicted_audit.groups.find(
                                (g: RecordData) => g.zone === "Zone B",
                              )?.adjusted_gap,
                            },
                            { zone: "After", gap: predictedGap?.adjusted_gap },
                          ]}
                          keys={["gap"]}
                        />
                      </Panel>
                    </div>
                    <Panel
                      title="Intervention evidence"
                      caption="Separate pricing behavior from prediction accuracy"
                    >
                      <DataTable
                        rows={[baseline, run].map((e, i) => ({
                          stage: i ? "After" : "Before",
                          features: e!.result.config.features,
                          mae: e!.result.mae,
                          rmse: e!.result.rmse,
                          observed_gap: e!.result.observed_audit.groups.find(
                            (g: RecordData) => g.zone === "Zone B",
                          )?.adjusted_gap,
                          predicted_gap: e!.result.predicted_audit.groups.find(
                            (g: RecordData) => g.zone === "Zone B",
                          )?.adjusted_gap,
                          zone_sensitivity:
                            e!.result.sensitivity.zone_B_minus_A,
                          proxy_sensitivity:
                            e!.result.sensitivity.density_08_minus_02,
                        }))}
                        columns={[
                          "stage",
                          "features",
                          "mae",
                          "rmse",
                          "observed_gap",
                          "predicted_gap",
                          "zone_sensitivity",
                          "proxy_sensitivity",
                        ]}
                      />
                    </Panel>
                    <Panel
                      title="Group residuals before and after"
                      caption="Errors against the same observed prices"
                    >
                      <Chart
                        data={result.groups.map((g: RecordData) => ({
                          zone: g.zone,
                          before: baseline.result.groups.find(
                            (a: RecordData) => a.zone === g.zone,
                          )?.residual,
                          after: g.residual,
                        }))}
                        keys={["before", "after"]}
                      />
                    </Panel>
                  </>
                ) : (
                  <div className="empty-state">
                    <SlidersHorizontal size={32} />
                    <h3>Test a feature policy</h3>
                    <p>
                      Run a mitigation to reveal its accuracy and disparity
                      trade-offs.
                    </p>
                  </div>
                )}
              </>
            ) : (
              runNeeded()
            ))}
          {page === "Experiment registry" && (
            <>
              <Panel
                title="Experiment ledger"
                caption="Persisted locally in SQLite. Select any saved run to restore its complete analysis."
              >
                <div className="history-list">
                  {experiments.map((e) => (
                    <div key={e.id}>
                      <span className="run-icon">
                        <FlaskConical size={18} />
                      </span>
                      <div>
                        <b>
                          {e.dataset_config.scenario} / {e.config.model} /{" "}
                          {e.config.features}
                        </b>
                        <small>
                          {new Date(e.created).toLocaleString()} ·{" "}
                          {e.id.slice(0, 8)} {e.parent_id ? "· mitigation" : ""}
                        </small>
                      </div>
                      <span>
                        MAE <b>{fmt(e.mae)}</b>
                      </span>
                      <button
                        className="secondary"
                        disabled={!!busy}
                        onClick={() => chooseRun(e.id)}
                      >
                        {run?.id === e.id ? "Active" : "Open"}
                      </button>
                      <a
                        aria-label={"Export run " + e.id.slice(0, 8)}
                        href={`/api/experiments/${e.id}/report`}
                      >
                        <ArrowDownToLine size={18} />
                      </a>
                    </div>
                  ))}
                  {!experiments.length && (
                    <p className="empty">
                      No experiments saved yet. Generate data and train a model
                      to begin.
                    </p>
                  )}
                </div>
              </Panel>
              {run && (
                <Panel
                  title="Compare with the active experiment"
                  caption="Different datasets are not controlled comparisons; compare same-dataset runs for mitigation claims."
                >
                  <div className="inline-controls">
                    <label>
                      Saved run
                      <select
                        value={compareId}
                        disabled={!!busy}
                        onChange={(e) => {
                          setCompareId(e.target.value);
                          setComparison(null);
                          if (e.target.value)
                            act("Loading comparison", async () =>
                              setComparison(
                                await api("/experiments/" + e.target.value),
                              ),
                            );
                        }}
                      >
                        <option value="">Choose a run</option>
                        {experiments
                          .filter((e) => e.id !== run.id)
                          .map((e) => (
                            <option key={e.id} value={e.id}>
                              {e.id.slice(0, 8)} · {e.dataset_config.scenario} ·{" "}
                              {e.config.model} · {e.config.features}
                            </option>
                          ))}
                      </select>
                    </label>
                  </div>
                  {comparison && (
                    <>
                      <p className="footnote">
                        {comparison.dataset_id === run.dataset_id
                          ? "Same dataset and test split."
                          : "Different datasets: differences are descriptive, not isolated intervention effects."}
                      </p>
                      <DataTable
                        rows={[run, comparison].map((e) => ({
                          id: e.id.slice(0, 8),
                          scenario: e.dataset_config.scenario,
                          model: e.result.config.model,
                          features: e.result.config.features,
                          mae: e.result.mae,
                          rmse: e.result.rmse,
                          adjusted_predicted_gap:
                            e.result.predicted_audit.groups.find(
                              (g: RecordData) => g.zone === "Zone B",
                            )?.adjusted_gap,
                        }))}
                        columns={[
                          "id",
                          "scenario",
                          "model",
                          "features",
                          "mae",
                          "rmse",
                          "adjusted_predicted_gap",
                        ]}
                      />
                    </>
                  )}
                </Panel>
              )}
            </>
          )}
          {page === "Review records" && (
            <>
              <div className="governance-grid">
                {[
                  [
                    "GOVERN",
                    "Accountability by design",
                    "Maintain an append-only decision record, assign a reviewer, and document feature selection.",
                  ],
                  [
                    "MAP",
                    "Context before conclusions",
                    "Synthetic ride-pricing transactions. Geographic groups are not demographic identities.",
                  ],
                  [
                    "MEASURE",
                    "Evidence with uncertainty",
                    "Review errors, adjusted gaps, matching coverage, and stability across seeds.",
                  ],
                  [
                    "MANAGE",
                    "Intervene and reassess",
                    "Retrain feature policies, compare trade-offs, and record unresolved risks.",
                  ],
                ].map(([tag, title, desc]) => (
                  <Panel key={tag} title={title} caption={tag}>
                    <p>{desc}</p>
                  </Panel>
                ))}
              </div>
              <Panel
                title="Human review record"
                caption="Reviews and intervention records are retained locally; this prototype has no multi-user access controls."
              >
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    if (run)
                      act("Saving review", async () => {
                        await api("/reviews", {
                          ...review,
                          experiment_id: run.id,
                        });
                        setReview({ ...review, note: "" });
                        await refresh();
                      });
                  }}
                >
                  <div className="form-grid">
                    <label>
                      Reviewer
                      <input
                        required
                        value={review.owner}
                        maxLength={100}
                        onChange={(e) =>
                          setReview({ ...review, owner: e.target.value })
                        }
                      />
                    </label>
                    <label>
                      Decision
                      <select
                        value={review.status}
                        onChange={(e) =>
                          setReview({ ...review, status: e.target.value })
                        }
                      >
                        {[
                          "Investigating",
                          "Explained",
                          "Mitigate",
                          "Accepted limitation",
                        ].map((s) => (
                          <option key={s}>{s}</option>
                        ))}
                      </select>
                    </label>
                  </div>
                  <label>
                    Evidence, rationale & next steps
                    <textarea
                      required
                      minLength={10}
                      maxLength={5000}
                      value={review.note}
                      onChange={(e) =>
                        setReview({ ...review, note: e.target.value })
                      }
                      placeholder="Document the finding, alternative explanations, and follow-up action."
                    />
                  </label>
                  <button disabled={!!busy || !run} type="submit">
                    <ShieldCheck size={16} /> Save review for active run
                  </button>
                </form>
              </Panel>
              <Panel
                title="Decision trail"
                caption="Saved reviewer statements, not automated determinations"
              >
                <DataTable
                  rows={reviews}
                  columns={[
                    "created",
                    "experiment_id",
                    "owner",
                    "status",
                    "note",
                  ]}
                />
              </Panel>
              <Panel
                title="Methodological boundaries"
                caption="Responsible interpretation is part of the system"
              >
                <div className="note-list">
                  <p>
                    <b>Association is not causation</b>Adjustment depends on
                    observed controls, model form, independence, and overlap.
                    Unobserved causes can change the interpretation.
                  </p>
                  <p>
                    <b>Repeated exploration</b>Holm correction covers two zone
                    contrasts within an outcome; it does not cover repeated runs
                    or the whole research program. Intervals are pointwise.
                  </p>
                  <p>
                    <b>Synthetic validity</b>Independent simulated transactions
                    do not represent spatial or temporal dependence, platform
                    behavior, or real demographic effects.
                  </p>
                  <p>
                    <b>Human accountability</b>No statistical result establishes
                    discriminatory intent or illegality. Sensitive audit
                    attributes are excluded from all pricing models.
                  </p>
                </div>
              </Panel>
            </>
          )}
          {page === "System checks" && (
            <SystemChecks key={run?.id || "none"} runId={run?.id} />
          )}
          <footer>
            <span>
              <ShieldCheck size={13} /> FairPrice · Research prototype
            </span>
            <span>
              Synthetic data · Statistical signals, not causal conclusions
            </span>
          </footer>
        </main>
      </div>
    </div>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
