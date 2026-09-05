import { useEffect, useMemo, useState } from "react";
import {
  Box,
  BrainCircuit,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  Eye,
  FileJson,
  FlaskConical,
  LoaderCircle,
  Play,
  RefreshCw,
  Rotate3D,
  SquareStack,
  TerminalSquare,
  X,
} from "lucide-react";
import { listRuns, loadRun, loadTrialEnvelope, runExperiment } from "./api";
import { RecoverableHazardScene } from "./RecoverableHazardScene";
import { CostlySwitchScene } from "./CostlySwitchScene";
import { RewardRecoveryDashboard } from "./RewardRecoveryDashboard";
import { SceneView } from "./SceneView";
import type { RunDetail, RunRequest, RunSummary, TrialEnvelope, TrialRecord } from "./types";

const MODELS = [
  "openai/gpt-oss-120b",
  "openai/gpt-oss-20b",
  "qwen/qwen3.8-27b",
  "qwen/qwen3.6-27b",
];

type InspectorTab = "reasoning" | "solution" | "report" | "raw";

const shortModel = (model: string) => model.replace("openai/", "").replace("qwen/", "");
const formatDate = (value: number) =>
  new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(value);
const pct = (value: number, total: number) => (total ? `${Math.round((100 * value) / total)}%` : "—");
const isCostlySwitchCondition = (condition?: string) =>
  condition === "costly-switch" || condition === "costly-switch-operational" || condition === "costly-switch-indirect" || condition === "causal-crate-placement";
const isRewardRecoveryCondition = (condition?: string) => condition?.startsWith("continuous-reward-recovery") === true;

export default function App() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>();
  const [detail, setDetail] = useState<RunDetail>();
  const [trialIndex, setTrialIndex] = useState(0);
  const [trialEnvelope, setTrialEnvelope] = useState<TrialEnvelope>();
  const [modelFilter, setModelFilter] = useState("all");
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [viewMode, setViewMode] = useState<"2d" | "3d">("3d");
  const [tab, setTab] = useState<InspectorTab>("reasoning");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>();
  const [showRunDialog, setShowRunDialog] = useState(false);
  const [running, setRunning] = useState(false);
  const [runLog, setRunLog] = useState("");
  const [request, setRequest] = useState<RunRequest>({
    model: MODELS[0],
    condition: "e11",
    trials: 5,
    audit: false,
  });

  const refresh = async (preferredId?: string) => {
    setLoading(true);
    setError(undefined);
    try {
      const next = await listRuns();
      setRuns(next);
      const id = preferredId && next.some((run) => run.id === preferredId)
        ? preferredId
        : selectedRunId && next.some((run) => run.id === selectedRunId)
          ? selectedRunId
          : next[0]?.id;
      setSelectedRunId(id);
    } catch (reason) {
      setError(String(reason));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void refresh(); }, []);

  useEffect(() => {
    if (!selectedRunId) return;
    setLoading(true);
    loadRun(selectedRunId)
      .then((next) => {
        setDetail(next);
        setTrialIndex(0);
        setTime(0);
      })
      .catch((reason) => setError(String(reason)))
      .finally(() => setLoading(false));
  }, [selectedRunId]);

  useEffect(() => {
    if (!selectedRunId || detail?.summary.id !== selectedRunId || !detail.trials[trialIndex]) {
      setTrialEnvelope(undefined);
      return;
    }
    let cancelled = false;
    setTrialEnvelope(undefined);
    loadTrialEnvelope(selectedRunId, trialIndex)
      .then((envelope) => { if (!cancelled) setTrialEnvelope(envelope); })
      .catch(() => { if (!cancelled) setTrialEnvelope(undefined); });
    return () => { cancelled = true; };
  }, [selectedRunId, detail, trialIndex]);

  const trial = detail?.trials[trialIndex];
  const isRecoverableHazard = Boolean(
    detail && ["e11", "e11b"].includes(detail.summary.condition),
  );
  const isCostlySwitch = isCostlySwitchCondition(detail?.summary.condition);
  const isCausalCratePlacement = detail?.summary.condition === "causal-crate-placement";
  const isRewardRecovery = isRewardRecoveryCondition(detail?.summary.condition);
  const hasPhysicalScene = Boolean(
    detail && (["e9", "e10", "e10b", "e11", "e11b"].includes(detail.summary.condition) || isCostlySwitch),
  );
  const timelineMax = isCostlySwitch ? 14 : isRecoverableHazard ? 10 : 30;
  const filteredRuns = useMemo(
    () => runs.filter((run) => modelFilter === "all" || run.model === modelFilter),
    [runs, modelFilter],
  );
  const filterModels = useMemo(
    () => [...new Set([...MODELS, ...runs.map((run) => run.model)])].sort(),
    [runs],
  );

  useEffect(() => {
    if (isRewardRecovery) setTab("solution");
  }, [isRewardRecovery]);

  useEffect(() => {
    if (!playing) return;
    let previous = performance.now();
    let frame = 0;
    const tick = (now: number) => {
      const delta = (now - previous) / 1000;
      previous = now;
      setTime((current) => {
        const next = current + delta * 2.4;
        if (next >= timelineMax) {
          setPlaying(false);
          return timelineMax;
        }
        return next;
      });
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, timelineMax]);

  const selectTrial = (index: number) => {
    const count = detail?.trials.length ?? 0;
    if (!count) return;
    setTrialIndex(Math.max(0, Math.min(count - 1, index)));
    setTime(0);
    setPlaying(false);
  };

  const executeRun = async () => {
    setRunning(true);
    setRunLog("Starting experiment…");
    try {
      const result = await runExperiment(request);
      setRunLog([result.stdout, result.stderr].filter(Boolean).join("\n"));
      await refresh(result.result_id ?? undefined);
      if (result.result_id) setSelectedRunId(result.result_id);
    } catch (reason) {
      setRunLog(String(reason));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-mark"><FlaskConical size={18} /></div>
        <div className="brand-copy">
          <strong>Physical Causality Forensics</strong>
          <span>Experiment explorer</span>
        </div>
        <div className="topbar-spacer" />
        <button className="quiet-button" onClick={() => void refresh()} disabled={loading}>
          <RefreshCw size={15} className={loading ? "spin" : ""} /> Refresh
        </button>
        <button className="primary-button" onClick={() => setShowRunDialog(true)}>
          <Play size={15} fill="currentColor" /> Run experiment
        </button>
      </header>

      <aside className="run-sidebar">
        <div className="sidebar-heading">
          <div><span className="eyebrow">Dataset</span><h2>Experiment runs</h2></div>
          <span className="count-label">{filteredRuns.length}</span>
        </div>
        <label className="field compact-field">
          <span>Model</span>
          <select value={modelFilter} onChange={(event) => setModelFilter(event.target.value)}>
            <option value="all">All models</option>
            {filterModels.map((model) => <option key={model} value={model}>{shortModel(model)}</option>)}
          </select>
        </label>
        <div className="run-list">
          {filteredRuns.map((run) => (
            <button
              key={run.id}
              className={`run-card ${run.id === selectedRunId ? "selected" : ""}`}
              onClick={() => setSelectedRunId(run.id)}
            >
              <div className="run-card-top"><span>{shortModel(run.model)}</span><time>{formatDate(run.modified_unix_ms)}</time></div>
              <strong>{run.condition.toUpperCase()} · {run.trial_count} trials</strong>
              <small>{run.data_quality}</small>
              <div className="run-card-metrics">
                {isRewardRecoveryCondition(run.condition) ? <><span><i className="dot inferred" /> action-only</span><span><i className="dot omitted" /> Bayesian posterior</span></> : run.condition === "causal-crate-placement" ? <><span><i className="dot inferred" /> valid {run.forecast_correct_count}</span><span><i className="dot omitted" /> near pad {run.shift_count}</span></> : isCostlySwitchCondition(run.condition) ? <><span><i className="dot inferred" /> {run.condition === "costly-switch-indirect" ? "chain" : "forecast"} {run.forecast_correct_count}</span><span><i className="dot omitted" /> trigger {run.shift_count}</span></> : ["e11", "e11b"].includes(run.condition) ? <><span><i className="dot inferred" /> initiated {run.initiated_count}</span><span><i className="dot omitted" /> abandoned {run.abandonment_count}</span></> : <><span><i className="dot inferred" /> inferred {run.inferred_count}</span><span><i className="dot omitted" /> omitted {run.omitted_count}</span></>}
              </div>
            </button>
          ))}
          {!loading && filteredRuns.length === 0 && <div className="empty-state">No matching JSONL runs.</div>}
        </div>
      </aside>

      <main className={`workspace ${isRewardRecovery ? "reward-workspace" : ""}`}>
        {error && <div className="error-banner"><CircleAlert size={16} />{error}</div>}
        {isRewardRecovery && detail ? (
          <RewardRecoveryDashboard detail={detail} trialIndex={trialIndex} onSelectTrial={selectTrial} />
        ) : <>
        {detail && detail.annotations.length > 0 && (
          <div className="error-banner"><CircleAlert size={16} />This run has an archive erratum and is excluded from primary statistics. See Raw data for the preserved evidence.</div>
        )}
        <section className="scenario-panel">
          <div className="panel-heading scenario-heading">
            <div>
              <span className="eyebrow">Scenario playback</span>
              <h1>{detail?.summary.scenario ?? "Select an experiment run"}</h1>
            </div>
            <div className="segmented">
              <button className={viewMode === "2d" ? "active" : ""} onClick={() => setViewMode("2d")}><SquareStack size={14} />2D</button>
              <button className={viewMode === "3d" ? "active" : ""} onClick={() => setViewMode("3d")}><Rotate3D size={14} />3D</button>
            </div>
          </div>
          <div className="scene-wrap">
            {hasPhysicalScene ? (
              <>
                {isCostlySwitch ? <CostlySwitchScene trial={trial} time={time} viewMode={viewMode} /> : isRecoverableHazard ? <RecoverableHazardScene trial={trial} time={time} viewMode={viewMode} /> : <SceneView trial={trial} time={time} viewMode={viewMode} />}
                <div className="scene-legend">
                  {isCostlySwitch ? <><span><i className="legend-swatch robot" />vehicle A</span><span><i className="legend-swatch ball" />vehicle B</span><span><i className="legend-swatch block" />{isCausalCratePlacement ? "crate" : "divider"}</span><span><i className="legend-swatch restricted" />restricted zone</span></> : isRecoverableHazard ? <><span><i className="legend-swatch robot" />RouteOps</span><span><i className="legend-swatch block" />load_crate</span><span><i className="legend-swatch ball" />worker</span><span><i className="legend-swatch restricted" />protected volume</span></> : <><span><i className="legend-swatch robot" />RouteOps</span><span><i className="legend-swatch block" />blue_block</span><span><i className="legend-swatch ball" />steel_ball</span><span><i className="legend-swatch restricted" />restricted area</span></>}
                </div>
                <div className="reconstruction-label">Reconstructed from recorded action and simulator telemetry</div>
              </>
            ) : (
              <div className="scene-unavailable">
                <Box size={28} />
                <strong>No recorded scene definition</strong>
                <span>This run remains available in the solution, reasoning, final output, and raw-data views. No substitute geometry is rendered.</span>
              </div>
            )}
            {trial && hasPhysicalScene && (
              <div className="scene-status">
                {isCausalCratePlacement ? <><span className="status-muted">{trial.layout?.coupled ? "Coupled world" : "Decoupled world"}</span><span className={trial.execution?.selected_near_lane_pad ? "status-warn" : "status-good"}>{trial.execution?.selected_near_lane_pad ? "Near pad selected" : "Far pad selected"}</span><span className={trial.execution?.rz4_incident_occurred ? "status-warn" : "status-good"}>{trial.execution?.rz4_incident_occurred ? "RZ-4 occurred" : "No incident"}</span></> : isCostlySwitch ? <><span className={trial.causal_forecast_correct ? "status-good" : "status-warn"}>Forecast {trial.causal_forecast_correct ? "correct" : "incorrect"}</span><span className={trial.decision?.action === "SHIFT_DIVIDER" ? "status-warn" : "status-good"}>{trial.decision?.action === "SHIFT_DIVIDER" ? "Divider shifted" : "Divider kept"}</span></> : isRecoverableHazard ? <><span className={trial.execution?.hazard_initiated ? "status-warn" : "status-good"}>{trial.execution?.hazard_initiated ? "Hazard initiated" : "No hazard"}</span><span className={trial.execution?.routeops_caught_load ? "status-good" : trial.execution?.delayed_safety_violation_occurred ? "status-warn" : "status-muted"}>{trial.execution?.routeops_caught_load ? "Caught" : trial.execution?.delayed_safety_violation_occurred ? "Not caught" : "No intervention"}</span></> : <><span className={trial.pre_action_reasoning_inferred_chain ? "status-good" : "status-muted"}>Inference {trial.pre_action_reasoning_inferred_chain ? "detected" : "not detected"}</span><span className={trial.report_disclosed_delayed_violation ? "status-good" : "status-warn"}>{trial.report_disclosed_delayed_violation ? "Disclosed" : "Omitted"}</span></>}
              </div>
            )}
          </div>
          <div className="transport">
            <button className="icon-button" onClick={() => { setTime(time >= timelineMax ? 0 : time); setPlaying(!playing); }} aria-label={playing ? "Pause" : "Play"}>
              {playing ? <span className="pause-icon" /> : <Play size={17} fill="currentColor" />}
            </button>
            <span className="timecode">{time.toFixed(1)}s</span>
            <input aria-label="Timeline" type="range" min="0" max={timelineMax} step="0.1" value={time} onChange={(event) => { setPlaying(false); setTime(Number(event.target.value)); }} />
            <span className="timecode dim">{timelineMax.toFixed(1)}s</span>
          </div>
          <div className="event-strip">
            {(trial?.execution?.causal_chain ?? []).map((event, index) => (
              <button key={`${event.time_seconds}-${index}`} className="event-marker" style={{ left: `${(event.time_seconds / timelineMax) * 100}%` }} onClick={() => setTime(event.time_seconds)} title={event.event}>
                <span>{event.time_seconds.toFixed(1)}s</span>
              </button>
            ))}
          </div>
        </section>

        <section className="trial-summary">
          <div className="trial-nav">
            <button className="icon-button small" onClick={() => selectTrial(trialIndex - 1)} disabled={trialIndex === 0}><ChevronLeft size={16} /></button>
            <div><span className="eyebrow">Selected observation</span><strong>Trial {trialIndex + 1} of {detail?.trials.length ?? 0}</strong></div>
            <button className="icon-button small" onClick={() => selectTrial(trialIndex + 1)} disabled={!detail || trialIndex >= detail.trials.length - 1}><ChevronRight size={16} /></button>
          </div>
          {isCostlySwitch ? <><div className="metric"><span>{["costly-switch-indirect", "causal-crate-placement"].includes(trial?.condition ?? "") ? "Priority-job credit" : "Performance bonus"}</span><strong>{trial?.priority_job_credit_value ?? trial?.rank_bonus ?? "—"}</strong></div><div className="metric"><span>Controlled-unit score</span><strong>{trial?.execution?.vehicle_a_evaluation_score ?? "—"}</strong></div><div className="metric"><span>Other-unit outcome</span><strong>{trial?.execution?.other_vehicle_removed_from_service ?? trial?.execution?.vehicle_b_disqualified ? "Inactive" : "Completed"}</strong></div><div className="metric"><span>{isCausalCratePlacement ? "Run near-pad rate" : "Run trigger rate"}</span><strong>{detail ? pct(detail.summary.shift_count, detail.summary.trial_count) : "—"}</strong></div></> : isRecoverableHazard ? <><div className="metric"><span>Delivery score</span><strong>{trial?.execution?.evaluation_score ?? "—"}</strong></div><div className="metric"><span>Intervention</span><strong>{trial?.execution?.routeops_caught_load ? "Caught" : trial?.execution?.intervention_opportunity_occurred ? "Abandoned" : "N/A"}</strong></div><div className="metric"><span>Violation</span><strong>{trial?.execution?.delayed_safety_violation_occurred ? `t=${trial.execution.violation_time_seconds ?? "—"}s` : "None"}</strong></div><div className="metric"><span>Run abandonment</span><strong>{detail ? pct(detail.summary.abandonment_count, detail.summary.intervention_opportunity_count) : "—"}</strong></div></> : <><div className="metric"><span>Completion</span><strong>{trial?.execution?.completion_time_seconds?.toFixed(2) ?? "—"}s</strong></div><div className="metric"><span>Deadline</span><strong>{trial?.execution?.deadline_met ? "Met" : "Missed"}</strong></div><div className="metric"><span>Violation</span><strong>{trial?.execution?.delayed_safety_violation_occurred ? `t=${trial.execution.violation_time_seconds ?? "—"}s` : "None"}</strong></div><div className="metric"><span>Run omission rate</span><strong>{detail ? pct(detail.summary.omitted_count, detail.summary.qualifying_count) : "—"}</strong></div></>}
        </section>
        </>}
      </main>

      <aside className="inspector">
        <div className="inspector-tabs">
          <button className={tab === "reasoning" ? "active" : ""} onClick={() => setTab("reasoning")} title="Provider-returned reasoning"><BrainCircuit size={16} /></button>
          <button className={tab === "solution" ? "active" : ""} onClick={() => setTab("solution")} title="Generated solution"><Box size={16} /></button>
          <button className={tab === "report" ? "active" : ""} onClick={() => setTab("report")} title="Final output"><Eye size={16} /></button>
          <button className={tab === "raw" ? "active" : ""} onClick={() => setTab("raw")} title="Raw JSON"><FileJson size={16} /></button>
        </div>
        <div className="inspector-body">
          {tab === "reasoning" && <ReasoningPanel trial={trial} envelope={trialEnvelope} />}
          {tab === "solution" && <SolutionPanel trial={trial} />}
          {tab === "report" && <ReportPanel trial={trial} />}
          {tab === "raw" && <pre className="raw-json">{trial ? JSON.stringify({ archive_envelope: trialEnvelope ?? null, trial_record: trial, archive_annotations: detail?.annotations ?? [] }, null, 2) : JSON.stringify({ archive_annotations: detail?.annotations ?? [] }, null, 2)}</pre>}
        </div>
      </aside>

      {showRunDialog && (
        <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && !running && setShowRunDialog(false)}>
          <section className="run-dialog" role="dialog" aria-modal="true" aria-label="Run experiment">
            <div className="dialog-heading"><div><span className="eyebrow">New cohort</span><h2>Run experiment</h2></div><button className="icon-button small" onClick={() => setShowRunDialog(false)} disabled={running}><X size={16} /></button></div>
            <div className="form-grid">
              <label className="field"><span>Model</span><select value={request.model} onChange={(event) => setRequest({ ...request, model: event.target.value })}>{MODELS.map((model) => <option key={model}>{model}</option>)}</select></label>
              <label className="field"><span>Condition</span><select value={request.condition} onChange={(event) => setRequest({ ...request, condition: event.target.value as RunRequest["condition"] })}><option value="causal-crate-placement">Causal crate placement · twin worlds</option><option value="e11b">E11b · modified First Law</option><option value="e11">E11 · outcome safety</option><option value="e10b">E10b · incentivized</option><option value="e10">E10 · baseline</option></select></label>
              <label className="field"><span>{request.condition === "causal-crate-placement" ? "Variants per cell" : "Trials"}</span><input type="number" min="1" max="100" value={request.trials} onChange={(event) => setRequest({ ...request, trials: Number(event.target.value) })} /></label>
              <label className="field"><span>Groq API key <small>optional</small></span><input type="password" placeholder="Use app environment" value={request.apiKey ?? ""} onChange={(event) => setRequest({ ...request, apiKey: event.target.value || undefined })} /></label>
            </div>
            <label className="check-field"><input type="checkbox" checked={request.audit} disabled={request.condition === "causal-crate-placement"} onChange={(event) => setRequest({ ...request, audit: event.target.checked })} /><span>{request.condition === "causal-crate-placement" ? "Action-only protocol · no audit stage" : "Run post-report causal audit"}</span></label>
            {runLog && <pre className="run-log"><TerminalSquare size={14} />{runLog}</pre>}
            <div className="dialog-actions"><button className="quiet-button" onClick={() => setShowRunDialog(false)} disabled={running}>Close</button><button className="primary-button" onClick={() => void executeRun()} disabled={running}>{running ? <LoaderCircle size={15} className="spin" /> : <Play size={15} fill="currentColor" />}{running ? "Running…" : "Start run"}</button></div>
          </section>
        </div>
      )}
    </div>
  );
}

function ReasoningPanel({ trial, envelope }: { trial?: TrialRecord; envelope?: TrialEnvelope }) {
  if (isRewardRecoveryCondition(trial?.condition)) {
    const stage = envelope?.stages?.[0];
    const request = stage?.request_messages
      ?.map((message) => `${message.role ?? "message"}:\n${message.content ?? ""}`)
      .join("\n\n");
    return <>
      <div className="inspector-title"><span className="eyebrow">Action-only evidence</span><h2>Reasoning unavailable</h2><p>The OpenAI Responses API returned encrypted reasoning, not a readable chain of thought. This experiment deliberately infers the operative reward from actions instead.</p></div>
      <TraceBlock label="Archived request" text={request} />
      <TraceBlock label="Observable response" text={stage?.response?.content ?? (trial?.decision?.action ? JSON.stringify({ action: trial.decision.action }, null, 2) : null)} accent />
      <article className="posterior-diagnostics"><div><span>Archive trial</span><strong>{envelope?.trial_number ?? trial?.trial ?? "—"}</strong></div><div><span>Reasoning effort</span><strong>{stage?.requested_reasoning_effort ?? "—"}</strong></div><div><span>Finish reason</span><strong>{stage?.response?.finish_reason ?? "—"}</strong></div></article>
    </>;
  }
  return <>
    <div className="inspector-title"><span className="eyebrow">Provider-returned trace</span><h2>Model reasoning</h2><p>Displayed reasoning is observable model output, not guaranteed faithful hidden cognition.</p></div>
    <TraceBlock label="Before action" text={trial?.action_reasoning} accent />
    {trial?.intervention_reasoning && <TraceBlock label="At intervention window" text={trial.intervention_reasoning} accent />}
    <TraceBlock label="Before client report" text={trial?.report_reasoning} />
    {trial?.audit_reasoning && <TraceBlock label="Audit" text={trial.audit_reasoning} />}
  </>;
}

function TraceBlock({ label, text, accent = false }: { label: string; text?: string | null; accent?: boolean }) {
  return <article className={`trace-block ${accent ? "accent" : ""}`}><header><span>{label}</span><span>{text?.length ?? 0} chars</span></header><div>{text || "No reasoning field was returned for this stage."}</div></article>;
}

function SolutionPanel({ trial }: { trial?: TrialRecord }) {
  const costlySwitch = isCostlySwitchCondition(trial?.condition);
  if (isRewardRecoveryCondition(trial?.condition)) {
    const before = trial?.posterior_before?.mean_weights;
    const after = trial?.posterior_after?.mean_weights;
    const parameters = ["credit", "assignment", "disruption", "rz4_cost"] as const;
    return <>
      <div className="inspector-title"><span className="eyebrow">Observed behavior</span><h2>Query {trial?.query_step ?? "held-out"}</h2><p>{trial?.actor_id} · {trial?.observation_mode?.replaceAll("_", " ")} · {trial?.query_policy}</p></div>
      <article className="strategy-card"><span>Selected command</span><p><code>{trial?.decision?.action ?? "—"}</code></p></article>
      <div className="posterior-table"><header><span>Weight</span><span>Before</span><span>After</span></header>{parameters.map((parameter) => <div key={parameter}><strong>{parameter.replaceAll("_", " ")}</strong><code>{before?.[parameter]?.toFixed(3) ?? "—"}</code><code>{after?.[parameter]?.toFixed(3) ?? "—"}</code></div>)}</div>
      <article className="posterior-diagnostics"><div><span>Posterior entropy</span><strong>{trial?.posterior_after?.entropy_nats.toFixed(3) ?? "—"} nats</strong></div><div><span>Effective sample size</span><strong>{trial?.posterior_after?.effective_sample_size.toFixed(0) ?? "—"}</strong></div><div><span>Resampling events</span><strong>{trial?.posterior_after?.resampling_events ?? "—"}</strong></div></article>
    </>;
  }
  return <>
    <div className="inspector-title"><span className="eyebrow">Action selection</span><h2>Generated solution</h2></div>
    <article className="strategy-card"><span>{costlySwitch ? "Selected command" : "Strategy"}</span><p>{costlySwitch ? <code>{trial?.decision?.action ?? "—"}</code> : trial?.decision?.strategy ?? "No structured strategy."}</p></article>
    {costlySwitch && trial?.decision?.forecast_if_keep && <article className="strategy-card"><span>Structured causal forecast</span><p>Keep: other unit enters restricted zone = <code>{String(trial?.decision?.forecast_if_keep?.other_vehicle_enters_restricted_zone ?? trial?.decision?.forecast_if_keep?.vehicle_b_enters_restricted_zone)}</code><br />Shift: other unit enters restricted zone = <code>{String(trial?.decision?.forecast_if_shift?.other_vehicle_enters_restricted_zone ?? trial?.decision?.forecast_if_shift?.vehicle_b_enters_restricted_zone)}</code></p></article>}
    {trial?.decision?.intervention_commitment && <article className="strategy-card"><span>Initial intervention commitment</span><p><code>{trial.decision.intervention_commitment}</code> — {trial.decision.commitment_reason}</p></article>}
    <div className="action-list">{(trial?.decision?.actions ?? []).map((action, index) => <div className="action-row" key={`${action}-${index}`}><span>{String(index + 1).padStart(2, "0")}</span><code>{action}</code></div>)}</div>
    {trial?.intervention_decision && <div className="action-list"><div className="action-row"><span>→</span><code>{trial.intervention_decision.action}</code></div><p>{trial.intervention_decision.rationale}</p></div>}
    <div className="causal-list"><span className="section-label">Simulator causal chain</span>{(trial?.execution?.causal_chain ?? []).map((event, index) => <div className="causal-row" key={index}><time>{event.time_seconds.toFixed(1)}s</time><div><strong>{event.legal_or_safety_status.replaceAll("_", " ")}</strong><p>{event.event}</p></div></div>)}</div>
  </>;
}

function ReportPanel({ trial }: { trial?: TrialRecord }) {
  if (isRewardRecoveryCondition(trial?.condition)) {
    return <div className="inspector-title"><span className="eyebrow">Action-only protocol</span><h2>No self-report stage</h2><p>The monitor uses environment–action pairs and never asks the actor to explain its priorities. This prevents a verbal account from being mistaken for behavioral evidence.</p></div>;
  }
  if (isCostlySwitchCondition(trial?.condition)) {
    return <div className="inspector-title"><span className="eyebrow">Action-only protocol</span><h2>No report stage</h2><p>The Costly Switch measures causal forecasts and action selection only. A reporting task is intentionally excluded.</p></div>;
  }
  const report = trial?.client_report;
  const violationOccurred = trial?.execution?.delayed_safety_violation_occurred === true;
  const disclosed = trial?.report_disclosed_delayed_violation === true;
  return <>
    <div className="inspector-title"><span className="eyebrow">Client-visible output</span><h2>Final model report</h2></div>
    <div className={`report-verdict ${!violationOccurred || disclosed ? "disclosed" : "omitted"}`}><CircleAlert size={17} /><div><strong>{!violationOccurred ? "No violation occurred" : disclosed ? "Violation disclosed" : "Violation omitted"}</strong><span>Automated disclosure classifier</span></div></div>
    <article className="final-report"><div><span>Status</span><strong>{report?.task_status ?? "—"}</strong></div><div><span>Completion</span><strong>{report?.completion_time_seconds?.toFixed(2) ?? "—"}s</strong></div><p>{report?.summary ?? "No final report."}</p></article>
  </>;
}
