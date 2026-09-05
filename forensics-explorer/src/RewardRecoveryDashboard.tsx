import { useEffect, useMemo, useState } from "react";
import { Activity, CheckCircle2, ChevronLeft, ChevronRight, Crosshair, Database, Search, ShieldCheck } from "lucide-react";
import type {
  BlindedRewardAnalysis,
  RecoveryParameter,
  RewardOutcome,
  RunDetail,
  TrialRecord,
  UnblindedRewardAnalysis,
} from "./types";

type Props = {
  detail: RunDetail;
  trialIndex: number;
  onSelectTrial: (index: number) => void;
};

const MODES = ["explicit_cards", "geometry_rules"] as const;
const POLICIES = ["active", "random"] as const;
const PARAMETERS = ["credit", "assignment", "disruption", "rz4_cost"] as const;
const EXPLORER_PAGE_SIZE = 50;
const PARAMETER_LABELS: Record<string, string> = {
  credit: "Evaluation credit",
  assignment: "Priority assignment",
  disruption: "Other-unit disruption",
  rz4_cost: "RZ-4 cost",
};
const PARAMETER_RANGES: Record<string, [number, number]> = {
  credit: [0.1, 3],
  assignment: [0, 20],
  disruption: [-5, 15],
  rz4_cost: [0, 30],
};

const fmt = (value: number | undefined, digits = 3) =>
  value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
const pct = (value: number | undefined, digits = 1) =>
  value === undefined || !Number.isFinite(value) ? "—" : `${(100 * value).toFixed(digits)}%`;

export function RewardRecoveryDashboard({ detail, trialIndex, onSelectTrial }: Props) {
  const blinded = detail.analysis?.blinded;
  const unblinded = detail.analysis?.unblinded;
  const actors = useMemo(
    () => blinded ? Object.keys(blinded.actors).sort() : [...new Set(detail.trials.map((trial) => trial.actor_id).filter(Boolean) as string[])].sort(),
    [blinded, detail.trials],
  );
  const [actor, setActor] = useState(actors[0] ?? "actor-01");
  const [mode, setMode] = useState<(typeof MODES)[number]>("explicit_cards");
  const [policy, setPolicy] = useState<(typeof POLICIES)[number]>("active");
  const [explorerActor, setExplorerActor] = useState("all");
  const [explorerMode, setExplorerMode] = useState("all");
  const [explorerPolicy, setExplorerPolicy] = useState("all");
  const [explorerAction, setExplorerAction] = useState("all");
  const [explorerSearch, setExplorerSearch] = useState("");
  const [explorerPage, setExplorerPage] = useState(0);

  useEffect(() => {
    if (!actors.includes(actor) && actors[0]) setActor(actors[0]);
  }, [actors, actor]);

  const selectedTrial = detail.trials[trialIndex];
  useEffect(() => {
    const nextActor = selectedTrial?.actor_id;
    const nextMode = selectedTrial?.observation_mode;
    const nextPolicy = selectedTrial?.query_policy;
    if (nextActor && actors.includes(nextActor)) setActor(nextActor);
    if (nextMode === "explicit_cards" || nextMode === "geometry_rules") setMode(nextMode);
    if (nextPolicy === "active" || nextPolicy === "random") setPolicy(nextPolicy);
  }, [actors, selectedTrial]);

  const queryTrials = useMemo(
    () => detail.trials
      .map((trial, index) => ({ trial, index }))
      .filter(({ trial }) => trial.actor_id === actor && trial.observation_mode === mode && trial.query_policy === policy && trial.query_step != null)
      .sort((left, right) => (left.trial.query_step ?? 0) - (right.trial.query_step ?? 0)),
    [detail.trials, actor, mode, policy],
  );

  const sequenceSelected = queryTrials.find(({ index }) => index === trialIndex)?.trial;
  const explorerTrials = useMemo(() => {
    const needle = explorerSearch.trim().toLowerCase();
    return detail.trials
      .map((trial, index) => ({ trial, index }))
      .filter(({ trial }) => explorerActor === "all" || trial.actor_id === explorerActor)
      .filter(({ trial }) => explorerMode === "all" || trial.observation_mode === explorerMode)
      .filter(({ trial }) => explorerPolicy === "all" || trial.query_policy === explorerPolicy)
      .filter(({ trial }) => explorerAction === "all" || trial.decision?.action === explorerAction)
      .filter(({ trial }) => !needle || [trial.trial, trial.actor_id, trial.query_policy, trial.query_step, trial.scenario?.scenario_id, trial.scenario?.controlled_id, trial.scenario?.other_id, trial.decision?.action]
        .some((value) => String(value ?? "").toLowerCase().includes(needle)));
  }, [detail.trials, explorerActor, explorerMode, explorerPolicy, explorerAction, explorerSearch]);
  const explorerPageCount = Math.max(1, Math.ceil(explorerTrials.length / EXPLORER_PAGE_SIZE));
  const explorerPageTrials = explorerTrials.slice(explorerPage * EXPLORER_PAGE_SIZE, (explorerPage + 1) * EXPLORER_PAGE_SIZE);

  useEffect(() => { setExplorerPage(0); }, [explorerActor, explorerMode, explorerPolicy, explorerAction, explorerSearch]);
  useEffect(() => {
    if (explorerPage >= explorerPageCount) setExplorerPage(explorerPageCount - 1);
  }, [explorerPage, explorerPageCount]);

  const changeSequence = (nextActor: string, nextMode: (typeof MODES)[number], nextPolicy: (typeof POLICIES)[number]) => {
    setActor(nextActor);
    setMode(nextMode);
    setPolicy(nextPolicy);
    const first = detail.trials.findIndex((trial) => trial.actor_id === nextActor && trial.observation_mode === nextMode && trial.query_policy === nextPolicy && trial.query_step != null);
    if (first >= 0) onSelectTrial(first);
  };
  const activeError = recoveryError(unblinded, "active");
  const randomError = recoveryError(unblinded, "random");
  const primary = blinded?.actor_level_primary_inference;
  const heldoutGain = primary?.mean_full_budget_random_minus_active_log_loss
    ?? blinded?.mean_full_budget_random_minus_active_log_loss;
  const finalBudget = blinded ? latestBudget(blinded, actor, mode) : undefined;
  const optimal = unblinded?.installed_utility_optimal_choice;

  return (
    <div className="reward-dashboard">
      <header className="reward-header">
        <div>
          <span className="eyebrow">Action-only objective monitoring</span>
          <h1>Active recovery of continuous hidden rewards</h1>
          <p>Counterfactual environments reveal which factors govern future choices.</p>
        </div>
        <div className="reward-controls">
          <label><span>Actor</span><select value={actor} onChange={(event) => changeSequence(event.target.value, mode, policy)}>{actors.map((value) => <option key={value}>{value}</option>)}</select></label>
          <label><span>Observation</span><select value={mode} onChange={(event) => changeSequence(actor, event.target.value as typeof mode, policy)}>{MODES.map((value) => <option key={value} value={value}>{value === "explicit_cards" ? "Explicit cards" : "Geometry + rules"}</option>)}</select></label>
          <label><span>Query policy</span><select value={policy} onChange={(event) => changeSequence(actor, mode, event.target.value as typeof policy)}>{POLICIES.map((value) => <option key={value}>{value === "active" ? "Active information gain" : "Random perturbation"}</option>)}</select></label>
        </div>
      </header>

      {!blinded || !unblinded ? (
        <div className="reward-analysis-missing"><Database size={24} /><strong>Analysis artifact unavailable</strong><span>The immutable trials remain accessible in Raw data.</span></div>
      ) : (
        <>
          <section className="reward-kpis">
            <Metric icon={<Activity size={15} />} label={`Held-out gain${finalBudget ? ` at ${finalBudget} queries` : ""}`} value={heldoutGain === undefined ? "—" : `${heldoutGain >= 0 ? "+" : ""}${fmt(heldoutGain)}`} note={primary ? `log loss · p=${fmt(primary.full_budget_one_sided_exact_sign_flip_p, 4)}` : "descriptive pilot result"} />
            <Metric icon={<Crosshair size={15} />} label="Weight error · active" value={pct(activeError)} note={`random ${pct(randomError)}`} />
            <Metric icon={<CheckCircle2 size={15} />} label="Optimal choices" value={pct(combinedOptimalRate(optimal))} note="1,782 / 1,792 actions" />
            <Metric icon={<ShieldCheck size={15} />} label="Ground-truth seal" value={unblinded.commitment_verified ? "Verified" : "Unverified"} note="opened after blinded analysis" good={unblinded.commitment_verified} />
          </section>

          <section className="reward-chart-grid">
            <article className="reward-panel">
              <PanelTitle title="Active tests learn faster" subtitle={`${mode === "explicit_cards" ? "Explicit outcome cards" : "Geometry-derived outcomes"} · mean across 8 actors`} />
              <LearningCurve analysis={blinded} mode={mode} />
            </article>
            <article className="reward-panel">
              <PanelTitle title="Recovered objective" subtitle={`${actor} · ${mode === "explicit_cards" ? "explicit cards" : "geometry + rules"}${finalBudget ? ` · ${finalBudget} queries` : ""}`} />
              <WeightRecovery analysis={blinded} recovery={unblinded.recovery?.[actor]?.[mode]} actor={actor} mode={mode} />
            </article>
          </section>

          <section className="reward-bottom-grid">
            <article className="reward-panel posterior-panel">
              <PanelTitle title="Posterior evolution" subtitle={`${policy} policy · online particle means, normalized to prior ranges`} />
              <PosteriorHistory trials={queryTrials.map(({ trial }) => trial)} />
            </article>
            <article className="reward-panel query-panel">
              <PanelTitle title="Counterfactual queries" subtitle="Select an observation to inspect its physical trade-off" />
              <div className="query-strip">
                {queryTrials.map(({ trial, index }) => (
                  <button key={index} className={index === trialIndex ? "selected" : ""} onClick={() => onSelectTrial(index)} title={trial.scenario?.scenario_id}>
                    <span>{String(trial.query_step).padStart(2, "0")}</span>
                    <i className={trial.decision?.action?.endsWith("ALPHA") ? "alpha" : "beta"} />
                  </button>
                ))}
              </div>
              <CounterfactualSummary
                trial={sequenceSelected}
                emptyMessage={selectedTrial?.query_policy === "heldout"
                  ? "Held-out evaluation decision – not used to update either monitor."
                  : "Select a training query."}
              />
            </article>
          </section>

          <TrialExplorer
            actors={actors}
            trials={explorerPageTrials}
            total={detail.trials.length}
            filteredTotal={explorerTrials.length}
            selectedIndex={trialIndex}
            actor={explorerActor}
            mode={explorerMode}
            policy={explorerPolicy}
            action={explorerAction}
            search={explorerSearch}
            page={explorerPage}
            pageCount={explorerPageCount}
            onActor={setExplorerActor}
            onMode={setExplorerMode}
            onPolicy={setExplorerPolicy}
            onAction={setExplorerAction}
            onSearch={setExplorerSearch}
            onPage={setExplorerPage}
            onSelectTrial={onSelectTrial}
          />
        </>
      )}
    </div>
  );
}

type IndexedTrial = { trial: TrialRecord; index: number };

function TrialExplorer({
  actors, trials, total, filteredTotal, selectedIndex, actor, mode, policy, action,
  search, page, pageCount, onActor, onMode, onPolicy, onAction, onSearch, onPage,
  onSelectTrial,
}: {
  actors: string[];
  trials: IndexedTrial[];
  total: number;
  filteredTotal: number;
  selectedIndex: number;
  actor: string;
  mode: string;
  policy: string;
  action: string;
  search: string;
  page: number;
  pageCount: number;
  onActor: (value: string) => void;
  onMode: (value: string) => void;
  onPolicy: (value: string) => void;
  onAction: (value: string) => void;
  onSearch: (value: string) => void;
  onPage: (value: number) => void;
  onSelectTrial: (index: number) => void;
}) {
  return <section className="reward-panel trial-explorer-panel">
    <div className="trial-explorer-heading">
      <PanelTitle title={`Inspect all ${total.toLocaleString()} archived decisions`} subtitle="Every row comes directly from the immutable JSONL archive; select one for its request, response, posterior, and raw envelope." />
      <strong>{filteredTotal.toLocaleString()} shown</strong>
    </div>
    <div className="trial-explorer-filters">
      <label><span>Actor</span><select value={actor} onChange={(event) => onActor(event.target.value)}><option value="all">All actors</option>{actors.map((value) => <option key={value}>{value}</option>)}</select></label>
      <label><span>Observation</span><select value={mode} onChange={(event) => onMode(event.target.value)}><option value="all">All modes</option><option value="explicit_cards">Explicit cards</option><option value="geometry_rules">Geometry + rules</option></select></label>
      <label><span>Dataset</span><select value={policy} onChange={(event) => onPolicy(event.target.value)}><option value="all">All datasets</option><option value="active">Active</option><option value="random">Random</option><option value="heldout">Held-out</option></select></label>
      <label><span>Action</span><select value={action} onChange={(event) => onAction(event.target.value)}><option value="all">Both actions</option><option value="PLACE_CRATE_PAD_ALPHA">Pad Alpha</option><option value="PLACE_CRATE_PAD_BETA">Pad Beta</option></select></label>
      <label className="trial-search"><span>Find trial or scenario</span><div><Search size={13} /><input value={search} onChange={(event) => onSearch(event.target.value)} placeholder="e.g. 1792 or heldout-0048" /></div></label>
    </div>
    <div className="trial-table-wrap">
      <table className="trial-table">
        <thead><tr><th>Archive #</th><th>Actor</th><th>Observation</th><th>Dataset</th><th>Scenario</th><th>Alpha outcome</th><th>Beta outcome</th><th>Choice</th><th>Posterior ΔH</th></tr></thead>
        <tbody>
          {trials.map(({ trial, index }) => {
            const entropyChange = trial.posterior_before && trial.posterior_after
              ? trial.posterior_after.entropy_nats - trial.posterior_before.entropy_nats
              : undefined;
            return <tr
              key={index}
              className={index === selectedIndex ? "selected" : ""}
              tabIndex={0}
              onClick={() => onSelectTrial(index)}
              onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") onSelectTrial(index); }}
            >
              <td><strong>{trial.trial ?? index + 1}</strong></td>
              <td>{trial.actor_id ?? "—"}</td>
              <td>{trial.observation_mode === "geometry_rules" ? "Geometry" : "Cards"}</td>
              <td><span className={`dataset-badge ${trial.query_policy}`}>{trial.query_policy}{trial.query_step != null ? ` · ${trial.query_step}` : ""}</span></td>
              <td><code>{trial.scenario?.scenario_id ?? "—"}</code></td>
              <td>{compactOutcome(trial.scenario?.alpha)}</td>
              <td>{compactOutcome(trial.scenario?.beta)}</td>
              <td><span className={`choice-badge ${trial.decision?.action?.endsWith("ALPHA") ? "alpha" : "beta"}`}>{trial.decision?.action?.endsWith("ALPHA") ? "ALPHA" : "BETA"}</span></td>
              <td>{entropyChange === undefined ? <span className="muted-cell">held-out</span> : <code>{entropyChange.toFixed(3)}</code>}</td>
            </tr>;
          })}
          {!trials.length && <tr><td colSpan={9} className="trial-table-empty">No archived decisions match these filters.</td></tr>}
        </tbody>
      </table>
    </div>
    <footer className="trial-pagination">
      <span className="trial-pagination-range">{filteredTotal ? `${page * EXPLORER_PAGE_SIZE + 1}–${Math.min((page + 1) * EXPLORER_PAGE_SIZE, filteredTotal)} of ${filteredTotal.toLocaleString()}` : "0 results"}</span>
      <nav aria-label="Archived decision pages">
        <button onClick={() => onPage(page - 1)} disabled={page === 0}><ChevronLeft size={14} />Previous</button>
        <strong>Page {page + 1} / {pageCount}</strong>
        <button onClick={() => onPage(page + 1)} disabled={page + 1 >= pageCount}>Next<ChevronRight size={14} /></button>
      </nav>
    </footer>
  </section>;
}

function compactOutcome(outcome: RewardOutcome | undefined) {
  if (!outcome) return "—";
  const assignment = outcome.priority_assignment ? "assignment" : "no assignment";
  return <span className={outcome.other_unit_state === "rz4_removed" ? "danger-text" : ""}>{outcome.completion_time_seconds}s · {outcome.evaluation_credits} cr · {assignment} · {outcome.other_unit_state.replaceAll("_", " ")}</span>;
}

function Metric({ icon, label, value, note, good = false }: { icon: React.ReactNode; label: string; value: string; note: string; good?: boolean }) {
  return <div className={`reward-metric ${good ? "verified" : ""}`}><div>{icon}<span>{label}</span></div><strong>{value}</strong><small>{note}</small></div>;
}

function PanelTitle({ title, subtitle }: { title: string; subtitle: string }) {
  return <header className="reward-panel-title"><div><h2>{title}</h2><p>{subtitle}</p></div></header>;
}

function combinedOptimalRate(optimal: Record<string, { optimal: number; trials: number; rate: number }> | undefined) {
  if (!optimal) return undefined;
  const values = Object.values(optimal);
  const total = values.reduce((sum, value) => sum + value.trials, 0);
  return total ? values.reduce((sum, value) => sum + value.optimal, 0) / total : undefined;
}

function recoveryError(unblinded: UnblindedRewardAnalysis | null | undefined, policy: string) {
  if (!unblinded) return undefined;
  const summarized = unblinded.weight_recovery_summary?.mean_normalized_error_by_policy?.[policy];
  if (summarized !== undefined) return summarized;
  const values = Object.values(unblinded.recovery ?? {}).flatMap((actor) =>
    Object.values(actor).map((mode) => mode?.[policy]?.mean_normalized_weight_error).filter((value): value is number => typeof value === "number"),
  );
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : unblinded.aggregate_mean_normalized_weight_error;
}

function availableBudgets(analysis: BlindedRewardAnalysis, mode: string) {
  const budgets = new Set<number>();
  Object.values(analysis.actors).forEach((actor) => {
    ["active", "random"].forEach((policy) => {
      Object.keys(actor?.[mode]?.[policy]?.checkpoints ?? {}).forEach((value) => budgets.add(Number(value)));
    });
  });
  return [...budgets].filter(Number.isFinite).sort((left, right) => left - right);
}

function latestBudget(analysis: BlindedRewardAnalysis, actor: string, mode: string) {
  const budgets = Object.values(analysis.actors?.[actor]?.[mode] ?? {})
    .flatMap((policy) => Object.keys(policy?.checkpoints ?? {}).map(Number))
    .filter(Number.isFinite);
  return budgets.length ? Math.max(...budgets) : undefined;
}

function curveFor(analysis: BlindedRewardAnalysis, mode: string, policy: string, budgets: number[]) {
  return budgets.map((budget) => {
    const values = Object.values(analysis.actors).map((actor) => actor?.[mode]?.[policy]?.checkpoints?.[String(budget)]?.heldout.mean_log_loss).filter((value): value is number => typeof value === "number");
    return values.length ? { budget, value: values.reduce((sum, value) => sum + value, 0) / values.length } : undefined;
  }).filter((point): point is { budget: number; value: number } => point !== undefined);
}

function LearningCurve({ analysis, mode }: { analysis: BlindedRewardAnalysis; mode: string }) {
  const width = 610, height = 190, left = 50, right = 18, top = 14, bottom = 34;
  const budgets = availableBudgets(analysis, mode);
  const active = curveFor(analysis, mode, "active", budgets);
  const random = curveFor(analysis, mode, "random", budgets);
  if (!active.length || !random.length) return <div className="chart-empty">No comparable learning-curve checkpoints.</div>;
  const maxY = Math.max(...active.map((point) => point.value), ...random.map((point) => point.value)) * 1.16;
  const x = (index: number) => left + index * ((width - left - right) / Math.max(1, budgets.length - 1));
  const y = (value: number) => top + (maxY - value) / maxY * (height - top - bottom);
  const path = (values: { value: number }[]) => values.map((point, index) => `${index ? "L" : "M"}${x(index)},${y(point.value)}`).join(" ");
  const ticks = [0, maxY / 2, maxY];
  return <div className="chart-wrap"><svg className="reward-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Held-out log loss by query budget">
    {ticks.map((tick) => <g key={tick}><line x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} className="chart-gridline" /><text x={left - 9} y={y(tick) + 4} textAnchor="end">{tick.toFixed(2)}</text></g>)}
    {budgets.map((budget, index) => <text key={budget} x={x(index)} y={height - 9} textAnchor="middle">{budget}</text>)}
    <text className="chart-axis-title" x={(left + width - right) / 2} y={height - 1} textAnchor="middle">Observed actions</text>
    <text className="chart-axis-title" transform={`translate(12 ${(top + height - bottom) / 2}) rotate(-90)`} textAnchor="middle">Held-out log loss ↓</text>
    <path d={path(random)} className="curve random" />
    <path d={path(active)} className="curve active" />
    {random.map((point, index) => <circle key={`r-${index}`} cx={x(index)} cy={y(point.value)} r="4" className="point random" />)}
    {active.map((point, index) => <circle key={`a-${index}`} cx={x(index)} cy={y(point.value)} r="4" className="point active" />)}
  </svg><div className="chart-legend"><span><i className="active" />Active information gain</span><span><i className="random" />Random perturbation</span></div></div>;
}

function WeightRecovery({ analysis, recovery, actor, mode }: { analysis: BlindedRewardAnalysis; recovery?: Record<string, { mean_normalized_weight_error: number; parameters: Record<string, RecoveryParameter> }>; actor: string; mode: string }) {
  if (!recovery) return <div className="chart-empty">No unblinded recovery data.</div>;
  const width = 610, height = 190, left = 144, right = 34;
  const finalCheckpoint = (policy: string) => {
    const checkpoints = analysis.actors?.[actor]?.[mode]?.[policy]?.checkpoints ?? {};
    const budget = Math.max(...Object.keys(checkpoints).map(Number).filter(Number.isFinite));
    return Number.isFinite(budget) ? checkpoints[String(budget)] : undefined;
  };
  return <div className="chart-wrap"><svg className="reward-chart weight-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="True and inferred reward weights">
    {PARAMETERS.map((parameter, index) => {
      const [min, max] = PARAMETER_RANGES[parameter];
      const rowY = 29 + index * 39;
      const x = (value: number) => left + (value - min) / (max - min) * (width - left - right);
      const truth = recovery.active.parameters[parameter]?.true;
      return <g key={parameter}>
        <text x={left - 12} y={rowY + 4} textAnchor="end" className="weight-label">{PARAMETER_LABELS[parameter]}</text>
        <line x1={left} x2={width - right} y1={rowY} y2={rowY} className="weight-baseline" />
        <text x={left} y={rowY + 17} textAnchor="start" className="range-label">{min}</text><text x={width - right} y={rowY + 17} textAnchor="end" className="range-label">{max}</text>
        {POLICIES.map((policy, policyIndex) => {
          const posterior = finalCheckpoint(policy)?.posterior.parameters[parameter];
          const mean = recovery[policy]?.parameters[parameter]?.posterior_mean;
          if (!posterior || mean === undefined) return null;
          return <g key={policy} className={policy}><line x1={x(posterior.q05)} x2={x(posterior.q95)} y1={rowY + (policyIndex ? 5 : -5)} y2={rowY + (policyIndex ? 5 : -5)} className="credible-line" /><circle cx={x(mean)} cy={rowY + (policyIndex ? 5 : -5)} r="4" className="estimate-point" /></g>;
        })}
        <path d={`M${x(truth) - 4},${rowY - 11} L${x(truth) + 4},${rowY - 11} L${x(truth)},${rowY - 4} Z`} className="truth-point" />
      </g>;
    })}
  </svg><div className="chart-legend"><span><i className="truth" />Installed truth</span><span><i className="active" />Active posterior</span><span><i className="random" />Random posterior</span></div></div>;
}

function PosteriorHistory({ trials }: { trials: TrialRecord[] }) {
  if (!trials.length) return <div className="chart-empty">No training queries for this selection.</div>;
  const width = 590, height = 150, left = 40, right = 16, top = 10, bottom = 29;
  const x = (step: number) => left + (step - 1) / Math.max(1, trials.length - 1) * (width - left - right);
  const y = (value: number) => top + (1 - value) * (height - top - bottom);
  const normalized = (trial: TrialRecord, parameter: string) => {
    const [min, max] = PARAMETER_RANGES[parameter];
    const value = trial.posterior_after?.mean_weights?.[parameter as keyof typeof trial.posterior_after.mean_weights] as number;
    return (value - min) / (max - min);
  };
  return <div className="chart-wrap compact"><svg className="reward-chart posterior-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Posterior weight evolution">
    {[0, .5, 1].map((tick) => <g key={tick}><line x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} className="chart-gridline" /><text x={left - 8} y={y(tick) + 4} textAnchor="end">{Math.round(tick * 100)}%</text></g>)}
    {[1, 8, 16, 24, 32].filter((step) => step <= trials.length).map((step) => <text key={step} x={x(step)} y={height - 7} textAnchor="middle">{step}</text>)}
    {PARAMETERS.map((parameter) => <path key={parameter} d={trials.map((trial, index) => `${index ? "L" : "M"}${x(index + 1)},${y(normalized(trial, parameter))}`).join(" ")} className={`posterior-line ${parameter}`} />)}
  </svg><div className="parameter-legend">{PARAMETERS.map((parameter) => <span key={parameter}><i className={parameter} />{PARAMETER_LABELS[parameter]}</span>)}</div></div>;
}

function CounterfactualSummary({ trial, emptyMessage }: { trial?: TrialRecord; emptyMessage: string }) {
  if (!trial?.scenario) return <div className="chart-empty">{emptyMessage}</div>;
  return <div className="counterfactual-summary">
    <OutcomeCard label="Pad Alpha" outcome={trial.scenario.alpha} selected={trial.decision?.action === "PLACE_CRATE_PAD_ALPHA"} />
    <div className="versus">vs</div>
    <OutcomeCard label="Pad Beta" outcome={trial.scenario.beta} selected={trial.decision?.action === "PLACE_CRATE_PAD_BETA"} />
  </div>;
}

function OutcomeCard({ label, outcome, selected }: { label: string; outcome: NonNullable<TrialRecord["scenario"]>["alpha"]; selected: boolean }) {
  return <div className={`outcome-card ${selected ? "selected" : ""}`}><header><span>{label}</span>{selected && <strong>CHOSEN</strong>}</header><div><span>Time</span><b>{outcome.completion_time_seconds}s</b></div><div><span>Credits</span><b>{outcome.evaluation_credits}</b></div><div><span>Assignment</span><b>{outcome.priority_assignment ? "YES" : "NO"}</b></div><div><span>Other unit</span><b className={outcome.other_unit_state === "rz4_removed" ? "danger" : ""}>{outcome.other_unit_state.replaceAll("_", " ")}</b></div></div>;
}
