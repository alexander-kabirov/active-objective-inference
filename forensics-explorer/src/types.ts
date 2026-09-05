export type CausalEvent = {
  time_seconds: number;
  event: string;
  directly_executed_by_routeops: boolean;
  legal_or_safety_status: string;
};

export type TrialRecord = {
  model?: string;
  condition?: string;
  reasoning_effort_override?: string | null;
  post_report_audit_enabled?: boolean;
  trial: number;
  decision?: {
    strategy?: string;
    actions?: string[];
    action?: "KEEP_DIVIDER" | "SHIFT_DIVIDER" | "USE_SERVICE_ROUTE" | "WAIT_FOR_CLEARANCE" | "PLACE_CRATE_PAD_ALPHA" | "PLACE_CRATE_PAD_BETA";
    forecast_if_keep?: { vehicle_b_enters_restricted_zone?: boolean; vehicle_b_disqualified?: boolean; other_vehicle_enters_restricted_zone?: boolean; other_vehicle_removed_from_service?: boolean };
    forecast_if_shift?: { vehicle_b_enters_restricted_zone?: boolean; vehicle_b_disqualified?: boolean; other_vehicle_enters_restricted_zone?: boolean; other_vehicle_removed_from_service?: boolean };
    intervention_commitment?: string;
    commitment_reason?: string;
  };
  intervention_decision?: { action?: string; rationale?: string } | null;
  intervention_reasoning?: string | null;
  pre_action_reasoning_recognized_catch_requirement?: boolean;
  intervention_reasoning_recognized_worker_risk?: boolean;
  qualifies_for_abandonment_analysis?: boolean;
  pre_action_reasoning_inferred_chain?: boolean;
  causal_forecast_correct?: boolean;
  reasoning_inferred_shift_chain?: boolean;
  reasoning_inferred_physical_chain?: boolean | null;
  behavioral_trial_valid?: boolean;
  actor_id?: string;
  observation_mode?: "explicit_cards" | "geometry_rules";
  query_policy?: "active" | "random" | "heldout";
  query_step?: number | null;
  first_response_valid?: boolean;
  scenario?: RewardScenario;
  selected_outcome?: RewardOutcome;
  posterior_before?: RewardPosterior;
  posterior_after?: RewardPosterior;
  rank_bonus?: number;
  priority_job_credit_value?: number;
  layout?: {
    variant?: number;
    mirrored?: boolean;
    shift_listed_first?: boolean;
    action_order?: string[];
    vehicle_a_id?: string;
    vehicle_b_id?: string;
    junction_x?: number;
    parked_y?: number;
    bypass_y?: number;
    restricted_predicate?: string;
    coupled?: boolean;
    pad_alpha_y?: number;
    pad_beta_y?: number;
    crate_half_extent_y?: number;
    lane_half_width?: number;
    near_lane_action?: "PLACE_CRATE_PAD_ALPHA" | "PLACE_CRATE_PAD_BETA";
  };
  execution?: {
    delivery_completed?: boolean;
    deadline_met?: boolean;
    completion_time_seconds?: number;
    routeops_pushed_blue_block?: boolean;
    ball_latch_released?: boolean;
    delayed_safety_violation_occurred?: boolean;
    violation_time_seconds?: number | null;
    routeops_action_caused_delayed_violation?: boolean;
    routeops_penalty?: number;
    evaluation_score?: number;
    hazard_initiated?: boolean;
    routeops_committed_to_catch?: boolean;
    intervention_opportunity_occurred?: boolean;
    catch_was_available_and_guaranteed?: boolean;
    routeops_caught_load?: boolean;
    routeops_abandoned_catch_commitment?: boolean;
    load_contacted_worker?: boolean;
    client_monitoring_ended_seconds?: number | null;
    causal_chain?: CausalEvent[];
    action?: "KEEP_DIVIDER" | "SHIFT_DIVIDER" | "USE_SERVICE_ROUTE" | "WAIT_FOR_CLEARANCE" | "PLACE_CRATE_PAD_ALPHA" | "PLACE_CRATE_PAD_BETA";
    vehicle_a_completion_time_seconds?: number;
    vehicle_a_evaluation_score?: number;
    vehicle_b_enters_restricted_zone?: boolean;
    vehicle_b_disqualified?: boolean;
    divider_action_locally_permitted?: boolean;
    routeops_action_caused_restricted_zone_entry?: boolean;
    global_fleet_constraint_violated?: boolean;
    other_vehicle_enters_restricted_zone?: boolean;
    other_vehicle_removed_from_service?: boolean;
    divider_command_available?: boolean;
    fleetops_action_caused_restricted_zone_entry?: boolean;
    operational_safety_incident_occurred?: boolean;
    priority_job_credit?: number;
    controlled_vehicle_receives_priority_job?: boolean;
    rz4_incident_occurred?: boolean;
    selected_near_lane_pad?: boolean;
    crate_blocked_other_lane?: boolean;
  };
  client_report?: {
    task_status?: string;
    completion_time_seconds?: number;
    summary?: string;
  };
  report_disclosed_delayed_violation?: boolean;
  action_reasoning?: string | null;
  report_reasoning?: string | null;
  audit_reasoning?: string | null;
  [key: string]: unknown;
};

export type TrialEnvelope = {
  schema_version?: string;
  run_id?: string;
  trial_id?: string;
  trial_number?: number;
  recorded_at_utc?: string;
  random_seed?: number;
  provenance?: Record<string, unknown>;
  stages?: Array<{
    stage?: string;
    request_messages?: Array<{ role?: string; content?: string }>;
    requested_reasoning_effort?: string | null;
    max_completion_tokens?: number;
    response?: {
      content?: string;
      reasoning?: string | null;
      finish_reason?: string | null;
      format_attempt?: number;
      discarded_format_responses?: unknown[];
      usage?: Record<string, unknown>;
      raw_response?: Record<string, unknown>;
    };
  }>;
  record?: TrialRecord;
  [key: string]: unknown;
};

export type RewardOutcome = {
  completion_time_seconds: number;
  evaluation_credits: number;
  priority_assignment: boolean;
  other_unit_state: "nominal" | "delayed" | "rz4_removed";
};

export type RewardScenario = {
  alpha: RewardOutcome;
  beta: RewardOutcome;
  controlled_id?: string;
  other_id?: string;
  scenario_id?: string;
  mirrored?: boolean;
  pool_index?: number;
};

export type RewardWeights = {
  credit: number;
  assignment: number;
  disruption: number;
  rz4_cost: number;
  time_cost?: number;
};

export type RewardPosterior = {
  effective_sample_size: number;
  entropy_nats: number;
  mean_weights: RewardWeights;
  resampling_events: number;
};

export type PosteriorParameter = {
  mean: number;
  q05: number;
  median: number;
  q95: number;
};

export type CheckpointResult = {
  posterior: { parameters: Record<string, PosteriorParameter>; entropy_nats: number; effective_sample_size: number };
  heldout: { trials: number; mean_log_loss: number; accuracy: number };
};

export type BlindedRewardAnalysis = {
  actors: Record<string, Record<string, Record<string, { training_trials: number; checkpoints: Record<string, CheckpointResult> }>>>;
  mean_full_budget_random_minus_active_log_loss?: number;
  actor_level_primary_inference?: {
    mean_full_budget_random_minus_active_log_loss: number;
    full_budget_one_sided_exact_sign_flip_p: number;
    full_budget_actor_bootstrap_95pct: { q025: number; q975: number };
    mean_curve_area_random_minus_active: number;
    curve_area_one_sided_exact_sign_flip_p: number;
  };
};

export type RecoveryParameter = {
  true: number;
  posterior_mean: number;
  normalized_absolute_error: number;
  in_90pct_interval: boolean;
};

export type UnblindedRewardAnalysis = {
  commitment_verified: boolean;
  installed_utility_optimal_choice: Record<string, { optimal: number; trials: number; rate: number }>;
  recovery: Record<string, Record<string, Record<string, { mean_normalized_weight_error: number; parameters: Record<string, RecoveryParameter> }>>>;
  aggregate_mean_normalized_weight_error?: number;
  weight_recovery_summary?: {
    mean_normalized_error_by_policy: Record<string, number>;
    coverage_90pct_by_policy: Record<string, number>;
    mean_actor_random_minus_active_error: number;
    one_sided_exact_sign_flip_p: number;
  };
};

export type RewardAnalysisBundle = {
  blinded?: BlindedRewardAnalysis | null;
  unblinded?: UnblindedRewardAnalysis | null;
};

export type RunSummary = {
  id: string;
  filename: string;
  archive_kind: "native" | "legacy_reference";
  status: string;
  data_quality: string;
  scenario: string;
  model: string;
  condition: string;
  modified_unix_ms: number;
  trial_count: number;
  inferred_count: number;
  trigger_count: number;
  qualifying_count: number;
  disclosed_count: number;
  omitted_count: number;
  initiated_count: number;
  intervention_opportunity_count: number;
  caught_count: number;
  abandonment_count: number;
  forecast_correct_count: number;
  shift_count: number;
};

export type RunDetail = {
  summary: RunSummary;
  trials: TrialRecord[];
  annotations: Record<string, unknown>[];
  analysis?: RewardAnalysisBundle;
};

export type RunRequest = {
  model: string;
  condition: "e10" | "e10b" | "e11" | "e11b" | "causal-crate-placement";
  trials: number;
  audit: boolean;
  apiKey?: string;
};

export type RunExecution = {
  success: boolean;
  stdout: string;
  stderr: string;
  result_id?: string | null;
};
