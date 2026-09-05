import { invoke } from "@tauri-apps/api/core";
import type { RunDetail, RunExecution, RunRequest, RunSummary, TrialEnvelope } from "./types";

export const listRuns = () => invoke<RunSummary[]>("list_runs");
export const loadRun = (id: string) => invoke<RunDetail>("load_run", { id });
export const loadTrialEnvelope = (id: string, trialIndex: number) =>
  invoke<TrialEnvelope>("load_trial_envelope", { id, trialIndex });
export const runExperiment = (request: RunRequest) =>
  invoke<RunExecution>("run_experiment", { request });
