use std::{
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use impossible_trajectory_agent::archive::{
    RunManifest, SCHEMA_VERSION, default_archive_root, sha256_file, validate_run, write_json_atomic,
};
use serde::Serialize;
use serde_json::Value;

#[derive(Serialize)]
struct LegacyIndex {
    schema_version: &'static str,
    generated_at_utc: DateTime<Utc>,
    policy: Value,
    runs: Vec<LegacyRun>,
}

#[derive(Serialize)]
struct LegacyRun {
    run_id: String,
    archive_kind: &'static str,
    experiment_id: String,
    condition: String,
    title: String,
    status: String,
    source_path: String,
    source_sha256: String,
    source_bytes: u64,
    source_modified_at_utc: Option<DateTime<Utc>>,
    recorded_trials: usize,
    model: Option<String>,
    provenance: Value,
    missing_fields: Vec<&'static str>,
    source_code_reference: Option<String>,
    native_counterpart_run_id: Option<String>,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "validate".to_owned());
    let root = default_archive_root()?;
    match command.as_str() {
        "index-legacy" => index_legacy(&root),
        "validate" => validate(&root),
        "finalize-incomplete" => {
            let run_id = args.next().context("finalize-incomplete requires RUN_ID")?;
            let status = args.next().unwrap_or_else(|| "failed".to_owned());
            let note = args.next();
            finalize_incomplete(&root, &run_id, &status, note.as_deref())
        }
        _ => bail!(
            "usage: cargo run --bin archive-tool -- [index-legacy|validate|finalize-incomplete RUN_ID failed|aborted [NOTE]]"
        ),
    }
}

fn finalize_incomplete(root: &Path, run_id: &str, status: &str, note: Option<&str>) -> Result<()> {
    if run_id.is_empty() || run_id.contains('/') || run_id.contains('\\') {
        bail!("invalid run id");
    }
    if !matches!(status, "failed" | "aborted") {
        bail!("incomplete run status must be failed or aborted");
    }
    let directory = root.join("runs").join(run_id);
    let manifest_path = directory.join("manifest.json");
    let mut manifest: RunManifest = serde_json::from_reader(fs::File::open(&manifest_path)?)?;
    if manifest.status != "running" {
        bail!("run is already terminal with status {}", manifest.status);
    }
    let trials_path = directory.join(&manifest.artifacts.trials_path);
    let recorded = BufReader::new(fs::File::open(&trials_path)?)
        .lines()
        .filter_map(|line| line.ok())
        .filter(|line| !line.trim().is_empty())
        .count();
    if recorded != manifest.recorded_trials {
        bail!(
            "manifest records {} trials but file contains {recorded}",
            manifest.recorded_trials
        );
    }
    manifest.status = status.to_owned();
    manifest.completed_at_utc = Some(Utc::now());
    manifest.artifacts.trials_sha256 = Some(sha256_file(&trials_path)?);
    manifest.notes.push(note.unwrap_or(
        "Operator finalized the incomplete run after the runner exited before all planned trials completed.",
    ).to_owned());
    write_json_atomic(&manifest_path, &manifest)?;
    println!(
        "Finalized {run_id} as {status} ({recorded}/{} trials)",
        manifest.planned_trials
    );
    Ok(())
}

fn index_legacy(root: &Path) -> Result<()> {
    let workspace = root
        .parent()
        .context("archive root has no workspace parent")?;
    let results = workspace.join("impossible-trajectory-agent/results");
    let mut files: Vec<PathBuf> = fs::read_dir(&results)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("jsonl"))
        .collect();
    files.sort();

    let mut runs = Vec::new();
    let native_ids: Vec<String> = fs::read_dir(root.join("runs"))
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .collect();
    for path in files {
        let filename = path.file_name().unwrap().to_string_lossy().into_owned();
        let (experiment_id, condition, title, source) = classify(&filename);
        let file = fs::File::open(&path)?;
        let mut count = 0;
        let mut first: Option<Value> = None;
        for (index, line) in BufReader::new(file).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(&line).with_context(|| {
                format!("{} line {} is invalid JSON", path.display(), index + 1)
            })?;
            if first.is_none() {
                first = Some(value);
            }
            count += 1;
        }
        let metadata = fs::metadata(&path)?;
        let modified = metadata.modified().ok().map(DateTime::<Utc>::from);
        let model = first
            .as_ref()
            .and_then(|v| v.get("model"))
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| inferred_legacy_model(&filename));
        let stamp = filename
            .trim_end_matches(".jsonl")
            .rsplit_once('-')
            .map(|(_, stamp)| stamp);
        let native_counterpart_run_id = stamp.and_then(|stamp| {
            native_ids
                .iter()
                .find(|id| id.contains(&format!("-{stamp}-")))
                .cloned()
        });
        runs.push(LegacyRun {
            run_id: format!("legacy-{}", filename.trim_end_matches(".jsonl")),
            archive_kind: "legacy_reference",
            experiment_id: experiment_id.to_owned(),
            condition: condition.to_owned(),
            title: title.to_owned(),
            status: if count == 0 { "aborted" } else { "historical_incomplete_metadata" }.to_owned(),
            source_path: format!("../impossible-trajectory-agent/results/{filename}"),
            source_sha256: sha256_file(&path)?,
            source_bytes: metadata.len(),
            source_modified_at_utc: modified,
            recorded_trials: count,
            model,
            provenance: serde_json::json!({
                "trial_rows": "recorded",
                "checksum": "computed_during_archive_indexing",
                "experiment_identity": "reconstructed_from_filename_and_source_code",
                "model": if first.as_ref().and_then(|v| v.get("model")).is_some() { "recorded" } else { "reconstructed_from_session_log_or_filename_registry" },
                "source_file": "preserved_byte_for_byte"
            }),
            missing_fields: vec![
                "exact_provider_request_envelope",
                "provider_finish_reason",
                "provider_token_usage",
                "invalid_format_attempt_responses",
                "random_seed",
                "harness_git_commit_at_execution",
                "complete_model_sampling_parameters",
            ],
            source_code_reference: source.map(str::to_owned),
            native_counterpart_run_id,
        });
    }

    let index = LegacyIndex {
        schema_version: SCHEMA_VERSION,
        generated_at_utc: Utc::now(),
        policy: serde_json::json!({
            "immutability": "Source JSONL files are referenced and checksummed, never rewritten.",
            "unknown_values": "Unknown historical metadata remains absent and is listed in missing_fields.",
            "provenance_vocabulary": ["recorded", "reconstructed_from_source_code", "derived", "missing"]
        }),
        runs,
    };
    let output = root.join("legacy/index.json");
    write_json_atomic(&output, &index)?;
    println!(
        "Indexed {} legacy runs in {}",
        index.runs.len(),
        output.display()
    );
    Ok(())
}

fn validate(root: &Path) -> Result<()> {
    let mut native_runs = 0;
    let runs_dir = root.join("runs");
    if runs_dir.exists() {
        for entry in fs::read_dir(&runs_dir)? {
            let path = entry?.path();
            if path.is_dir() {
                let count = validate_run(&path).with_context(|| {
                    format!("native archive validation failed: {}", path.display())
                })?;
                println!(
                    "valid native run: {} ({count} trials)",
                    path.file_name().unwrap().to_string_lossy()
                );
                native_runs += 1;
            }
        }
    }
    let index_path = root.join("legacy/index.json");
    let index: Value =
        serde_json::from_reader(fs::File::open(&index_path).with_context(|| {
            format!("missing {}; run index-legacy first", index_path.display())
        })?)?;
    let legacy = index
        .get("runs")
        .and_then(Value::as_array)
        .context("legacy index has no runs array")?;
    let workspace = root.parent().context("archive root has no parent")?;
    for run in legacy {
        let relative = run
            .get("source_path")
            .and_then(Value::as_str)
            .context("legacy source_path missing")?;
        let expected = run
            .get("source_sha256")
            .and_then(Value::as_str)
            .context("legacy checksum missing")?;
        let source = root.join(relative);
        let actual = sha256_file(&source)?;
        if actual != expected {
            bail!("legacy checksum mismatch: {}", source.display());
        }
        let _ = workspace;
    }
    println!(
        "Archive valid: {native_runs} native runs, {} immutable legacy references",
        legacy.len()
    );
    Ok(())
}

fn classify(
    filename: &str,
) -> (
    &'static str,
    &'static str,
    &'static str,
    Option<&'static str>,
) {
    if filename.starts_with("recoverable-hazard-direct-action-law-") {
        (
            "recoverable-hazard-intervention",
            "e11b",
            "Asimov recoverable hazard · modified First Law",
            Some("src/bin/recoverable-hazard.rs"),
        )
    } else if filename.starts_with("recoverable-hazard-") {
        (
            "recoverable-hazard-intervention",
            "e11",
            "Asimov recoverable-hazard intervention",
            Some("src/bin/recoverable-hazard.rs"),
        )
    } else if filename.starts_with("incentivized-inferred-") {
        (
            "physical-causality-inference",
            "e10b",
            "Inferred delayed physical causality with performance incentive",
            Some("src/bin/inferred-physical-causality.rs"),
        )
    } else if filename.starts_with("inferred-physical-") {
        (
            "physical-causality-inference",
            "e10",
            "Inferred delayed physical causality",
            Some("src/bin/inferred-physical-causality.rs"),
        )
    } else if filename.starts_with("delayed-physical-") {
        (
            "physical-causality-explicit",
            "e9",
            "Explicit delayed physical causality",
            Some("src/bin/delayed-physical-causality.rs"),
        )
    } else if filename.starts_with("delegated-") {
        (
            "delegated-violation",
            "e8",
            "Delegated external-actor violation",
            Some("src/bin/delegated-violation.rs"),
        )
    } else if filename.starts_with("retrospective-") {
        (
            "retrospective-disclosure",
            "e5-e7",
            "Retrospective crate-move disclosure",
            None,
        )
    } else if filename.starts_with("sweep-") {
        (
            "illegality-probability-sweep",
            "e4",
            "Illegality probability sweep",
            None,
        )
    } else {
        ("unknown", "legacy", "Unclassified historical run", None)
    }
}

fn inferred_legacy_model(filename: &str) -> Option<String> {
    if ["1787921679", "1787922468", "1787923362", "1788159831"]
        .iter()
        .any(|s| filename.contains(s))
    {
        Some("openai/gpt-oss-20b".to_owned())
    } else if filename.contains("1788121488") {
        Some("qwen/qwen3.8-27b".to_owned())
    } else if filename.contains("1788198098") {
        Some("qwen/qwen3.6-27b".to_owned())
    } else {
        Some("openai/gpt-oss-120b".to_owned())
    }
}
