use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub const SCHEMA_VERSION: &str = "1.0.0";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentMetadata {
    pub id: String,
    pub condition: String,
    pub title: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub provider: String,
    pub model: String,
    pub endpoint: String,
    pub reasoning_effort_override: Option<String>,
    pub parameters: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HarnessMetadata {
    pub crate_name: String,
    pub crate_version: String,
    pub binary: String,
    pub git_commit: Option<String>,
    pub git_worktree_dirty: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    pub trials_path: String,
    pub trials_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunManifest {
    pub schema_version: String,
    pub run_id: String,
    pub archive_kind: String,
    pub experiment: ExperimentMetadata,
    pub status: String,
    pub created_at_utc: DateTime<Utc>,
    pub completed_at_utc: Option<DateTime<Utc>>,
    pub planned_trials: usize,
    pub recorded_trials: usize,
    pub model: ModelMetadata,
    pub scenario: Value,
    pub prompts: BTreeMap<String, String>,
    pub harness: HarnessMetadata,
    pub randomization: Value,
    pub artifacts: ArtifactMetadata,
    pub provenance: Value,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StageRecord {
    pub stage: String,
    pub request_messages: Value,
    pub requested_reasoning_effort: String,
    pub max_completion_tokens: u32,
    pub response: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrialEnvelope {
    pub schema_version: String,
    pub run_id: String,
    pub trial_id: String,
    pub trial_number: usize,
    pub recorded_at_utc: DateTime<Utc>,
    pub random_seed: Option<u64>,
    pub stages: Vec<StageRecord>,
    pub record: Value,
    pub provenance: Value,
}

pub struct RunArchive {
    directory: PathBuf,
    manifest: RunManifest,
    trials: File,
}

impl RunManifest {
    pub fn new(
        experiment: ExperimentMetadata,
        planned_trials: usize,
        model: ModelMetadata,
        scenario: Value,
        prompts: BTreeMap<String, String>,
        binary: &str,
        randomization: Value,
    ) -> Result<Self> {
        let run_id = new_run_id(&experiment.condition)?;
        let (git_commit, git_worktree_dirty) = git_state();
        Ok(Self {
            schema_version: SCHEMA_VERSION.to_owned(),
            run_id,
            archive_kind: "native".to_owned(),
            experiment,
            status: "running".to_owned(),
            created_at_utc: Utc::now(),
            completed_at_utc: None,
            planned_trials,
            recorded_trials: 0,
            model,
            scenario,
            prompts,
            harness: HarnessMetadata {
                crate_name: env!("CARGO_PKG_NAME").to_owned(),
                crate_version: env!("CARGO_PKG_VERSION").to_owned(),
                binary: binary.to_owned(),
                git_commit,
                git_worktree_dirty,
            },
            randomization,
            artifacts: ArtifactMetadata {
                trials_path: "trials.jsonl".to_owned(),
                trials_sha256: None,
            },
            provenance: serde_json::json!({
                "manifest": "recorded_at_run_start",
                "trials": "recorded_at_trial_completion"
            }),
            notes: Vec::new(),
        })
    }
}

impl RunArchive {
    pub fn create(root: &Path, manifest: RunManifest) -> Result<Self> {
        validate_manifest(&manifest)?;
        let directory = root.join("runs").join(&manifest.run_id);
        fs::create_dir_all(&directory)
            .with_context(|| format!("failed to create {}", directory.display()))?;
        let trials_path = directory.join(&manifest.artifacts.trials_path);
        let trials = OpenOptions::new()
            .create_new(true)
            .append(true)
            .open(&trials_path)
            .with_context(|| format!("failed to create {}", trials_path.display()))?;
        write_json_atomic(&directory.join("manifest.json"), &manifest)?;
        Ok(Self {
            directory,
            manifest,
            trials,
        })
    }

    pub fn run_id(&self) -> &str {
        &self.manifest.run_id
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    pub fn append(&mut self, envelope: &TrialEnvelope) -> Result<()> {
        if envelope.schema_version != SCHEMA_VERSION
            || envelope.run_id != self.manifest.run_id
            || envelope.trial_number != self.manifest.recorded_trials + 1
            || envelope.trial_id
                != format!("{}-t{:04}", self.manifest.run_id, envelope.trial_number)
        {
            bail!("trial envelope violates archive ordering or identity invariants");
        }
        serde_json::to_writer(&mut self.trials, envelope)?;
        self.trials.write_all(b"\n")?;
        self.trials.flush()?;
        self.trials.sync_data()?;
        self.manifest.recorded_trials += 1;
        write_json_atomic(&self.directory.join("manifest.json"), &self.manifest)
    }

    pub fn finish(mut self, status: &str) -> Result<()> {
        if !matches!(status, "completed" | "aborted" | "failed") {
            bail!("invalid terminal run status: {status}");
        }
        self.trials.flush()?;
        self.trials.sync_all()?;
        let trials_path = self.directory.join(&self.manifest.artifacts.trials_path);
        self.manifest.artifacts.trials_sha256 = Some(sha256_file(&trials_path)?);
        self.manifest.status = status.to_owned();
        self.manifest.completed_at_utc = Some(Utc::now());
        validate_manifest(&self.manifest)?;
        write_json_atomic(&self.directory.join("manifest.json"), &self.manifest)
    }
}

pub fn default_archive_root() -> Result<PathBuf> {
    if let Some(root) = std::env::var_os("EXPERIMENT_ARCHIVE_ROOT") {
        return Ok(PathBuf::from(root));
    }
    let cwd = std::env::current_dir()?;
    if cwd.file_name().and_then(|name| name.to_str()) == Some("impossible-trajectory-agent") {
        Ok(cwd
            .parent()
            .context("crate has no parent directory")?
            .join("experiment-archive"))
    } else {
        Ok(cwd.join("experiment-archive"))
    }
}

pub fn trial_envelope(
    run_id: &str,
    trial_number: usize,
    random_seed: Option<u64>,
    stages: Vec<StageRecord>,
    record: Value,
) -> TrialEnvelope {
    TrialEnvelope {
        schema_version: SCHEMA_VERSION.to_owned(),
        run_id: run_id.to_owned(),
        trial_id: format!("{run_id}-t{trial_number:04}"),
        trial_number,
        recorded_at_utc: Utc::now(),
        random_seed,
        stages,
        record,
        provenance: serde_json::json!({
            "request_and_response": "recorded",
            "simulation": "recorded",
            "classifications": "derived_by_harness"
        }),
    }
}

pub fn validate_manifest(manifest: &RunManifest) -> Result<()> {
    if manifest.schema_version != SCHEMA_VERSION {
        bail!(
            "unsupported manifest schema version {}",
            manifest.schema_version
        );
    }
    if manifest.run_id.is_empty()
        || manifest.experiment.id.is_empty()
        || manifest.model.model.is_empty()
    {
        bail!("manifest identity fields cannot be empty");
    }
    if manifest.recorded_trials > manifest.planned_trials {
        bail!("recorded_trials exceeds planned_trials");
    }
    if manifest.status == "completed" && manifest.recorded_trials != manifest.planned_trials {
        bail!("completed run does not contain all planned trials");
    }
    Ok(())
}

pub fn validate_run(directory: &Path) -> Result<usize> {
    let manifest: RunManifest =
        serde_json::from_reader(File::open(directory.join("manifest.json"))?)?;
    validate_manifest(&manifest)?;
    let trials_path = directory.join(&manifest.artifacts.trials_path);
    let reader = BufReader::new(File::open(&trials_path)?);
    let mut count = 0;
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let trial: TrialEnvelope = serde_json::from_str(&line)
            .with_context(|| format!("invalid trial at line {}", index + 1))?;
        count += 1;
        if trial.run_id != manifest.run_id || trial.trial_number != count {
            bail!("trial identity/order mismatch at line {}", index + 1);
        }
    }
    if count != manifest.recorded_trials {
        bail!(
            "manifest records {} trials but file contains {count}",
            manifest.recorded_trials
        );
    }
    if let Some(expected) = &manifest.artifacts.trials_sha256 {
        let actual = sha256_file(&trials_path)?;
        if &actual != expected {
            bail!("trials checksum mismatch");
        }
    }
    Ok(count)
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let parent = path.parent().context("atomic JSON path has no parent")?;
    fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name().unwrap().to_string_lossy(),
        std::process::id()
    ));
    {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&tmp)?;
        serde_json::to_writer_pretty(&mut file, value)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    Ok(())
}

fn new_run_id(condition: &str) -> Result<String> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?;
    let safe: String = condition
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    Ok(format!(
        "{}-{}-{:09}-{}",
        safe,
        now.as_secs(),
        now.subsec_nanos(),
        std::process::id()
    ))
}

fn git_state() -> (Option<String>, Option<bool>) {
    let commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_owned());
    let dirty = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| !o.stdout.is_empty());
    (commit, dirty)
}
