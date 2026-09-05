# What You Do Is Who You Are

**Active objective inference from agent behavior through counterfactual test selection**

This repository contains the experiment code, immutable trial archive, analysis outputs, and visual explorer for a study of whether deliberately chosen test situations reveal an agent's hidden behavioral objective more efficiently than random observation.

## Repository contents

- `impossible-trajectory-agent/` – Rust experiment runner and analysis scripts.
- `experiment-archive/` – versioned manifests, trial records, annotations, schemas, and sealed objective files.
- `analysis/` – blinded and unblinded machine-readable analysis outputs.
- `forensics-explorer/` – Tauri/React interface for experiment- and trial-level inspection.

## Run the visual explorer

```text
cd forensics-explorer
npm install
npm run tauri dev
```

The explorer reads the archived observations; it does not replace them with mock data.

## Build and test

```text
cd impossible-trajectory-agent
cargo test

cd ../forensics-explorer
npm install
npm run build
cargo test --manifest-path src-tauri/Cargo.toml
```

API credentials are read from environment variables and are not included in this repository.
