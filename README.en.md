# keiba-axis-model

[![日本語](https://img.shields.io/badge/lang-日本語-green.svg)](README.md)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/package%20manager-uv-4B5563)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A horse racing prediction workflow that outputs top-3 finish probabilities and a recommended axis horse using a time-series ensemble model plus Plackett-Luce simulation.

## Overview

This repository manages race-specific inputs under `races/<slug>/`, generates prediction artifacts with `predict`, and settles finished races with `settle` to feed results back into the training history.

- `predict`: writes `predictions.csv`, `recommended_axis_horse.json`, CV metrics, feature importance, and simulation diagnostics
- `settle`: writes `settled_entry.csv` and `post_race_report.json`, and updates `data/training/race_results_master.csv` when the full result is available
- `init-race`: creates a scaffold for a new race directory

`races/hanshin-daishoten-2026-03-22/` is included as a sample race bundle. Running the workflow generates the corresponding artifacts under `outputs/Hanshin Daishoten/`.

## Setup

### Prerequisites

- Python 3.10 or newer
- `uv`

### Install

```bash
uv sync
```

### Smoke Test

```bash
uv run python main.py --help
uv run python main.py list-races
```

## User Flow

```mermaid
flowchart TD
    A[Start] --> B{Does the target race already exist?}
    B -->|Yes| C[uv run python main.py list-races]
    B -->|No| D[uv run python main.py init-race --race ...]
    C --> E[Choose the target races/{slug}/ bundle]
    D --> E
    E --> F[Prepare entry.csv, history.csv, and race.yaml]
    F --> G[uv run python main.py predict --race {slug}]
    G --> H[Review artifacts under outputs/{race_name}/]
    H --> I{Has the official result been published?}
    I -->|Not yet| H
    I -->|Yes| J[Update result.csv]
    J --> K[uv run python main.py settle --race {slug}]
    K --> L[Append to data/training/race_results_master.csv]
    L --> M[Repeat the same loop for the next race]
```

## Usage

### 1. Predict an Existing Race

```bash
uv run python main.py predict --race hanshin-daishoten-2026-03-22
```

Main outputs:

- `outputs/Hanshin Daishoten/predictions.csv`
- `outputs/Hanshin Daishoten/recommended_axis_horse.json`
- `outputs/Hanshin Daishoten/evaluation_summary.json`
- `outputs/Hanshin Daishoten/feature_importance.csv`
- `outputs/Hanshin Daishoten/simulation_diagnostics.csv`

### 2. Create a New Race Bundle

```bash
uv run python main.py init-race \
  --race spring-tenno-sho-2026-05-03 \
  --race-name "Tenno Sho Spring" \
  --race-date 2026-05-03
```

Files created:

- `races/spring-tenno-sho-2026-05-03/race.yaml`
- `races/spring-tenno-sho-2026-05-03/entry.csv`
- `races/spring-tenno-sho-2026-05-03/history.csv`
- `races/spring-tenno-sho-2026-05-03/result.csv`
- `races/spring-tenno-sho-2026-05-03/README.md`

Populate `entry.csv` and `history.csv` manually, or add a race-local data collection script like the sample `refresh_race_data.py`.

### 3. Settle a Finished Race

```bash
uv run python main.py settle --race hanshin-daishoten-2026-03-22
```

This updates:

- `outputs/Hanshin Daishoten/settled_entry.csv`
- `outputs/Hanshin Daishoten/post_race_report.json`
- `data/training/race_results_master.csv`

Notes:

- The cumulative training history is updated only when `result.csv` contains the full result
- A partial Top-K result still generates `post_race_report.json`

### 4. Common Options

| Command | Option | Purpose |
| --- | --- | --- |
| `predict` / `settle` | `--race` | Select a race slug under `races/` |
| `predict` / `settle` | `--race-dir` | Point directly to a race directory |
| `predict` / `settle` | `--training-history` | Override the cumulative training CSV |
| `predict` / `settle` | `--output-root` | Override the output root directory |
| `predict` | `--config` | Replace the base config file |
| `init-race` | `--force` | Overwrite scaffold files |

## Use Cases

### 1. Standardize Weekly Graded Stakes Forecasting

Each race lives in its own `races/<slug>/` bundle, so you can run the same workflow every week while keeping race-specific conditions in `race.yaml`.

### 2. Review Why the Model Chose an Axis Horse

The workflow keeps `predictions.csv`, `recommended_axis_horse.json`, `feature_importance.csv`, and `calibration_curve.csv` together, which makes it easier to review probabilities, model drivers, and calibration quality in one place.

### 3. Turn Post-Race Review into Future Training Data

After `settle`, you can inspect `post_race_report.json` to compare predicted leaders with the actual order of finish. When the full result is present, the run also appends to `data/training/race_results_master.csv` for future races.

## Architecture

```mermaid
flowchart TD
    A[data/training/race_results_master.csv<br/>cumulative training history]
    B[races/{slug}/entry.csv<br/>race card]
    C[races/{slug}/history.csv<br/>recent form history]
    D[races/{slug}/race.yaml<br/>race profile]
    E[races/{slug}/result.csv<br/>race result]

    F[main.py<br/>CLI]
    G[workflow.py<br/>race-level orchestration]
    H[data_loader.py<br/>validation and normalization]
    I[features.py<br/>feature engineering]
    J[hybrid_model.py<br/>ranker + classifier + regressor]
    K[simulation.py<br/>Plackett-Luce simulation]
    L[prediction.py<br/>CV, blending, outputs]

    M[outputs/{race_name}/<br/>prediction artifacts]
    N[data/training/race_results_master.csv<br/>updated history]

    A --> G
    B --> F
    C --> F
    D --> F
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    G --> N
```

### Model Breakdown

- `workflow.py` merges `src/keiba_predictor/config.yaml` with each race-local `race.yaml` and controls prediction and settlement
- `prediction.py` runs time-series CV, feature generation, hybrid model training, and artifact generation
- `hybrid_model.py` combines ranking, classification, and regression models, while `simulation.py` estimates finish distributions with Monte Carlo simulation

## Directory Layout

```text
.
├── main.py
├── pyproject.toml
├── data/
│   └── training/
│       └── race_results_master.csv
├── outputs/
│   └── Hanshin Daishoten/
├── races/
│   └── hanshin-daishoten-2026-03-22/
│       ├── README.md
│       ├── entry.csv
│       ├── history.csv
│       ├── race.yaml
│       ├── refresh_race_data.py
│       └── result.csv
└── src/
    └── keiba_predictor/
        ├── config.yaml
        ├── data_loader.py
        ├── features.py
        ├── hybrid_model.py
        ├── model.py
        ├── prediction.py
        ├── simulation.py
        └── workflow.py
```

## Key Output Files

| File | Description |
| --- | --- |
| `predictions.csv` | Full prediction table for all horses |
| `recommended_axis_horse.json` | Detailed payload for the selected axis horse |
| `evaluation_summary.json` | Aggregated CV metrics |
| `cv_fold_metrics.csv` | Fold-level evaluation output |
| `feature_importance.csv` | Feature importance ranking |
| `calibration_curve.csv` | Probability calibration summary |
| `simulation_diagnostics.csv` | Simulation convergence diagnostics |
| `settled_entry.csv` | Race card with finish ranks attached |
| `post_race_report.json` | Comparison between predicted leaders and actual results |

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
