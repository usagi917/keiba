# keiba-axis-model

[![日本語](https://img.shields.io/badge/lang-日本語-green.svg)](README.md)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/package%20manager-uv-4B5563)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A race-bundle-based horse racing workflow that produces top-3 probabilities, an axis horse candidate, and post-race review artifacts.

## Overview

This repository manages one input bundle per race under `races/<slug>/` and runs the weekly workflow through `main.py`.

- `list-races`: list available race bundles
- `init-race`: scaffold a new race directory
- `predict`: merge cumulative history with race-local history and generate prediction artifacts
- `settle`: ingest `result.csv`, write review artifacts, and update cumulative training history

Bundled sample race directories:

- `races/hanshin-daishoten-2026-03-22/`
- `races/takamatsunomiya-kinen-2026-03-29/`

`outputs/` also contains previously generated artifact snapshots.

## Current Status

This README reflects what was verified on 2026-03-29.

- `uv run pytest`: 68 passed
- `uv run python main.py --help`
- `uv run python main.py list-races`
- `uv run python main.py init-race ...`
- `uv run python main.py settle ...`

The `predict` command is implemented, but the bundled sample race data currently fails with the CatBoost ranker error `Groupwise loss/metrics require nontrivial groups`. Treat the existing files under `outputs/` as the reference artifact schema for now.

## Setup

### Prerequisites

- Python 3.10 or newer
- `uv`

### Install

```bash
uv sync
```

### Development Commands

```bash
uv run python main.py --help
uv run python main.py list-races
uv run pytest
```

## Quick Start

### 1. List available races

```bash
uv run python main.py list-races
```

### 2. Create a new race bundle

```bash
uv run python main.py init-race \
  --race spring-tenno-sho-2026-05-03 \
  --race-name "Tenno Sho Spring" \
  --race-date 2026-05-03
```

Generated files:

- `races/spring-tenno-sho-2026-05-03/race.yaml`
- `races/spring-tenno-sho-2026-05-03/entry.csv`
- `races/spring-tenno-sho-2026-05-03/history.csv`
- `races/spring-tenno-sho-2026-05-03/result.csv`
- `races/spring-tenno-sho-2026-05-03/README.md`

### 3. Prediction command shape

```bash
uv run python main.py predict --race hanshin-daishoten-2026-03-22
```

You can also use the shorthand form.

```bash
uv run python main.py hanshin-daishoten-2026-03-22
```

Notes:

- `predict` builds its training set by combining `data/training/race_results_master.csv` with `races/<slug>/history.csv`
- It automatically excludes the target `race_id` and records on or after the target race date
- With the currently bundled sample data, this command fails, so use the existing files under `outputs/` to inspect the expected artifact set

### 4. Settle a finished race

```bash
uv run python main.py settle --race takamatsunomiya-kinen-2026-03-29
```

`settle` can run even when `predictions.csv` is missing.

- It always writes `settled_entry.csv` and `post_race_report.json`
- It writes `post_race_analysis.csv` only when prediction artifacts exist
- It updates `data/training/race_results_master.csv` only when `result.csv` contains the full result

### 5. Common options

| Command | Option | Purpose |
| --- | --- | --- |
| `predict` / `settle` | `--race` | Select a slug under `races/` |
| `predict` / `settle` | `--race-dir` | Point directly to a race directory |
| `predict` / `settle` | `--training-history` | Override the cumulative training CSV |
| `predict` / `settle` | `--output-root` | Override the output root |
| `predict` | `--config` | Override the base config file |
| `init-race` | `--races-root` | Override the race root directory |
| `init-race` | `--force` | Overwrite scaffold files |

## Race Bundle Layout

Each `races/<slug>/` directory can contain the following files.

| File | Required | Purpose |
| --- | --- | --- |
| `race.yaml` | Required | Race conditions and config overrides. `target_race_profile.surface`, `distance`, and `course` are required. `use_odds` toggles odds/popularity-derived features |
| `entry.csv` | Required | Current race entrants |
| `history.csv` | Required | Extra history rows to include for this race |
| `result.csv` | Required after the race | Official finishing result |
| `race_result_meta.json` | Optional | Winning time, lap splits, going, and other supplemental race metadata |
| `README.md` | Optional | Race-local notes |
| `refresh_race_data.py` | Optional | Race-local data refresh script |

The `result.csv` template columns are:

```text
horse_id,horse_name,finish_rank,result_time,result_margin,result_last3f,result_final_odds,result_final_popularity,result_body_weight,result_body_weight_diff
```

## Output Files

### Files produced by `predict`

| File | Description |
| --- | --- |
| `predictions.csv` | Full prediction table for all horses |
| `recommended_axis_horse.json` | Detailed payload for the selected axis horse |
| `evaluation_summary.json` | CV metrics plus data diagnostics |
| `cv_fold_metrics.csv` | Fold-level evaluation results |
| `calibration_curve.csv` | Aggregated top-3 calibration data |
| `feature_importance.csv` | Ranked feature importance table |
| `simulation_diagnostics.csv` | Simulation convergence diagnostics |
| `schema_report.json` | Column normalization and schema-validation report |
| `effective_config.json` | Final merged config from base config plus `race.yaml` |
| `run_context.json` | Input and output paths used for the run |
| `top3_probability_bar.png` | Dashboard-style plot for top candidates |
| `calibration_plot.png` | Calibration visualization |
| `feature_importance.png` | Feature importance visualization |

### Files produced by `settle`

| File | Description |
| --- | --- |
| `settled_entry.csv` | Race card augmented with finishing result columns |
| `post_race_report.json` | Summary comparison between predictions and actual result |
| `post_race_analysis.csv` | Per-horse ranking error analysis. Written only when `predictions.csv` exists |

## Model and Preprocessing

- `data_loader.py` handles CSV encoding fallback, column alias normalization, dtype coercion, and schema validation
- `features.py` builds derived features plus aggregate horse, jockey, trainer, distance-band, and grade features
- `hybrid_model.py` combines a CatBoost ranker, a CatBoost classifier, and scikit-learn regression/classification ensembles
- `simulation.py` estimates rank distributions with Plackett-Luce and Gaussian-style simulations
- `workflow.py` merges `config.yaml` and `race.yaml`, then orchestrates prediction and settlement

## Directory Layout

```text
.
├── main.py
├── pyproject.toml
├── README.md
├── README.en.md
├── data/
│   └── training/
│       └── race_results_master.csv
├── outputs/
│   ├── Hanshin Daishoten/
│   └── Takamatsunomiya Kinen/
├── races/
│   ├── hanshin-daishoten-2026-03-22/
│   └── takamatsunomiya-kinen-2026-03-29/
├── src/
│   └── keiba_predictor/
│       ├── config.yaml
│       ├── data_loader.py
│       ├── features.py
│       ├── hybrid_model.py
│       ├── model.py
│       ├── prediction.py
│       ├── simulation.py
│       └── workflow.py
└── tests/
    ├── test_data_loader.py
    ├── test_features.py
    ├── test_hybrid_model.py
    ├── test_model.py
    └── test_workflow.py
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
