# keiba-axis-model

[![日本語](https://img.shields.io/badge/lang-日本語-green.svg)](README.md)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/package%20manager-uv-4B5563)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A horse racing prediction workflow that turns race-level input bundles into top-3 probabilities, an axis horse, partner candidates, and post-race review reports.

## Overview

`keiba-axis-model` is a Python project for managing one race card, local history, race configuration, and result per `races/<slug>/` directory, then running weekly prediction and settlement from the same CLI.

Main features:

- `predict`: combine cumulative training history with race-local history and estimate top-3 probability, win probability, and rank distribution
- `recommended_axis_horse.json`: output one axis horse candidate and the scores used for selection
- `recommended_partners.json` / `ticket_candidates.csv`: output partner candidates against the selected axis horse
- `settle`: ingest `result.csv` and generate comparison reports between predictions and actual results
- `scripts/backtest.py`: run walk-forward backtests for the market baseline and a single ranker

Bundled sample races:

- `races/hanshin-daishoten-2026-03-22/`
- `races/takamatsunomiya-kinen-2026-03-29/`
- `races/osaka-hai-2026-04-05/`
- `races/oka-sho-2026-04-12/`
- `races/satsuki-sho-2026-04-19/`

## Current Status

This README reflects the following checks run on 2026-04-20.

```bash
uv run python main.py --help
uv run python main.py list-races
uv run pytest
```

Verified results:

- `main.py` exposes `predict`, `settle`, `init-race`, and `list-races`
- `list-races` detects 5 bundled sample races
- `uv run pytest`: 203 passed, 1 warning

## Setup

### Prerequisites

- Python 3.10 or newer
- `uv`

### Install

```bash
uv sync
```

### Common Development Commands

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

Example output:

```text
hanshin-daishoten-2026-03-22: Hanshin Daishoten (2026-03-22)
oka-sho-2026-04-12: Oka Sho (2026-04-12)
osaka-hai-2026-04-05: Osaka Hai (2026-04-05)
satsuki-sho-2026-04-19: Satsuki Sho (2026-04-19)
takamatsunomiya-kinen-2026-03-29: Takamatsunomiya Kinen (2026-03-29)
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

### 3. Run a prediction

```bash
uv run python main.py predict --race satsuki-sho-2026-04-19
```

The shorthand form is also supported.

```bash
uv run python main.py satsuki-sho-2026-04-19
```

`predict` builds its training set from:

- `data/training/race_results_master.csv`
- `races/<slug>/history.csv`

The target race's own `race_id` and records on or after the target race date are excluded from the prediction training data.

### 4. Settle a finished race

```bash
uv run python main.py settle --race satsuki-sho-2026-04-19
```

`settle` reads `races/<slug>/result.csv`.

- Always writes `settled_entry.csv` and `post_race_report.json`
- Writes `post_race_analysis.csv` only when prediction artifacts exist
- Updates `data/training/race_results_master.csv` when `result.csv` contains the full-field result

### 5. Run a backtest

```bash
uv run python scripts/backtest.py --data data/training/race_results_master.csv
```

You can also specify the number of folds and the minimum number of training races.

```bash
uv run python scripts/backtest.py \
  --data data/training/race_results_master.csv \
  --n-splits 5 \
  --min-train-races 20
```

## CLI Reference

| Command | Purpose |
| --- | --- |
| `predict` | Generate prediction artifacts for a race |
| `settle` | Ingest `result.csv`, write post-race reports, and update training history |
| `init-race` | Create a new `races/<slug>/` scaffold |
| `list-races` | List available race bundles |

| Command | Option | Purpose |
| --- | --- | --- |
| `predict` / `settle` | `--race` | Select a slug under `races/` |
| `predict` / `settle` | `--race-dir` | Point directly to a race directory |
| `predict` / `settle` | `--races-root` | Override the parent race directory |
| `predict` / `settle` | `--training-history` | Override the cumulative training CSV |
| `predict` / `settle` | `--output-root` | Override the output root |
| `predict` | `--config` | Override the base config file |
| `init-race` | `--race-name` | Set the display name |
| `init-race` | `--race-date` | Set the race date as `YYYY-MM-DD` |
| `init-race` | `--force` | Overwrite scaffold files |

## Race Bundle Layout

Each `races/<slug>/` directory can contain the following files.

| File | Required | Purpose |
| --- | --- | --- |
| `race.yaml` | Required | Race conditions and config overrides. `target_race_profile.surface`, `distance`, and `course` are required |
| `entry.csv` | Required | Entrants for the target race |
| `history.csv` | Required | Additional history rows for this race |
| `result.csv` | Required after the race | Finishing order and result fields |
| `race_result_meta.json` | Optional | Winning time, lap splits, going, and other supplemental metadata |
| `README.md` | Optional | Race-local notes |
| `refresh_race_data.py` | Optional | Race-local data refresh script |

The `result.csv` template columns are:

```text
horse_id,horse_name,finish_rank,result_time,result_margin,result_last3f,result_final_odds,result_final_popularity,result_body_weight,result_body_weight_diff
```

## Configuration

The base configuration lives in `src/keiba_predictor/config.yaml`.

Main config areas:

- `similarity_weights`: similarity weighting for historical training rows
- `model.ranker`: CatBoost ranker settings
- `model.classifier`: CatBoost classifier and scikit-learn classifier settings
- `model.regressor`: scikit-learn regressor settings
- `cv`: time-series CV fold counts and minimum training race counts
- `simulation`: rank distribution simulation and calibration settings
- `partner`: partner candidate count, popularity constraints, and optimization weights

Each race's `race.yaml` can override `target_race_profile` and `use_odds`.

```yaml
race_name: Satsuki Sho
race_date: 2026-04-19
use_odds: true
target_race_profile:
  surface: turf
  distance: 2000
  course: Nakayama
```

## Output Files

### Files produced by `predict`

| File | Description |
| --- | --- |
| `predictions.csv` | Full prediction table with top-3 probability, win probability, mean rank, axis score, and partner score |
| `recommended_axis_horse.json` | Detailed payload for the selected axis horse |
| `recommended_partners.json` | Partner candidate set for the axis horse |
| `ticket_candidates.csv` | Axis-partner pairs for ticket planning |
| `evaluation_summary.json` | CV metrics plus data diagnostics |
| `cv_fold_metrics.csv` | Fold-level evaluation results |
| `calibration_curve.csv` | Aggregated top-3 calibration data |
| `feature_importance.csv` | Ranked feature importance table |
| `simulation_diagnostics.csv` | Simulation convergence diagnostics |
| `schema_report.json` | Column normalization and schema validation report |
| `effective_config.json` | Final merged config from base config plus `race.yaml` |
| `run_context.json` | Input and output paths used for the run |
| `top3_probability_bar.png` | Chart for top candidates' top-3 and win probabilities |
| `calibration_plot.png` | Calibration visualization |
| `feature_importance.png` | Feature importance visualization |

### Files produced by `settle`

| File | Description |
| --- | --- |
| `settled_entry.csv` | Race card augmented with result fields |
| `post_race_report.json` | Comparison report for top predictions, axis horse, partner candidates, and actual results |
| `post_race_analysis.csv` | Per-horse ranking error analysis. Written only when `predictions.csv` exists |

## Model and Preprocessing

- `data_loader.py`: CSV encoding fallback, column alias normalization, dtype coercion, and required-column validation
- `features.py`: base features, aggregate horse/jockey/trainer/distance/class features, and relative rank features
- `hybrid_model.py`: CatBoost ranker/classifier plus scikit-learn regression and classification ensembles
- `simulation.py`: Plackett-Luce and Gaussian-style rank distribution estimation plus conditional top-3 probabilities
- `prediction.py`: axis horse scoring, partner candidate scoring, and artifact writing
- `workflow.py`: merges `config.yaml` with `race.yaml`, then orchestrates prediction and settlement

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
├── races/
├── scripts/
│   └── backtest.py
├── src/
│   └── keiba_predictor/
│       ├── config.yaml
│       ├── backtest_baseline.py
│       ├── data_loader.py
│       ├── features.py
│       ├── hybrid_model.py
│       ├── model.py
│       ├── prediction.py
│       ├── simulation.py
│       └── workflow.py
└── tests/
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
