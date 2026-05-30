from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


_COURSES = ["Hanshin", "Tokyo", "Chukyo", "Nakayama", "Kyoto"]
_JOCKEYS = [f"J{i:02d}" for i in range(1, 11)]
_TRAINERS = [f"T{i:02d}" for i in range(1, 11)]
_SURFACES = ["turf", "dirt"]
_CLASSES = ["Open", "3Win", "Allowance"]
_GRADES = ["G1", "G2", "G3", "Listed", ""]
_SEXES = ["male", "female", "gelding"]


def _generate_race_dataframe(
    n_races: int = 50,
    horses_per_race: int = 10,
    seed: int = 42,
    start_date: str = "2023-01-01",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range(start_date, periods=n_races, freq="7D")
    horse_pool = [f"H{i:03d}" for i in range(1, 41)]

    for race_idx in range(n_races):
        race_id = f"R{race_idx:04d}"
        race_date = dates[race_idx]
        course = rng.choice(_COURSES)
        surface = rng.choice(_SURFACES)
        distance = int(rng.choice([1200, 1400, 1600, 1800, 2000, 2200, 2400, 3000]))
        grade = rng.choice(_GRADES)
        cls = rng.choice(_CLASSES)
        turn = rng.choice(["left", "right"])
        field_size = horses_per_race

        entrants = rng.choice(horse_pool, size=horses_per_race, replace=False)
        finish_order = rng.permutation(horses_per_race)

        for pos, horse_id in enumerate(entrants):
            finish_rank = int(finish_order[pos]) + 1
            odds_val = round(float(rng.exponential(10.0)) + 1.0, 1)
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "course": course,
                    "surface": surface,
                    "distance": distance,
                    "grade": grade,
                    "class": cls,
                    "field_size": field_size,
                    "turn": turn,
                    "horse_id": horse_id,
                    "horse_name": f"Horse_{horse_id}",
                    "draw": pos + 1,
                    "horse_number": pos + 1,
                    "sex": rng.choice(_SEXES),
                    "age": int(rng.integers(3, 8)),
                    "carried_weight": round(float(rng.uniform(53.0, 58.0)), 1),
                    "jockey": rng.choice(_JOCKEYS),
                    "trainer": rng.choice(_TRAINERS),
                    "body_weight": int(rng.integers(430, 520)),
                    "body_weight_diff": int(rng.integers(-10, 11)),
                    "odds": odds_val,
                    "popularity": pos + 1,
                    "last_finish": int(rng.integers(1, 11)),
                    "last_distance": int(rng.choice([1200, 1600, 2000, 2400])),
                    "last_class": rng.choice(_CLASSES),
                    "last_margin": round(float(rng.uniform(-2.0, 5.0)), 1),
                    "last_3f": round(float(rng.uniform(33.0, 38.0)), 1),
                    "days_since_last": int(rng.integers(14, 180)),
                    "finish_rank": finish_rank,
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_history() -> pd.DataFrame:
    """50 races x 10 horses = 500 rows of synthetic data."""
    return _generate_race_dataframe(n_races=50, horses_per_race=10, seed=42)


@pytest.fixture
def synthetic_small_history() -> pd.DataFrame:
    """10 races x 8 horses = 80 rows (small data regime)."""
    return _generate_race_dataframe(n_races=10, horses_per_race=8, seed=99)


@pytest.fixture
def synthetic_entry() -> pd.DataFrame:
    """Single race entry with 10 horses (no finish_rank)."""
    rng = np.random.default_rng(123)
    horse_pool = [f"H{i:03d}" for i in range(1, 11)]
    rows = []
    for pos, horse_id in enumerate(horse_pool):
        rows.append(
            {
                "race_id": "R9999",
                "race_date": pd.Timestamp("2026-03-29"),
                "course": "Chukyo",
                "surface": "turf",
                "distance": 1200,
                "grade": "G1",
                "class": "Open",
                "field_size": 10,
                "turn": "left",
                "horse_id": horse_id,
                "horse_name": f"Horse_{horse_id}",
                "draw": pos + 1,
                "horse_number": pos + 1,
                "sex": rng.choice(_SEXES),
                "age": int(rng.integers(3, 8)),
                "carried_weight": round(float(rng.uniform(53.0, 58.0)), 1),
                "jockey": _JOCKEYS[pos],
                "trainer": _TRAINERS[pos],
                "body_weight": int(rng.integers(430, 520)),
                "body_weight_diff": int(rng.integers(-10, 11)),
                "odds": round(float(rng.exponential(10.0)) + 1.0, 1),
                "popularity": pos + 1,
                "last_finish": int(rng.integers(1, 11)),
                "last_distance": int(rng.choice([1200, 1600, 2000])),
                "last_class": rng.choice(_CLASSES),
                "last_margin": round(float(rng.uniform(-2.0, 5.0)), 1),
                "last_3f": round(float(rng.uniform(33.0, 38.0)), 1),
                "days_since_last": int(rng.integers(14, 180)),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def base_config() -> dict:
    """Minimal config dict matching config.yaml structure."""
    return {
        "seed": 42,
        "target_race_profile": {
            "name": "Test Race",
            "race_date": "2026-03-29",
            "surface": "turf",
            "distance": 1200,
            "course": "Chukyo",
            "turn": "left",
            "grade": "G1",
            "class": "Open",
            "season_label": "spring",
            "season_months": [3, 4, 5],
        },
        "similarity_weights": {
            "exact_surface_distance": 5.0,
            "near_surface_distance": 3.0,
            "near_distance_tolerance": 200,
            "wide_surface_distance": 1.5,
            "wide_distance_tolerance": 600,
            "age_4plus": 1.0,
            "open_graded": 1.0,
            "same_course": 0.8,
            "same_turn": 0.6,
            "same_season": 0.4,
        },
        "model": {
            "ranker": {
                "loss_function": "QueryRMSE",
                "eval_metric": "NDCG:top=3",
                "iterations": 50,
                "learning_rate": 0.05,
                "depth": 4,
                "l2_leaf_reg": 8.0,
                "random_strength": 0.5,
                "min_data_in_leaf": 5,
                "bootstrap_type": "Bernoulli",
                "subsample": 0.8,
                "thread_count": -1,
            },
            "classifier": {
                "loss_function": "Logloss",
                "eval_metric": "Logloss",
                "iterations": 50,
                "learning_rate": 0.05,
                "depth": 4,
                "l2_leaf_reg": 8.0,
                "random_strength": 0.5,
                "min_data_in_leaf": 5,
                "bootstrap_type": "Bernoulli",
                "subsample": 0.8,
                "auto_class_weights": "Balanced",
                "thread_count": -1,
                "bases": ["logistic", "random_forest", "extra_trees"],
                "logistic": {"C": 0.7, "max_iter": 1000, "solver": "lbfgs", "class_weight": "balanced"},
                "random_forest": {"n_estimators": 50, "max_depth": 6, "min_samples_leaf": 3, "max_features": "sqrt", "class_weight": "balanced", "n_jobs": -1},
                "extra_trees": {"n_estimators": 50, "max_depth": 6, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": "balanced", "n_jobs": -1},
            },
            "regressor": {
                "bases": ["ridge", "random_forest", "extra_trees"],
                "ridge": {"alpha": 1.0},
                "random_forest": {"n_estimators": 50, "max_depth": 6, "min_samples_leaf": 3, "max_features": "sqrt", "n_jobs": -1},
                "extra_trees": {"n_estimators": 50, "max_depth": 6, "min_samples_leaf": 2, "max_features": "sqrt", "n_jobs": -1},
            },
            "blending": {"min_weight": 0.08, "residual_sigma_scale": 0.35},
        },
        "cv": {
            "n_splits": 3,
            "min_train_races": 5,
            "classifier_inner_splits": 2,
            "classifier_inner_min_train_races": 3,
            "regression_inner_splits": 2,
            "regression_inner_min_train_races": 3,
            "stacking_inner_splits": 2,
            "stacking_inner_min_train_races": 3,
        },
        "simulation": {
            "stages": [5000, 10000],
            "convergence_threshold": 0.005,
            "block_size": 5000,
            "calibration_trials": 2000,
            "calibration_block_size": 2000,
            "temperature_grid": [0.5, 1.0, 2.0],
        },
    }
