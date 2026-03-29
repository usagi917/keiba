from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba_predictor.features import (
    add_aggregate_features,
    add_basic_features,
    add_targets,
    infer_feature_columns,
)
from keiba_predictor.hybrid_model import (
    _resolve_group_holdout_split,
    fit_probability_blender,
    fit_score_temperature,
    time_series_race_splits,
)


class TestTimeSeriesRaceSplits:
    def test_creates_correct_number_of_splits(self, synthetic_history):
        splits = time_series_race_splits(synthetic_history, n_splits=3, min_train_races=5)
        assert len(splits) == 3

    def test_train_before_validation(self, synthetic_history):
        splits = time_series_race_splits(synthetic_history, n_splits=3, min_train_races=5)
        for train_idx, val_idx in splits:
            train_dates = synthetic_history.loc[train_idx, "race_date"]
            val_dates = synthetic_history.loc[val_idx, "race_date"]
            assert train_dates.max() <= val_dates.min()

    def test_no_overlap_between_train_and_val(self, synthetic_history):
        splits = time_series_race_splits(synthetic_history, n_splits=3, min_train_races=5)
        for train_idx, val_idx in splits:
            assert len(set(train_idx) & set(val_idx)) == 0

    def test_raises_with_too_few_races(self):
        df = pd.DataFrame({
            "race_id": ["R1", "R1"],
            "race_date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
        })
        with pytest.raises(ValueError, match="レース数"):
            time_series_race_splits(df, n_splits=3, min_train_races=1)

    def test_small_data_regime(self, synthetic_small_history):
        splits = time_series_race_splits(
            synthetic_small_history, n_splits=2, min_train_races=3,
        )
        assert len(splits) >= 1
        for train_idx, val_idx in splits:
            assert len(train_idx) > 0
            assert len(val_idx) > 0


class TestResolveGroupHoldoutSplit:
    def test_preserves_full_race_groups(self):
        group_ids = ["R1"] * 10 + ["R2"] * 10 + ["R3"] * 10 + ["R4"] * 10 + ["R5"] * 10
        split_row = _resolve_group_holdout_split(group_ids, train_fraction=0.85)
        assert split_row == 40

    def test_returns_none_when_single_race(self):
        assert _resolve_group_holdout_split(["R1"] * 12, train_fraction=0.85) is None


class TestFitScoreTemperature:
    def test_returns_float(self, base_config):
        race_ids = ["R1"] * 10
        scores = np.linspace(0.2, 0.8, 10)
        labels = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        temp = fit_score_temperature(race_ids, scores, labels, base_config, seed=42)
        assert isinstance(temp, float)
        assert temp > 0

    def test_returns_1_for_uniform_labels(self, base_config):
        race_ids = ["R1"] * 5
        scores = np.linspace(0.2, 0.8, 5)
        labels = [1, 1, 1, 1, 1]
        temp = fit_score_temperature(race_ids, scores, labels, base_config, seed=42)
        assert temp == 1.0


class TestFitProbabilityBlender:
    def test_blends_multiple_sources(self):
        n = 100
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "source_a": rng.uniform(0.1, 0.9, n),
            "source_b": rng.uniform(0.1, 0.9, n),
        })
        labels = (rng.random(n) > 0.7).astype(int)
        blender = fit_probability_blender(
            df, feature_cols=["source_a", "source_b"],
            labels=labels, random_state=42,
        )
        proba = blender.predict_proba(df)
        assert len(proba) == n
        assert np.all(proba > 0)
        assert np.all(proba < 1)

    def test_single_source_passthrough(self):
        n = 50
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"source_a": rng.uniform(0.1, 0.9, n)})
        labels = (rng.random(n) > 0.7).astype(int)
        blender = fit_probability_blender(
            df, feature_cols=["source_a"],
            labels=labels, random_state=42,
        )
        proba = blender.predict_proba(df)
        assert len(proba) == n
