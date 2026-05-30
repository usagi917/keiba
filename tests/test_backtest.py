from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.backtest import (
    WalkForwardResult,
    compute_market_baseline_predictions,
    compute_single_ranker_predictions,
    evaluate_predictions,
    walk_forward_backtest,
)


def _make_history(n_races: int = 20, horses_per_race: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2023-01-01", periods=n_races, freq="7D")
    for i in range(n_races):
        race_id = f"R{i:04d}"
        field_size = horses_per_race
        finish_order = rng.permutation(field_size)
        for j in range(field_size):
            odds = round(float(rng.exponential(8.0)) + 1.0, 1)
            rows.append({
                "race_id": race_id,
                "race_date": dates[i],
                "horse_id": f"H{j:03d}",
                "finish_rank": int(finish_order[j]) + 1,
                "field_size": field_size,
                "odds": odds,
                "popularity": j + 1,
            })
    return pd.DataFrame(rows)


class TestComputeMarketBaselinePredictions:
    def test_returns_predictions_for_all_rows(self):
        df = _make_history(n_races=5)
        preds = compute_market_baseline_predictions(df)
        assert len(preds) == len(df)

    def test_predictions_have_top3_prob_column(self):
        df = _make_history(n_races=5)
        preds = compute_market_baseline_predictions(df)
        assert "top3_prob" in preds.columns

    def test_probabilities_sum_to_roughly_1_per_race(self):
        df = _make_history(n_races=5)
        preds = compute_market_baseline_predictions(df)
        for race_id, group in preds.groupby("race_id"):
            assert group["top3_prob"].sum() == pytest.approx(3.0, abs=0.5)

    def test_lower_odds_gets_higher_probability(self):
        """低オッズ馬は高い確率を得ること"""
        df = pd.DataFrame([
            {"race_id": "R1", "race_date": pd.Timestamp("2024-01-01"), "horse_id": "H1",
             "finish_rank": 1, "field_size": 3, "odds": 1.5, "popularity": 1},
            {"race_id": "R1", "race_date": pd.Timestamp("2024-01-01"), "horse_id": "H2",
             "finish_rank": 2, "field_size": 3, "odds": 5.0, "popularity": 2},
            {"race_id": "R1", "race_date": pd.Timestamp("2024-01-01"), "horse_id": "H3",
             "finish_rank": 3, "field_size": 3, "odds": 20.0, "popularity": 3},
        ])
        preds = compute_market_baseline_predictions(df)
        probs = preds.set_index("horse_id")["top3_prob"]
        assert probs["H1"] > probs["H2"] > probs["H3"]


class TestEvaluatePredictions:
    def test_returns_metrics_dict(self):
        df = _make_history(n_races=5)
        preds = compute_market_baseline_predictions(df)
        merged = df.merge(preds[["race_id", "horse_id", "top3_prob"]], on=["race_id", "horse_id"])
        metrics = evaluate_predictions(merged)
        assert "top3_brier_score" in metrics
        assert "top3_precision" in metrics
        assert "n_races" in metrics

    def test_brier_score_in_valid_range(self):
        df = _make_history(n_races=10)
        preds = compute_market_baseline_predictions(df)
        merged = df.merge(preds[["race_id", "horse_id", "top3_prob"]], on=["race_id", "horse_id"])
        merged["is_top3"] = (merged["finish_rank"] <= 3).astype(int)
        metrics = evaluate_predictions(merged)
        assert 0.0 <= metrics["top3_brier_score"] <= 1.0

    def test_precision_in_valid_range(self):
        df = _make_history(n_races=10)
        preds = compute_market_baseline_predictions(df)
        merged = df.merge(preds[["race_id", "horse_id", "top3_prob"]], on=["race_id", "horse_id"])
        metrics = evaluate_predictions(merged)
        assert 0.0 <= metrics["top3_precision"] <= 1.0


class TestWalkForwardBacktest:
    def test_returns_walk_forward_result(self):
        df = _make_history(n_races=20)
        result = walk_forward_backtest(df, n_splits=3, min_train_races=5)
        assert isinstance(result, WalkForwardResult)

    def test_result_has_market_baseline(self):
        df = _make_history(n_races=20)
        result = walk_forward_backtest(df, n_splits=3, min_train_races=5)
        assert "market_baseline" in result.metrics_by_model

    def test_result_has_correct_n_folds(self):
        df = _make_history(n_races=20)
        result = walk_forward_backtest(df, n_splits=3, min_train_races=5)
        assert result.n_folds == 3

    def test_metrics_structure_per_model(self):
        df = _make_history(n_races=20)
        result = walk_forward_backtest(df, n_splits=3, min_train_races=5)
        for model_name, metrics in result.metrics_by_model.items():
            assert "top3_brier_score" in metrics
            assert "n_races" in metrics


class TestComputeSingleRankerPredictions:
    def test_requires_finish_rank_in_history(self):
        df = _make_history(n_races=5)
        entry = df[df["race_id"] == "R0004"].copy()
        history = df[df["race_id"] != "R0004"].copy()
        preds = compute_single_ranker_predictions(history=history, entry=entry)
        assert "top3_prob" in preds.columns
        assert len(preds) == len(entry)
