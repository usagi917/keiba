from __future__ import annotations

import json
from pathlib import Path

import pytest

from keiba_predictor.backtest_baseline import (
    compare_with_baseline,
    compute_axis_baseline,
    compute_baselines_by_condition,
    compute_partner_baseline,
    save_baselines,
)


class TestComputeAxisBaseline:
    def test_returns_required_keys(self, synthetic_history, base_config):
        result = compute_axis_baseline(synthetic_history, base_config)
        assert "axis_top3_hit_rate" in result
        assert "n_races" in result

    def test_hit_rate_in_valid_range(self, synthetic_history, base_config):
        result = compute_axis_baseline(synthetic_history, base_config)
        assert 0.0 <= result["axis_top3_hit_rate"] <= 1.0

    def test_n_races_is_positive(self, synthetic_history, base_config):
        result = compute_axis_baseline(synthetic_history, base_config)
        assert result["n_races"] > 0


class TestComputePartnerBaseline:
    def test_returns_required_keys(self, synthetic_history, base_config):
        result = compute_partner_baseline(synthetic_history, base_config)
        assert "top5_cover_rate" in result
        assert "top5_cover_count_mean" in result

    def test_cover_rate_in_valid_range(self, synthetic_history, base_config):
        result = compute_partner_baseline(synthetic_history, base_config)
        assert 0.0 <= result["top5_cover_rate"] <= 1.0

    def test_cover_count_mean_non_negative(self, synthetic_history, base_config):
        result = compute_partner_baseline(synthetic_history, base_config)
        assert result["top5_cover_count_mean"] >= 0.0


class TestSaveBaselines:
    def test_saves_json_file(self, tmp_path):
        baselines = {"axis_top3_hit_rate": 0.35, "n_races": 40}
        out_path = tmp_path / "baselines.json"
        save_baselines(baselines, out_path)
        assert out_path.exists()

    def test_saved_content_is_correct(self, tmp_path):
        baselines = {"axis_top3_hit_rate": 0.35, "n_races": 40}
        out_path = tmp_path / "baselines.json"
        save_baselines(baselines, out_path)
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert loaded["axis_top3_hit_rate"] == pytest.approx(0.35)
        assert loaded["n_races"] == 40


class TestCompareWithBaseline:
    def test_returns_improvement_rate(self):
        new_metrics = {"axis_top3_hit_rate": 0.5}
        baseline = {"axis_top3_hit_rate": 0.4}
        result = compare_with_baseline(new_metrics, baseline)
        assert "axis_top3_hit_rate_improvement_pct" in result
        assert result["axis_top3_hit_rate_improvement_pct"] == pytest.approx(25.0, rel=1e-3)

    def test_returns_improved_bool_true(self):
        new_metrics = {"axis_top3_hit_rate": 0.5}
        baseline = {"axis_top3_hit_rate": 0.4}
        result = compare_with_baseline(new_metrics, baseline)
        assert result["axis_top3_hit_rate_improved"] is True

    def test_returns_improved_bool_false(self):
        new_metrics = {"axis_top3_hit_rate": 0.3}
        baseline = {"axis_top3_hit_rate": 0.4}
        result = compare_with_baseline(new_metrics, baseline)
        assert result["axis_top3_hit_rate_improved"] is False

    def test_missing_baseline_file_warns_and_returns_empty(self, tmp_path):
        new_metrics = {"axis_top3_hit_rate": 0.5}
        result = compare_with_baseline(new_metrics, tmp_path / "nonexistent.json")
        assert result == {}

    def test_baseline_dict_accepted_directly(self):
        new_metrics = {"top5_cover_rate": 0.6, "top5_cover_count_mean": 1.8}
        baseline = {"top5_cover_rate": 0.5, "top5_cover_count_mean": 1.5}
        result = compare_with_baseline(new_metrics, baseline)
        assert "top5_cover_rate_improvement_pct" in result
        assert result["top5_cover_rate_improved"] is True


class TestSyntheticBaselineRejection:
    def test_compare_rejects_synthetic_flag(self):
        new_metrics = {"axis_top3_hit_rate": 0.5}
        baseline = {"axis_top3_hit_rate": 0.4, "is_synthetic": True}
        with pytest.warns(UserWarning, match="synthetic|合成"):
            result = compare_with_baseline(new_metrics, baseline)
        assert result == {}

    def test_compare_rejects_synthetic_note(self):
        new_metrics = {"axis_top3_hit_rate": 0.5}
        baseline = {
            "axis_top3_hit_rate": 0.4,
            "_note": "synthetic data (50 races x 10 horses) — replace with real data",
        }
        with pytest.warns(UserWarning):
            result = compare_with_baseline(new_metrics, baseline)
        assert result == {}

    def test_compare_accepts_real_baseline(self):
        new_metrics = {"axis_top3_hit_rate": 0.5}
        baseline = {"axis_top3_hit_rate": 0.4, "is_synthetic": False}
        result = compare_with_baseline(new_metrics, baseline)
        assert result["axis_top3_hit_rate_improved"] is True


class TestGenerateRealBaselines:
    def test_bundles_real_metrics_and_metadata(self, synthetic_history, base_config):
        from keiba_predictor.backtest_baseline import generate_real_baselines

        result = generate_real_baselines(
            synthetic_history, base_config, generated_at="2026-05-31T00:00:00",
            source_file="data/training/race_results_master.csv",
        )
        assert result["is_synthetic"] is False
        assert result["n_races"] > 0
        assert "axis_top3_hit_rate" in result
        assert "favorite_baseline" in result
        assert "edge_vs_market" in result
        assert "data_sufficiency" in result
        assert result["generated_at"] == "2026-05-31T00:00:00"
        assert result["source_file"] == "data/training/race_results_master.csv"


class TestComputeBaselinesByCondition:
    def test_returns_distance_class_surface_slices(self, monkeypatch):
        import pandas as pd
        import keiba_predictor.backtest_baseline as module

        prepared_history = pd.DataFrame(
            [
                {"race_id": "R1", "surface": "turf", "distance": 1200, "class": "maiden"},
                {"race_id": "R1", "surface": "turf", "distance": 1200, "class": "maiden"},
                {"race_id": "R2", "surface": "dirt", "distance": 2000, "class": "Open"},
                {"race_id": "R2", "surface": "dirt", "distance": 2000, "class": "Open"},
            ]
        )
        oof_predictions = pd.DataFrame(
            [
                {"race_id": "R1", "horse_id": "H1", "finish_rank": 1, "is_top3": 1, "ensemble_top3_prob": 0.8},
                {"race_id": "R1", "horse_id": "H2", "finish_rank": 4, "is_top3": 0, "ensemble_top3_prob": 0.4},
                {"race_id": "R2", "horse_id": "H3", "finish_rank": 5, "is_top3": 0, "ensemble_top3_prob": 0.7},
                {"race_id": "R2", "horse_id": "H4", "finish_rank": 2, "is_top3": 1, "ensemble_top3_prob": 0.6},
            ]
        )

        monkeypatch.setattr(
            module,
            "_prepare_history",
            lambda history_df, config: (prepared_history, pd.Series([1.0] * len(prepared_history)), ["surface"], ["surface"]),
        )
        monkeypatch.setattr(
            module,
            "evaluate_time_series_cv",
            lambda history, feature_cols, no_odds_feature_cols, sample_weight, config: {"oof_predictions": oof_predictions},
        )

        result = compute_baselines_by_condition(history_df=prepared_history, config={})

        assert set(result.keys()) == {"distance_band", "class", "surface"}
        assert "sprint" in result["distance_band"]
        assert "intermediate" in result["distance_band"]
        assert "maiden" in result["class"]
        assert "open" in result["class"]
        assert "turf" in result["surface"]
        assert "dirt" in result["surface"]
        assert "axis_top3_hit_rate" in result["surface"]["turf"]
        assert "cover_rate" in result["surface"]["dirt"]
