from __future__ import annotations

import pandas as pd
import pytest

from keiba_predictor.prediction import _prepare_top3_dashboard_frame
from keiba_predictor.workflow import (
    build_post_race_analysis,
    build_post_race_report,
    build_settled_entry,
    validate_result_frame,
)


def _sample_entry_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "race_id": "R2026",
                "horse_id": "H001",
                "horse_name": "Alpha",
                "draw": 1,
                "horse_number": 1,
                "finish_rank": pd.NA,
            },
            {
                "race_id": "R2026",
                "horse_id": "H002",
                "horse_name": "Beta",
                "draw": 2,
                "horse_number": 2,
                "finish_rank": pd.NA,
            },
            {
                "race_id": "R2026",
                "horse_id": "H003",
                "horse_name": "Gamma",
                "draw": 3,
                "horse_number": 3,
                "finish_rank": pd.NA,
            },
        ]
    )


def _sample_result_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "horse_id": "H001",
                "horse_name": "Alpha",
                "finish_rank": 1,
                "result_time": "1:06.3",
                "result_margin": "",
                "result_last3f": "32.4",
                "result_final_odds": "3.5",
                "result_final_popularity": "1",
                "result_body_weight": "530",
                "result_body_weight_diff": "-2",
            },
            {
                "horse_id": "H002",
                "horse_name": "Beta",
                "finish_rank": 2,
                "result_time": "1:06.6",
                "result_margin": "2",
                "result_last3f": "32.5",
                "result_final_odds": "67.0",
                "result_final_popularity": "15",
                "result_body_weight": "500",
                "result_body_weight_diff": "-18",
            },
            {
                "horse_id": "H003",
                "horse_name": "Gamma",
                "finish_rank": 3,
                "result_time": "1:06.6",
                "result_margin": "クビ",
                "result_last3f": "33.2",
                "result_final_odds": "19.4",
                "result_final_popularity": "7",
                "result_body_weight": "516",
                "result_body_weight_diff": "+3",
            },
        ]
    )


def _sample_predictions_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "race_id": "R2026",
                "horse_id": "H002",
                "horse_display_name": "Beta",
                "selected_top3_model": "classifier_top3_prob",
                "consensus_top3_score": 0.90,
                "top3_ci_width": 0.04,
                "top3_prob": 0.80,
                "win_prob": 0.50,
                "mean_rank": 1.5,
            },
            {
                "race_id": "R2026",
                "horse_id": "H001",
                "horse_display_name": "Alpha",
                "selected_top3_model": "classifier_top3_prob",
                "consensus_top3_score": 0.80,
                "top3_ci_width": 0.05,
                "top3_prob": 0.70,
                "win_prob": 0.30,
                "mean_rank": 2.5,
            },
            {
                "race_id": "R2026",
                "horse_id": "H003",
                "horse_display_name": "Gamma",
                "selected_top3_model": "classifier_top3_prob",
                "consensus_top3_score": 0.70,
                "top3_ci_width": 0.06,
                "top3_prob": 0.20,
                "win_prob": 0.20,
                "mean_rank": 3.0,
            },
        ]
    )


class TestValidateResultFrame:
    def test_parses_optional_result_columns_and_infers_race_id(self):
        entry_df = _sample_entry_df()
        normalized, is_full_result = validate_result_frame(entry_df, _sample_result_df())

        assert is_full_result is True
        assert normalized["race_id"].tolist() == ["R2026", "R2026", "R2026"]
        assert normalized["result_time_seconds"].tolist() == pytest.approx([66.3, 66.6, 66.6])
        assert normalized["result_final_popularity"].tolist() == [1, 15, 7]
        assert normalized["result_body_weight_diff"].tolist() == [-2, -18, 3]


class TestBuildSettledEntry:
    def test_preserves_optional_result_columns(self):
        entry_df = _sample_entry_df()
        result_df, _ = validate_result_frame(entry_df, _sample_result_df())

        settled = build_settled_entry(entry_df, result_df).sort_values("horse_id").reset_index(drop=True)

        assert "result_time" in settled.columns
        assert "result_time_seconds" in settled.columns
        assert "result_final_odds" in settled.columns
        assert settled.loc[0, "horse_name"] == "Alpha"
        assert settled.loc[0, "result_time"] == "1:06.3"
        assert settled.loc[0, "result_time_seconds"] == pytest.approx(66.3)
        assert settled.loc[2, "result_body_weight_diff"] == 3


class TestPostRaceAnalysis:
    def test_builds_analysis_and_report_metrics(self):
        entry_df = _sample_entry_df()
        result_df, _ = validate_result_frame(entry_df, _sample_result_df())
        predictions_df = _sample_predictions_df()
        settled_df = build_settled_entry(entry_df, result_df)

        analysis = build_post_race_analysis(predictions_df=predictions_df, result_df=result_df)
        assert analysis is not None

        by_horse = analysis.set_index("horse_id")
        assert by_horse.loc["H001", "predicted_rank"] == 2
        assert by_horse.loc["H001", "actual_finish_rank"] == 1
        assert by_horse.loc["H001", "abs_rank_error"] == 1
        assert by_horse.loc["H001", "published_top3_prob"] == pytest.approx(0.8)
        assert by_horse.loc["H001", "result_time_seconds"] == pytest.approx(66.3)
        assert bool(by_horse.loc["H001", "actual_is_win"]) is True

        report = build_post_race_report(
            predictions_df=predictions_df,
            settled_entry_df=settled_df,
            result_df=result_df,
            is_full_result=True,
            appended_to_training_history=True,
            race_meta={"winning_time": "1:06.3", "winning_time_seconds": 66.3},
            analysis_df=analysis,
        )

        assert report["prediction_status"] == "ok"
        assert report["selected_top3_model"] == "classifier_top3_prob"
        assert report["top3_overlap_count"] == 3
        assert report["axis_hit_top3"] is True
        assert report["known_rank_mean_abs_error"] == pytest.approx((1.5 + 0.5 + 0.0) / 3)
        assert report["rank_order_mean_abs_error"] == pytest.approx((1.0 + 1.0 + 0.0) / 3)
        assert report["top3_brier_score"] == pytest.approx((0.01 + 0.04 + 0.09) / 3)
        assert report["win_brier_score"] == pytest.approx((0.25 + 0.49 + 0.04) / 3)
        assert report["rank_spearman"] == pytest.approx(0.5)
        assert report["winner_predicted_rank"] == 2
        assert report["winner_mean_rank"] == pytest.approx(2.5)
        assert report["winner_win_prob"] == pytest.approx(0.3)
        assert report["winner_top3_prob"] == pytest.approx(0.8)
        assert report["actual_winner"]["horse_id"] == "H001"
        assert report["actual_winner"]["finish_rank"] == 1
        assert report["actual_winner"]["result_time_seconds"] == pytest.approx(66.3)
        assert report["largest_prediction_gaps"][0]["selected_top3_model"] == "classifier_top3_prob"
        assert report["largest_prediction_gaps"][0]["published_top3_prob"] == pytest.approx(0.8)


class TestPredictionDashboardOrdering:
    def test_dashboard_uses_published_top3_score(self):
        pred_df = pd.DataFrame(
            [
                {"horse_id": "H001", "horse_display_name": "Alpha", "horse_number": 1, "consensus_top3_score": 0.60, "top3_prob": 0.95, "win_prob": 0.10},
                {"horse_id": "H002", "horse_display_name": "Beta", "horse_number": 2, "consensus_top3_score": 0.90, "top3_prob": 0.30, "win_prob": 0.20},
                {"horse_id": "H003", "horse_display_name": "Gamma", "horse_number": 3, "consensus_top3_score": 0.80, "top3_prob": 0.20, "win_prob": 0.30},
            ]
        )

        ordered = _prepare_top3_dashboard_frame(pred_df)

        assert ordered["horse_id"].tolist() == ["H002", "H003", "H001"]
        assert ordered["published_top3_prob"].tolist() == pytest.approx([0.9, 0.8, 0.6])
