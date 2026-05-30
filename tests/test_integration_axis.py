"""Phase 1 統合テスト: axis_score が予測パイプライン全体を通じて出力される。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from keiba_predictor.prediction import run_prediction
from keiba_predictor.workflow import (
    build_post_race_analysis,
    build_post_race_report,
    build_settled_entry,
    validate_result_frame,
)


class TestRunPredictionAxisScore:
    """run_prediction() → predictions に axis_score 列が含まれる。"""

    def test_predictions_contains_axis_score(
        self, synthetic_history: pd.DataFrame, synthetic_entry: pd.DataFrame, base_config: dict, tmp_path: Path
    ) -> None:
        result = run_prediction(
            history_input=synthetic_history,
            entry_input=synthetic_entry,
            use_odds=True,
            config=base_config,
            output_dir=tmp_path,
        )
        pred_df = result["predictions"]
        assert "axis_score" in pred_df.columns, "predictions に axis_score 列がない"
        assert pred_df["axis_score"].notna().all(), "axis_score に NaN がある"
        assert (pred_df["axis_score"] >= 0.0).all()
        assert (pred_df["axis_score"] <= 1.0).all()

    def test_predictions_csv_contains_axis_score(
        self, synthetic_history: pd.DataFrame, synthetic_entry: pd.DataFrame, base_config: dict, tmp_path: Path
    ) -> None:
        run_prediction(
            history_input=synthetic_history,
            entry_input=synthetic_entry,
            use_odds=True,
            config=base_config,
            output_dir=tmp_path,
        )
        saved = pd.read_csv(tmp_path / "predictions.csv")
        assert "axis_score" in saved.columns

    def test_axis_row_has_axis_score(
        self, synthetic_history: pd.DataFrame, synthetic_entry: pd.DataFrame, base_config: dict, tmp_path: Path
    ) -> None:
        result = run_prediction(
            history_input=synthetic_history,
            entry_input=synthetic_entry,
            use_odds=True,
            config=base_config,
            output_dir=tmp_path,
        )
        axis_row = result["axis_row"]
        assert "axis_score" in axis_row.index, "axis_row に axis_score がない"


class TestPostRaceReportAxisEvaluation:
    """settle_race 相当の flow で axis_evaluation が post_race_report に含まれる。"""

    def _make_result_df(self, entry_df: pd.DataFrame) -> pd.DataFrame:
        """entry_df の最初の 3 頭に finish_rank 1-3 を付与した result_df を生成。"""
        n = len(entry_df)
        horses = entry_df["horse_id"].tolist()
        rows = [
            {
                "horse_id": horses[i],
                "finish_rank": i + 1,
            }
            for i in range(n)
        ]
        result_raw = pd.DataFrame(rows)
        result_raw["race_id"] = entry_df["race_id"].iloc[0]
        return result_raw

    def test_post_race_report_has_axis_evaluation(
        self, synthetic_history: pd.DataFrame, synthetic_entry: pd.DataFrame, base_config: dict, tmp_path: Path
    ) -> None:
        result = run_prediction(
            history_input=synthetic_history,
            entry_input=synthetic_entry,
            use_odds=True,
            config=base_config,
            output_dir=tmp_path,
        )
        pred_df = result["predictions"]

        result_raw = self._make_result_df(synthetic_entry)
        result_df, is_full = validate_result_frame(synthetic_entry, result_raw)
        settled_df = build_settled_entry(synthetic_entry, result_df)
        analysis_df = build_post_race_analysis(pred_df, result_df)

        if "horse_name" not in result_df.columns and "horse_name" in synthetic_entry.columns:
            result_df = result_df.merge(
                synthetic_entry[["horse_id", "horse_name"]], on="horse_id", how="left"
            )

        report = build_post_race_report(
            predictions_df=pred_df,
            settled_entry_df=settled_df,
            result_df=result_df,
            is_full_result=is_full,
            appended_to_training_history=False,
            analysis_df=analysis_df,
        )

        assert "axis_evaluation" in report, "post_race_report に axis_evaluation がない"
        ae = report["axis_evaluation"]
        for key in ["axis_score", "axis_rank_in_field", "axis_tail_risk", "hit_top3", "finish_rank", "selection_reason"]:
            assert key in ae, f"axis_evaluation に {key} がない"
