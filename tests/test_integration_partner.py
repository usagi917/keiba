"""Phase 2 統合テスト: Partner V1 が予測パイプライン全体を通じて出力される。"""
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


class TestRunPredictionPartnerOutputs:
    """run_prediction() → recommended_partners.json + ticket_candidates.csv が生成される。"""

    def test_recommended_partners_json_created(
        self, synthetic_history: pd.DataFrame, synthetic_entry: pd.DataFrame, base_config: dict, tmp_path: Path
    ) -> None:
        run_prediction(
            history_input=synthetic_history,
            entry_input=synthetic_entry,
            use_odds=True,
            config=base_config,
            output_dir=tmp_path,
        )
        partners_path = tmp_path / "recommended_partners.json"
        assert partners_path.exists(), "recommended_partners.json が生成されていない"
        payload = json.loads(partners_path.read_text(encoding="utf-8"))
        assert "axis" in payload
        assert "partners" in payload
        assert isinstance(payload["partners"], list)

    def test_ticket_candidates_csv_created(
        self, synthetic_history: pd.DataFrame, synthetic_entry: pd.DataFrame, base_config: dict, tmp_path: Path
    ) -> None:
        run_prediction(
            history_input=synthetic_history,
            entry_input=synthetic_entry,
            use_odds=True,
            config=base_config,
            output_dir=tmp_path,
        )
        ticket_path = tmp_path / "ticket_candidates.csv"
        assert ticket_path.exists(), "ticket_candidates.csv が生成されていない"
        df = pd.read_csv(ticket_path)
        for col in ["axis_horse_id", "partner_horse_id", "partner_rank", "partner_score"]:
            assert col in df.columns, f"ticket_candidates.csv に {col} がない"

    def test_predictions_contains_partner_score(
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
        assert "partner_score_vs_axis" in pred_df.columns, "predictions に partner_score_vs_axis がない"


class TestPostRaceReportPartnerEvaluation:
    """settle_race 相当の flow で partner_evaluation が post_race_report に含まれる。"""

    def _make_result_df(self, entry_df: pd.DataFrame) -> pd.DataFrame:
        n = len(entry_df)
        horses = entry_df["horse_id"].tolist()
        rows = [{"horse_id": horses[i], "finish_rank": i + 1} for i in range(n)]
        result_raw = pd.DataFrame(rows)
        result_raw["race_id"] = entry_df["race_id"].iloc[0]
        return result_raw

    def test_post_race_report_has_partner_evaluation(
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
            output_dir=tmp_path,
        )

        assert "partner_evaluation" in report, "post_race_report に partner_evaluation がない"
        pe = report["partner_evaluation"]
        for key in ["n_partners", "actual_top3_covered", "cover_rate", "missed_horses", "axis_plus_partners_top3_covered"]:
            assert key in pe, f"partner_evaluation に {key} がない"
