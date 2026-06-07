from __future__ import annotations

import pandas as pd

from keiba_predictor.prediction import optimize_partner_set


def _candidate_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "horse_id": "A",
                "partner_conditional_top3": 0.50,
                "partner_score_vs_axis": 0.60,
                "consensus_top3_score": 0.55,
                "popularity": 1,
                "component_model_std": 0.10,
            },
            {
                "horse_id": "B",
                "partner_conditional_top3": 0.48,
                "partner_score_vs_axis": 0.59,
                "consensus_top3_score": 0.52,
                "popularity": 2,
                "component_model_std": 0.11,
            },
            {
                "horse_id": "C",
                "partner_conditional_top3": 0.44,
                "partner_score_vs_axis": 0.58,
                "consensus_top3_score": 0.49,
                "popularity": 6,
                "component_model_std": 0.12,
            },
            {
                "horse_id": "D",
                "partner_conditional_top3": 0.20,
                "partner_score_vs_axis": 0.30,
                "consensus_top3_score": 0.28,
                "popularity": 7,
                "component_model_std": 0.95,
            },
        ]
    )


def test_optimize_partner_set_selects_best_combination() -> None:
    candidates = _candidate_df()
    constraints = {
        "popularity_cap": 3,
        "max_popular": 2,
        "max_uncertain": 1,
        "uncertainty_percentile": 80,
    }

    result = optimize_partner_set(candidates, axis_horse="AXIS", constraints=constraints, n_partners=2)

    assert result["horse_id"].tolist() == ["A", "B"]


def test_optimize_partner_set_skips_constraint_violations() -> None:
    candidates = _candidate_df()
    constraints = {
        "popularity_cap": 3,
        "max_popular": 1,
        "max_uncertain": 1,
        "uncertainty_percentile": 80,
    }

    result = optimize_partner_set(candidates, axis_horse="AXIS", constraints=constraints, n_partners=2)

    assert result["horse_id"].tolist() == ["A", "C"]
    assert int((result["popularity"] <= 3).sum()) <= 1
