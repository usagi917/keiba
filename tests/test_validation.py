from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from keiba_predictor.validation import (
    CVPlan,
    assert_no_result_leakage,
    assess_data_sufficiency,
    blend_with_market,
    compute_favorite_baseline,
    market_implied_top3,
    market_shrinkage_weight,
    resolve_cv_plan,
    summarize_edge,
)


# ---------------------------------------------------------------------------
# resolve_cv_plan: 沈黙クランプの可視化
# ---------------------------------------------------------------------------

def test_resolve_cv_plan_replicates_clamp_on_tiny_data():
    """9レース・min_train_races=50 は実効5にクランプされ、各fold 1レース検証になる。"""
    plan = resolve_cv_plan(n_races=9, n_splits=4, min_train_races=50)
    assert isinstance(plan, CVPlan)
    assert plan.feasible is True
    assert plan.effective_n_splits == 4
    assert plan.effective_min_train_races == 5
    assert plan.min_train_races_clamped is True
    assert plan.n_splits_clamped is False
    # 9 - 5 = 4 検証レースを 4 fold → 1レース/fold
    assert plan.val_races_per_fold == pytest.approx(1.0)
    assert plan.is_clamped is True


def test_resolve_cv_plan_no_clamp_with_enough_data():
    plan = resolve_cv_plan(n_races=300, n_splits=4, min_train_races=50)
    assert plan.effective_n_splits == 4
    assert plan.effective_min_train_races == 50
    assert plan.min_train_races_clamped is False
    assert plan.n_splits_clamped is False
    assert plan.is_clamped is False


def test_resolve_cv_plan_infeasible_below_3_races():
    plan = resolve_cv_plan(n_races=2, n_splits=4, min_train_races=50)
    assert plan.feasible is False


# ---------------------------------------------------------------------------
# assess_data_sufficiency: 低信頼フラグと regime
# ---------------------------------------------------------------------------

def _mini_history(n_races: int, horses: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2020-01-01", periods=n_races, freq="7D")
    for r in range(n_races):
        order = rng.permutation(horses)
        for pos in range(horses):
            rows.append(
                {
                    "race_id": f"R{r:04d}",
                    "race_date": dates[r],
                    "horse_id": f"H{(r * horses + pos) % 200:03d}",
                    "finish_rank": int(order[pos]) + 1,
                    "popularity": pos + 1,
                    "odds": float(pos + 1) * 2.0,
                    "field_size": horses,
                }
            )
    return pd.DataFrame(rows)


def test_assess_data_sufficiency_flags_low_confidence_on_9_races():
    df = _mini_history(9)
    config = {"cv": {"n_splits": 4, "min_train_races": 50}}
    report = assess_data_sufficiency(df, config)
    assert report["n_races"] == 9
    assert report["n_positives"] == 9 * 3
    assert report["is_low_confidence"] is True
    assert report["regime"] == "low"
    assert report["effective_min_train_races"] == 5
    assert any("min_train_races" in w for w in report["warnings"])


def test_assess_data_sufficiency_full_regime_on_large_data():
    df = _mini_history(220, horses=8, seed=1)
    config = {"cv": {"n_splits": 4, "min_train_races": 50}}
    report = assess_data_sufficiency(df, config)
    assert report["n_races"] == 220
    assert report["regime"] == "full"
    assert report["is_low_confidence"] is False
    assert report["warnings"] == []


# ---------------------------------------------------------------------------
# compute_favorite_baseline: 人気馬追随ベースライン
# ---------------------------------------------------------------------------

def test_compute_favorite_baseline_uses_popularity_one_as_axis():
    # race A: 人気1が2着(hit) / race B: 人気1が5着(miss)
    rows = []
    for rid, fav_rank in [("A", 2), ("B", 5)]:
        for pop in range(1, 7):
            finish = fav_rank if pop == 1 else (1 if pop == 2 else pop)
            rows.append(
                {
                    "race_id": rid,
                    "horse_id": f"{rid}{pop}",
                    "popularity": pop,
                    "odds": float(pop),
                    "finish_rank": finish,
                    "field_size": 6,
                }
            )
    df = pd.DataFrame(rows)
    result = compute_favorite_baseline(df, n_partners=3)
    assert result["n_races"] == 2
    # 軸(人気1)的中は1/2レース
    assert result["axis_top3_hit_rate"] == pytest.approx(0.5)
    assert "favorite_top3_cover_rate" in result


def test_compute_favorite_baseline_falls_back_to_odds_when_no_popularity():
    rows = []
    for pop in range(1, 6):
        rows.append(
            {
                "race_id": "A",
                "horse_id": f"A{pop}",
                "odds": float(pop),  # 最低オッズ=pop1 が軸
                "finish_rank": 1 if pop == 1 else pop,
                "field_size": 5,
            }
        )
    df = pd.DataFrame(rows)
    result = compute_favorite_baseline(df, n_partners=3)
    assert result["axis_top3_hit_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# summarize_edge: モデル vs 市場 vs シャドウ
# ---------------------------------------------------------------------------

def test_summarize_edge_reports_lift_over_favorite():
    # 2レース。oof: ensemble が race A で正解馬を最上位、race B でも正解
    oof = pd.DataFrame(
        {
            "race_id": ["A", "A", "A", "B", "B", "B"],
            "finish_rank": [1, 4, 5, 1, 4, 5],
            "ensemble_top3_prob": [0.9, 0.2, 0.1, 0.8, 0.3, 0.2],
            "shadow_no_odds_top3_prob": [0.5, 0.4, 0.3, 0.5, 0.4, 0.3],
            "popularity": [1, 2, 3, 3, 1, 2],
            "odds": [2.0, 3.0, 4.0, 4.0, 2.0, 3.0],
            "field_size": [3, 3, 3, 3, 3, 3],
        }
    )
    edge = summarize_edge(oof)
    assert edge["model_axis_top3_hit_rate"] == pytest.approx(1.0)
    # シャドウ race A は1番目が正解(finish1), race B は同点→先頭(finish1) 両方hit
    assert "shadow_axis_top3_hit_rate" in edge
    assert "favorite_axis_top3_hit_rate" in edge
    assert edge["lift_model_over_favorite"] == pytest.approx(
        edge["model_axis_top3_hit_rate"] - edge["favorite_axis_top3_hit_rate"]
    )
    assert "lift_shadow_over_favorite" in edge


# ---------------------------------------------------------------------------
# assert_no_result_leakage: result_* 列を特徴量に入れない
# ---------------------------------------------------------------------------

def test_assert_no_result_leakage_raises_on_result_columns():
    with pytest.raises(ValueError, match="result_"):
        assert_no_result_leakage(["odds", "result_final_odds", "age"])


def test_assert_no_result_leakage_passes_clean_columns():
    # 例外を出さない
    assert_no_result_leakage(["odds", "popularity", "age", "last_3f"]) is None


# ---------------------------------------------------------------------------
# 市場ブレンド (Phase 3 ユーティリティ; 既定では発火させない)
# ---------------------------------------------------------------------------

def test_market_implied_top3_ranks_favorites_higher():
    df = pd.DataFrame(
        {
            "race_id": ["A"] * 4,
            "odds": [2.0, 4.0, 8.0, 16.0],
            "field_size": [4] * 4,
        }
    )
    prob = market_implied_top3(df)
    assert len(prob) == 4
    assert prob[0] >= prob[1] >= prob[2] >= prob[3]
    assert np.all(prob >= 0.0) and np.all(prob <= 1.0)


def test_blend_with_market_convex_combination():
    model = np.array([0.8, 0.2])
    market = np.array([0.4, 0.6])
    out = blend_with_market(model, market, weight=0.5)
    assert out[0] == pytest.approx(0.6)
    assert out[1] == pytest.approx(0.4)


def test_market_shrinkage_weight_higher_for_sparse_data():
    w_low = market_shrinkage_weight({"regime": "low"}, {})
    w_partial = market_shrinkage_weight({"regime": "partial"}, {})
    w_full = market_shrinkage_weight({"regime": "full"}, {})
    assert 0.0 <= w_full < w_partial < w_low <= 1.0
