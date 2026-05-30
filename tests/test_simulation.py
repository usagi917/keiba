from __future__ import annotations

import numpy as np
import pandas as pd

from keiba_predictor.simulation import compute_tail_risk, conditional_top3_probs, plackett_luce_rank_distribution


def _make_rank_df(n: int, seed: int = 42) -> pd.DataFrame:
    names = [f"Horse{i}" for i in range(n)]
    strengths = np.linspace(1.0, 2.0, n)
    rank_df, _ = plackett_luce_rank_distribution(
        horse_names=names,
        strengths=strengths,
        temperature=1.0,
        stages=[2000],
        threshold=0.01,
        seed=seed,
    )
    return rank_df


def test_compute_tail_risk_range() -> None:
    rank_df = _make_rank_df(18)
    result = compute_tail_risk(rank_df, threshold_percentile=50)
    assert result.shape == (18,)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_compute_tail_risk_18_threshold50_equals_rank10_to_18() -> None:
    """18頭立て、threshold=50 → rank_10〜rank_18 の確率合計であることを検証。"""
    rank_df = _make_rank_df(18)
    result = compute_tail_risk(rank_df, threshold_percentile=50)
    expected = sum(rank_df[f"rank_{k}_prob"].to_numpy() for k in range(10, 19))
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_compute_tail_risk_strong_horse_has_lower_risk() -> None:
    """rank_df は top3_prob 降順でソート済み: 先頭馬は末尾馬より tail_risk が低い。"""
    rank_df = _make_rank_df(10, seed=99)
    result = compute_tail_risk(rank_df, threshold_percentile=50)
    assert result[0] < result[-1]


# ---------------------------------------------------------------------------
# 2-1: conditional_top3_probs
# ---------------------------------------------------------------------------

def test_conditional_top3_probs_length() -> None:
    """戻り値の長さ = 頭数。"""
    strengths = np.linspace(2.0, 1.0, 8)
    result = conditional_top3_probs(strengths, axis_idx=0, temperature=1.0, n_trials=2000, seed=42)
    assert len(result) == len(strengths)


def test_conditional_top3_probs_axis_is_nan() -> None:
    """axis_idx 自身の値は NaN。"""
    strengths = np.linspace(2.0, 1.0, 8)
    result = conditional_top3_probs(strengths, axis_idx=0, temperature=1.0, n_trials=2000, seed=42)
    assert np.isnan(result[0])


def test_conditional_top3_probs_non_axis_in_0_1() -> None:
    """axis 以外の全馬の条件付き確率は [0, 1]。"""
    strengths = np.linspace(2.0, 1.0, 8)
    result = conditional_top3_probs(strengths, axis_idx=0, temperature=1.0, n_trials=3000, seed=42)
    non_nan = result[~np.isnan(result)]
    assert np.all(non_nan >= 0.0)
    assert np.all(non_nan <= 1.0)


def test_conditional_top3_probs_sum_approx_2() -> None:
    """軸以外の条件付きTop3確率の合計 ≈ 2.0（残り2枠）。"""
    strengths = np.linspace(2.0, 1.0, 10)
    result = conditional_top3_probs(strengths, axis_idx=0, temperature=1.0, n_trials=5000, seed=42)
    non_nan_sum = float(np.nansum(result))
    assert abs(non_nan_sum - 2.0) < 0.3


def test_conditional_top3_probs_strong_horse_higher() -> None:
    """強い馬（strength 高）の条件付き確率 > 弱い馬。"""
    # axis は最弱馬: idx=5。最強馬 idx=0 > 中間 idx=3
    strengths = np.array([2.0, 1.8, 1.5, 1.2, 0.8, 0.3])
    result = conditional_top3_probs(strengths, axis_idx=5, temperature=1.0, n_trials=8000, seed=42)
    assert result[0] > result[3]


def test_conditional_top3_probs_reproducible() -> None:
    """同じ seed なら結果が完全一致。"""
    strengths = np.linspace(2.0, 1.0, 8)
    r1 = conditional_top3_probs(strengths, axis_idx=0, temperature=1.0, n_trials=1000, seed=99)
    r2 = conditional_top3_probs(strengths, axis_idx=0, temperature=1.0, n_trials=1000, seed=99)
    np.testing.assert_array_equal(r1, r2)
