from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import io

from keiba_predictor.prediction import (
    build_prediction_table,
    compute_axis_features,
    compute_axis_score,
    compute_partner_scores,
    print_prediction_console_summary,
    select_axis_horse,
    select_partners,
)


def test_build_prediction_table_uses_sparse_shadow_regression_blend() -> None:
    entry_df = pd.DataFrame(
        [
            {
                "race_id": "R2026",
                "horse_id": "H001",
                "horse_name": "Alpha",
                "horse_number": 1,
                "top3_model_sparse_context": 1.0,
            },
            {
                "race_id": "R2026",
                "horse_id": "H002",
                "horse_name": "Beta",
                "horse_number": 2,
                "top3_model_sparse_context": 0.0,
            },
        ]
    )
    component_df = pd.DataFrame(
        [
            {
                "entity_id": 0,
                "race_id": "R2026",
                "top3_prob": 0.20,
                "classifier_top3_prob": 0.30,
                "aux_classifier_top3_prob": 0.35,
                "regression_top3_prob": 0.50,
                "shadow_no_odds_top3_prob": 0.40,
                "win_prob": 0.10,
                "top2_prob": 0.20,
                "mean_rank": 5.0,
                "top3_ci_low": 0.15,
                "top3_ci_high": 0.55,
                "top3_ci_width": 0.40,
                "component_model_std": 0.08,
                "rank_strength": 0.5,
                "shadow_rank_strength": 0.4,
                "rank_temperature": 1.0,
            },
            {
                "entity_id": 1,
                "race_id": "R2026",
                "top3_prob": 0.25,
                "classifier_top3_prob": 0.32,
                "aux_classifier_top3_prob": 0.34,
                "regression_top3_prob": 0.28,
                "shadow_no_odds_top3_prob": 0.27,
                "win_prob": 0.12,
                "top2_prob": 0.22,
                "mean_rank": 4.8,
                "top3_ci_low": 0.18,
                "top3_ci_high": 0.36,
                "top3_ci_width": 0.18,
                "component_model_std": 0.05,
                "rank_strength": 0.6,
                "shadow_rank_strength": 0.5,
                "rank_temperature": 1.0,
            },
        ]
    )

    pred_df = build_prediction_table(
        entry_df=entry_df,
        component_df=component_df,
        selected_top3_col="classifier_top3_prob",
    )

    sparse_row = pred_df.loc[pred_df["horse_id"] == "H001"].iloc[0]
    normal_row = pred_df.loc[pred_df["horse_id"] == "H002"].iloc[0]

    assert sparse_row["selected_top3_model"] == "sparse_shadow_regression_blend"
    assert sparse_row["consensus_top3_score"] == pytest.approx((0.6 * 0.40) + (0.4 * 0.50))
    assert normal_row["selected_top3_model"] == "classifier_top3_prob"
    assert normal_row["consensus_top3_score"] == pytest.approx(0.32)


# ---------------------------------------------------------------------------
# 1-1 / 1-2: compute_axis_features
# ---------------------------------------------------------------------------

def _make_axis_pred_df(n: int = 5, with_odds: bool = False) -> pd.DataFrame:
    data: dict = {
        "horse_id": [f"H{i}" for i in range(n)],
        "consensus_top3_score": np.linspace(0.6, 0.2, n).tolist(),
        "mean_rank": np.linspace(2.0, 5.0, n).tolist(),
        "top3_ci_width": np.linspace(0.10, 0.35, n).tolist(),
        "component_model_std": np.linspace(0.05, 0.20, n).tolist(),
        "rank_stability_std": np.linspace(0.3, 1.2, n).tolist(),
        **{f"rank_{k}_prob": [1 / n] * n for k in range(1, n + 1)},
    }
    if with_odds:
        data["odds"] = [2.0, 5.0, 8.0, 15.0, 30.0][:n]
    return pd.DataFrame(data)


def test_compute_axis_features_required_columns() -> None:
    pred_df = _make_axis_pred_df(n=5)
    result = compute_axis_features(pred_df)
    for col in [
        "axis_top3_prob",
        "axis_mean_rank",
        "axis_ci_width",
        "axis_model_std",
        "axis_rank_stability",
        "axis_tail_risk",
        "axis_market_edge",
    ]:
        assert col in result.columns, f"{col} が出力に含まれていない"
        assert result[col].notna().all(), f"{col} に NaN がある"
    # odds なし → axis_market_edge は 0.0
    assert (result["axis_market_edge"] == 0.0).all()


def test_compute_axis_features_market_edge_with_odds() -> None:
    pred_df = _make_axis_pred_df(n=3, with_odds=True)
    result = compute_axis_features(pred_df)
    assert "axis_market_edge" in result.columns
    assert result["axis_market_edge"].notna().all()


def test_compute_axis_features_preserves_original_columns() -> None:
    pred_df = _make_axis_pred_df(n=4)
    result = compute_axis_features(pred_df)
    assert "consensus_top3_score" in result.columns
    assert "horse_id" in result.columns


# ---------------------------------------------------------------------------
# 1-3: compute_axis_score
# ---------------------------------------------------------------------------

def test_compute_axis_score_range() -> None:
    n = 5
    features_df = pd.DataFrame({
        "axis_top3_prob": [0.6, 0.5, 0.4, 0.3, 0.2],
        "axis_tail_risk": [0.1, 0.2, 0.3, 0.4, 0.5],
        "axis_ci_width": [0.1, 0.15, 0.2, 0.25, 0.3],
        "axis_model_std": [0.05, 0.08, 0.1, 0.12, 0.15],
        "axis_rank_stability": [0.3, 0.5, 0.7, 0.9, 1.1],
    })
    result = compute_axis_score(features_df)
    assert isinstance(result, pd.Series)
    assert len(result) == n
    assert (result >= 0.0).all()
    assert (result <= 1.0).all()


def test_compute_axis_score_ordering() -> None:
    """top3_prob が高く tail_risk が低い馬のスコアが高い。"""
    features_df = pd.DataFrame({
        "axis_top3_prob": [0.8, 0.2],
        "axis_tail_risk": [0.1, 0.5],
        "axis_ci_width": [0.1, 0.1],
        "axis_model_std": [0.05, 0.05],
        "axis_rank_stability": [0.3, 0.3],
    })
    result = compute_axis_score(features_df)
    assert result.iloc[0] > result.iloc[1]


def test_compute_axis_score_uniform_input() -> None:
    """全馬同一入力 → 全馬同一スコア。"""
    n = 5
    features_df = pd.DataFrame({
        "axis_top3_prob": [0.5] * n,
        "axis_tail_risk": [0.3] * n,
        "axis_ci_width": [0.2] * n,
        "axis_model_std": [0.1] * n,
        "axis_rank_stability": [0.5] * n,
    })
    result = compute_axis_score(features_df)
    assert result.nunique() == 1


# ---------------------------------------------------------------------------
# 1-4: select_axis_horse
# ---------------------------------------------------------------------------

def _make_pred_df_for_axis(axis_scores: list[float], consensus: list[float]) -> pd.DataFrame:
    n = len(axis_scores)
    return pd.DataFrame({
        "horse_id": [f"H{i}" for i in range(n)],
        "consensus_top3_score": consensus,
        "top3_prob": [0.3] * n,
        "top3_ci_width": [0.1] * n,
        "component_model_std": [0.05] * n,
        "shadow_no_odds_rank": [float(i + 1) for i in range(n)],
        "rank_stability_std": [0.5] * n,
        "classifier_top3_prob": [0.3] * n,
        "axis_score": axis_scores,
    })


def test_select_axis_horse_uses_axis_score_when_present() -> None:
    """axis_score が存在するとき、最高スコアの馬が選ばれる。"""
    pred_df = _make_pred_df_for_axis(
        axis_scores=[0.7, 0.4, 0.5],
        consensus=[0.3, 0.5, 0.4],  # consensus では H1 が最高だが axis_score では H0
    )
    result = select_axis_horse(pred_df)
    assert result["horse_id"] == "H0"


def test_select_axis_horse_fallback_without_axis_score() -> None:
    """axis_score がないとき、従来の consensus_top3_score でソートする。"""
    pred_df = pd.DataFrame({
        "horse_id": ["H0", "H1"],
        "consensus_top3_score": [0.3, 0.5],  # H1 が最高
        "top3_prob": [0.3, 0.5],
        "top3_ci_width": [0.1, 0.1],
        "component_model_std": [0.05, 0.05],
        "shadow_no_odds_rank": [1.0, 1.0],
        "rank_stability_std": [0.5, 0.5],
        "classifier_top3_prob": [0.3, 0.5],
    })
    result = select_axis_horse(pred_df)
    assert result["horse_id"] == "H1"


def test_select_axis_horse_result_contains_axis_score() -> None:
    """戻り値（pd.Series）に axis_score が含まれる。"""
    pred_df = _make_pred_df_for_axis(
        axis_scores=[0.7, 0.4],
        consensus=[0.3, 0.5],
    )
    result = select_axis_horse(pred_df)
    assert "axis_score" in result.index


# ---------------------------------------------------------------------------
# 1-5: build_prediction_table が axis_score 列を含む
# ---------------------------------------------------------------------------

def _make_component_df_with_ranks(n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row: dict = {
            "entity_id": i,
            "race_id": "R2026",
            "top3_prob": 0.3,
            "classifier_top3_prob": 0.3,
            "aux_classifier_top3_prob": 0.3,
            "regression_top3_prob": 0.3,
            "shadow_no_odds_top3_prob": 0.3,
            "win_prob": 0.15,
            "top2_prob": 0.25,
            "mean_rank": 3.0,
            "top3_ci_low": 0.2,
            "top3_ci_high": 0.5,
            "top3_ci_width": 0.3,
            "component_model_std": 0.08,
            "rank_strength": 0.5,
            "shadow_rank_strength": 0.4,
            "rank_temperature": 1.0,
        }
        for k in range(1, n + 1):
            row[f"rank_{k}_prob"] = 1.0 / n
        rows.append(row)
    return pd.DataFrame(rows)


def test_build_prediction_table_has_axis_score() -> None:
    n = 3
    entry_df = pd.DataFrame([
        {"race_id": "R2026", "horse_id": f"H{i}", "horse_name": f"Horse{i}", "horse_number": i + 1}
        for i in range(n)
    ])
    component_df = _make_component_df_with_ranks(n)
    result = build_prediction_table(entry_df, component_df, "classifier_top3_prob")
    assert "axis_score" in result.columns
    assert result["axis_score"].notna().all()
    assert (result["axis_score"] >= 0.0).all()
    assert (result["axis_score"] <= 1.0).all()


# ---------------------------------------------------------------------------
# 2-2: compute_partner_scores
# ---------------------------------------------------------------------------

def _make_partner_pred_df(n: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {
        "horse_id": [f"H{i}" for i in range(n)],
        "horse_display_name": [f"Horse{i}" for i in range(n)],
        "consensus_top3_score": np.linspace(0.7, 0.2, n).tolist(),
        "rank_strength": np.linspace(1.5, 0.5, n).tolist(),
        "rank_temperature": [1.0] * n,
        "popularity": list(range(1, n + 1)),
        "component_model_std": np.linspace(0.05, 0.20, n).tolist(),
    }
    return pd.DataFrame(data)


def test_compute_partner_scores_output_columns() -> None:
    """出力列: partner_conditional_top3, partner_unconditional_top3, partner_lift が含まれる。"""
    pred_df = _make_partner_pred_df(8)
    strengths = pred_df["rank_strength"].to_numpy()
    result = compute_partner_scores(
        pred_df=pred_df,
        axis_horse_id="H0",
        sim_strengths=strengths,
        sim_temperature=1.0,
        config={},
    )
    for col in ["partner_conditional_top3", "partner_unconditional_top3", "partner_lift"]:
        assert col in result.columns, f"{col} が出力に含まれていない"


def test_compute_partner_scores_axis_conditional_is_nan() -> None:
    """軸馬自身の partner_conditional_top3 は NaN。"""
    pred_df = _make_partner_pred_df(8)
    strengths = pred_df["rank_strength"].to_numpy()
    result = compute_partner_scores(
        pred_df=pred_df,
        axis_horse_id="H0",
        sim_strengths=strengths,
        sim_temperature=1.0,
        config={},
    )
    axis_row = result.loc[result["horse_id"] == "H0"].iloc[0]
    assert np.isnan(axis_row["partner_conditional_top3"])


def test_compute_partner_scores_lift_positive() -> None:
    """強い馬（上位馬）の partner_lift が定義されている（NaN でない）。"""
    pred_df = _make_partner_pred_df(8)
    strengths = pred_df["rank_strength"].to_numpy()
    result = compute_partner_scores(
        pred_df=pred_df,
        axis_horse_id="H7",  # 最弱 axis
        sim_strengths=strengths,
        sim_temperature=1.0,
        config={},
    )
    non_axis = result.loc[result["horse_id"] != "H7"]
    assert non_axis["partner_lift"].notna().any()


# ---------------------------------------------------------------------------
# 2-3: select_partners
# ---------------------------------------------------------------------------

def _make_pred_df_for_partners(n: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    data = {
        "horse_id": [f"H{i}" for i in range(n)],
        "horse_display_name": [f"Horse{i}" for i in range(n)],
        "consensus_top3_score": np.linspace(0.8, 0.1, n).tolist(),
        "partner_conditional_top3": [np.nan] + np.linspace(0.6, 0.1, n - 1).tolist(),
        "partner_unconditional_top3": np.linspace(0.5, 0.1, n).tolist(),
        "partner_lift": [np.nan] + np.linspace(1.5, 0.5, n - 1).tolist(),
        "partner_score_vs_axis": [np.nan] + np.linspace(0.7, 0.1, n - 1).tolist(),
        "popularity": list(range(1, n + 1)),
        "component_model_std": np.linspace(0.05, 0.25, n).tolist(),
    }
    return pd.DataFrame(data)


def test_select_partners_returns_n_partners() -> None:
    """戻り値が n_partners 行。"""
    pred_df = _make_pred_df_for_partners(10)
    result = select_partners(pred_df, axis_horse_id="H0", config={}, n_partners=5)
    assert len(result) == 5


def test_select_partners_excludes_axis() -> None:
    """軸馬が含まれていない。"""
    pred_df = _make_pred_df_for_partners(10)
    result = select_partners(pred_df, axis_horse_id="H0", config={}, n_partners=5)
    assert "H0" not in result["horse_id"].tolist()


def test_select_partners_sorted_by_score() -> None:
    """partner_score_vs_axis 降順でソートされている。"""
    pred_df = _make_pred_df_for_partners(10)
    result = select_partners(pred_df, axis_horse_id="H0", config={}, n_partners=5)
    scores = result["partner_score_vs_axis"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_select_partners_fewer_than_n_horses() -> None:
    """頭数 <= n_partners+1 (軸含む) のとき軸以外全員が相手。"""
    n = 5
    pred_df = _make_pred_df_for_partners(n)
    # axis + 残り4頭 → 4頭を全員返す
    result = select_partners(pred_df, axis_horse_id="H0", config={}, n_partners=8)
    assert len(result) == n - 1


def test_select_partners_popularity_cap() -> None:
    """popularity <= 3 の馬は max_popular(=3) 頭まで。"""
    n = 10
    pred_df = _make_pred_df_for_partners(n)
    config = {"partner": {"max_popular": 2, "popularity_cap": 3}}
    result = select_partners(pred_df, axis_horse_id="H0", config=config, n_partners=5)
    # 結果中で popularity <= 3 (かつ軸以外) は最大 2 頭
    popular_count = int((result["popularity"] <= 3).sum())
    assert popular_count <= 2


# ---------------------------------------------------------------------------
# 2-4: config デフォルト値テスト
# ---------------------------------------------------------------------------

def test_select_partners_config_defaults() -> None:
    """config['partner'] がなくてもデフォルト値で動作する。"""
    pred_df = _make_pred_df_for_partners(10)
    result = select_partners(pred_df, axis_horse_id="H0", config={}, n_partners=5)
    assert len(result) == 5


def test_select_partners_config_n_partners_override() -> None:
    """config['partner']['n_partners'] で頭数を変更できる。"""
    pred_df = _make_pred_df_for_partners(10)
    config = {"partner": {"n_partners": 3}}
    result = select_partners(pred_df, axis_horse_id="H0", config=config, n_partners=3)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# 2-7: print_prediction_console_summary に相手情報が含まれる
# ---------------------------------------------------------------------------

def _make_minimal_result(axis_id: str = "H0") -> dict:
    n = 5
    pred_df = _make_pred_df_for_partners(n)
    pred_df["win_prob"] = 0.1
    pred_df["top3_prob"] = 0.3
    pred_df["top3_ci_width"] = 0.1
    pred_df["mean_rank"] = 3.0
    pred_df["horse_number"] = list(range(1, n + 1))
    pred_df["axis_score"] = np.linspace(0.8, 0.2, n)
    pred_df["axis_tail_risk"] = np.linspace(0.1, 0.5, n)
    axis_row = pred_df.loc[pred_df["horse_id"] == axis_id].iloc[0]
    partners_df = pred_df.loc[pred_df["horse_id"] != axis_id].head(3).copy()
    return {
        "feature_cols": ["f1"],
        "eval_result": {"summary": {}},
        "predictions": pred_df,
        "axis_row": axis_row,
        "partner_rows": partners_df,
        "schema_info": {},
        "feature_importance": pd.DataFrame({"feature": ["f1"], "importance": [1.0]}),
        "simulation_diagnostics": pd.DataFrame(),
        "output_paths": {},
    }


def test_print_prediction_console_summary_includes_partners(capsys) -> None:
    """partner_rows がある場合、コンソール出力に相手情報が含まれる。"""
    result = _make_minimal_result()
    print_prediction_console_summary(result)
    captured = capsys.readouterr()
    assert "partner" in captured.out.lower() or "相手" in captured.out


def test_compute_partner_scores_aligns_strengths_with_sorted_pred_df() -> None:
    """build_prediction_table は consensus_top3_score でソートするため、sim_strengths は
    ソート後の pred_df 順で渡されなければ axis_idx と整合しない。このテストは、entry_df 順
    と pred_df 順が異なる状況でも、軸馬の partner_conditional_top3 が NaN になる（軸紐付け
    が正しい）ことを検証する。
    """
    n = 4
    entry_df = pd.DataFrame(
        [
            {"race_id": "R", "horse_id": f"H{i}", "horse_name": f"Horse{i}", "horse_number": i + 1}
            for i in range(n)
        ]
    )
    component_df = _make_component_df_with_ranks(n)
    # top3_prob を entry 順と逆にして、consensus_top3_score ソートで順序が変わる状況を作る
    component_df["top3_prob"] = [0.20, 0.40, 0.60, 0.80]
    component_df["classifier_top3_prob"] = component_df["top3_prob"].to_numpy()
    component_df["rank_strength"] = [0.5, 1.0, 1.5, 2.0]

    pred_df = build_prediction_table(entry_df, component_df, "top3_prob")
    # entry の H0 は最下位 top3_prob なので pred_df では最下位。順序が entry と異なる。
    assert pred_df["horse_id"].tolist() != entry_df["horse_id"].tolist()

    # pred_df の rank_strength は pred_df の順序で並んでいる前提
    strengths = pd.to_numeric(pred_df["rank_strength"], errors="coerce").to_numpy(dtype=float)
    axis_horse_id = "H0"
    result = compute_partner_scores(
        pred_df=pred_df,
        axis_horse_id=axis_horse_id,
        sim_strengths=strengths,
        sim_temperature=1.0,
        config={},
    )
    # 軸馬の行で partner_conditional_top3 が NaN になっていれば、axis_idx と strengths が整合
    axis_row = result.loc[result["horse_id"] == axis_horse_id].iloc[0]
    assert np.isnan(axis_row["partner_conditional_top3"]), "axis の partner_conditional_top3 は NaN であるべき"
    # 非軸馬はいずれも NaN にならない
    non_axis = result.loc[result["horse_id"] != axis_horse_id, "partner_conditional_top3"]
    assert non_axis.notna().all(), "非軸馬の partner_conditional_top3 に NaN が混入している"
