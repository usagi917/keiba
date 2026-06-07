"""検証基盤 (validation harness)。

このモジュールは「予測が本当に高精度か」を正直に測定するための土台を提供する。
診断で判明した次の問題を構造的に防ぐ:

- `cv.min_train_races` の沈黙クランプを `resolve_cv_plan` で可視化する。
- レース数が設計下限を下回るときに低信頼フラグと警告を出す (`assess_data_sufficiency`)。
- 「1番人気追随」ベースラインを計算し、モデルの上乗せ(lift)を測る (`compute_favorite_baseline`)。
- オッズ抜きシャドウ vs オッズ使用モデル vs 市場の差分を要約する (`summarize_edge`)。
- `result_*` 列が特徴量に紛れ込むのを防ぐ (`assert_no_result_leakage`)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# 学習時に絶対に特徴量へ入れてはならない結果(post-race)列のプレフィックス。
RESULT_COLUMN_PREFIX = "result_"

# ブレンダー regime と整合するデータ量閾値 (hybrid_model._select_blender_regime と一致させる)。
LOW_DATA_RACE_THRESHOLD = 50
FULL_DATA_RACE_THRESHOLD = 200


@dataclass
class CVPlan:
    """時系列CVの「設定値」と「実効値」を保持し、沈黙クランプを可視化する。"""

    n_races: int
    configured_n_splits: int
    configured_min_train_races: int
    effective_n_splits: int
    effective_min_train_races: int
    n_splits_clamped: bool
    min_train_races_clamped: bool
    val_races_per_fold: float
    feasible: bool

    @property
    def is_clamped(self) -> bool:
        return self.n_splits_clamped or self.min_train_races_clamped


def resolve_cv_plan(n_races: int, n_splits: int, min_train_races: int) -> CVPlan:
    """`time_series_race_splits` のクランプ規則を純粋関数として再現する。

    実装は hybrid_model.time_series_race_splits:213-214 と同じ式を使う。
    実効値と「クランプされたか」を返すことで、設定上の下限が黙って無効化される
    挙動を呼び出し側から検知できるようにする。
    """
    n_races = int(n_races)
    configured_n_splits = int(n_splits)
    configured_min_train_races = int(min_train_races)
    feasible = n_races >= 3

    effective_n_splits = max(1, min(configured_n_splits, n_races - 1)) if n_races >= 2 else 1
    effective_min_train_races = max(1, min(configured_min_train_races, n_races - effective_n_splits))

    n_val_races = max(0, n_races - effective_min_train_races)
    val_races_per_fold = (n_val_races / effective_n_splits) if effective_n_splits > 0 else 0.0

    return CVPlan(
        n_races=n_races,
        configured_n_splits=configured_n_splits,
        configured_min_train_races=configured_min_train_races,
        effective_n_splits=effective_n_splits,
        effective_min_train_races=effective_min_train_races,
        n_splits_clamped=effective_n_splits != configured_n_splits,
        min_train_races_clamped=effective_min_train_races != configured_min_train_races,
        val_races_per_fold=float(val_races_per_fold),
        feasible=feasible,
    )


def _data_regime(n_races: int) -> str:
    if n_races < LOW_DATA_RACE_THRESHOLD:
        return "low"
    if n_races < FULL_DATA_RACE_THRESHOLD:
        return "partial"
    return "full"


def assess_data_sufficiency(history_df: pd.DataFrame, config: Dict[str, object]) -> Dict[str, object]:
    """学習データの十分性を評価し、低信頼フラグと人間向けの警告を返す。

    レース数が設計下限(50)未満、または時系列CVが沈黙クランプされている場合に
    `is_low_confidence=True` とし、何が問題かを `warnings` に文字列で列挙する。
    """
    cv_cfg = dict(config.get("cv", {}) or {})
    configured_n_splits = int(cv_cfg.get("n_splits", 4))
    configured_min_train_races = int(cv_cfg.get("min_train_races", 50))

    if "race_id" in history_df.columns:
        n_races = int(history_df["race_id"].nunique())
    else:
        n_races = 0
    n_rows = int(len(history_df))
    if "finish_rank" in history_df.columns:
        finish = pd.to_numeric(history_df["finish_rank"], errors="coerce")
        n_positives = int((finish <= 3).sum())
    else:
        n_positives = 0

    plan = resolve_cv_plan(n_races, configured_n_splits, configured_min_train_races)
    regime = _data_regime(n_races)

    warnings_list: List[str] = []
    if plan.min_train_races_clamped:
        warnings_list.append(
            f"min_train_races={configured_min_train_races} は実効 {plan.effective_min_train_races} に"
            f"沈黙クランプされました (n_races={n_races})。設定上の下限は強制されていません。"
        )
    if plan.n_splits_clamped:
        warnings_list.append(
            f"n_splits={configured_n_splits} は実効 {plan.effective_n_splits} にクランプされました "
            f"(n_races={n_races})。"
        )
    if regime == "low":
        warnings_list.append(
            f"n_races={n_races} は設計下限 {LOW_DATA_RACE_THRESHOLD} を下回ります。"
            "アンサンブルは分類器単体に縮退し、報告メトリクスはノイズが支配的です。"
        )
    if plan.feasible and plan.val_races_per_fold < 2.0:
        warnings_list.append(
            f"検証fold あたり {plan.val_races_per_fold:.2f} レースしかありません。"
            "メトリクスは統計的にほぼ無意味です。"
        )
    if not plan.feasible:
        warnings_list.append(
            f"n_races={n_races} は時系列CVの最小要件(3)未満です。検証できません。"
        )

    is_low_confidence = (regime == "low") or plan.is_clamped or (not plan.feasible)

    return {
        "n_races": n_races,
        "n_rows": n_rows,
        "n_positives": n_positives,
        "regime": regime,
        "is_low_confidence": is_low_confidence,
        "feasible": plan.feasible,
        "configured_n_splits": plan.configured_n_splits,
        "configured_min_train_races": plan.configured_min_train_races,
        "effective_n_splits": plan.effective_n_splits,
        "effective_min_train_races": plan.effective_min_train_races,
        "val_races_per_fold": plan.val_races_per_fold,
        "warnings": warnings_list,
    }


def _favorite_key(df: pd.DataFrame) -> Optional[pd.Series]:
    """軸(本命)を選ぶためのキー。小さいほど人気上位。

    popularity を最優先、無ければ odds を使う。両方無ければ None。
    """
    if "popularity" in df.columns:
        pop = pd.to_numeric(df["popularity"], errors="coerce")
        if pop.notna().any():
            return pop
    if "odds" in df.columns:
        odds = pd.to_numeric(df["odds"], errors="coerce")
        if odds.notna().any():
            return odds
    return None


def compute_favorite_baseline(history_df: pd.DataFrame, n_partners: int = 5) -> Dict[str, object]:
    """「1番人気追随」ベースラインを計算する。

    軸 = 各レースの最人気馬(popularity最小、無ければodds最小)。
    これがTop3に入った割合 (axis_top3_hit_rate) と、上位n_partners頭で実際の
    Top3を何割カバーしたか (favorite_top3_cover_rate) を返す。
    モデルの「高精度」はこのベースラインへの上乗せでしか意味を持たない。
    """
    if "finish_rank" not in history_df.columns:
        raise ValueError("compute_favorite_baseline には finish_rank が必要です。")

    hit_flags: List[bool] = []
    cover_rates: List[float] = []
    cover_counts: List[float] = []

    for _, group in history_df.groupby("race_id"):
        key = _favorite_key(group)
        if key is None or not key.notna().any():
            continue
        finish = pd.to_numeric(group["finish_rank"], errors="coerce")
        axis_idx = key.idxmin()
        axis_finish = finish.loc[axis_idx]
        if pd.isna(axis_finish):
            continue
        hit_flags.append(bool(axis_finish <= 3))

        field_size = len(group)
        denom = min(3, field_size)
        top_n_idx = key.nsmallest(n_partners).index
        covered = int((finish.loc[top_n_idx] <= 3).sum())
        cover_rates.append(float(covered / denom) if denom > 0 else 0.0)
        cover_counts.append(float(covered))

    n_races = len(hit_flags)
    return {
        "axis_top3_hit_rate": float(np.mean(hit_flags)) if hit_flags else 0.0,
        "favorite_top3_cover_rate": float(np.mean(cover_rates)) if cover_rates else 0.0,
        "favorite_top3_cover_count_mean": float(np.mean(cover_counts)) if cover_counts else 0.0,
        "n_partners": int(n_partners),
        "n_races": int(n_races),
    }


def _axis_top3_hit_rate(df: pd.DataFrame, score_col: str) -> float:
    """score_col が各レース最大の馬(=軸)がTop3に入った割合。"""
    if score_col not in df.columns:
        return float("nan")
    flags: List[bool] = []
    for _, group in df.groupby("race_id"):
        score = pd.to_numeric(group[score_col], errors="coerce")
        if not score.notna().any():
            continue
        axis_idx = score.idxmax()
        finish = pd.to_numeric(group["finish_rank"], errors="coerce").loc[axis_idx]
        if pd.isna(finish):
            continue
        flags.append(bool(finish <= 3))
    return float(np.mean(flags)) if flags else float("nan")


def summarize_edge(
    oof_df: pd.DataFrame,
    favorite_baseline: Optional[Dict[str, object]] = None,
    n_partners: int = 5,
) -> Dict[str, object]:
    """モデル / オッズ抜きシャドウ / 1番人気 の軸的中率と上乗せ(lift)を要約する。

    - lift_model_over_favorite: オッズ使用モデルが人気馬追随をどれだけ上回るか。
    - lift_shadow_over_favorite: オッズ"抜き"モデルが人気馬を上回るか
      (= 市場を再現しているだけでない「本物のエッジ」の代理指標)。
    """
    model_axis = _axis_top3_hit_rate(oof_df, "ensemble_top3_prob")
    shadow_axis = _axis_top3_hit_rate(oof_df, "shadow_no_odds_top3_prob")

    if favorite_baseline is None:
        favorite_baseline = compute_favorite_baseline(oof_df, n_partners=n_partners)
    favorite_axis = float(favorite_baseline.get("axis_top3_hit_rate", float("nan")))

    def _lift(a: float, b: float) -> float:
        if np.isnan(a) or np.isnan(b):
            return float("nan")
        return float(a - b)

    return {
        "model_axis_top3_hit_rate": model_axis,
        "shadow_axis_top3_hit_rate": shadow_axis,
        "favorite_axis_top3_hit_rate": favorite_axis,
        "lift_model_over_favorite": _lift(model_axis, favorite_axis),
        "lift_shadow_over_favorite": _lift(shadow_axis, favorite_axis),
        "lift_model_over_shadow": _lift(model_axis, shadow_axis),
        "n_races": int(favorite_baseline.get("n_races", 0)),
    }


def market_implied_top3(df: pd.DataFrame) -> np.ndarray:
    """オッズ(無ければ人気)から各馬のTop3確率の市場推定値を計算する。

    レース内で正規化し、合計がおよそ min(3, 出走頭数) になるようスケールする。
    市場は近オラクルなので、これはモデルが希薄データ時に縮約すべき強い事前分布。
    """
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=float)
    odds = pd.to_numeric(df.get("odds", pd.Series([np.nan] * n, index=df.index)), errors="coerce")
    implied = (1.0 / odds.where(odds > 1.0)).fillna(0.0)

    if "popularity" in df.columns:
        pop = pd.to_numeric(df["popularity"], errors="coerce")
        max_pop = pop.groupby(df["race_id"]).transform("max")
        pop_score = (1.0 - (pop - 1.0) / max_pop.replace(0, np.nan)).fillna(0.0)
        implied = implied.where(implied > 0, pop_score)

    race_sum = implied.groupby(df["race_id"]).transform("sum").replace(0, np.nan)
    if "field_size" in df.columns:
        field = pd.to_numeric(df["field_size"], errors="coerce")
    else:
        field = df.groupby("race_id")["race_id"].transform("count")
    top_k = np.minimum(3.0, field)
    prob = ((implied / race_sum) * top_k).fillna(0.0)
    return np.clip(prob.to_numpy(dtype=float), 0.0, 1.0)


def blend_with_market(
    model_prob: Sequence[float],
    market_prob: Sequence[float],
    weight: float,
) -> np.ndarray:
    """モデル確率と市場確率の凸結合 (1-weight)*model + weight*market。"""
    w = float(np.clip(weight, 0.0, 1.0))
    model_arr = np.asarray(model_prob, dtype=float)
    market_arr = np.asarray(market_prob, dtype=float)
    return (1.0 - w) * model_arr + w * market_arr


# regime 別の既定縮約重み。希薄なほど市場へ強く寄せる。
_DEFAULT_SHRINKAGE_WEIGHTS = {"low": 0.6, "partial": 0.3, "full": 0.0}


def market_shrinkage_weight(data_sufficiency: Dict[str, object], config: Dict[str, object]) -> float:
    """データ十分性(regime)に応じた市場縮約重みを返す。

    config["model"]["blending"]["market_shrinkage_weights"] で上書き可能。
    希薄データ(low)では市場へ強く寄せ、データが揃う(full)と寄せない。
    """
    weights = dict(_DEFAULT_SHRINKAGE_WEIGHTS)
    override = (
        dict(config.get("model", {}) or {})
        .get("blending", {})
        .get("market_shrinkage_weights", {})
        if isinstance(config, dict)
        else {}
    )
    weights.update(override or {})
    regime = str(data_sufficiency.get("regime", "full"))
    return float(weights.get(regime, 0.0))


def assert_no_result_leakage(feature_cols: Sequence[str]) -> None:
    """特徴量列に post-race の `result_*` 列が紛れ込んでいないか検査する。

    紛れ込んでいれば ValueError を送出する。学習に未来情報を混入させないための
    明示的なガード。
    """
    offending = [
        str(col)
        for col in feature_cols
        if str(col).lower().startswith(RESULT_COLUMN_PREFIX)
    ]
    if offending:
        raise ValueError(
            "特徴量に post-race の result_* 列が含まれています (データリーク): "
            + ", ".join(offending)
        )
