from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    from .features import (
        AGGREGATE_FEATURE_COLUMNS,
        ODDS_RELATED_FEATURES,
        RAW_MODEL_FEATURES,
        SPARSE_CONTEXT_FEATURE_COLUMNS,
        add_aggregate_features,
        add_basic_features,
        add_targets,
        compute_similarity_weights,
        get_distance_band,
        is_turf_value,
    )
    from .hybrid_model import evaluate_time_series_cv
except ImportError:  # pragma: no cover - direct script fallback
    from features import (
        AGGREGATE_FEATURE_COLUMNS,
        ODDS_RELATED_FEATURES,
        RAW_MODEL_FEATURES,
        SPARSE_CONTEXT_FEATURE_COLUMNS,
        add_aggregate_features,
        add_basic_features,
        add_targets,
        compute_similarity_weights,
        get_distance_band,
        is_turf_value,
    )
    from hybrid_model import evaluate_time_series_cv

try:
    from .validation import (
        assert_no_result_leakage,
        assess_data_sufficiency,
        compute_favorite_baseline,
        summarize_edge,
    )
except ImportError:  # pragma: no cover - direct script fallback
    from validation import (
        assert_no_result_leakage,
        assess_data_sufficiency,
        compute_favorite_baseline,
        summarize_edge,
    )


def _infer_feature_cols_from_history(
    history_df: pd.DataFrame,
    use_odds: bool,
) -> tuple[List[str], List[str]]:
    """entry なしで history のみから feature_cols を推定する。"""
    candidates = list(RAW_MODEL_FEATURES) + list(AGGREGATE_FEATURE_COLUMNS) + list(SPARSE_CONTEXT_FEATURE_COLUMNS)
    feature_cols = [c for c in candidates if c in history_df.columns and history_df[c].notna().any()]
    assert_no_result_leakage(feature_cols)
    if not use_odds:
        no_odds_cols = feature_cols
        feature_cols_with_odds: List[str] = []
        return no_odds_cols, no_odds_cols
    no_odds_cols = [c for c in feature_cols if c not in ODDS_RELATED_FEATURES]
    return feature_cols, no_odds_cols


def _prepare_history(
    history_df: pd.DataFrame,
    config: Dict[str, object],
) -> tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """history_df に前処理を施して (history, sample_weight, feature_cols, no_odds_feature_cols) を返す。"""
    df = history_df.copy()
    df = add_targets(df)
    df = df.dropna(subset=["finish_percentile", "is_top3"]).reset_index(drop=True)

    # entry が不要な特徴量処理のため、空の entry を渡す
    empty_entry = pd.DataFrame(columns=df.columns).astype(df.dtypes)
    df, _ = add_basic_features(df, empty_entry)
    df, _ = add_aggregate_features(df, empty_entry)

    feature_cols, no_odds_feature_cols = _infer_feature_cols_from_history(df, use_odds=True)
    sample_weight = compute_similarity_weights(df, config)
    return df, sample_weight, feature_cols, no_odds_feature_cols


def _run_cv_and_get_oof(
    history_df: pd.DataFrame,
    config: Dict[str, object],
) -> pd.DataFrame:
    """時系列CVを実行して OOF 予測 DataFrame を返す。

    戻り値の列: race_id, finish_rank, is_top3, ensemble_top3_prob, ...
    """
    df, sample_weight, feature_cols, no_odds_feature_cols = _prepare_history(history_df, config)
    eval_result = evaluate_time_series_cv(df, feature_cols, no_odds_feature_cols, sample_weight, config)
    oof_df: pd.DataFrame = eval_result["oof_predictions"]
    return oof_df


def compute_axis_baseline(
    history_df: pd.DataFrame,
    config: Dict[str, object],
) -> Dict[str, object]:
    """現行の7基準軸選定ロジック（ensemble_top3_prob 最大馬）でのTop3的中率を計算する。

    Returns:
        dict with keys:
            axis_top3_hit_rate: float - 軸馬がTop3に入った割合
            n_races: int - 評価レース数
    """
    oof_df = _run_cv_and_get_oof(history_df, config)

    hit_flags: List[bool] = []
    for race_id, group in oof_df.groupby("race_id"):
        if "ensemble_top3_prob" not in group.columns:
            continue
        axis_idx = group["ensemble_top3_prob"].idxmax()
        axis_row = group.loc[axis_idx]
        is_hit = bool(int(axis_row["finish_rank"]) <= 3)
        hit_flags.append(is_hit)

    n_races = len(hit_flags)
    axis_top3_hit_rate = float(np.mean(hit_flags)) if hit_flags else 0.0

    return {
        "axis_top3_hit_rate": axis_top3_hit_rate,
        "n_races": n_races,
    }


def compute_partner_baseline(
    history_df: pd.DataFrame,
    config: Dict[str, object],
    n_partners: int = 5,
) -> Dict[str, object]:
    """上位N頭の単純切り出しでactual Top3をカバーした割合を計算する。

    Returns:
        dict with keys:
            top5_cover_rate: float - 上位N頭でTop3を全体の何割カバーしたか（レース平均）
            top5_cover_count_mean: float - 上位N頭に含まれるactual Top3頭数の平均
    """
    oof_df = _run_cv_and_get_oof(history_df, config)

    cover_rates: List[float] = []
    cover_counts: List[float] = []
    for race_id, group in oof_df.groupby("race_id"):
        if "ensemble_top3_prob" not in group.columns:
            continue
        top_n = group.nlargest(n_partners, "ensemble_top3_prob")
        actual_top3_count = int((group["finish_rank"] <= 3).sum())
        covered = int((top_n["finish_rank"] <= 3).sum())
        cover_count = float(covered)
        cover_rate = float(covered / actual_top3_count) if actual_top3_count > 0 else 0.0
        cover_rates.append(cover_rate)
        cover_counts.append(cover_count)

    top5_cover_rate = float(np.mean(cover_rates)) if cover_rates else 0.0
    top5_cover_count_mean = float(np.mean(cover_counts)) if cover_counts else 0.0

    return {
        "top5_cover_rate": top5_cover_rate,
        "top5_cover_count_mean": top5_cover_count_mean,
    }


def _normalize_class_bucket(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    if any(token in text for token in ["g1", "g2", "g3", "重賞", "graded", "listed"]):
        return "graded"
    if any(token in text for token in ["open", "オープン"]):
        return "open"
    if any(token in text for token in ["maiden", "未勝利", "newcomer", "新馬"]):
        return "maiden"
    return "allowance"


def _normalize_surface_bucket(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    return "turf" if is_turf_value(value) else "dirt"


def _build_race_level_backtest_rows(
    prepared_history: pd.DataFrame,
    oof_df: pd.DataFrame,
    n_partners: int,
) -> pd.DataFrame:
    race_meta_cols = [col for col in ["race_id", "surface", "distance", "class"] if col in prepared_history.columns]
    race_meta = prepared_history[race_meta_cols].drop_duplicates(subset=["race_id"]).copy()
    merged = oof_df.merge(race_meta, on="race_id", how="left")

    rows: list[dict[str, object]] = []
    for race_id, group in merged.groupby("race_id"):
        if "ensemble_top3_prob" not in group.columns:
            continue
        axis_idx = group["ensemble_top3_prob"].idxmax()
        axis_row = group.loc[axis_idx]
        top_n = group.nlargest(n_partners, "ensemble_top3_prob")
        actual_top3_count = int((group["finish_rank"] <= 3).sum())
        covered = int((top_n["finish_rank"] <= 3).sum())
        distance = pd.to_numeric(pd.Series([axis_row.get("distance")]), errors="coerce").iloc[0]
        rows.append(
            {
                "race_id": race_id,
                "axis_hit_top3": bool(int(axis_row["finish_rank"]) <= 3),
                "cover_rate": float(covered / actual_top3_count) if actual_top3_count > 0 else 0.0,
                "distance_band": get_distance_band(distance) or "unknown",
                "class_bucket": _normalize_class_bucket(axis_row.get("class")),
                "surface_bucket": _normalize_surface_bucket(axis_row.get("surface")),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_condition_metrics(metrics_df: pd.DataFrame, column: str) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, group in metrics_df.groupby(column):
        result[str(key)] = {
            "axis_top3_hit_rate": float(group["axis_hit_top3"].mean()),
            "cover_rate": float(group["cover_rate"].mean()),
            "n_races": int(len(group)),
        }
    return result


def compute_baselines_by_condition(
    history_df: pd.DataFrame,
    config: Dict[str, object],
    n_partners: int = 5,
) -> Dict[str, object]:
    prepared_history, sample_weight, feature_cols, no_odds_feature_cols = _prepare_history(history_df, config)
    eval_result = evaluate_time_series_cv(prepared_history, feature_cols, no_odds_feature_cols, sample_weight, config)
    oof_df: pd.DataFrame = eval_result["oof_predictions"]
    metrics_df = _build_race_level_backtest_rows(prepared_history, oof_df, n_partners=n_partners)
    if len(metrics_df) == 0:
        return {"distance_band": {}, "class": {}, "surface": {}}

    return {
        "distance_band": _aggregate_condition_metrics(metrics_df, "distance_band"),
        "class": _aggregate_condition_metrics(metrics_df, "class_bucket"),
        "surface": _aggregate_condition_metrics(metrics_df, "surface_bucket"),
    }


def save_baselines(baselines: Dict[str, object], path: Union[str, Path]) -> None:
    """ベースライン結果を JSON ファイルに保存する。"""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(baselines, f, ensure_ascii=False, indent=2)


def _is_synthetic_baseline(baseline_dict: Dict[str, object]) -> bool:
    """ベースラインが合成データ由来のプレースホルダかどうかを判定する。"""
    if baseline_dict.get("is_synthetic") is True:
        return True
    note = str(baseline_dict.get("_note", "")).lower()
    return "synthetic" in note or "合成" in note


def _axis_top3_hit_rate_from_oof(oof_df: pd.DataFrame) -> tuple[float, int]:
    """OOF 予測から軸(ensemble_top3_prob最大馬)のTop3的中率とレース数を返す。"""
    hit_flags: List[bool] = []
    for _, group in oof_df.groupby("race_id"):
        if "ensemble_top3_prob" not in group.columns:
            continue
        axis_row = group.loc[group["ensemble_top3_prob"].idxmax()]
        hit_flags.append(bool(int(axis_row["finish_rank"]) <= 3))
    return (float(np.mean(hit_flags)) if hit_flags else 0.0), len(hit_flags)


def _partner_cover_from_oof(oof_df: pd.DataFrame, n_partners: int) -> tuple[float, float]:
    """OOF 予測から上位N頭のTop3カバー率・平均カバー頭数を返す。"""
    cover_rates: List[float] = []
    cover_counts: List[float] = []
    for _, group in oof_df.groupby("race_id"):
        if "ensemble_top3_prob" not in group.columns:
            continue
        top_n = group.nlargest(n_partners, "ensemble_top3_prob")
        actual_top3 = int((group["finish_rank"] <= 3).sum())
        covered = int((top_n["finish_rank"] <= 3).sum())
        cover_rates.append(float(covered / actual_top3) if actual_top3 > 0 else 0.0)
        cover_counts.append(float(covered))
    return (
        float(np.mean(cover_rates)) if cover_rates else 0.0,
        float(np.mean(cover_counts)) if cover_counts else 0.0,
    )


def generate_real_baselines(
    history_df: pd.DataFrame,
    config: Dict[str, object],
    n_partners: int = 5,
    source_file: Optional[Union[str, Path]] = None,
    generated_at: Optional[str] = None,
) -> Dict[str, object]:
    """実データから正直なベースラインを1回のCV実行で生成する。

    合成プレースホルダを置き換えるためのもの。次を束ねる:
    - axis_top3_hit_rate / top{n}_cover (現行ロジック)
    - favorite_baseline (1番人気追随ベースライン)
    - edge_vs_market (モデル vs シャドウ vs 人気馬の上乗せ)
    - data_sufficiency (レース数・低信頼フラグ・警告)
    - メタデータ (is_synthetic=False, n_races, date_range, source, generated_at)
    """
    prepared, sample_weight, feature_cols, no_odds_feature_cols = _prepare_history(history_df, config)
    eval_result = evaluate_time_series_cv(prepared, feature_cols, no_odds_feature_cols, sample_weight, config)
    oof_df: pd.DataFrame = eval_result["oof_predictions"]
    data_sufficiency = eval_result.get("data_sufficiency") or assess_data_sufficiency(prepared, config)

    axis_hit_rate, n_eval_races = _axis_top3_hit_rate_from_oof(oof_df)
    cover_rate, cover_count_mean = _partner_cover_from_oof(oof_df, n_partners)
    favorite = compute_favorite_baseline(prepared, n_partners=n_partners)
    edge = summarize_edge(oof_df, favorite_baseline=favorite, n_partners=n_partners)

    date_range: Dict[str, Optional[str]] = {"start": None, "end": None}
    if "race_date" in prepared.columns:
        dates = pd.to_datetime(prepared["race_date"], errors="coerce").dropna()
        if not dates.empty:
            date_range = {"start": str(dates.min().date()), "end": str(dates.max().date())}

    if generated_at is None:
        generated_at = datetime.now(timezone.utc).isoformat()

    return {
        "is_synthetic": False,
        "generated_at": generated_at,
        "source_file": str(source_file) if source_file is not None else None,
        "n_races": int(data_sufficiency.get("n_races", n_eval_races)),
        "n_eval_races": int(n_eval_races),
        "date_range": date_range,
        "axis_top3_hit_rate": float(axis_hit_rate),
        f"top{n_partners}_cover_rate": float(cover_rate),
        f"top{n_partners}_cover_count_mean": float(cover_count_mean),
        "favorite_baseline": favorite,
        "edge_vs_market": edge,
        "data_sufficiency": data_sufficiency,
    }


def compare_with_baseline(
    new_metrics: Dict[str, object],
    baseline: Union[Dict[str, object], str, Path],
) -> Dict[str, object]:
    """新しい指標をベースラインと比較して改善率・改善有無を返す。

    Args:
        new_metrics: 新しい評価指標の dict
        baseline: ベースラインの dict、またはベースライン JSON ファイルのパス

    Returns:
        各指標について {key_improvement_pct: float, key_improved: bool} を含む dict。
        ベースラインファイルが存在しない場合は空の dict を返す（警告のみ）。
    """
    if isinstance(baseline, (str, Path)):
        baseline_path = Path(baseline)
        if not baseline_path.exists():
            warnings.warn(
                f"ベースラインファイルが見つかりません: {baseline_path}。比較をスキップします。",
                UserWarning,
                stacklevel=2,
            )
            return {}
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_dict: Dict[str, object] = json.load(f)
    else:
        baseline_dict = dict(baseline)

    if _is_synthetic_baseline(baseline_dict):
        warnings.warn(
            "ベースラインが synthetic(合成データ)プレースホルダです。実データから再生成してください "
            "(scripts/generate_baselines.py)。比較をスキップします。",
            UserWarning,
            stacklevel=2,
        )
        return {}

    result: Dict[str, object] = {}
    for key, new_val in new_metrics.items():
        if key not in baseline_dict:
            continue
        old_val = baseline_dict[key]
        try:
            new_f = float(new_val)  # type: ignore[arg-type]
            old_f = float(old_val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if old_f == 0.0:
            improvement_pct = float("inf") if new_f > 0 else 0.0
        else:
            improvement_pct = (new_f - old_f) / abs(old_f) * 100.0
        result[f"{key}_improvement_pct"] = improvement_pct
        result[f"{key}_improved"] = bool(new_f > old_f)

    return result
