"""Walk-forward backtest framework for comparing prediction models.

Baselines:
  - market_baseline: Implied probability from odds (Plato-Luce normalized)
  - single_ranker:   Simple CatBoost Ranker trained on prior races only

Usage:
    uv run python scripts/backtest.py --data data/training/race_results_master.csv
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    metrics_by_model: Dict[str, Dict[str, float]]
    fold_details: List[Dict[str, object]] = field(default_factory=list)
    n_folds: int = 0


# ---------------------------------------------------------------------------
# Market baseline
# ---------------------------------------------------------------------------

def compute_market_baseline_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """オッズから implied probability を計算し、top3確率として正規化する。

    オッズ欠損馬は人気順 rank_score で代替する。
    """
    out = df[["race_id", "horse_id"]].copy()
    odds = pd.to_numeric(df.get("odds", pd.Series(dtype=float)), errors="coerce")
    implied = (1.0 / odds.where(odds > 1.0)).fillna(0.0)

    # Fallback: popularity rank score
    if "popularity" in df.columns:
        pop = pd.to_numeric(df["popularity"], errors="coerce")
        max_pop = pop.groupby(df["race_id"]).transform("max")
        pop_score = (1.0 - (pop - 1) / max_pop.replace(0, np.nan)).fillna(0.0)
        implied = implied.where(implied > 0, pop_score)

    # Softmax-like normalization within each race → scale so group sum ≈ min(3, n)
    race_sum = implied.groupby(df["race_id"]).transform("sum").replace(0, np.nan)
    field_size = pd.to_numeric(df.get("field_size", df.groupby(df["race_id"])["race_id"].transform("count")), errors="coerce")
    top_k = np.minimum(3.0, field_size)
    out["top3_prob"] = np.clip((implied / race_sum) * top_k, 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Single ranker baseline
# ---------------------------------------------------------------------------

def compute_single_ranker_predictions(
    history: pd.DataFrame,
    entry: pd.DataFrame,
) -> pd.DataFrame:
    """履歴データで CatBoost Ranker を学習し、entry の top3 確率を推定する。

    特徴量: 利用可能な数値カラムのみ使用。
    オッズ欠損はフィールド内中央値で補完。
    """
    try:
        from keiba_predictor.features import (
            add_aggregate_features,
            add_basic_features,
            add_targets,
            infer_feature_columns,
        )
        from keiba_predictor.hybrid_model import fit_ranker_model
        from keiba_predictor.simulation import (
            normalize_scores_by_group,
            plackett_luce_group_topk_probabilities,
        )
    except ImportError as e:
        logger.warning("keiba_predictor not available: %s. Falling back to market baseline.", e)
        return compute_market_baseline_predictions(entry)

    if len(history) < 20:
        logger.warning("学習データ不足 (%d行)。market baseline で代替。", len(history))
        return compute_market_baseline_predictions(entry)

    try:
        history_w_targets = add_targets(history.copy())
        history_feats, entry_feats = add_basic_features(history_w_targets, entry.copy())
        history_feats, entry_feats = add_aggregate_features(history_feats, entry_feats)
        feat_cols = infer_feature_columns(history_feats, entry_feats, use_odds=True)

        config: dict = {
            "model": {
                "ranker": {
                    "loss_function": "QueryRMSE",
                    "eval_metric": "NDCG:top=3",
                    "iterations": 100,
                    "learning_rate": 0.05,
                    "depth": 4,
                    "l2_leaf_reg": 8.0,
                    "random_strength": 1.0,
                    "min_data_in_leaf": 3,
                    "bootstrap_type": "Bernoulli",
                    "subsample": 0.8,
                    "thread_count": -1,
                }
            }
        }
        weights = np.ones(len(history_feats))
        ranker = fit_ranker_model(
            train_df=history_feats,
            feature_cols=feat_cols,
            sample_weight=weights,
            random_state=42,
            config=config,
        )
        race_ids = entry_feats["race_id"].astype(str).to_numpy()
        scores = ranker.predict_scores(entry_feats)
        strength = normalize_scores_by_group(race_ids, scores)
        top3_prob = plackett_luce_group_topk_probabilities(
            group_ids=race_ids,
            strengths=strength,
            topk=3,
            temperature=1.0,
            n_trials=2000,
            seed=42,
            block_size=2000,
        )
        out = entry[["race_id", "horse_id"]].copy()
        out["top3_prob"] = np.clip(top3_prob, 0.0, 1.0)
        return out
    except Exception as e:
        logger.warning("Ranker 学習失敗: %s。market baseline で代替。", e)
        return compute_market_baseline_predictions(entry)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(df: pd.DataFrame) -> Dict[str, float]:
    """予測 DataFrame から評価指標を計算する。

    df には以下が必要:
      - race_id, horse_id, top3_prob, finish_rank (または is_top3)
    """
    if "is_top3" not in df.columns:
        df = df.copy()
        df["is_top3"] = (df["finish_rank"] <= 3).astype(int)

    race_ids = df["race_id"].unique()
    n_races = len(race_ids)
    is_top3 = df["is_top3"].to_numpy(dtype=float)
    top3_prob = df["top3_prob"].to_numpy(dtype=float)

    brier = float(np.mean((top3_prob - is_top3) ** 2))

    # Top-3 precision: 各レースで top3_prob 上位3頭の正解率
    precisions = []
    for rid in race_ids:
        mask = df["race_id"] == rid
        race = df[mask].sort_values("top3_prob", ascending=False)
        n_sel = min(3, len(race))
        selected = race.head(n_sel)
        actual_top3 = int((selected["is_top3"] == 1).sum())
        precisions.append(actual_top3 / n_sel if n_sel > 0 else 0.0)

    return {
        "top3_brier_score": brier,
        "top3_precision": float(np.mean(precisions)),
        "n_races": n_races,
    }


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

def walk_forward_backtest(
    history: pd.DataFrame,
    n_splits: int = 5,
    min_train_races: int = 10,
    models: Optional[List[str]] = None,
) -> WalkForwardResult:
    """Walk-forward バックテストを実行する。

    history を時系列分割し、各 fold で指定モデルの予測精度を評価する。
    """
    if models is None:
        models = ["market_baseline", "single_ranker"]

    # race_id を日付順でソートして fold を作成
    race_dates = (
        history.groupby("race_id")["race_date"]
        .min()
        .sort_values()
        .reset_index()
    )
    n_races = len(race_dates)
    if n_races < min_train_races + 1:
        raise ValueError(f"バックテストに必要なレース数不足: {n_races} < {min_train_races + 1}")

    # 等間隔分割
    fold_boundaries: List[int] = []
    val_per_fold = max(1, (n_races - min_train_races) // n_splits)
    for i in range(n_splits):
        split_point = min_train_races + i * val_per_fold
        if split_point >= n_races:
            break
        fold_boundaries.append(split_point)

    actual_n_folds = len(fold_boundaries)
    all_metrics: Dict[str, List[Dict[str, float]]] = {m: [] for m in models}
    fold_details: List[Dict[str, object]] = []

    for fold_idx, split_point in enumerate(fold_boundaries):
        train_race_ids = set(race_dates.iloc[:split_point]["race_id"])
        val_start = split_point
        val_end = min(split_point + val_per_fold, n_races)
        val_race_ids = set(race_dates.iloc[val_start:val_end]["race_id"])

        train_df = history[history["race_id"].isin(train_race_ids)].copy()
        val_df = history[history["race_id"].isin(val_race_ids)].copy()

        if len(val_df) == 0:
            continue

        fold_info: Dict[str, object] = {
            "fold": fold_idx,
            "n_train_races": len(train_race_ids),
            "n_val_races": len(val_race_ids),
        }

        for model_name in models:
            try:
                if model_name == "market_baseline":
                    preds = compute_market_baseline_predictions(val_df)
                elif model_name == "single_ranker":
                    preds = compute_single_ranker_predictions(history=train_df, entry=val_df)
                else:
                    logger.warning("未知モデル: %s", model_name)
                    continue

                merged = val_df.merge(
                    preds[["race_id", "horse_id", "top3_prob"]],
                    on=["race_id", "horse_id"],
                    how="left",
                )
                merged["top3_prob"] = merged["top3_prob"].fillna(0.3)
                metrics = evaluate_predictions(merged)
                all_metrics[model_name].append(metrics)
                fold_info[f"{model_name}_brier"] = metrics["top3_brier_score"]
            except Exception as e:
                logger.warning("Fold %d / %s 評価失敗: %s", fold_idx, model_name, e)

        fold_details.append(fold_info)

    # fold 平均
    aggregated: Dict[str, Dict[str, float]] = {}
    for model_name, fold_metrics in all_metrics.items():
        if not fold_metrics:
            continue
        keys = fold_metrics[0].keys()
        aggregated[model_name] = {
            k: float(np.mean([fm[k] for fm in fold_metrics]))
            for k in keys
        }

    return WalkForwardResult(
        metrics_by_model=aggregated,
        fold_details=fold_details,
        n_folds=actual_n_folds,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument("--data", required=True, help="Path to race_results_master.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-train-races", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv(args.data)
    if "race_date" in df.columns:
        df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")

    result = walk_forward_backtest(
        df,
        n_splits=args.n_splits,
        min_train_races=args.min_train_races,
        models=["market_baseline", "single_ranker"],
    )

    print("\n=== Walk-Forward Backtest Results ===")
    for model_name, metrics in result.metrics_by_model.items():
        print(f"\n[{model_name}]")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    print(f"\nFolds evaluated: {result.n_folds}")
