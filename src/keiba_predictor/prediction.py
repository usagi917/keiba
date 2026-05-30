from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import yaml

try:
    from .data_loader import DataValidationError, load_datasets, prepare_datasets
    from .features import add_aggregate_features, add_basic_features, add_targets, compute_similarity_weights, infer_feature_columns
    from .hybrid_model import (
        evaluate_time_series_cv,
        fit_hybrid_race_model,
        get_feature_importance,
        predict_hybrid_model,
    )
    from .simulation import compute_tail_risk, conditional_top3_probs
    from .validation import summarize_edge
except ImportError:  # pragma: no cover - direct script fallback
    from data_loader import DataValidationError, load_datasets, prepare_datasets
    from features import add_aggregate_features, add_basic_features, add_targets, compute_similarity_weights, infer_feature_columns
    from hybrid_model import (
        evaluate_time_series_cv,
        fit_hybrid_race_model,
        get_feature_importance,
        predict_hybrid_model,
    )
    from simulation import compute_tail_risk, conditional_top3_probs
    from validation import summarize_edge


DISPLAY_COLS = [
    "horse_display_name",
    "consensus_top3_score",
    "win_prob",
    "top2_prob",
    "top3_prob",
    "regression_top3_prob",
    "mean_rank",
    "top3_ci_low",
    "top3_ci_high",
    "calibrated_top3_prob",
    "classifier_top3_prob",
    "aux_classifier_top3_prob",
    "shadow_no_odds_top3_prob",
    "component_model_std",
    "rank_stability_std",
]


def parse_bool(value: str) -> bool:
    value_l = str(value).strip().lower()
    if value_l in {"true", "1", "yes", "y"}:
        return True
    if value_l in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"bool値に解釈できません: {value}")


def deep_merge_dicts(base: Dict[str, object], override: Dict[str, object]) -> Dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def normalize_config_override(raw: Dict[str, object]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    profile = dict(raw.get("target_race_profile", {}) or {})
    alias_map = {
        "race_name": "name",
        "name": "name",
        "race_date": "race_date",
        "surface": "surface",
        "distance": "distance",
        "course": "course",
        "turn": "turn",
        "grade": "grade",
        "class": "class",
        "season_label": "season_label",
        "season_months": "season_months",
    }
    for source_key, target_key in alias_map.items():
        if source_key in raw and target_key not in profile:
            profile[target_key] = raw[source_key]
    if profile:
        normalized["target_race_profile"] = profile
    if "similarity_weights" in raw:
        normalized["similarity_weights"] = raw["similarity_weights"]
    if "use_odds" in raw:
        normalized["use_odds"] = raw["use_odds"]
    return normalized


def load_config(path: str | Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config は mapping である必要があります: {path}")
    return loaded


def ensure_output_dir(base_dir: str | Path) -> Path:
    outdir = Path(base_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def get_display_name(df: pd.DataFrame) -> pd.Series:
    if "horse_name" in df.columns and df["horse_name"].notna().any():
        return df["horse_name"].astype("string").fillna(df["horse_id"].astype("string"))
    return df["horse_id"].astype("string")


def configure_matplotlib_fonts() -> None:
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    candidates = [
        "Hiragino Sans",
        "YuGothic",
        "Hiragino Kaku Gothic ProN",
        "Osaka",
    ]
    selected_font = next((name for name in candidates if name in available_fonts), None)
    if selected_font is not None:
        plt.rcParams["font.family"] = selected_font
    plt.rcParams["axes.unicode_minus"] = False


def _prepare_top3_dashboard_frame(pred_df: pd.DataFrame) -> pd.DataFrame:
    ordered = pred_df.copy()
    if "consensus_top3_score" in ordered.columns:
        ordered["published_top3_prob"] = pd.to_numeric(ordered["consensus_top3_score"], errors="coerce")
    else:
        ordered["published_top3_prob"] = pd.to_numeric(ordered["top3_prob"], errors="coerce")
    return ordered.sort_values(["published_top3_prob", "win_prob"], ascending=[False, False]).reset_index(drop=True)


def save_top3_bar_chart(pred_df: pd.DataFrame, outdir: Path, axis_horse_id: str, race_name: str = "") -> Path:
    ordered = _prepare_top3_dashboard_frame(pred_df).head(12)
    plot_df = ordered.iloc[::-1].copy()
    plot_df["label"] = plot_df["horse_number"].astype(int).astype(str) + " " + plot_df["horse_display_name"].astype(str)

    axis_horse_key = str(axis_horse_id)
    bar_colors = ["#d1495b" if str(horse_id) == axis_horse_key else "#2f6db3" for horse_id in plot_df["horse_id"]]
    selected_model = "consensus_top3_score"
    if "selected_top3_model" in ordered.columns and ordered["selected_top3_model"].notna().any():
        selected_model = str(ordered["selected_top3_model"].dropna().iloc[0])

    fig = plt.figure(figsize=(15.5, 8))
    grid = GridSpec(1, 2, figure=fig, width_ratios=[2.0, 1.45], wspace=0.08)
    ax = fig.add_subplot(grid[0, 0])
    ax_table = fig.add_subplot(grid[0, 1])

    ax.barh(plot_df["label"], plot_df["published_top3_prob"], color=bar_colors, alpha=0.92, label="最終Top3確率")
    ax.scatter(plot_df["win_prob"], plot_df["label"], color="#f28e2b", s=55, zorder=3, label="勝利確率")

    for _, row in plot_df.iterrows():
        ax.text(
            float(row["published_top3_prob"]) + 0.006,
            row["label"],
            f'{row["published_top3_prob"]:.1%}',
            va="center",
            ha="left",
            fontsize=9,
            color="#1f2933",
        )

    subtitle = f"棒=最終Top3確率 / 橙=勝利確率 / 採用モデル={selected_model}"
    ax.set_xlabel("確率")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlim(0.0, max(0.32, float(plot_df["published_top3_prob"].max()) + 0.06))
    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False)

    summary_df = ordered.head(8).copy()
    summary_df["Top3"] = summary_df["published_top3_prob"].map(lambda v: f"{v:.1%}")
    summary_df["Win"] = summary_df["win_prob"].map(lambda v: f"{v:.1%}")
    summary_df["Odds"] = summary_df["odds"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "-")
    summary_df["人気"] = summary_df["popularity"].map(lambda v: f"{int(v)}" if pd.notna(v) else "-")
    summary_df["馬"] = summary_df["horse_number"].astype(int).astype(str) + " " + summary_df["horse_display_name"].astype(str)

    ax_table.axis("off")
    table = ax_table.table(
        cellText=summary_df[["馬", "Top3", "Win", "Odds", "人気"]].values.tolist(),
        colLabels=["馬", "Top3", "Win", "Odds", "人気"],
        colColours=["#e9eef5"] * 5,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e2ec")
        if col_idx == 0:
            cell.set_width(0.52)
        elif col_idx in {1, 2}:
            cell.set_width(0.16)
        else:
            cell.set_width(0.14)
        if row_idx == 0:
            cell.set_text_props(weight="bold", color="#102a43")
            cell.set_height(cell.get_height() * 1.08)
            continue
        is_axis = str(summary_df.iloc[row_idx - 1]["horse_id"]) == axis_horse_key
        if is_axis:
            cell.set_facecolor("#fde8ea")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f8fbff")
        else:
            cell.set_facecolor("#ffffff")
        if col_idx == 0 and is_axis:
            cell.get_text().set_text(f'{summary_df.iloc[row_idx - 1]["馬"]}  <- 軸')

    ax_table.set_title("上位8頭サマリー", loc="left", fontsize=12, fontweight="bold")
    dashboard_title = f"{race_name} 予測ダッシュボード" if race_name else "予測ダッシュボード"
    fig.suptitle(dashboard_title, x=0.125, y=0.975, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.125, 0.935, subtitle, fontsize=10, color="#52606d")
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.12, right=0.98)
    filepath = outdir / "top3_probability_bar.png"
    fig.savefig(filepath, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return filepath


def save_calibration_plot(calibration_df: pd.DataFrame, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(calibration_df["pred_mean"], calibration_df["true_rate"], marker="o")
    ax.set_title("Calibration plot")
    ax.set_xlabel("Predicted top3 probability")
    ax.set_ylabel("Observed top3 rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    filepath = outdir / "calibration_plot.png"
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    return filepath


def save_feature_importance_plot(feature_importance: pd.DataFrame, outdir: Path, top_n: int) -> Path:
    top_df = feature_importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_df["feature"], top_df["importance"])
    ax.set_title(f"Feature importance (top {top_n})")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    filepath = outdir / "feature_importance.png"
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    return filepath


def compute_axis_features(pred_df: pd.DataFrame) -> pd.DataFrame:
    """pred_df に axis_* 特徴量列を追加して返す。

    - axis_top3_prob    : consensus_top3_score のコピー
    - axis_mean_rank    : mean_rank のコピー
    - axis_ci_width     : top3_ci_width のコピー
    - axis_model_std    : component_model_std のコピー
    - axis_rank_stability: rank_stability_std のコピー
    - axis_tail_risk    : rank_{k}_prob 列から算出（なければ 0.0）
    - axis_market_edge  : consensus - implied_prob（odds なしなら 0.0）
    """
    df = pred_df.copy()
    df["axis_top3_prob"] = pd.to_numeric(df.get("consensus_top3_score"), errors="coerce").fillna(0.0)
    df["axis_mean_rank"] = pd.to_numeric(df.get("mean_rank"), errors="coerce").fillna(float("inf"))
    df["axis_ci_width"] = pd.to_numeric(df.get("top3_ci_width"), errors="coerce").fillna(1.0)
    df["axis_model_std"] = pd.to_numeric(df.get("component_model_std"), errors="coerce").fillna(1.0)
    df["axis_rank_stability"] = pd.to_numeric(df.get("rank_stability_std"), errors="coerce").fillna(1.0)
    df["axis_tail_risk"] = compute_tail_risk(df, threshold_percentile=50)

    if "odds" in df.columns:
        odds = pd.to_numeric(df["odds"], errors="coerce")
        implied = (1.0 / odds.where(odds > 1.0)).fillna(0.0)
        implied_sum = float(implied.sum())
        implied_norm = implied / implied_sum if implied_sum > 0.0 else pd.Series(0.0, index=df.index)
        df["axis_market_edge"] = df["axis_top3_prob"] - implied_norm
    else:
        df["axis_market_edge"] = 0.0

    return df


def compute_axis_score(axis_features_df: pd.DataFrame) -> pd.Series:
    """axis_* 特徴量から [0, 1] の axis_score を算出する。

    重み: top3_prob×0.35, (1-tail_risk)×0.25, (1-norm_ci)×0.15,
          (1-norm_std)×0.15, (1-norm_stability)×0.10
    """
    df = axis_features_df

    def _min_max_norm(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce").fillna(0.0)
        lo, hi = float(s.min()), float(s.max())
        if hi <= lo:
            return pd.Series(0.0, index=s.index, dtype=float)
        return (s - lo) / (hi - lo)

    top3_prob = pd.to_numeric(df.get("axis_top3_prob", 0.0), errors="coerce").fillna(0.0)
    tail_risk = pd.to_numeric(df.get("axis_tail_risk", 0.0), errors="coerce").fillna(0.0)
    ci_norm = _min_max_norm(df.get("axis_ci_width", pd.Series(0.0, index=df.index)))
    std_norm = _min_max_norm(df.get("axis_model_std", pd.Series(0.0, index=df.index)))
    stab_norm = _min_max_norm(df.get("axis_rank_stability", pd.Series(0.0, index=df.index)))

    raw = (
        top3_prob * 0.35
        + (1.0 - tail_risk) * 0.25
        + (1.0 - ci_norm) * 0.15
        + (1.0 - std_norm) * 0.15
        + (1.0 - stab_norm) * 0.10
    )
    score = 1.0 / (1.0 + np.exp(-raw))
    return pd.Series(score.to_numpy(dtype=float), index=df.index, dtype=float)


def _compute_selection_reason(axis_row: pd.Series) -> str:
    """axis_score の上位 2 寄与要因を文字列化する。"""
    components = {
        "axis_top3_prob": float(axis_row.get("axis_top3_prob") or 0.0),
        "axis_tail_risk_inv": 1.0 - float(axis_row.get("axis_tail_risk") or 0.0),
        "axis_market_edge": float(axis_row.get("axis_market_edge") or 0.0),
    }
    top2 = sorted(components.items(), key=lambda kv: kv[1], reverse=True)[:2]
    return ", ".join(f"{k}={v:.3f}" for k, v in top2)


def select_axis_horse(pred_df: pd.DataFrame) -> pd.Series:
    if "axis_score" in pred_df.columns:
        return pred_df.sort_values(
            ["axis_score", "consensus_top3_score"],
            ascending=[False, False],
        ).iloc[0]
    # 従来の 7 基準フォールバック
    sort_cols = [
        "consensus_top3_score",
        "top3_prob",
        "top3_ci_width",
        "component_model_std",
        "shadow_no_odds_rank",
        "rank_stability_std",
        "classifier_top3_prob",
    ]
    ascending = [False, False, True, True, True, False, False]
    return pred_df.sort_values(sort_cols, ascending=ascending).iloc[0]


_PARTNER_DEFAULTS: Dict[str, object] = {
    "n_partners": 5,
    "selection_strategy": "optimized",
    "weights": {
        "conditional_top3": 0.4,
        "partner_lift": 0.3,
        "consensus_top3": 0.3,
    },
    "popularity_cap": 3,
    "max_popular": 3,
    "max_uncertain": 2,
    "uncertainty_percentile": 80,
    "n_trials": 5000,
    "seed_offset": 7000,
}


def _partner_cfg(config: Dict[str, object]) -> Dict[str, object]:
    """config['partner'] とデフォルト値をマージして返す。"""
    raw = dict(config.get("partner", {}) or {})
    merged = dict(_PARTNER_DEFAULTS)
    merged.update(raw)
    if "weights" in raw and isinstance(raw["weights"], dict):
        base_weights = dict(_PARTNER_DEFAULTS["weights"])  # type: ignore[arg-type]
        base_weights.update(raw["weights"])
        merged["weights"] = base_weights
    return merged


def _partner_constraints_from_cfg(cfg: Dict[str, object]) -> Dict[str, object]:
    return {
        "popularity_cap": int(cfg["popularity_cap"]),
        "max_popular": int(cfg["max_popular"]),
        "max_uncertain": int(cfg["max_uncertain"]),
        "uncertainty_percentile": float(cfg["uncertainty_percentile"]),
    }


def _partner_uncertain_threshold(candidates: pd.DataFrame, constraints: Dict[str, object]) -> float:
    std_col = pd.to_numeric(
        candidates.get("component_model_std", pd.Series(0.0, index=candidates.index)),
        errors="coerce",
    ).fillna(0.0)
    if len(std_col) == 0:
        return float("inf")
    return float(np.percentile(std_col.to_numpy(dtype=float), float(constraints["uncertainty_percentile"])))


def _partner_constraint_ok(
    candidate_set: pd.DataFrame,
    constraints: Dict[str, object],
    uncertain_threshold: float,
) -> bool:
    popularity = pd.to_numeric(candidate_set.get("popularity"), errors="coerce")
    popular_count = int((popularity <= int(constraints["popularity_cap"])).fillna(False).sum())
    if popular_count > int(constraints["max_popular"]):
        return False

    std_col = pd.to_numeric(candidate_set.get("component_model_std"), errors="coerce").fillna(0.0)
    uncertain_count = int((std_col >= uncertain_threshold).sum()) if np.isfinite(uncertain_threshold) else 0
    if uncertain_count > int(constraints["max_uncertain"]):
        return False

    return True


def optimize_partner_set(
    candidates_df: pd.DataFrame,
    axis_horse: str | pd.Series | Dict[str, object],
    constraints: Dict[str, object],
    n_partners: int,
) -> pd.DataFrame:
    """候補集合を全列挙して、制約を満たす最良の相手集合を返す。"""
    axis_horse_id = (
        str(axis_horse.get("horse_id", ""))  # type: ignore[union-attr]
        if isinstance(axis_horse, (pd.Series, dict))
        else str(axis_horse)
    )
    candidates = candidates_df.copy()
    if "horse_id" in candidates.columns:
        candidates = candidates.loc[candidates["horse_id"].astype("string") != axis_horse_id].copy()
    if len(candidates) <= n_partners:
        return candidates.sort_values("partner_score_vs_axis", ascending=False).reset_index(drop=True)

    if "partner_score_vs_axis" not in candidates.columns:
        candidates["partner_score_vs_axis"] = pd.to_numeric(
            candidates.get("partner_conditional_top3", 0.0),
            errors="coerce",
        ).fillna(0.0)

    uncertain_threshold = _partner_uncertain_threshold(candidates, constraints)
    score_series = pd.to_numeric(candidates.get("partner_conditional_top3"), errors="coerce").fillna(0.0)
    tiebreak_score = pd.to_numeric(candidates.get("partner_score_vs_axis"), errors="coerce").fillna(0.0)
    consensus_score = pd.to_numeric(candidates.get("consensus_top3_score"), errors="coerce").fillna(0.0)
    candidates = candidates.assign(
        _optimizer_score=score_series,
        _optimizer_tiebreak=tiebreak_score,
        _optimizer_consensus=consensus_score,
    ).reset_index(drop=True)

    best_indices: tuple[int, ...] | None = None
    best_key: tuple[float, float, float] | None = None
    for combo in combinations(candidates.index.tolist(), n_partners):
        subset = candidates.loc[list(combo)]
        if not _partner_constraint_ok(subset, constraints, uncertain_threshold):
            continue
        key = (
            float(subset["_optimizer_score"].sum()),
            float(subset["_optimizer_tiebreak"].sum()),
            float(subset["_optimizer_consensus"].sum()),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_indices = combo

    if best_indices is None:
        fallback = candidates.sort_values("partner_score_vs_axis", ascending=False).head(n_partners).copy()
        return fallback.drop(columns=["_optimizer_score", "_optimizer_tiebreak", "_optimizer_consensus"], errors="ignore").reset_index(drop=True)

    selected = candidates.loc[list(best_indices)].sort_values("partner_score_vs_axis", ascending=False).copy()
    return selected.drop(columns=["_optimizer_score", "_optimizer_tiebreak", "_optimizer_consensus"], errors="ignore").reset_index(drop=True)


def _select_partners_greedy(
    candidates: pd.DataFrame,
    constraints: Dict[str, object],
    n_partners: int,
) -> pd.DataFrame:
    candidates = candidates.sort_values("partner_score_vs_axis", ascending=False).reset_index(drop=True)
    uncertain_threshold = _partner_uncertain_threshold(candidates, constraints)
    selected: list[pd.Series] = []
    popular_count = 0
    uncertain_count = 0

    for _, row in candidates.iterrows():
        if len(selected) >= n_partners:
            break
        pop = pd.to_numeric(pd.Series([row.get("popularity")]), errors="coerce").iloc[0]
        is_popular = pd.notna(pop) and int(pop) <= int(constraints["popularity_cap"])
        std_val = float(pd.to_numeric(pd.Series([row.get("component_model_std")]), errors="coerce").fillna(0.0).iloc[0])
        is_uncertain = bool(np.isfinite(uncertain_threshold) and std_val >= uncertain_threshold)

        if is_popular and popular_count >= int(constraints["max_popular"]):
            continue
        if is_uncertain and uncertain_count >= int(constraints["max_uncertain"]):
            continue

        selected.append(row)
        if is_popular:
            popular_count += 1
        if is_uncertain:
            uncertain_count += 1

    if not selected:
        return candidates.head(n_partners).reset_index(drop=True)
    return pd.DataFrame(selected).reset_index(drop=True).head(n_partners)


def compute_partner_scores(
    pred_df: pd.DataFrame,
    axis_horse_id: str,
    sim_strengths: np.ndarray,
    sim_temperature: float,
    config: Dict[str, object],
) -> pd.DataFrame:
    """各馬の条件付き/無条件Top3確率と lift を算出して列追加した DataFrame を返す。

    追加列:
    - partner_conditional_top3: 軸がTop3のときの条件付きTop3確率（軸自身は NaN）
    - partner_unconditional_top3: 無条件Top3確率
    - partner_lift: conditional / unconditional
    """
    cfg = _partner_cfg(config)
    strengths = np.asarray(sim_strengths, dtype=float)
    n_trials = int(cfg["n_trials"])
    seed = int(config.get("seed", 42)) + int(cfg["seed_offset"])

    horse_ids = pred_df["horse_id"].astype("string").tolist()
    try:
        axis_idx = horse_ids.index(str(axis_horse_id))
    except ValueError:
        axis_idx = 0

    cond_probs = conditional_top3_probs(
        strengths=strengths,
        axis_idx=axis_idx,
        temperature=sim_temperature,
        n_trials=n_trials,
        seed=seed,
    )

    df = pred_df.copy()
    df["partner_conditional_top3"] = cond_probs
    if "consensus_top3_score" in df.columns:
        df["partner_unconditional_top3"] = pd.to_numeric(df["consensus_top3_score"], errors="coerce").fillna(0.0).to_numpy()
    else:
        df["partner_unconditional_top3"] = np.zeros(len(df))

    unconditional = df["partner_unconditional_top3"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(unconditional > 0, cond_probs / unconditional, np.nan)
    df["partner_lift"] = lift

    # partner_score_vs_axis を事前計算（select_partners でも使用）
    cfg = _partner_cfg(config)
    weights = dict(cfg["weights"])  # type: ignore[arg-type]
    cond_col = pd.to_numeric(df["partner_conditional_top3"], errors="coerce").fillna(0.0)
    lift_col = pd.to_numeric(df["partner_lift"], errors="coerce").fillna(1.0)
    consens_col = pd.to_numeric(df.get("consensus_top3_score", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    score = (
        cond_col * float(weights.get("conditional_top3", 0.4))
        + lift_col * float(weights.get("partner_lift", 0.3))
        + consens_col * float(weights.get("consensus_top3", 0.3))
    )
    score_arr = score.to_numpy(dtype=float).copy()
    score_arr[axis_idx] = float("nan")
    df["partner_score_vs_axis"] = score_arr

    return df


def select_partners(
    pred_df: pd.DataFrame,
    axis_horse_id: str,
    config: Dict[str, object],
    n_partners: int = 5,
) -> pd.DataFrame:
    """partner_score_vs_axis 降順で相手馬を選定して返す。

    制約:
    - 軸馬は除外
    - popularity <= popularity_cap の馬は max_popular 頭まで
    - component_model_std の上位 uncertainty_percentile% に入る馬は max_uncertain 頭まで
    """
    cfg = _partner_cfg(config)
    weights = dict(cfg["weights"])  # type: ignore[arg-type]
    constraints = _partner_constraints_from_cfg(cfg)

    candidates = pred_df.loc[pred_df["horse_id"].astype("string") != str(axis_horse_id)].copy()

    # partner_score_vs_axis を算出（なければ計算）
    if "partner_score_vs_axis" not in candidates.columns or candidates["partner_score_vs_axis"].isna().all():
        cond = pd.to_numeric(candidates.get("partner_conditional_top3", 0.0), errors="coerce").fillna(0.0)
        lift = pd.to_numeric(candidates.get("partner_lift", 1.0), errors="coerce").fillna(1.0)
        consens = pd.to_numeric(candidates.get("consensus_top3_score", 0.0), errors="coerce").fillna(0.0)
        candidates["partner_score_vs_axis"] = (
            cond * float(weights.get("conditional_top3", 0.4))
            + lift * float(weights.get("partner_lift", 0.3))
            + consens * float(weights.get("consensus_top3", 0.3))
        )
    if len(candidates) <= n_partners:
        return candidates.sort_values("partner_score_vs_axis", ascending=False).reset_index(drop=True)

    selection_strategy = str(cfg.get("selection_strategy", "optimized")).strip().lower()
    if selection_strategy == "greedy":
        return _select_partners_greedy(candidates, constraints=constraints, n_partners=n_partners)
    return optimize_partner_set(candidates, axis_horse=axis_horse_id, constraints=constraints, n_partners=n_partners)


def select_best_top3_probability_source(eval_result: Dict[str, object]) -> str:
    summary = dict(eval_result.get("summary", {}) or {})
    candidates = {
        "ensemble_top3_prob": summary.get("brier_score"),
        "classifier_top3_prob": summary.get("classifier_brier_score"),
        "aux_classifier_top3_prob": summary.get("aux_classifier_brier_score"),
        "regression_top3_prob": summary.get("regression_brier_score"),
        "shadow_no_odds_top3_prob": summary.get("shadow_brier_score"),
        "top3_prob": summary.get("ranker_brier_score"),
    }
    valid = {col: float(score) for col, score in candidates.items() if score is not None and pd.notna(score)}
    if not valid:
        return "ensemble_top3_prob"
    return min(valid, key=valid.get)


def lookup_top3_brier_score(summary: Dict[str, object], probability_col: str) -> float | None:
    mapping = {
        "ensemble_top3_prob": "brier_score",
        "classifier_top3_prob": "classifier_brier_score",
        "aux_classifier_top3_prob": "aux_classifier_brier_score",
        "regression_top3_prob": "regression_brier_score",
        "shadow_no_odds_top3_prob": "shadow_brier_score",
        "top3_prob": "ranker_brier_score",
    }
    metric_key = mapping.get(probability_col)
    if metric_key is None:
        return None
    score = summary.get(metric_key)
    return float(score) if score is not None and pd.notna(score) else None


def _resolve_sparse_top3_model_selection(
    pred_df: pd.DataFrame,
    default_top3_col: str,
) -> tuple[pd.Series, pd.Series]:
    selected_model = pd.Series(default_top3_col, index=pred_df.index, dtype="string")
    consensus = pd.to_numeric(pred_df[default_top3_col], errors="coerce")

    _sparse_raw = pred_df.get("top3_model_sparse_context")
    if _sparse_raw is None:
        sparse_context = pd.Series(False, index=pred_df.index)
    else:
        sparse_context = pd.to_numeric(_sparse_raw, errors="coerce").fillna(0.0) > 0.0
    if sparse_context.any():
        shadow = pd.to_numeric(pred_df["shadow_no_odds_top3_prob"], errors="coerce")
        regression = pd.to_numeric(pred_df["regression_top3_prob"], errors="coerce")
        sparse_consensus = (0.6 * shadow) + (0.4 * regression)
        consensus = consensus.where(~sparse_context, sparse_consensus)
        selected_model = selected_model.where(~sparse_context, "sparse_shadow_regression_blend")

    return selected_model, consensus


def build_prediction_table(
    entry_df: pd.DataFrame,
    component_df: pd.DataFrame,
    selected_top3_col: str,
) -> pd.DataFrame:
    base = entry_df.copy().reset_index(drop=True)
    base["horse_display_name"] = get_display_name(base)
    use_cols = [c for c in component_df.columns if c not in {"entity_id", "race_id"}]
    merged = pd.concat([base, component_df[use_cols].reset_index(drop=True)], axis=1)
    merged["sim_rank"] = merged["top3_prob"].rank(ascending=False, method="min")
    merged["clf_rank"] = merged["classifier_top3_prob"].rank(ascending=False, method="min")
    merged["aux_clf_rank"] = merged["aux_classifier_top3_prob"].rank(ascending=False, method="min")
    merged["reg_rank"] = merged["regression_top3_prob"].rank(ascending=False, method="min")
    merged["shadow_no_odds_rank"] = merged["shadow_no_odds_top3_prob"].rank(ascending=False, method="min")
    merged["rank_stability_std"] = merged[
        ["sim_rank", "clf_rank", "aux_clf_rank", "reg_rank", "shadow_no_odds_rank"]
    ].std(axis=1)
    selected_model, consensus = _resolve_sparse_top3_model_selection(merged, selected_top3_col)
    merged["selected_top3_model"] = selected_model
    merged["consensus_top3_score"] = consensus
    merged["calibrated_top3_prob"] = consensus

    merged = compute_axis_features(merged)
    merged["axis_score"] = compute_axis_score(merged)

    ordered_cols = [
        "horse_display_name",
        "consensus_top3_score",
        "win_prob",
        "top2_prob",
        "top3_prob",
        "regression_top3_prob",
        "mean_rank",
        "top3_ci_low",
        "top3_ci_high",
        "top3_ci_width",
        "calibrated_top3_prob",
        "classifier_top3_prob",
        "aux_classifier_top3_prob",
        "shadow_no_odds_top3_prob",
        "component_model_std",
        "rank_strength",
        "shadow_rank_strength",
        "rank_temperature",
        "sim_rank",
        "clf_rank",
        "aux_clf_rank",
        "reg_rank",
        "shadow_no_odds_rank",
        "rank_stability_std",
        "axis_score",
        "axis_top3_prob",
        "axis_tail_risk",
        "axis_ci_width",
        "axis_model_std",
        "axis_rank_stability",
        "axis_market_edge",
    ]
    remain = [c for c in merged.columns if c not in ordered_cols]
    merged = merged[ordered_cols + remain]
    return merged.sort_values(["consensus_top3_score", "top3_ci_width"], ascending=[False, True]).reset_index(drop=True)


def prepare_prediction_inputs(
    history_input: str | Path | pd.DataFrame,
    entry_input: str | Path | pd.DataFrame,
    use_odds: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    if isinstance(history_input, pd.DataFrame) and isinstance(entry_input, pd.DataFrame):
        return prepare_datasets(history_raw=history_input, entry_raw=entry_input, use_odds=use_odds)
    if isinstance(history_input, pd.DataFrame) or isinstance(entry_input, pd.DataFrame):
        raise TypeError("history_input と entry_input は両方とも DataFrame か、両方とも path で渡してください。")
    return load_datasets(history_input, entry_input, use_odds=use_odds)


def validate_prediction_frames(history: pd.DataFrame, entry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    history_out = history.dropna(subset=["race_id", "race_date", "finish_rank"]).reset_index(drop=True)
    entry_out = entry.dropna(subset=["horse_id"]).reset_index(drop=True)
    if len(history_out) == 0:
        raise SystemExit("[ERROR] 学習に使える history.csv 行がありません。")
    if len(entry_out) == 0:
        raise SystemExit("[ERROR] 予測対象の entry 行がありません。")
    return history_out, entry_out


def _build_partners_payload(
    axis_row: pd.Series,
    partner_rows: pd.DataFrame,
    config: Dict[str, object],
) -> Dict[str, object]:
    axis_payload: Dict[str, object] = {
        "horse_id": str(axis_row.get("horse_id", "")),
        "name": str(axis_row.get("horse_display_name", axis_row.get("horse_name", ""))),
        "axis_score": float(axis_row.get("axis_score") or 0.0),
    }
    partners_list = []
    for rank_idx, (_, pr) in enumerate(partner_rows.iterrows(), start=1):
        entry: Dict[str, object] = {
            "horse_id": str(pr.get("horse_id", "")),
            "name": str(pr.get("horse_display_name", pr.get("horse_name", ""))),
            "partner_score": float(pr.get("partner_score_vs_axis") or 0.0),
            "conditional_top3": float(pr.get("partner_conditional_top3") or 0.0)
            if not pd.isna(pr.get("partner_conditional_top3", float("nan")))
            else None,
            "lift": float(pr.get("partner_lift") or 0.0)
            if not pd.isna(pr.get("partner_lift", float("nan")))
            else None,
        }
        partners_list.append(entry)

    cfg = _partner_cfg(config)
    return {
        "axis": axis_payload,
        "partners": partners_list,
        "set_confidence": float(
            np.mean([p["partner_score"] for p in partners_list]) if partners_list else 0.0
        ),
        "selection_config": {
            "n_partners": int(cfg["n_partners"]),
            "popularity_cap": int(cfg["popularity_cap"]),
            "max_popular": int(cfg["max_popular"]),
        },
    }


def write_prediction_outputs(
    outdir: Path,
    pred_df: pd.DataFrame,
    eval_result: Dict[str, object],
    feature_importance_df: pd.DataFrame,
    sim_diag_df: pd.DataFrame,
    schema_info: Dict[str, object],
    axis_row: pd.Series,
    config: Dict[str, object],
    partner_rows: pd.DataFrame | None = None,
) -> Dict[str, str]:
    pred_path = outdir / "predictions.csv"
    fold_metric_path = outdir / "cv_fold_metrics.csv"
    calibration_path = outdir / "calibration_curve.csv"
    importance_path = outdir / "feature_importance.csv"
    diag_path = outdir / "simulation_diagnostics.csv"
    schema_path = outdir / "schema_report.json"
    summary_path = outdir / "evaluation_summary.json"
    axis_path = outdir / "recommended_axis_horse.json"

    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    eval_result["fold_metrics"].to_csv(fold_metric_path, index=False, encoding="utf-8-sig")
    eval_result["calibration_curve"].to_csv(calibration_path, index=False, encoding="utf-8-sig")
    feature_importance_df.to_csv(importance_path, index=False, encoding="utf-8-sig")
    sim_diag_df.to_csv(diag_path, index=False, encoding="utf-8-sig")

    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_info, f, ensure_ascii=False, indent=2, default=str)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(eval_result["summary"], f, ensure_ascii=False, indent=2)
    with open(axis_path, "w", encoding="utf-8") as f:
        json.dump(axis_row.to_dict(), f, ensure_ascii=False, indent=2, default=str)

    output_paths: Dict[str, str] = {
        "predictions": str(pred_path),
        "cv_fold_metrics": str(fold_metric_path),
        "calibration_curve": str(calibration_path),
        "feature_importance": str(importance_path),
        "simulation_diagnostics": str(diag_path),
        "schema_report": str(schema_path),
        "evaluation_summary": str(summary_path),
        "recommended_axis_horse": str(axis_path),
    }

    if partner_rows is not None and len(partner_rows) > 0:
        partners_path = outdir / "recommended_partners.json"
        partners_payload = _build_partners_payload(axis_row, partner_rows, config)
        with open(partners_path, "w", encoding="utf-8") as f:
            json.dump(partners_payload, f, ensure_ascii=False, indent=2, default=str)
        output_paths["recommended_partners"] = str(partners_path)

        axis_id = str(axis_row.get("horse_id", ""))
        ticket_rows = []
        for rank_idx, (_, pr) in enumerate(partner_rows.iterrows(), start=1):
            ticket_rows.append({
                "axis_horse_id": axis_id,
                "partner_horse_id": str(pr.get("horse_id", "")),
                "partner_rank": rank_idx,
                "partner_score": float(pr.get("partner_score_vs_axis") or 0.0),
            })
        ticket_path = outdir / "ticket_candidates.csv"
        pd.DataFrame(ticket_rows).to_csv(ticket_path, index=False, encoding="utf-8-sig")
        output_paths["ticket_candidates"] = str(ticket_path)

    race_name = str(config.get("target_race_profile", {}).get("name", ""))
    top3_plot_path = save_top3_bar_chart(pred_df, outdir, axis_horse_id=str(axis_row["horse_id"]), race_name=race_name)
    calibration_plot_path = save_calibration_plot(eval_result["calibration_curve"], outdir)
    feature_plot_path = save_feature_importance_plot(
        feature_importance_df,
        outdir,
        top_n=int(config.get("plot", {}).get("feature_importance_top_n", 20)),
    )
    output_paths["top3_bar_chart"] = str(top3_plot_path)
    output_paths["calibration_plot"] = str(calibration_plot_path)
    output_paths["feature_importance_plot"] = str(feature_plot_path)

    return output_paths


def run_prediction(
    history_input: str | Path | pd.DataFrame,
    entry_input: str | Path | pd.DataFrame,
    use_odds: bool,
    config: Dict[str, object],
    output_dir: str | Path,
) -> Dict[str, object]:
    outdir = ensure_output_dir(output_dir)
    configure_matplotlib_fonts()

    try:
        history, entry, schema_info = prepare_prediction_inputs(history_input, entry_input, use_odds=use_odds)
    except (IOError, DataValidationError, ValueError) as exc:
        raise SystemExit(f"[ERROR] データ読込または検証に失敗しました: {exc}")

    history, entry = validate_prediction_frames(history, entry)

    history = add_targets(history)
    history = history.dropna(subset=["finish_percentile", "is_top3"]).reset_index(drop=True)
    history, entry = add_basic_features(history, entry)
    history, entry = add_aggregate_features(history, entry)

    feature_cols = infer_feature_columns(history, entry, use_odds=use_odds)
    no_odds_feature_cols = infer_feature_columns(history, entry, use_odds=False)
    sample_weight = compute_similarity_weights(history, config)

    n_unique_races = int(history["race_id"].nunique()) if "race_id" in history.columns else 0
    n_unique_horses = int(history["horse_id"].nunique()) if "horse_id" in history.columns else 0
    feature_nan_pct = {
        col: round(float(history[col].isna().mean()) * 100, 1)
        for col in feature_cols
        if history[col].isna().any()
    }
    data_diagnostics = {
        "history_rows": len(history),
        "entry_rows": len(entry),
        "unique_races": n_unique_races,
        "unique_horses": n_unique_horses,
        "n_features": len(feature_cols),
        "n_no_odds_features": len(no_odds_feature_cols),
        "feature_nan_pct": feature_nan_pct,
    }
    sparse_entry_count = int(pd.to_numeric(entry.get("top3_model_sparse_context"), errors="coerce").fillna(0.0).sum())
    sparse_entry_ratio = float(sparse_entry_count / len(entry)) if len(entry) > 0 else 0.0

    eval_result = evaluate_time_series_cv(history, feature_cols, no_odds_feature_cols, sample_weight, config)
    selected_top3_col = select_best_top3_probability_source(eval_result)
    eval_result["summary"]["selected_top3_source"] = selected_top3_col
    eval_result["summary"]["selected_top3_brier_score"] = lookup_top3_brier_score(eval_result["summary"], selected_top3_col)
    eval_result["summary"]["sparse_entry_count"] = sparse_entry_count
    eval_result["summary"]["sparse_entry_ratio"] = sparse_entry_ratio
    data_diagnostics["sparse_entry_count"] = sparse_entry_count
    data_diagnostics["sparse_entry_ratio"] = sparse_entry_ratio
    eval_result["summary"]["data_diagnostics"] = data_diagnostics

    # 検証基盤: データ十分性(低信頼フラグ)とモデル vs 市場(人気馬/シャドウ)の上乗せを surface する。
    eval_result["summary"]["data_sufficiency"] = eval_result.get("data_sufficiency", {})
    eval_result["summary"]["edge_vs_market"] = summarize_edge(eval_result["oof_predictions"])

    hybrid_model = fit_hybrid_race_model(
        train_df=history,
        feature_cols=feature_cols,
        no_odds_feature_cols=no_odds_feature_cols,
        sample_weight=sample_weight.to_numpy(dtype=float),
        random_state=int(config.get("seed", 42)) + 3000,
        config=config,
    )
    component_df, sim_diag_df = predict_hybrid_model(
        model=hybrid_model,
        entry_df=entry,
        config=config,
        seed=int(config.get("seed", 42)) + 9000,
    )
    pred_df = build_prediction_table(entry_df=entry, component_df=component_df, selected_top3_col=selected_top3_col)
    axis_row = select_axis_horse(pred_df)
    feature_importance_df = get_feature_importance(hybrid_model)

    # Phase 2: Partner スコア算出
    # pred_df は consensus_top3_score で並び替え済みのため、strengths も pred_df の順で取得して axis_idx と整合させる。
    if "rank_strength" in pred_df.columns:
        sim_strengths = pd.to_numeric(pred_df["rank_strength"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        sim_strengths = np.zeros(len(pred_df))
    if "rank_temperature" in pred_df.columns:
        sim_temperature = float(pd.to_numeric(pred_df["rank_temperature"], errors="coerce").iloc[0])
    else:
        sim_temperature = 1.0
    pred_df = compute_partner_scores(
        pred_df=pred_df,
        axis_horse_id=str(axis_row["horse_id"]),
        sim_strengths=sim_strengths,
        sim_temperature=sim_temperature,
        config=config,
    )
    cfg = _partner_cfg(config)
    n_partners = int(cfg["n_partners"])
    partner_rows = select_partners(pred_df, axis_horse_id=str(axis_row["horse_id"]), config=config, n_partners=n_partners)

    output_paths = write_prediction_outputs(
        outdir=outdir,
        pred_df=pred_df,
        eval_result=eval_result,
        feature_importance_df=feature_importance_df,
        sim_diag_df=sim_diag_df,
        schema_info=schema_info,
        axis_row=axis_row,
        config=config,
        partner_rows=partner_rows,
    )

    return {
        "feature_cols": feature_cols,
        "eval_result": eval_result,
        "predictions": pred_df,
        "axis_row": axis_row,
        "partner_rows": partner_rows,
        "schema_info": schema_info,
        "feature_importance": feature_importance_df,
        "simulation_diagnostics": sim_diag_df,
        "output_paths": output_paths,
    }


def print_prediction_console_summary(result: Dict[str, object]) -> None:
    pred_df = result["predictions"]
    axis_row = result["axis_row"]
    eval_result = result["eval_result"]
    output_paths = result["output_paths"]

    display_cols = [c for c in DISPLAY_COLS if c in pred_df.columns]
    axis_display_cols = [c for c in DISPLAY_COLS if c in axis_row.index]

    summary = eval_result.get("summary", {}) if isinstance(eval_result, dict) else {}
    sufficiency = summary.get("data_sufficiency", {}) or {}
    if sufficiency.get("is_low_confidence"):
        print("=" * 72)
        print(
            f"[⚠ 低信頼] レース数={sufficiency.get('n_races')} / regime={sufficiency.get('regime')}。"
            "検証が統計的に不十分です。以下の予測は参考値です。"
        )
        for warn_msg in sufficiency.get("warnings", []):
            print(f"  - {warn_msg}")
        print("=" * 72)
    edge = summary.get("edge_vs_market", {}) or {}
    if edge:
        print("\n[INFO] モデル vs 市場 (軸Top3的中率 / 上乗せ)")
        print(json.dumps(edge, ensure_ascii=False, indent=2))

    print("[INFO] feature columns")
    print(result["feature_cols"])
    print("\n[INFO] CV summary")
    print(json.dumps(eval_result["summary"], ensure_ascii=False, indent=2))
    print("\n[INFO] prediction table")
    print(pred_df[display_cols].to_string(index=False))
    print("\n[INFO] recommended axis horse")
    print(axis_row[axis_display_cols].to_string())

    partner_rows = result.get("partner_rows")
    if partner_rows is not None and len(partner_rows) > 0:
        print("\n[INFO] recommended partner horses (相手候補)")
        partner_cols = [c for c in ["horse_display_name", "partner_score_vs_axis", "partner_conditional_top3", "partner_lift", "consensus_top3_score"] if c in partner_rows.columns]
        print(partner_rows[partner_cols].to_string(index=False))

    print("\n[INFO] outputs")
    for key, path in output_paths.items():
        print(f"{key}={path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="競馬の軸馬・Top3確率予測")
    parser.add_argument("--history", required=True, help="history.csv のパス")
    parser.add_argument("--entry", required=True, help="entry.csv のパス")
    parser.add_argument("--use-odds", required=True, type=parse_bool, help="odds/popularity を使うか")
    parser.add_argument("--config", default="config.yaml", help="config.yaml のパス")
    parser.add_argument("--race-config", help="race.yaml のような追加設定ファイル")
    parser.add_argument("--output-dir", default="outputs", help="出力先ディレクトリ")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.race_config:
        config = deep_merge_dicts(config, normalize_config_override(load_config(args.race_config)))

    result = run_prediction(
        history_input=args.history,
        entry_input=args.entry,
        use_odds=args.use_odds,
        config=config,
        output_dir=args.output_dir,
    )
    print_prediction_console_summary(result)


if __name__ == "__main__":
    main()
