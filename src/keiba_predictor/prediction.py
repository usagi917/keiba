from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter
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
except ImportError:  # pragma: no cover - direct script fallback
    from data_loader import DataValidationError, load_datasets, prepare_datasets
    from features import add_aggregate_features, add_basic_features, add_targets, compute_similarity_weights, infer_feature_columns
    from hybrid_model import (
        evaluate_time_series_cv,
        fit_hybrid_race_model,
        get_feature_importance,
        predict_hybrid_model,
    )


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


def save_top3_bar_chart(pred_df: pd.DataFrame, outdir: Path, axis_horse_id: str) -> Path:
    ordered = pred_df.sort_values(["top3_prob", "win_prob"], ascending=[False, False]).reset_index(drop=True).head(12)
    plot_df = ordered.iloc[::-1].copy()
    plot_df["label"] = plot_df["horse_number"].astype(int).astype(str) + " " + plot_df["horse_display_name"].astype(str)

    axis_horse_key = str(axis_horse_id)
    bar_colors = ["#d1495b" if str(horse_id) == axis_horse_key else "#2f6db3" for horse_id in plot_df["horse_id"]]
    consensus_variation = plot_df["consensus_top3_score"].round(6).nunique(dropna=True)

    fig = plt.figure(figsize=(15.5, 8))
    grid = GridSpec(1, 2, figure=fig, width_ratios=[2.0, 1.45], wspace=0.08)
    ax = fig.add_subplot(grid[0, 0])
    ax_table = fig.add_subplot(grid[0, 1])

    ax.barh(plot_df["label"], plot_df["top3_prob"], color=bar_colors, alpha=0.92, label="Top3確率")
    ax.scatter(plot_df["win_prob"], plot_df["label"], color="#f28e2b", s=55, zorder=3, label="勝利確率")
    if consensus_variation > 3:
        ax.scatter(
            plot_df["consensus_top3_score"],
            plot_df["label"],
            color="#2a9d8f",
            marker="D",
            s=45,
            zorder=3,
            label="最終採用スコア",
        )

    for _, row in plot_df.iterrows():
        ax.text(
            float(row["top3_prob"]) + 0.006,
            row["label"],
            f'{row["top3_prob"]:.1%}',
            va="center",
            ha="left",
            fontsize=9,
            color="#1f2933",
        )

    if consensus_variation > 3:
        subtitle = "棒=Top3確率 / 橙=勝利確率 / 緑=最終採用スコア"
    else:
        subtitle = "棒=Top3確率 / 橙=勝利確率 / 最終採用スコアは横並びのため省略"
    ax.set_xlabel("確率")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlim(0.0, max(0.32, float(plot_df["top3_prob"].max()) + 0.06))
    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False)

    summary_df = ordered.head(8).copy()
    summary_df["Top3"] = summary_df["top3_prob"].map(lambda v: f"{v:.1%}")
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
    fig.suptitle("高松宮記念 予測ダッシュボード", x=0.125, y=0.975, ha="left", fontsize=17, fontweight="bold")
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


def select_axis_horse(pred_df: pd.DataFrame) -> pd.Series:
    sort_cols = [
        "consensus_top3_score",
        "top3_prob",
        "top3_ci_width",
        "component_model_std",
        "shadow_no_odds_rank",
        "rank_stability_std",
        "classifier_top3_prob",
    ]
    ascending = [False, False, True, True, True, True, False]
    return pred_df.sort_values(sort_cols, ascending=ascending).iloc[0]


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
    merged["selected_top3_model"] = selected_top3_col
    merged["consensus_top3_score"] = pd.to_numeric(merged[selected_top3_col], errors="coerce")
    merged["calibrated_top3_prob"] = pd.to_numeric(merged[selected_top3_col], errors="coerce")

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


def write_prediction_outputs(
    outdir: Path,
    pred_df: pd.DataFrame,
    eval_result: Dict[str, object],
    feature_importance_df: pd.DataFrame,
    sim_diag_df: pd.DataFrame,
    schema_info: Dict[str, object],
    axis_row: pd.Series,
    config: Dict[str, object],
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

    top3_plot_path = save_top3_bar_chart(pred_df, outdir, axis_horse_id=str(axis_row["horse_id"]))
    calibration_plot_path = save_calibration_plot(eval_result["calibration_curve"], outdir)
    feature_plot_path = save_feature_importance_plot(
        feature_importance_df,
        outdir,
        top_n=int(config.get("plot", {}).get("feature_importance_top_n", 20)),
    )

    return {
        "predictions": str(pred_path),
        "cv_fold_metrics": str(fold_metric_path),
        "calibration_curve": str(calibration_path),
        "feature_importance": str(importance_path),
        "simulation_diagnostics": str(diag_path),
        "schema_report": str(schema_path),
        "evaluation_summary": str(summary_path),
        "recommended_axis_horse": str(axis_path),
        "top3_bar_chart": str(top3_plot_path),
        "calibration_plot": str(calibration_plot_path),
        "feature_importance_plot": str(feature_plot_path),
    }


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

    eval_result = evaluate_time_series_cv(history, feature_cols, no_odds_feature_cols, sample_weight, config)
    selected_top3_col = select_best_top3_probability_source(eval_result)
    eval_result["summary"]["selected_top3_source"] = selected_top3_col
    eval_result["summary"]["selected_top3_brier_score"] = lookup_top3_brier_score(eval_result["summary"], selected_top3_col)
    eval_result["summary"]["data_diagnostics"] = data_diagnostics

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

    output_paths = write_prediction_outputs(
        outdir=outdir,
        pred_df=pred_df,
        eval_result=eval_result,
        feature_importance_df=feature_importance_df,
        sim_diag_df=sim_diag_df,
        schema_info=schema_info,
        axis_row=axis_row,
        config=config,
    )

    return {
        "feature_cols": feature_cols,
        "eval_result": eval_result,
        "predictions": pred_df,
        "axis_row": axis_row,
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

    print("[INFO] feature columns")
    print(result["feature_cols"])
    print("\n[INFO] CV summary")
    print(json.dumps(eval_result["summary"], ensure_ascii=False, indent=2))
    print("\n[INFO] prediction table")
    print(pred_df[DISPLAY_COLS].to_string(index=False))
    print("\n[INFO] recommended axis horse")
    print(axis_row[DISPLAY_COLS].to_string())
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
