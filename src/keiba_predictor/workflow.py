from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .data_loader import read_csv_safely
from .prediction import deep_merge_dicts, load_config, print_prediction_console_summary, run_prediction


RACE_DATA_COLUMNS = [
    "race_id",
    "race_date",
    "race_month",
    "season",
    "course",
    "surface",
    "distance",
    "turn_direction",
    "grade",
    "class",
    "field_size",
    "horse_id",
    "horse_name",
    "draw",
    "horse_number",
    "sex",
    "age",
    "carried_weight",
    "jockey",
    "trainer",
    "body_weight",
    "body_weight_diff",
    "odds",
    "popularity",
    "last_finish",
    "last_distance",
    "last_class",
    "last_margin",
    "last_3f",
    "days_since_last",
    "finish_rank",
]
RESULT_TEMPLATE_COLUMNS = [
    "horse_id",
    "horse_name",
    "finish_rank",
    "result_time",
    "result_margin",
    "result_last3f",
    "result_final_odds",
    "result_final_popularity",
    "result_body_weight",
    "result_body_weight_diff",
]
RESULT_META_FILENAME = "race_result_meta.json"
SEASON_BY_MONTH = {
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
    12: "winter",
}
SEASON_MONTHS = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
}


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def find_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("必要ファイルが見つかりません: " + ", ".join(str(path) for path in candidates))


def _parse_race_time_to_seconds(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if ":" not in text:
        try:
            return float(text)
        except ValueError:
            return None
    try:
        minutes_text, seconds_text = text.split(":", 1)
        return (float(minutes_text) * 60.0) + float(seconds_text)
    except ValueError:
        return None


def _coerce_optional_result_columns(result_df: pd.DataFrame) -> pd.DataFrame:
    normalized = result_df.copy()

    int_like_columns = [
        "result_final_popularity",
        "result_body_weight",
        "result_body_weight_diff",
    ]
    float_like_columns = [
        "result_last3f",
        "result_final_odds",
    ]

    for column in int_like_columns + float_like_columns:
        if column not in normalized.columns:
            continue
        raw = normalized[column]
        text = raw.astype("string").str.strip()
        text = text.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NaN": pd.NA})
        parsed = pd.to_numeric(text, errors="coerce")
        invalid_mask = text.notna() & parsed.isna()
        if invalid_mask.any():
            sample = text[invalid_mask].iloc[0]
            raise SystemExit(f"[ERROR] result.csv の {column} に数値でない値があります: {sample}")
        normalized[column] = parsed
        if column in int_like_columns:
            normalized[column] = normalized[column].astype("Int64")

    if "result_time" in normalized.columns:
        time_text = normalized["result_time"].astype("string").str.strip()
        time_text = time_text.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NaN": pd.NA})
        parsed_seconds = time_text.map(_parse_race_time_to_seconds)
        invalid_mask = time_text.notna() & parsed_seconds.isna()
        if invalid_mask.any():
            sample = time_text[invalid_mask].iloc[0]
            raise SystemExit(f"[ERROR] result.csv の result_time に時刻として解釈できない値があります: {sample}")
        normalized["result_time_seconds"] = parsed_seconds.astype(float)

    return normalized


def load_result_meta(race_dir: Path) -> Dict[str, object] | None:
    meta_path = race_dir / RESULT_META_FILENAME
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[ERROR] {meta_path.name} の JSON 解析に失敗しました: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"[ERROR] {meta_path.name} は JSON object である必要があります。")

    meta = dict(payload)
    winning_time = meta.get("winning_time")
    if winning_time and "winning_time_seconds" not in meta:
        seconds = _parse_race_time_to_seconds(winning_time)
        if seconds is not None:
            meta["winning_time_seconds"] = seconds
    return meta


def resolve_race_directory(races_root: Path, race: str | None, race_dir: str | None) -> Path:
    if race_dir:
        resolved = Path(race_dir).expanduser().resolve()
    elif race:
        resolved = (races_root.expanduser() / race).resolve()
    else:
        raise SystemExit("[ERROR] `--race` または `--race-dir` を指定してください。")

    if not resolved.exists():
        raise SystemExit(f"[ERROR] レースディレクトリが見つかりません: {resolved}")
    if not resolved.is_dir():
        raise SystemExit(f"[ERROR] レースディレクトリではありません: {resolved}")
    return resolved


def list_race_directories(races_root: Path) -> list[Path]:
    if not races_root.exists():
        return []
    return sorted(
        [path for path in races_root.iterdir() if path.is_dir() and (path / "race.yaml").exists()],
        key=lambda path: path.name,
    )


def summarize_race_directory(race_dir: Path) -> Dict[str, str]:
    summary = {
        "slug": race_dir.name,
        "race_name": race_dir.name,
        "race_date": "",
    }
    race_yaml_path = race_dir / "race.yaml"
    if not race_yaml_path.exists():
        return summary
    try:
        config = load_config(race_yaml_path)
    except Exception:
        return summary
    profile = dict(config.get("target_race_profile", {}) or {})
    summary["race_name"] = str(config.get("race_name") or profile.get("name") or race_dir.name)
    summary["race_date"] = str(config.get("race_date") or profile.get("race_date") or "")
    return summary


def load_race_inputs(race_dir: Path) -> Tuple[Path, Path, Path]:
    race_yaml_path = find_existing_path(race_dir / "race.yaml")
    entry_path = find_existing_path(race_dir / "entry.csv")
    history_path = find_existing_path(race_dir / "history.csv", race_dir / "historical.csv")
    return race_yaml_path, entry_path, history_path


def build_output_folder_name(race_dir: Path) -> str:
    race_name = summarize_race_directory(race_dir)["race_name"].strip() or race_dir.name
    invalid_chars = '<>:"/\\|?*'
    sanitized = "".join("_" if ch in invalid_chars else ch for ch in race_name)
    sanitized = " ".join(sanitized.split()).strip(" .")
    return sanitized or race_dir.name


def build_prediction_output_dir(output_root: Path, race_dir: Path) -> Path:
    outdir = output_root / build_output_folder_name(race_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def normalize_race_config(raw: Dict[str, object]) -> Dict[str, object]:
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


_REQUIRED_RACE_PROFILE_FIELDS = ["surface", "distance", "course"]


def _validate_race_profile(config: Dict[str, object], race_yaml_path: Path) -> None:
    profile = dict(config.get("target_race_profile", {}) or {})
    missing = [f for f in _REQUIRED_RACE_PROFILE_FIELDS if not profile.get(f)]
    if missing:
        raise ValueError(
            f"race.yaml ({race_yaml_path}) の target_race_profile に必須フィールドがありません: {missing}. "
            f"各レースの race.yaml で surface, distance, course を指定してください。"
        )


def load_effective_config(base_config_path: Path, race_yaml_path: Path) -> Tuple[Dict[str, object], bool]:
    base_config = load_config(base_config_path)
    race_config = normalize_race_config(load_config(race_yaml_path))
    use_odds = bool(race_config.pop("use_odds", True))
    merged = deep_merge_dicts(base_config, race_config)
    _validate_race_profile(merged, race_yaml_path)
    return merged, use_odds


def combine_history_sources(training_history_path: Path, weekly_history_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if training_history_path.exists():
        frames.append(read_csv_safely(training_history_path))
    frames.append(read_csv_safely(weekly_history_path))

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "race_id" in combined.columns:
        combined["race_id"] = combined["race_id"].astype("string")
    if "horse_id" in combined.columns:
        combined["horse_id"] = combined["horse_id"].astype("string")
    if {"race_id", "horse_id"}.issubset(combined.columns):
        combined = combined.drop_duplicates(subset=["race_id", "horse_id"], keep="last")
    return combined.reset_index(drop=True)


def filter_history_for_prediction(history_df: pd.DataFrame, entry_df: pd.DataFrame) -> pd.DataFrame:
    filtered = history_df.copy()

    if "race_id" in filtered.columns and "race_id" in entry_df.columns:
        target_race_ids = set(entry_df["race_id"].dropna().astype("string"))
        filtered["race_id"] = filtered["race_id"].astype("string")
        filtered = filtered.loc[~filtered["race_id"].isin(target_race_ids)]

    if "race_date" in filtered.columns and "race_date" in entry_df.columns:
        entry_dates = pd.to_datetime(entry_df["race_date"], errors="coerce").dropna().unique().tolist()
        if len(entry_dates) == 1:
            target_date = pd.Timestamp(entry_dates[0])
            history_dates = pd.to_datetime(filtered["race_date"], errors="coerce")
            filtered = filtered.loc[history_dates.isna() | (history_dates < target_date)]

    return filtered.reset_index(drop=True)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def predict_race(
    race_dir: Path,
    base_config_path: Path,
    training_history_path: Path,
    output_root: Path,
) -> Dict[str, object]:
    race_yaml_path, entry_path, history_path = load_race_inputs(race_dir)
    output_dir = build_prediction_output_dir(output_root, race_dir)
    config, use_odds = load_effective_config(base_config_path, race_yaml_path)

    entry_df = read_csv_safely(entry_path)
    history_df = filter_history_for_prediction(
        combine_history_sources(training_history_path, history_path),
        entry_df,
    )
    result = run_prediction(
        history_input=history_df,
        entry_input=entry_df,
        use_odds=use_odds,
        config=config,
        output_dir=output_dir,
    )
    save_json(output_dir / "effective_config.json", config)
    save_json(
        output_dir / "run_context.json",
        {
            "race_dir": str(race_dir),
            "entry_path": str(entry_path),
            "history_path": str(history_path),
            "training_history_path": str(training_history_path),
            "output_dir": str(output_dir),
            "use_odds": use_odds,
        },
    )
    print_prediction_console_summary(result)
    return result


def validate_result_frame(entry_df: pd.DataFrame, result_df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    normalized = _coerce_optional_result_columns(result_df)
    if "race_id" not in normalized.columns:
        race_ids = entry_df["race_id"].dropna().astype("string").unique().tolist()
        if len(race_ids) != 1:
            raise SystemExit("[ERROR] result.csv に race_id がなく、entry.csv からも一意に補完できません。")
        normalized["race_id"] = race_ids[0]

    required_cols = {"race_id", "horse_id", "finish_rank"}
    missing = required_cols - set(normalized.columns)
    if missing:
        raise SystemExit(f"[ERROR] result.csv の必須列が不足しています: {sorted(missing)}")

    normalized["horse_id"] = normalized["horse_id"].astype("string")
    normalized["race_id"] = normalized["race_id"].astype("string")
    normalized["finish_rank"] = pd.to_numeric(normalized["finish_rank"], errors="coerce")
    if normalized["finish_rank"].isna().any():
        raise SystemExit("[ERROR] result.csv の finish_rank に数値でない値があります。")
    if not (normalized["finish_rank"] % 1 == 0).all():
        raise SystemExit("[ERROR] result.csv の finish_rank は整数である必要があります。")
    normalized["finish_rank"] = normalized["finish_rank"].astype(int)
    if (normalized["finish_rank"] <= 0).any():
        raise SystemExit("[ERROR] result.csv の finish_rank は 1 以上である必要があります。")
    if normalized["horse_id"].duplicated().any():
        raise SystemExit("[ERROR] result.csv の horse_id が重複しています。")
    if normalized["finish_rank"].duplicated().any():
        raise SystemExit("[ERROR] result.csv の finish_rank が重複しています。")

    entry_keys = entry_df[["race_id", "horse_id"]].copy()
    entry_keys["race_id"] = entry_keys["race_id"].astype("string")
    entry_keys["horse_id"] = entry_keys["horse_id"].astype("string")
    joined = normalized.merge(entry_keys, on=["race_id", "horse_id"], how="left", indicator=True)
    unknown = joined.loc[joined["_merge"] != "both", ["race_id", "horse_id"]]
    if len(unknown) > 0:
        raise SystemExit(f"[ERROR] result.csv に entry.csv に存在しない馬が含まれています: {unknown.to_dict(orient='records')}")

    ranks = sorted(normalized["finish_rank"].tolist())
    expected_partial = list(range(1, len(normalized) + 1))
    is_full_result = len(normalized) == len(entry_df) and ranks == list(range(1, len(entry_df) + 1))
    if not is_full_result and ranks != expected_partial:
        raise SystemExit("[ERROR] partial result.csv は 1着から連番の TopK 形式である必要があります。")
    return normalized.sort_values("finish_rank").reset_index(drop=True), is_full_result


def build_settled_entry(entry_df: pd.DataFrame, result_df: pd.DataFrame) -> pd.DataFrame:
    base = entry_df.copy()
    base["race_id"] = base["race_id"].astype("string")
    base["horse_id"] = base["horse_id"].astype("string")
    if "finish_rank" in base.columns:
        base = base.drop(columns=["finish_rank"])
    result_merge_columns = [col for col in result_df.columns if col != "horse_name"]
    result_keys = result_df[result_merge_columns].copy()
    result_keys["race_id"] = result_keys["race_id"].astype("string")
    result_keys["horse_id"] = result_keys["horse_id"].astype("string")
    return base.merge(result_keys, on=["race_id", "horse_id"], how="left")


def upsert_training_history(training_history_path: Path, settled_df: pd.DataFrame) -> pd.DataFrame:
    ensure_parent_dir(training_history_path)
    if training_history_path.exists():
        training_df = read_csv_safely(training_history_path)
        combined = pd.concat([training_df, settled_df], ignore_index=True, sort=False)
    else:
        combined = settled_df.copy()
    if "race_id" in combined.columns:
        combined["race_id"] = combined["race_id"].astype("string")
    if "horse_id" in combined.columns:
        combined["horse_id"] = combined["horse_id"].astype("string")
    combined = combined.drop_duplicates(subset=["race_id", "horse_id"], keep="last").reset_index(drop=True)
    combined.to_csv(training_history_path, index=False, encoding="utf-8-sig")
    return combined


def load_predictions_frame(output_dir: Path) -> pd.DataFrame | None:
    predictions_path = output_dir / "predictions.csv"
    if not predictions_path.exists():
        return None
    return read_csv_safely(predictions_path)


def _sort_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    ranked = predictions_df.copy()
    ranked["horse_id"] = ranked["horse_id"].astype("string")
    return ranked.sort_values(["consensus_top3_score", "top3_ci_width"], ascending=[False, True]).reset_index(drop=True)


def build_post_race_analysis(
    predictions_df: pd.DataFrame | None,
    result_df: pd.DataFrame,
) -> pd.DataFrame | None:
    if predictions_df is None:
        return None

    ranked = _sort_predictions(predictions_df)
    ranked["predicted_rank"] = range(1, len(ranked) + 1)
    if "finish_rank" in ranked.columns:
        ranked = ranked.drop(columns=["finish_rank"])

    result_norm = result_df.copy()
    result_norm["horse_id"] = result_norm["horse_id"].astype("string")
    result_norm = result_norm.rename(columns={"finish_rank": "actual_finish_rank"})
    result_merge_columns = [col for col in result_norm.columns if col != "horse_name"]
    analysis = ranked.merge(result_norm[result_merge_columns], on=["race_id", "horse_id"], how="left")
    analysis["actual_is_top3"] = analysis["actual_finish_rank"].le(3)
    analysis["actual_is_win"] = analysis["actual_finish_rank"].eq(1)
    analysis["predicted_is_top3"] = analysis["predicted_rank"].le(3)
    analysis["rank_error"] = analysis["predicted_rank"] - analysis["actual_finish_rank"]
    analysis["abs_rank_error"] = analysis["rank_error"].abs()
    analysis["mean_rank_error"] = analysis["mean_rank"] - analysis["actual_finish_rank"]
    analysis["abs_mean_rank_error"] = analysis["mean_rank_error"].abs()
    return analysis.sort_values(["actual_finish_rank", "predicted_rank"], na_position="last").reset_index(drop=True)


def build_post_race_report(
    predictions_df: pd.DataFrame | None,
    settled_entry_df: pd.DataFrame,
    result_df: pd.DataFrame,
    is_full_result: bool,
    appended_to_training_history: bool,
    race_meta: Dict[str, object] | None = None,
    analysis_df: pd.DataFrame | None = None,
) -> Dict[str, object]:
    result_norm = result_df.copy()
    result_norm["horse_id"] = result_norm["horse_id"].astype("string")
    actual_top3 = result_df[result_df["finish_rank"] <= 3].sort_values("finish_rank")
    report: Dict[str, object] = {
        "race_id": str(settled_entry_df["race_id"].iloc[0]),
        "field_size": int(len(settled_entry_df)),
        "known_result_count": int(len(result_df)),
        "is_full_result": bool(is_full_result),
        "appended_to_training_history": bool(appended_to_training_history),
        "actual_top3": actual_top3[["horse_id", "horse_name", "finish_rank"]].to_dict(orient="records")
        if "horse_name" in actual_top3.columns
        else actual_top3[["horse_id", "finish_rank"]].to_dict(orient="records"),
    }
    if race_meta:
        report["race_meta"] = race_meta

    if predictions_df is None:
        report["prediction_status"] = "predictions.csv not found"
        return report

    ranked = _sort_predictions(predictions_df)
    predicted_top3 = ranked.head(3)
    predicted_top3_ids = set(predicted_top3["horse_id"].astype("string"))
    actual_top3_ids = set(result_norm[result_norm["finish_rank"] <= 3]["horse_id"].astype("string"))
    axis_row = ranked.iloc[0]
    winner_row = result_norm.sort_values("finish_rank").iloc[0]

    report["predicted_top3"] = predicted_top3[
        ["horse_id", "horse_display_name", "consensus_top3_score", "top3_prob", "win_prob"]
    ].to_dict(orient="records")
    report["predicted_axis_horse"] = {
        "horse_id": str(axis_row["horse_id"]),
        "horse_display_name": str(axis_row["horse_display_name"]),
        "consensus_top3_score": float(axis_row["consensus_top3_score"]),
    }
    actual_winner: Dict[str, object] = {}
    for key in [col for col in ["horse_id", "horse_name", "finish_rank", "result_time", "result_time_seconds"] if col in winner_row.index]:
        value = winner_row[key]
        if pd.isna(value):
            continue
        if key in {"horse_id", "horse_name", "result_time"}:
            actual_winner[key] = str(value)
        elif key == "finish_rank":
            actual_winner[key] = int(value)
        elif key.endswith("_seconds"):
            actual_winner[key] = float(value)
        else:
            actual_winner[key] = value
    report["actual_winner"] = actual_winner
    report["top3_overlap_count"] = int(len(predicted_top3_ids & actual_top3_ids))
    report["axis_hit_top3"] = bool(str(axis_row["horse_id"]) in actual_top3_ids)

    if analysis_df is not None and len(analysis_df) > 0:
        valid = analysis_df[analysis_df["actual_finish_rank"].notna()].copy()
        if len(valid) > 0:
            report["prediction_status"] = "ok"
            report["known_rank_mean_abs_error"] = float(valid["abs_mean_rank_error"].mean())
            report["rank_order_mean_abs_error"] = float(valid["abs_rank_error"].mean())
            report["top3_brier_score"] = float(((valid["top3_prob"] - valid["actual_is_top3"].astype(float)) ** 2).mean())
            report["win_brier_score"] = float(((valid["win_prob"] - valid["actual_is_win"].astype(float)) ** 2).mean())
            spearman = valid["predicted_rank"].corr(valid["actual_finish_rank"], method="spearman")
            if pd.notna(spearman):
                report["rank_spearman"] = float(spearman)

            axis_analysis = valid.loc[valid["horse_id"].astype("string") == str(axis_row["horse_id"])]
            if len(axis_analysis) > 0:
                report["axis_finish_rank"] = int(axis_analysis.iloc[0]["actual_finish_rank"])

            winner_analysis = valid.loc[valid["horse_id"].astype("string") == str(winner_row["horse_id"])]
            if len(winner_analysis) > 0:
                winner_analysis_row = winner_analysis.iloc[0]
                report["winner_predicted_rank"] = int(winner_analysis_row["predicted_rank"])
                report["winner_mean_rank"] = float(winner_analysis_row["mean_rank"])
                report["winner_win_prob"] = float(winner_analysis_row["win_prob"])
                report["winner_top3_prob"] = float(winner_analysis_row["top3_prob"])

            miss_columns = [
                "horse_id",
                "horse_display_name",
                "predicted_rank",
                "actual_finish_rank",
                "mean_rank",
                "rank_error",
                "mean_rank_error",
                "top3_prob",
                "win_prob",
            ]
            report["largest_prediction_gaps"] = valid.sort_values(
                ["abs_rank_error", "abs_mean_rank_error"],
                ascending=[False, False],
            )[miss_columns].head(5).to_dict(orient="records")
    return report


def settle_race(
    race_dir: Path,
    training_history_path: Path,
    output_root: Path,
) -> Dict[str, object]:
    _, entry_path, _ = load_race_inputs(race_dir)
    result_path = find_existing_path(race_dir / "result.csv")
    output_dir = build_prediction_output_dir(output_root, race_dir)

    entry_df = read_csv_safely(entry_path)
    result_df_raw = read_csv_safely(result_path)
    result_df, is_full_result = validate_result_frame(entry_df, result_df_raw)
    settled_entry_df = build_settled_entry(entry_df, result_df)
    settled_entry_path = output_dir / "settled_entry.csv"
    settled_entry_df.to_csv(settled_entry_path, index=False, encoding="utf-8-sig")

    appended_to_training_history = False
    training_history_rows = None
    if is_full_result:
        training_history_rows = upsert_training_history(
            training_history_path,
            settled_entry_df[settled_entry_df["finish_rank"].notna()].copy(),
        )
        appended_to_training_history = True

    predictions_df = load_predictions_frame(output_dir)
    analysis_df = build_post_race_analysis(predictions_df=predictions_df, result_df=result_df)
    analysis_path = None
    if analysis_df is not None:
        analysis_path = output_dir / "post_race_analysis.csv"
        analysis_df.to_csv(analysis_path, index=False, encoding="utf-8-sig")
    if "horse_name" not in result_df.columns and "horse_name" in entry_df.columns:
        result_df = result_df.merge(entry_df[["horse_id", "horse_name"]], on="horse_id", how="left")
    result_meta = load_result_meta(race_dir)
    report = build_post_race_report(
        predictions_df=predictions_df,
        settled_entry_df=settled_entry_df,
        result_df=result_df,
        is_full_result=is_full_result,
        appended_to_training_history=appended_to_training_history,
        race_meta=result_meta,
        analysis_df=analysis_df,
    )
    save_json(output_dir / "post_race_report.json", report)

    summary = {
        "result_path": str(result_path),
        "settled_entry_path": str(settled_entry_path),
        "post_race_analysis_path": str(analysis_path) if analysis_path is not None else None,
        "post_race_report_path": str(output_dir / "post_race_report.json"),
        "is_full_result": is_full_result,
        "appended_to_training_history": appended_to_training_history,
        "training_history_path": str(training_history_path),
        "training_history_rows": int(len(training_history_rows)) if training_history_rows is not None else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _season_for_date(race_date: date | None) -> tuple[str, list[int]]:
    if race_date is None:
        return "", []
    season_label = SEASON_BY_MONTH[race_date.month]
    return season_label, SEASON_MONTHS[season_label]


def create_race_directory(
    race_dir: Path,
    race_name: str = "",
    race_date_text: str = "",
    force: bool = False,
) -> Dict[str, str]:
    parsed_date = date.fromisoformat(race_date_text) if race_date_text else None
    season_label, season_months = _season_for_date(parsed_date)
    default_race_name = race_name or race_dir.name.replace("-", " ").title()
    date_literal = race_date_text or ""
    season_months_literal = "[" + ", ".join(str(month) for month in season_months) + "]"

    race_yaml = "\n".join(
        [
            f"race_name: {_yaml_string(default_race_name)}",
            f"race_date: {_yaml_string(date_literal)}",
            "use_odds: true",
            "",
            "target_race_profile:",
            f"  name: {_yaml_string(default_race_name)}",
            f"  race_date: {_yaml_string(date_literal)}",
            '  surface: "turf"',
            "  distance: 2000",
            '  course: "Tokyo"',
            '  turn: "left"',
            '  grade: "G3"',
            '  class: "Open"',
            f"  season_label: {_yaml_string(season_label)}",
            f"  season_months: {season_months_literal}",
            "",
        ]
    )
    readme = "\n".join(
        [
            f"# {default_race_name}",
            "",
            "## Files",
            "- `race.yaml`: 対象レースの条件と上書き設定",
            "- `entry.csv`: 今週の出走馬データ",
            "- `history.csv`: 学習に使う近走データ",
            "- `result.csv`: レース後に記入する結果ファイル",
            "",
            "## Commands",
            f"- 予測: `uv run python main.py --race {race_dir.name}`",
            f"- 結果反映: `uv run python main.py settle --race {race_dir.name}`",
            "",
            "## Outputs",
            f"- 予測結果: `outputs/{build_output_folder_name(race_dir)}/`",
            "- 学習用累積ファイル: `data/training/race_results_master.csv`",
            "",
        ]
    )

    files = {
        "race_yaml": race_dir / "race.yaml",
        "entry_csv": race_dir / "entry.csv",
        "history_csv": race_dir / "history.csv",
        "result_csv": race_dir / "result.csv",
        "readme": race_dir / "README.md",
    }
    if not force:
        existing = [path.name for path in files.values() if path.exists()]
        if existing:
            raise FileExistsError(f"既存ファイルがあります: {existing}")

    race_dir.mkdir(parents=True, exist_ok=True)
    files["race_yaml"].write_text(race_yaml, encoding="utf-8")
    files["entry_csv"].write_text(",".join(RACE_DATA_COLUMNS) + "\n", encoding="utf-8")
    files["history_csv"].write_text(",".join(RACE_DATA_COLUMNS) + "\n", encoding="utf-8")
    files["result_csv"].write_text(",".join(RESULT_TEMPLATE_COLUMNS) + "\n", encoding="utf-8")
    files["readme"].write_text(readme, encoding="utf-8")

    return {name: str(path) for name, path in files.items()}
