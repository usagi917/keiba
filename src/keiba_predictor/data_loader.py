from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import difflib
import re

import numpy as np
import pandas as pd


CANONICAL_ALIASES: Dict[str, List[str]] = {
    "race_id": ["raceid", "race_id", "レースid", "レースID", "race"],
    "race_date": ["racedate", "race_date", "date", "開催日", "日付"],
    "course": ["course", "track", "venue", "競馬場", "コース"],
    "surface": ["surface", "surface_type", "馬場", "芝ダ", "芝ダート"],
    "distance": ["distance", "dist", "距離"],
    "grade": ["grade", "race_grade", "グレード", "格"],
    "class": ["class", "race_class", "クラス"],
    "field_size": ["fieldsize", "field_size", "頭数"],
    "horse_id": ["horseid", "horse_id", "horse", "馬id", "馬ID"],
    "horse_name": ["horsename", "horse_name", "name", "馬名"],
    "draw": ["draw", "gate", "枠", "枠順"],
    "horse_number": ["horsenumber", "horse_number", "馬番", "馬番号"],
    "sex": ["sex", "gender", "性別", "性"],
    "age": ["age", "年齢", "齢", "歳"],
    "carried_weight": ["carriedweight", "carried_weight", "斤量"],
    "jockey": ["jockey", "騎手"],
    "trainer": ["trainer", "調教師"],
    "body_weight": ["bodyweight", "body_weight", "馬体重"],
    "body_weight_diff": ["bodyweightdiff", "body_weight_diff", "馬体重増減"],
    "odds": ["odds", "win_odds", "単勝", "単勝オッズ"],
    "popularity": ["popularity", "fav_rank", "favorite_rank", "人気"],
    "last_finish": ["lastfinish", "last_finish", "前走着順"],
    "last_distance": ["lastdistance", "last_distance", "前走距離"],
    "last_class": ["lastclass", "last_class", "前走クラス"],
    "last_margin": ["lastmargin", "last_margin", "前走着差"],
    "last_3f": ["last3f", "last_3f", "前走上がり", "上がり", "上がり3f", "上がり3F"],
    "days_since_last": ["dayssincelast", "days_since_last", "休み明け日数", "間隔"],
    "finish_rank": ["finishrank", "finish_rank", "着順"],
    "turn": ["turn", "direction", "course_direction", "turn_direction", "回り", "方向"],
}

NUMERIC_HINTS = {
    "distance",
    "field_size",
    "draw",
    "horse_number",
    "age",
    "carried_weight",
    "body_weight",
    "body_weight_diff",
    "odds",
    "popularity",
    "last_finish",
    "last_distance",
    "last_margin",
    "last_3f",
    "days_since_last",
    "finish_rank",
}

REQUIRED_HISTORY_BASE = ["race_id", "race_date", "finish_rank"]
ENTITY_CANDIDATES = ["horse_id", "horse_name"]
RAW_FEATURE_CANDIDATES = [
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
    "surface",
    "distance",
    "course",
    "grade",
    "class",
    "field_size",
]


class DataValidationError(ValueError):
    pass


def normalize_colname(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace(" ", "").replace("-", "").replace("_", "")
    s = re.sub(r"[^\w一-龥ぁ-んァ-ヴー]+", "", s)
    return s


def _build_alias_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for canonical, aliases in CANONICAL_ALIASES.items():
        lookup[normalize_colname(canonical)] = canonical
        for alias in aliases:
            lookup[normalize_colname(alias)] = canonical
    return lookup


ALIAS_LOOKUP = _build_alias_lookup()


def read_csv_safely(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    encodings = ["utf-8-sig", "utf-8", "cp932"]
    errors: List[str] = []
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:  # pragma: no cover - detail preserved for runtime diagnosis
            errors.append(f"encoding={enc}: {exc}")
    raise IOError(f"CSVの読込に失敗しました: {path}\n" + "\n".join(errors))


def auto_map_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, List[str]]]:
    mapped: Dict[str, str] = {}
    suggestions: Dict[str, List[str]] = {}
    used_targets = set()
    current_columns = list(df.columns)

    for col in current_columns:
        norm = normalize_colname(col)
        target = ALIAS_LOOKUP.get(norm)
        if target is None:
            continue
        if target in used_targets:
            continue
        if col == target:
            used_targets.add(target)
            continue
        if target in current_columns:
            continue
        mapped[col] = target
        used_targets.add(target)

    renamed = df.rename(columns=mapped).copy()

    available_norms = {normalize_colname(c): c for c in renamed.columns}
    missing_canonicals = [c for c in CANONICAL_ALIASES.keys() if c not in renamed.columns]
    for canonical in missing_canonicals:
        matches = difflib.get_close_matches(
            normalize_colname(canonical),
            list(available_norms.keys()),
            n=3,
            cutoff=0.75,
        )
        if matches:
            suggestions[canonical] = [available_norms[m] for m in matches]

    return renamed, mapped, suggestions


def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "race_date" in out.columns:
        out["race_date"] = pd.to_datetime(out["race_date"], errors="coerce")
    for col in NUMERIC_HINTS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def ensure_entity_identifier(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    out = df.copy()
    if "horse_id" in out.columns and out["horse_id"].notna().any():
        if "horse_name" in out.columns:
            out["horse_id"] = out["horse_id"].astype("string").fillna(out["horse_name"].astype("string"))
        else:
            out["horse_id"] = out["horse_id"].astype("string")
        return out, "horse_id"

    if "horse_name" in out.columns and out["horse_name"].notna().any():
        out["horse_id"] = out["horse_name"].astype("string")
        return out, "horse_name_fallback"

    raise DataValidationError("horse_id も horse_name も存在しないため、馬単位の識別子を作成できません。")


def infer_field_size(df: pd.DataFrame, is_entry: bool) -> pd.DataFrame:
    out = df.copy()
    if "field_size" in out.columns and out["field_size"].notna().all():
        return out

    if "race_id" in out.columns and out["race_id"].notna().any():
        out["field_size"] = out.groupby("race_id")["race_id"].transform("size")
        return out

    if is_entry:
        out["field_size"] = len(out)
        return out

    raise DataValidationError("history.csv の field_size を推定できません。race_id または field_size が必要です。")


def validate_schema(history: pd.DataFrame, entry: pd.DataFrame, use_odds: bool) -> Dict[str, List[str]]:
    missing_history = [c for c in REQUIRED_HISTORY_BASE if c not in history.columns]
    missing_entry: List[str] = []

    if not any(c in history.columns for c in ENTITY_CANDIDATES):
        missing_history.append("horse_id or horse_name")
    if not any(c in entry.columns for c in ENTITY_CANDIDATES):
        missing_entry.append("horse_id or horse_name")

    if missing_history or missing_entry:
        raise DataValidationError(
            "必須カラム不足です。"
            f"\nmissing_history={missing_history}"
            f"\nmissing_entry={missing_entry}"
        )

    usable_feature_candidates = set(RAW_FEATURE_CANDIDATES)
    if not use_odds:
        usable_feature_candidates -= {"odds", "popularity"}

    common_features = sorted(list((set(history.columns) & set(entry.columns)) & usable_feature_candidates))
    optional_missing_history = sorted(list(usable_feature_candidates - set(history.columns)))
    optional_missing_entry = sorted(list(usable_feature_candidates - set(entry.columns)))

    aggregate_prerequisites: List[str] = []
    if {"horse_id", "surface", "distance"}.issubset(history.columns) and "horse_id" in entry.columns:
        aggregate_prerequisites.append("horse_long_turf")
    if {"horse_id", "course"}.issubset(history.columns) and {"horse_id", "course"}.issubset(entry.columns):
        aggregate_prerequisites.append("horse_course")
    if {"jockey", "surface", "distance"}.issubset(history.columns) and "jockey" in entry.columns:
        aggregate_prerequisites.append("jockey_long_turf")
    if {"trainer", "surface", "distance"}.issubset(history.columns) and "trainer" in entry.columns:
        aggregate_prerequisites.append("trainer_long_turf")

    if len(common_features) == 0 and len(aggregate_prerequisites) == 0:
        raise DataValidationError(
            "history.csv と entry.csv の共通特徴量が見つかりません。"
            "raw feature も aggregate prerequisite も不足しています。"
        )

    return {
        "common_feature_candidates": common_features,
        "aggregate_prerequisites": aggregate_prerequisites,
        "optional_missing_history": optional_missing_history,
        "optional_missing_entry": optional_missing_entry,
    }


def align_entry_columns(entry: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    out = entry.copy()
    for col in history.columns:
        if col not in out.columns:
            out[col] = np.nan
    return out


def prepare_datasets(
    history_raw: pd.DataFrame,
    entry_raw: pd.DataFrame,
    use_odds: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    history_mapped, history_mapping, history_suggestions = auto_map_columns(history_raw)
    entry_mapped, entry_mapping, entry_suggestions = auto_map_columns(entry_raw)

    history = coerce_dtypes(history_mapped)
    entry = coerce_dtypes(entry_mapped)

    history, history_entity_source = ensure_entity_identifier(history)
    entry, entry_entity_source = ensure_entity_identifier(entry)

    history = infer_field_size(history, is_entry=False)
    entry = infer_field_size(entry, is_entry=True)

    schema_report = validate_schema(history, entry, use_odds=use_odds)
    entry = align_entry_columns(entry, history)

    info: Dict[str, object] = {
        "history_mapping": history_mapping,
        "entry_mapping": entry_mapping,
        "history_mapping_suggestions": history_suggestions,
        "entry_mapping_suggestions": entry_suggestions,
        "history_entity_source": history_entity_source,
        "entry_entity_source": entry_entity_source,
        **schema_report,
    }
    return history, entry, info


def load_datasets(
    history_path: str | Path,
    entry_path: str | Path,
    use_odds: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    history_raw = read_csv_safely(history_path)
    entry_raw = read_csv_safely(entry_path)
    return prepare_datasets(history_raw=history_raw, entry_raw=entry_raw, use_odds=use_odds)
