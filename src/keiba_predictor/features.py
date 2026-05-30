from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .validation import assert_no_result_leakage
except ImportError:  # pragma: no cover - direct script fallback
    from validation import assert_no_result_leakage


LONG_DISTANCE_THRESHOLD = 2400
TOP3_PRIOR_MEAN = 0.30
FINISH_PCT_PRIOR_MEAN = 0.50
MARKET_EDGE_PRIOR_MEAN = 0.0
SMOOTHING_STRENGTH = 5.0
RECENT_WINDOWS = (3, 5)
RECENCY_DECAY_RATE = 0.85

DISTANCE_BANDS = {
    "sprint": (0, 1400),
    "mile": (1401, 1800),
    "intermediate": (1801, 2200),
    "long": (2201, 2800),
    "stayer": (2801, 9999),
}

CLASS_RANK = {
    "newcomer": 1, "新馬": 1,
    "maiden": 2, "未勝利": 2,
    "allowance": 3, "1勝": 3, "1win": 3,
    "2win": 4, "2勝": 4,
    "3win": 5, "3勝": 5,
    "open": 6, "オープン": 6,
}


def get_distance_band(distance: object) -> Optional[str]:
    try:
        d = float(distance)
    except Exception:
        return None
    for band, (lo, hi) in DISTANCE_BANDS.items():
        if lo <= d <= hi:
            return band
    return None


def is_graded(grade: object) -> bool:
    if grade is None or pd.isna(grade):
        return False
    s = str(grade).strip().lower()
    return any(token in s for token in ["g1", "g2", "g3", "重賞", "listed"])

AGGREGATE_PREFIXES = [
    "horse_overall",
    "horse_surface",
    "horse_long_turf",
    "horse_course",
    "horse_dist_band",
    "horse_graded",
    "horse_condition",
    "jockey_overall",
    "jockey_long_turf",
    "jockey_dist_band",
    "jockey_course",
    "jockey_graded",
    "jockey_trainer",
    "trainer_overall",
    "trainer_long_turf",
    "trainer_dist_band",
]
AGGREGATE_FEATURE_COLUMNS = [
    feature
    for prefix in AGGREGATE_PREFIXES
    for feature in [
        f"{prefix}_starts",
        f"{prefix}_top3_rate",
        f"{prefix}_top3_rate_smooth",
        f"{prefix}_finish_pct_mean",
        f"{prefix}_market_edge_mean",
    ]
]
AGGREGATE_FEATURE_COLUMNS += [
    f"horse_recent{window}_top3_rate"
    for window in RECENT_WINDOWS
]
AGGREGATE_FEATURE_COLUMNS += [
    f"horse_recent{window}_finish_pct_mean"
    for window in RECENT_WINDOWS
]
AGGREGATE_FEATURE_COLUMNS += [
    f"horse_recent{window}_market_edge_mean"
    for window in RECENT_WINDOWS
]
AGGREGATE_FEATURE_COLUMNS += [
    "horse_recent3_finish_pct_ewm",
    "horse_recent5_finish_pct_ewm",
    "horse_finish_trend_3",
    "horse_finish_std_3",
    "horse_best_finish_5",
]

SPARSE_CONTEXT_FEATURE_COLUMNS = [
    "horse_overall_data_sparse",
    "horse_course_data_sparse",
    "horse_dist_band_data_sparse",
    "jockey_overall_data_sparse",
    "trainer_overall_data_sparse",
    "top3_model_sparse_context",
]

RAW_MODEL_FEATURES = [
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
    "last_3f_rank",
    "days_since_last",
    "distance_change",
    "surface",
    "distance",
    "course",
    "grade",
    "class",
    "field_size",
    "distance_change_abs",
    "is_distance_up",
    "is_distance_down",
    "draw_ratio",
    "horse_number_ratio",
    "distance_log",
    "field_size_log",
    "age_x_carried_weight",
    "days_since_last_log1p",
    "log_odds",
    "implied_prob",
    "popularity_inv",
    "last_margin_neg",
    "last_3f_neg",
    "body_weight_missing",
    "body_weight_diff_missing",
    "odds_missing",
    "odds_rank_score",
    "popularity_rank_score",
    "last_finish_rank_score",
    "last_3f_rank_score",
    "carried_weight_rank_score",
    "draw_rank_score",
    "implied_prob_share",
    "carried_weight_field_delta",
    "age_field_delta",
    "log_odds_field_delta",
    "last_finish_field_delta",
    "last_3f_field_delta",
    "class_transition",
    "is_class_up",
    "is_class_down",
    "is_fresh",
    "is_layoff",
    "rest_bucket",
    "draw_inner",
    "draw_outer",
    "last_3f_per_furlong",
    "relative_3f_speed",
    "track_condition",
    "running_style",
    "passing_position",
    "track_bias",
    "running_style_balance",
    "same_style_count",
]

ODDS_RELATED_FEATURES = {
    "odds",
    "popularity",
    "log_odds",
    "implied_prob",
    "popularity_inv",
    "odds_missing",
    "odds_rank_score",
    "popularity_rank_score",
    "implied_prob_share",
    "log_odds_field_delta",
}

RACE_RELATIVE_RANK_FEATURES = [
    ("odds", "odds_rank_score", True),
    ("popularity", "popularity_rank_score", True),
    ("last_finish", "last_finish_rank_score", True),
    ("last_3f", "last_3f_rank_score", True),
    ("carried_weight", "carried_weight_rank_score", True),
    ("draw", "draw_rank_score", True),
]
RACE_DELTA_FEATURES = [
    ("carried_weight", "carried_weight_field_delta"),
    ("age", "age_field_delta"),
    ("log_odds", "log_odds_field_delta"),
    ("last_finish", "last_finish_field_delta"),
    ("last_3f", "last_3f_field_delta"),
]


@dataclass
class RunningStats:
    count: int = 0
    top3: int = 0
    finish_pct_sum: float = 0.0
    market_edge_count: int = 0
    market_edge_sum: float = 0.0

    def update(self, finish_percentile: float, is_top3: int, market_edge: Optional[float] = None) -> None:
        if pd.isna(finish_percentile):
            return
        self.count += 1
        self.top3 += int(is_top3)
        self.finish_pct_sum += float(finish_percentile)
        if market_edge is not None and not pd.isna(market_edge):
            self.market_edge_count += 1
            self.market_edge_sum += float(market_edge)

    def starts(self) -> float:
        return float(self.count) if self.count > 0 else np.nan

    def top3_rate(self) -> float:
        return float(self.top3 / self.count) if self.count > 0 else np.nan

    def top3_rate_smooth(
        self,
        prior_mean: float = TOP3_PRIOR_MEAN,
        prior_strength: float = SMOOTHING_STRENGTH,
    ) -> float:
        denom = self.count + prior_strength
        if denom <= 0:
            return float(prior_mean)
        return float((self.top3 + (prior_strength * prior_mean)) / denom)

    def finish_pct_mean_smooth(
        self,
        prior_mean: float = FINISH_PCT_PRIOR_MEAN,
        prior_strength: float = SMOOTHING_STRENGTH,
    ) -> float:
        denom = self.count + prior_strength
        if denom <= 0:
            return float(prior_mean)
        return float((self.finish_pct_sum + (prior_strength * prior_mean)) / denom)

    def market_edge_mean_smooth(
        self,
        prior_mean: float = MARKET_EDGE_PRIOR_MEAN,
        prior_strength: float = SMOOTHING_STRENGTH,
    ) -> float:
        denom = self.market_edge_count + prior_strength
        if denom <= 0:
            return float(prior_mean)
        return float((self.market_edge_sum + (prior_strength * prior_mean)) / denom)


def _safe_key(value: object) -> Optional[str]:
    if value is None:
        return None
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s


def _string_contains(series: pd.Series, keywords: Iterable[str]) -> pd.Series:
    pattern = "|".join([rf"(?:{k})" for k in keywords])
    return series.astype("string").str.contains(pattern, case=False, regex=True, na=False)


def is_turf_value(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    s = str(value).strip().lower()
    return any(token in s for token in ["芝", "turf", "grass"])


def is_long_turf(surface: object, distance: object, threshold: int = LONG_DISTANCE_THRESHOLD) -> bool:
    try:
        dist = float(distance)
    except Exception:
        return False
    return is_turf_value(surface) and dist >= threshold


def _to_string_key(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _normalize_running_style_label(value: object) -> Optional[str]:
    text = _safe_key(value)
    if text is None:
        return None
    lowered = text.lower()
    if any(token in lowered for token in ["逃", "先", "front", "lead", "逃げ", "先行"]):
        return "front"
    if any(token in lowered for token in ["差", "追", "close", "stalk", "追込", "差し"]):
        return "close"
    return lowered


def _ewm_mean(values: List[float], decay: float = RECENCY_DECAY_RATE) -> float:
    """指数減衰重み付き平均。末尾 (最新) に最も高い重みを付ける。"""
    if not values:
        return float("nan")
    w = np.array([decay ** (len(values) - 1 - i) for i in range(len(values))], dtype=float)
    w /= w.sum()
    return float(np.dot(w, values))


def _linear_slope(values: List[float]) -> float:
    """等間隔時系列の線形回帰スロープ。正値 = 改善傾向。"""
    n = len(values)
    if n < 2:
        return float("nan")
    x = np.arange(n, dtype=float)
    x -= x.mean()
    y = np.array(values, dtype=float) - float(np.mean(values))
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return 0.0
    return float(np.dot(x, y) / denom)


def _target_profile(config: Dict[str, object]) -> Dict[str, object]:
    return dict(config.get("target_race_profile", {}) or {})


def _market_edge(finish_percentile: object, odds_rank_score: object, implied_prob_share: object) -> Optional[float]:
    finish_pct = _to_float(finish_percentile)
    if finish_pct is None:
        return None

    market_score = _to_float(odds_rank_score)
    if market_score is None:
        market_score = _to_float(implied_prob_share)
    if market_score is None:
        return None
    return float(finish_pct - market_score)


def _season_label_from_months(months: object) -> Optional[str]:
    month_tuple = tuple(sorted(int(month) for month in months or []))
    mapping = {
        (1, 2, 12): "winter",
        (3, 4, 5): "spring",
        (6, 7, 8): "summer",
        (9, 10, 11): "autumn",
    }
    if month_tuple in mapping:
        return mapping[month_tuple]
    return None


def _weight_with_legacy_fallback(
    cfg: Dict[str, object],
    primary_key: str,
    legacy_keys: Iterable[str],
    default: float,
) -> float:
    if primary_key in cfg:
        return float(cfg[primary_key])
    for key in legacy_keys:
        if key in cfg:
            return float(cfg[key])
    return float(default)


def add_targets(history: pd.DataFrame) -> pd.DataFrame:
    out = history.copy()
    if "field_size" not in out.columns:
        raise ValueError("field_size が必要です。")
    if "finish_rank" not in out.columns:
        raise ValueError("finish_rank が必要です。")

    denom = (out["field_size"] - 1).replace(0, np.nan)
    finish_percentile = 1.0 - ((out["finish_rank"] - 1.0) / denom)
    single_runner = denom.isna()
    finish_percentile = finish_percentile.where(~single_runner, (out["finish_rank"] == 1).astype(float))
    out["finish_percentile"] = finish_percentile.clip(0.0, 1.0)
    out["is_top3"] = (out["finish_rank"] <= 3).astype(int)
    return out


# 前走(last_*)特徴量を、各 runner の「直前の出走」から引くためのマッピング。
# value 側は long-history 上の列名 (前走の結果)。
_LAG_SOURCE_MAP = {
    "last_finish": "finish_rank",
    "last_distance": "distance",
    "last_class": "class",
    "last_3f": "result_last3f",
    "last_margin": "result_margin",
}


def build_lag_features(history: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    """各馬を「自分の直前の出走」に紐づけて last_* / days_since_last を生成する。

    20年分のスクレイプ履歴を取り込む際、各 runner に前走情報が無いケースで、
    時系列順に strict に過去のみを参照して前走の結果から埋める (look-ahead なし)。
    これがデータ取り込みの本当の難所であり、行数を増やすこと自体より重要。

    Args:
        history: race_id / race_date / horse_id / finish_rank などを含む長期履歴。
        overwrite: True なら既存の last_* 値も上書きする。False(既定)では
            欠損(NaN)のみを補完し、既存の非NaN値は保持する。
    """
    if "horse_id" not in history.columns or "race_date" not in history.columns:
        return history.copy()

    df = history.copy()
    df["__lag_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    order = df.sort_values(["horse_id", "__lag_date", "race_id"], kind="mergesort").index
    ordered = df.loc[order]
    grouped = ordered.groupby("horse_id", sort=False)

    computed: Dict[str, pd.Series] = {}
    for target_col, source_col in _LAG_SOURCE_MAP.items():
        if source_col in ordered.columns:
            computed[target_col] = grouped[source_col].shift(1)
    prior_date = grouped["__lag_date"].shift(1)
    computed["days_since_last"] = (ordered["__lag_date"] - prior_date).dt.days

    for target_col, series in computed.items():
        aligned = series.reindex(df.index)
        if overwrite or target_col not in df.columns:
            df[target_col] = aligned
        else:
            df[target_col] = df[target_col].where(df[target_col].notna(), aligned)

    return df.drop(columns="__lag_date")


def _add_row_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"distance", "last_distance"}.issubset(out.columns):
        out["distance_change"] = out["distance"] - out["last_distance"]
        out["distance_change_abs"] = out["distance_change"].abs()
        out["is_distance_up"] = (out["distance_change"] > 0).astype(float)
        out["is_distance_down"] = (out["distance_change"] < 0).astype(float)

    if {"draw", "field_size"}.issubset(out.columns):
        denom = pd.to_numeric(out["field_size"], errors="coerce").replace(0, np.nan)
        out["draw_ratio"] = pd.to_numeric(out["draw"], errors="coerce") / denom

    if {"horse_number", "field_size"}.issubset(out.columns):
        denom = pd.to_numeric(out["field_size"], errors="coerce").replace(0, np.nan)
        out["horse_number_ratio"] = pd.to_numeric(out["horse_number"], errors="coerce") / denom

    if "distance" in out.columns:
        dist = pd.to_numeric(out["distance"], errors="coerce")
        out["distance_log"] = np.log(dist.where(dist > 0))

    if "field_size" in out.columns:
        field_size = pd.to_numeric(out["field_size"], errors="coerce")
        out["field_size_log"] = np.log1p(field_size.where(field_size >= 0))

    if {"age", "carried_weight"}.issubset(out.columns):
        out["age_x_carried_weight"] = pd.to_numeric(out["age"], errors="coerce") * pd.to_numeric(
            out["carried_weight"], errors="coerce"
        )

    if "days_since_last" in out.columns:
        rest = pd.to_numeric(out["days_since_last"], errors="coerce")
        out["days_since_last_log1p"] = np.log1p(rest.where(rest >= 0))

    if "odds" in out.columns:
        odds = pd.to_numeric(out["odds"], errors="coerce")
        valid_odds = odds.where(odds > 1.0)
        out["log_odds"] = np.log(valid_odds)
        out["implied_prob"] = 1.0 / valid_odds
        out["odds_missing"] = odds.isna().astype(float)

    if "popularity" in out.columns:
        popularity = pd.to_numeric(out["popularity"], errors="coerce")
        out["popularity_inv"] = 1.0 / popularity.where(popularity > 0)

    if "last_margin" in out.columns:
        out["last_margin_neg"] = -pd.to_numeric(out["last_margin"], errors="coerce")

    if "last_3f" in out.columns:
        out["last_3f_neg"] = -pd.to_numeric(out["last_3f"], errors="coerce")

    if "body_weight" in out.columns:
        out["body_weight_missing"] = out["body_weight"].isna().astype(float)

    if "body_weight_diff" in out.columns:
        out["body_weight_diff_missing"] = out["body_weight_diff"].isna().astype(float)

    if {"class", "last_class"}.issubset(out.columns):
        current_rank = out["class"].astype(str).str.strip().str.lower().map(CLASS_RANK)
        previous_rank = out["last_class"].astype(str).str.strip().str.lower().map(CLASS_RANK)
        out["class_transition"] = current_rank - previous_rank
        out["is_class_up"] = (out["class_transition"] > 0).astype(float)
        out["is_class_down"] = (out["class_transition"] < 0).astype(float)

    if "days_since_last" in out.columns:
        rest = pd.to_numeric(out["days_since_last"], errors="coerce")
        out["is_fresh"] = (rest <= 35).astype(float)
        out["is_layoff"] = (rest >= 120).astype(float)
        out["rest_bucket"] = np.select(
            [rest <= 21, rest <= 35, rest <= 90, rest <= 180],
            ["very_short", "short", "normal", "long"],
            default="very_long",
        )
        out["rest_bucket"] = out["rest_bucket"].where(rest.notna(), other=pd.NA)

    if {"draw", "field_size"}.issubset(out.columns):
        draw = pd.to_numeric(out["draw"], errors="coerce")
        field_size = pd.to_numeric(out["field_size"], errors="coerce")
        out["draw_inner"] = (draw <= 4).astype(float)
        out["draw_outer"] = (draw > field_size * 0.7).astype(float)

    if "last_3f" in out.columns and "distance" in out.columns:
        last_3f = pd.to_numeric(out["last_3f"], errors="coerce")
        dist = pd.to_numeric(out["distance"], errors="coerce")
        out["last_3f_per_furlong"] = last_3f / 3.0
        furlongs = dist / 200.0
        out["relative_3f_speed"] = last_3f / furlongs.replace(0, np.nan)

    return out


def _race_complete_mask(df: pd.DataFrame) -> pd.Series:
    if not {"race_id", "field_size"}.issubset(df.columns):
        return pd.Series(False, index=df.index)

    observed = df.groupby("race_id")["race_id"].transform("size")
    field_size = pd.to_numeric(df["field_size"], errors="coerce")
    return observed.eq(field_size) & field_size.notna()


def _rank_score(series: pd.Series, group_key: pd.Series, ascending: bool) -> pd.Series:
    rank = series.groupby(group_key).rank(method="average", ascending=ascending, na_option="keep")
    count = series.groupby(group_key).transform("count")
    denom = (count - 1).replace(0, np.nan)
    score = 1.0 - ((rank - 1.0) / denom)
    return score.where(count > 1, 1.0)


def _center_delta(series: pd.Series, group_key: pd.Series) -> pd.Series:
    group_mean = series.groupby(group_key).transform("mean")
    return series - group_mean


def _add_race_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "race_id" not in out.columns:
        return out

    complete_mask = _race_complete_mask(out)
    race_key = out["race_id"]

    for source_col, target_col, ascending in RACE_RELATIVE_RANK_FEATURES:
        if source_col not in out.columns:
            continue
        numeric = pd.to_numeric(out[source_col], errors="coerce")
        score = _rank_score(numeric, race_key, ascending=ascending)
        out[target_col] = score.where(complete_mask & numeric.notna(), np.nan)

    for source_col, target_col in RACE_DELTA_FEATURES:
        if source_col not in out.columns:
            continue
        numeric = pd.to_numeric(out[source_col], errors="coerce")
        delta = _center_delta(numeric, race_key)
        out[target_col] = delta.where(complete_mask & numeric.notna(), np.nan)

    if "implied_prob" in out.columns:
        implied = pd.to_numeric(out["implied_prob"], errors="coerce")
        implied_sum = implied.groupby(race_key).transform("sum")
        share = implied / implied_sum.replace(0, np.nan)
        out["implied_prob_share"] = share.where(complete_mask & implied.notna(), np.nan)

    return out


def _add_running_style_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "race_id" not in out.columns or "running_style" not in out.columns:
        return out

    normalized_style = out["running_style"].map(_normalize_running_style_label)
    work = pd.DataFrame(
        {
            "race_id": out["race_id"],
            "_style": normalized_style,
        },
        index=out.index,
    )
    valid = work["_style"].notna()
    if not valid.any():
        return out

    style_counts = work.loc[valid].groupby(["race_id", "_style"])["_style"].transform("size")
    group_sizes = work.groupby("race_id")["race_id"].transform("size")
    out.loc[valid, "same_style_count"] = style_counts.to_numpy(dtype=float) - 1.0
    out.loc[valid, "running_style_balance"] = style_counts.to_numpy(dtype=float) / group_sizes.loc[valid].to_numpy(dtype=float)
    return out


def add_basic_features(history: pd.DataFrame, entry: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    history_out = _add_running_style_features(_add_race_relative_features(_add_row_derived_features(history)))
    entry_out = _add_running_style_features(_add_race_relative_features(_add_row_derived_features(entry)))
    return history_out, entry_out


def compute_similarity_weights(history: pd.DataFrame, config: Dict[str, object]) -> pd.Series:
    out = pd.Series(np.ones(len(history), dtype=float), index=history.index, name="sample_weight")
    cfg = config.get("similarity_weights", {})
    profile = _target_profile(config)
    target_surface = _to_string_key(profile.get("surface"))
    target_distance = _to_float(profile.get("distance"))
    target_course = _to_string_key(profile.get("course"))
    target_turn = _to_string_key(profile.get("turn"))
    season_months = profile.get("season_months", profile.get("spring_months", [3, 4, 5]))
    season_label = _to_string_key(profile.get("season_label")) or _season_label_from_months(season_months)

    if {"surface", "distance"}.issubset(history.columns):
        turf = history["surface"].apply(is_turf_value)
        dist = pd.to_numeric(history["distance"], errors="coerce")
        if target_surface is not None and target_distance is not None:
            exact_mask = history["surface"].astype("string").str.lower().eq(target_surface.lower()) & dist.eq(target_distance)
            near_tolerance = float(cfg.get("near_distance_tolerance", 200.0))
            wide_tolerance = float(cfg.get("wide_distance_tolerance", 600.0))
            near_mask = history["surface"].astype("string").str.lower().eq(target_surface.lower()) & dist.between(
                target_distance - near_tolerance,
                target_distance + near_tolerance,
                inclusive="both",
            )
            wide_mask = history["surface"].astype("string").str.lower().eq(target_surface.lower()) & dist.between(
                target_distance - wide_tolerance,
                target_distance + wide_tolerance,
                inclusive="both",
            )
            surface_slug = target_surface.lower()
            legacy_exact = [f"{surface_slug}_{int(target_distance)}"] if float(target_distance).is_integer() else []
            legacy_near = [
                f"{surface_slug}_{int(target_distance - near_tolerance)}_{int(target_distance + near_tolerance)}"
            ]
            legacy_wide = [
                f"{surface_slug}_{int(target_distance - wide_tolerance)}_{int(target_distance + wide_tolerance)}"
            ]
            out += np.where(
                exact_mask,
                _weight_with_legacy_fallback(cfg, "exact_surface_distance", legacy_exact, 5.0),
                0.0,
            )
            out += np.where(
                near_mask,
                _weight_with_legacy_fallback(cfg, "near_surface_distance", legacy_near, 3.0),
                0.0,
            )
            out += np.where(
                wide_mask,
                _weight_with_legacy_fallback(cfg, "wide_surface_distance", legacy_wide, 1.5),
                0.0,
            )
        else:
            out += np.where(turf & dist.eq(3000), float(cfg.get("turf_3000", 5.0)), 0.0)
            out += np.where(turf & dist.between(2600, 3200, inclusive="both"), float(cfg.get("turf_2600_3200", 3.0)), 0.0)
            out += np.where(turf & dist.between(2400, 3600, inclusive="both"), float(cfg.get("turf_2400_3600", 1.5)), 0.0)

    if "age" in history.columns:
        age = pd.to_numeric(history["age"], errors="coerce")
        out += np.where(age >= 4, float(cfg.get("age_4plus", 1.0)), 0.0)

    grade_hit = pd.Series(False, index=history.index)
    class_hit = pd.Series(False, index=history.index)
    if "grade" in history.columns:
        grade_hit = _string_contains(history["grade"], ["G1", "G2", "G3", "重賞", "graded"])
    if "class" in history.columns:
        class_hit = _string_contains(history["class"], ["open", "オープン", "listed", "L"])
    out += np.where(grade_hit | class_hit, float(cfg.get("open_graded", 1.0)), 0.0)

    if "course" in history.columns and target_course is not None:
        course_hit = history["course"].astype("string").str.lower().eq(target_course.lower())
        out += np.where(
            course_hit,
            _weight_with_legacy_fallback(cfg, "same_course", [target_course.lower()], 0.8),
            0.0,
        )

    for turn_col in ["turn", "direction", "course_direction", "turn_direction"]:
        if turn_col in history.columns and target_turn is not None:
            turn_hit = history[turn_col].astype("string").str.lower().eq(target_turn.lower())
            out += np.where(
                turn_hit,
                _weight_with_legacy_fallback(cfg, "same_turn", [f"{target_turn.lower()}_turn"], 0.6),
                0.0,
            )
            break

    if "race_date" in history.columns:
        race_month = pd.to_datetime(history["race_date"], errors="coerce").dt.month
        out += np.where(
            race_month.isin(season_months),
            _weight_with_legacy_fallback(
                cfg,
                "same_season",
                [season_label] if season_label is not None else [],
                0.4,
            ),
            0.0,
        )

    return out.astype(float)


def _assign_stats(feature_store: Dict[str, np.ndarray], pos: int, prefix: str, stats: Optional[RunningStats]) -> None:
    if stats is None or stats.count == 0:
        feature_store[f"{prefix}_starts"][pos] = 0.0
        feature_store[f"{prefix}_top3_rate"][pos] = 0.0
        feature_store[f"{prefix}_top3_rate_smooth"][pos] = TOP3_PRIOR_MEAN
        feature_store[f"{prefix}_finish_pct_mean"][pos] = FINISH_PCT_PRIOR_MEAN
        feature_store[f"{prefix}_market_edge_mean"][pos] = MARKET_EDGE_PRIOR_MEAN
        return
    feature_store[f"{prefix}_starts"][pos] = stats.starts()
    feature_store[f"{prefix}_top3_rate"][pos] = stats.top3_rate()
    feature_store[f"{prefix}_top3_rate_smooth"][pos] = stats.top3_rate_smooth()
    feature_store[f"{prefix}_finish_pct_mean"][pos] = stats.finish_pct_mean_smooth()
    feature_store[f"{prefix}_market_edge_mean"][pos] = stats.market_edge_mean_smooth()


def _assign_recent_stats(
    feature_store: Dict[str, np.ndarray],
    pos: int,
    history_records: Optional[List[Tuple[float, int, Optional[float]]]],
) -> None:
    records = history_records or []
    for window in RECENT_WINDOWS:
        col_prefix = f"horse_recent{window}"
        recent = records[-window:]
        if len(recent) == 0:
            feature_store[f"{col_prefix}_top3_rate"][pos] = TOP3_PRIOR_MEAN
            feature_store[f"{col_prefix}_finish_pct_mean"][pos] = FINISH_PCT_PRIOR_MEAN
            feature_store[f"{col_prefix}_market_edge_mean"][pos] = MARKET_EDGE_PRIOR_MEAN
            continue
        finish_values = [float(finish_pct) for finish_pct, _, _ in recent if not pd.isna(finish_pct)]
        top3_values = [int(is_top3) for _, is_top3, _ in recent]
        market_values = [float(market_edge) for _, _, market_edge in recent if market_edge is not None and not pd.isna(market_edge)]
        feature_store[f"{col_prefix}_top3_rate"][pos] = float(np.mean(top3_values)) if top3_values else np.nan
        feature_store[f"{col_prefix}_finish_pct_mean"][pos] = float(np.mean(finish_values)) if finish_values else np.nan
        feature_store[f"{col_prefix}_market_edge_mean"][pos] = float(np.mean(market_values)) if market_values else np.nan

    # --- EWM features (window 3 and 5) ---
    for window, ewm_col in [(3, "horse_recent3_finish_pct_ewm"), (5, "horse_recent5_finish_pct_ewm")]:
        recent = records[-window:]
        finish_vals = [float(fp) for fp, _, _ in recent if not pd.isna(fp)]
        if ewm_col in feature_store:
            feature_store[ewm_col][pos] = _ewm_mean(finish_vals) if finish_vals else FINISH_PCT_PRIOR_MEAN

    # --- Trend and std (window 3) ---
    recent3 = records[-3:]
    finish3 = [float(fp) for fp, _, _ in recent3 if not pd.isna(fp)]
    if "horse_finish_trend_3" in feature_store:
        feature_store["horse_finish_trend_3"][pos] = _linear_slope(finish3) if len(finish3) >= 2 else np.nan
    if "horse_finish_std_3" in feature_store:
        feature_store["horse_finish_std_3"][pos] = float(np.std(finish3)) if len(finish3) >= 2 else np.nan

    # --- Best finish (window 5) ---
    recent5 = records[-5:]
    finish5 = [float(fp) for fp, _, _ in recent5 if not pd.isna(fp)]
    if "horse_best_finish_5" in feature_store:
        feature_store["horse_best_finish_5"][pos] = float(max(finish5)) if finish5 else np.nan


def _empty_feature_store(n_rows: int) -> Dict[str, np.ndarray]:
    return {col: np.full(n_rows, np.nan, dtype=float) for col in AGGREGATE_FEATURE_COLUMNS}


def _add_sparse_context_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _starts(column: str) -> pd.Series:
        if column not in out.columns:
            return pd.Series(0.0, index=out.index, dtype=float)
        return pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    horse_overall_starts = _starts("horse_overall_starts")
    horse_course_starts = _starts("horse_course_starts")
    horse_dist_band_starts = _starts("horse_dist_band_starts")
    jockey_overall_starts = _starts("jockey_overall_starts")
    trainer_overall_starts = _starts("trainer_overall_starts")

    out["horse_overall_data_sparse"] = (horse_overall_starts < 3.0).astype(float)
    out["horse_course_data_sparse"] = (horse_course_starts < 1.0).astype(float)
    out["horse_dist_band_data_sparse"] = (horse_dist_band_starts < 1.0).astype(float)
    out["jockey_overall_data_sparse"] = (jockey_overall_starts < 3.0).astype(float)
    out["trainer_overall_data_sparse"] = (trainer_overall_starts < 3.0).astype(float)
    out["top3_model_sparse_context"] = (
        (out["horse_overall_data_sparse"] > 0.0)
        & (out["horse_course_data_sparse"] > 0.0)
        & (out["horse_dist_band_data_sparse"] > 0.0)
    ).astype(float)
    return out


def _build_history_aggregate_features(history: pd.DataFrame) -> pd.DataFrame:
    work = history.copy().sort_values(["race_date", "race_id"], kind="mergesort").reset_index(drop=True)
    work["_race_day"] = pd.to_datetime(work["race_date"], errors="coerce").dt.normalize()

    feature_store = _empty_feature_store(len(work))
    horse_overall: Dict[str, RunningStats] = defaultdict(RunningStats)
    horse_surface: Dict[Tuple[str, str], RunningStats] = defaultdict(RunningStats)
    horse_long: Dict[str, RunningStats] = defaultdict(RunningStats)
    horse_course: Dict[Tuple[str, str], RunningStats] = defaultdict(RunningStats)
    horse_dist_band: Dict[Tuple[str, str], RunningStats] = defaultdict(RunningStats)
    horse_graded: Dict[str, RunningStats] = defaultdict(RunningStats)
    horse_condition: Dict[Tuple[str, str], RunningStats] = defaultdict(RunningStats)
    jockey_overall: Dict[str, RunningStats] = defaultdict(RunningStats)
    jockey_long: Dict[str, RunningStats] = defaultdict(RunningStats)
    jockey_dist_band: Dict[Tuple[str, str], RunningStats] = defaultdict(RunningStats)
    jockey_course: Dict[Tuple[str, str], RunningStats] = defaultdict(RunningStats)
    jockey_graded: Dict[str, RunningStats] = defaultdict(RunningStats)
    jockey_trainer: Dict[Tuple[str, str], RunningStats] = defaultdict(RunningStats)
    trainer_overall: Dict[str, RunningStats] = defaultdict(RunningStats)
    trainer_long: Dict[str, RunningStats] = defaultdict(RunningStats)
    trainer_dist_band: Dict[Tuple[str, str], RunningStats] = defaultdict(RunningStats)
    horse_recent: Dict[str, List[Tuple[float, int, Optional[float]]]] = defaultdict(list)

    for _, day_df in work.groupby("_race_day", sort=True):
        for row in day_df.itertuples():
            pos = int(row.Index)
            horse_key = _safe_key(getattr(row, "horse_id", None))
            jockey_key = _safe_key(getattr(row, "jockey", None))
            trainer_key = _safe_key(getattr(row, "trainer", None))
            course_key = _safe_key(getattr(row, "course", None))
            surface_key = _safe_key(getattr(row, "surface", None))
            condition_key = _safe_key(getattr(row, "track_condition", None))
            dist_band = get_distance_band(getattr(row, "distance", None))

            _assign_stats(feature_store, pos, "horse_overall", horse_overall.get(horse_key))
            if horse_key is not None and surface_key is not None:
                _assign_stats(feature_store, pos, "horse_surface", horse_surface.get((horse_key, surface_key)))
            else:
                _assign_stats(feature_store, pos, "horse_surface", None)
            _assign_stats(feature_store, pos, "horse_long_turf", horse_long.get(horse_key))
            if horse_key is not None and course_key is not None:
                _assign_stats(feature_store, pos, "horse_course", horse_course.get((horse_key, course_key)))
            else:
                _assign_stats(feature_store, pos, "horse_course", None)
            if horse_key is not None and dist_band is not None:
                _assign_stats(feature_store, pos, "horse_dist_band", horse_dist_band.get((horse_key, dist_band)))
            else:
                _assign_stats(feature_store, pos, "horse_dist_band", None)
            _assign_stats(feature_store, pos, "horse_graded", horse_graded.get(horse_key))
            if horse_key is not None and condition_key is not None:
                _assign_stats(feature_store, pos, "horse_condition", horse_condition.get((horse_key, condition_key)))
            else:
                _assign_stats(feature_store, pos, "horse_condition", None)
            _assign_stats(feature_store, pos, "jockey_overall", jockey_overall.get(jockey_key))
            _assign_stats(feature_store, pos, "jockey_long_turf", jockey_long.get(jockey_key))
            if jockey_key is not None and dist_band is not None:
                _assign_stats(feature_store, pos, "jockey_dist_band", jockey_dist_band.get((jockey_key, dist_band)))
            else:
                _assign_stats(feature_store, pos, "jockey_dist_band", None)
            if jockey_key is not None and course_key is not None:
                _assign_stats(feature_store, pos, "jockey_course", jockey_course.get((jockey_key, course_key)))
            else:
                _assign_stats(feature_store, pos, "jockey_course", None)
            _assign_stats(feature_store, pos, "jockey_graded", jockey_graded.get(jockey_key))
            if jockey_key is not None and trainer_key is not None:
                _assign_stats(feature_store, pos, "jockey_trainer", jockey_trainer.get((jockey_key, trainer_key)))
            else:
                _assign_stats(feature_store, pos, "jockey_trainer", None)
            _assign_stats(feature_store, pos, "trainer_overall", trainer_overall.get(trainer_key))
            _assign_stats(feature_store, pos, "trainer_long_turf", trainer_long.get(trainer_key))
            if trainer_key is not None and dist_band is not None:
                _assign_stats(feature_store, pos, "trainer_dist_band", trainer_dist_band.get((trainer_key, dist_band)))
            else:
                _assign_stats(feature_store, pos, "trainer_dist_band", None)
            _assign_recent_stats(feature_store, pos, horse_recent.get(horse_key))

        for row in day_df.itertuples():
            horse_key = _safe_key(getattr(row, "horse_id", None))
            jockey_key = _safe_key(getattr(row, "jockey", None))
            trainer_key = _safe_key(getattr(row, "trainer", None))
            course_key = _safe_key(getattr(row, "course", None))
            surface_key = _safe_key(getattr(row, "surface", None))
            condition_key = _safe_key(getattr(row, "track_condition", None))
            finish_pct = getattr(row, "finish_percentile")
            is_top3 = getattr(row, "is_top3")
            long_turf = is_long_turf(getattr(row, "surface", None), getattr(row, "distance", None))
            dist_band = get_distance_band(getattr(row, "distance", None))
            graded = is_graded(getattr(row, "grade", None))
            market_edge = _market_edge(
                finish_percentile=finish_pct,
                odds_rank_score=getattr(row, "odds_rank_score", None),
                implied_prob_share=getattr(row, "implied_prob_share", None),
            )

            if horse_key is not None:
                horse_overall[horse_key].update(finish_pct, is_top3, market_edge=market_edge)
                horse_recent[horse_key].append((finish_pct, is_top3, market_edge))
                if surface_key is not None:
                    horse_surface[(horse_key, surface_key)].update(finish_pct, is_top3, market_edge=market_edge)
                if long_turf:
                    horse_long[horse_key].update(finish_pct, is_top3, market_edge=market_edge)
                if course_key is not None:
                    horse_course[(horse_key, course_key)].update(finish_pct, is_top3, market_edge=market_edge)
                if dist_band is not None:
                    horse_dist_band[(horse_key, dist_band)].update(finish_pct, is_top3, market_edge=market_edge)
                if graded:
                    horse_graded[horse_key].update(finish_pct, is_top3, market_edge=market_edge)
                if condition_key is not None:
                    horse_condition[(horse_key, condition_key)].update(finish_pct, is_top3, market_edge=market_edge)
            if jockey_key is not None:
                jockey_overall[jockey_key].update(finish_pct, is_top3, market_edge=market_edge)
                if long_turf:
                    jockey_long[jockey_key].update(finish_pct, is_top3, market_edge=market_edge)
                if dist_band is not None:
                    jockey_dist_band[(jockey_key, dist_band)].update(finish_pct, is_top3, market_edge=market_edge)
                if course_key is not None:
                    jockey_course[(jockey_key, course_key)].update(finish_pct, is_top3, market_edge=market_edge)
                if graded:
                    jockey_graded[jockey_key].update(finish_pct, is_top3, market_edge=market_edge)
                if trainer_key is not None:
                    jockey_trainer[(jockey_key, trainer_key)].update(finish_pct, is_top3, market_edge=market_edge)
            if trainer_key is not None:
                trainer_overall[trainer_key].update(finish_pct, is_top3, market_edge=market_edge)
                if long_turf:
                    trainer_long[trainer_key].update(finish_pct, is_top3, market_edge=market_edge)
                if dist_band is not None:
                    trainer_dist_band[(trainer_key, dist_band)].update(finish_pct, is_top3, market_edge=market_edge)

    for col, values in feature_store.items():
        work[col] = values
    return work.drop(columns=["_race_day"])


@dataclass
class AggregatedLookups:
    horse_overall: Dict[str, RunningStats]
    horse_surface: Dict[Tuple[str, str], RunningStats]
    horse_long: Dict[str, RunningStats]
    horse_course: Dict[Tuple[str, str], RunningStats]
    horse_dist_band: Dict[Tuple[str, str], RunningStats]
    horse_graded: Dict[str, RunningStats]
    horse_condition: Dict[Tuple[str, str], RunningStats]
    jockey_overall: Dict[str, RunningStats]
    jockey_long: Dict[str, RunningStats]
    jockey_dist_band: Dict[Tuple[str, str], RunningStats]
    jockey_course: Dict[Tuple[str, str], RunningStats]
    jockey_graded: Dict[str, RunningStats]
    jockey_trainer: Dict[Tuple[str, str], RunningStats]
    trainer_overall: Dict[str, RunningStats]
    trainer_long: Dict[str, RunningStats]
    trainer_dist_band: Dict[Tuple[str, str], RunningStats]
    horse_recent: Dict[str, List[Tuple[float, int, Optional[float]]]]


def _build_final_lookups(history_subset: pd.DataFrame) -> AggregatedLookups:
    horse_overall: Dict[str, RunningStats] = defaultdict(RunningStats)
    horse_surface: Dict[Tuple[str, str], RunningStats] = defaultdict(RunningStats)
    horse_long: Dict[str, RunningStats] = defaultdict(RunningStats)
    horse_course: Dict[Tuple[str, str], RunningStats] = defaultdict(RunningStats)
    horse_dist_band: Dict[Tuple[str, str], RunningStats] = defaultdict(RunningStats)
    horse_graded: Dict[str, RunningStats] = defaultdict(RunningStats)
    horse_condition: Dict[Tuple[str, str], RunningStats] = defaultdict(RunningStats)
    jockey_overall: Dict[str, RunningStats] = defaultdict(RunningStats)
    jockey_long: Dict[str, RunningStats] = defaultdict(RunningStats)
    jockey_dist_band: Dict[Tuple[str, str], RunningStats] = defaultdict(RunningStats)
    jockey_course: Dict[Tuple[str, str], RunningStats] = defaultdict(RunningStats)
    jockey_graded: Dict[str, RunningStats] = defaultdict(RunningStats)
    jockey_trainer: Dict[Tuple[str, str], RunningStats] = defaultdict(RunningStats)
    trainer_overall: Dict[str, RunningStats] = defaultdict(RunningStats)
    trainer_long: Dict[str, RunningStats] = defaultdict(RunningStats)
    trainer_dist_band: Dict[Tuple[str, str], RunningStats] = defaultdict(RunningStats)
    horse_recent: Dict[str, List[Tuple[float, int, Optional[float]]]] = defaultdict(list)

    ordered = history_subset.sort_values(["race_date", "race_id"], kind="mergesort")
    for row in ordered.itertuples(index=False):
        horse_key = _safe_key(getattr(row, "horse_id", None))
        jockey_key = _safe_key(getattr(row, "jockey", None))
        trainer_key = _safe_key(getattr(row, "trainer", None))
        course_key = _safe_key(getattr(row, "course", None))
        surface_key = _safe_key(getattr(row, "surface", None))
        condition_key = _safe_key(getattr(row, "track_condition", None))
        finish_pct = getattr(row, "finish_percentile")
        is_top3 = getattr(row, "is_top3")
        long_turf = is_long_turf(getattr(row, "surface", None), getattr(row, "distance", None))
        dist_band = get_distance_band(getattr(row, "distance", None))
        graded = is_graded(getattr(row, "grade", None))
        market_edge = _market_edge(
            finish_percentile=finish_pct,
            odds_rank_score=getattr(row, "odds_rank_score", None),
            implied_prob_share=getattr(row, "implied_prob_share", None),
        )

        if horse_key is not None:
            horse_overall[horse_key].update(finish_pct, is_top3, market_edge=market_edge)
            horse_recent[horse_key].append((finish_pct, is_top3, market_edge))
            if surface_key is not None:
                horse_surface[(horse_key, surface_key)].update(finish_pct, is_top3, market_edge=market_edge)
            if long_turf:
                horse_long[horse_key].update(finish_pct, is_top3, market_edge=market_edge)
            if course_key is not None:
                horse_course[(horse_key, course_key)].update(finish_pct, is_top3, market_edge=market_edge)
            if dist_band is not None:
                horse_dist_band[(horse_key, dist_band)].update(finish_pct, is_top3, market_edge=market_edge)
            if graded:
                horse_graded[horse_key].update(finish_pct, is_top3, market_edge=market_edge)
            if condition_key is not None:
                horse_condition[(horse_key, condition_key)].update(finish_pct, is_top3, market_edge=market_edge)
        if jockey_key is not None:
            jockey_overall[jockey_key].update(finish_pct, is_top3, market_edge=market_edge)
            if long_turf:
                jockey_long[jockey_key].update(finish_pct, is_top3, market_edge=market_edge)
            if dist_band is not None:
                jockey_dist_band[(jockey_key, dist_band)].update(finish_pct, is_top3, market_edge=market_edge)
            if course_key is not None:
                jockey_course[(jockey_key, course_key)].update(finish_pct, is_top3, market_edge=market_edge)
            if graded:
                jockey_graded[jockey_key].update(finish_pct, is_top3, market_edge=market_edge)
            if trainer_key is not None:
                jockey_trainer[(jockey_key, trainer_key)].update(finish_pct, is_top3, market_edge=market_edge)
        if trainer_key is not None:
            trainer_overall[trainer_key].update(finish_pct, is_top3, market_edge=market_edge)
            if long_turf:
                trainer_long[trainer_key].update(finish_pct, is_top3, market_edge=market_edge)
            if dist_band is not None:
                trainer_dist_band[(trainer_key, dist_band)].update(finish_pct, is_top3, market_edge=market_edge)

    return AggregatedLookups(
        horse_overall=horse_overall,
        horse_surface=horse_surface,
        horse_long=horse_long,
        horse_course=horse_course,
        horse_dist_band=horse_dist_band,
        horse_graded=horse_graded,
        horse_condition=horse_condition,
        jockey_overall=jockey_overall,
        jockey_long=jockey_long,
        jockey_dist_band=jockey_dist_band,
        jockey_course=jockey_course,
        jockey_graded=jockey_graded,
        jockey_trainer=jockey_trainer,
        trainer_overall=trainer_overall,
        trainer_long=trainer_long,
        trainer_dist_band=trainer_dist_band,
        horse_recent=horse_recent,
    )


def _build_entry_aggregate_features(entry: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    work = entry.copy().reset_index(drop=True)
    if "race_date" in work.columns and work["race_date"].notna().any():
        work["_race_day"] = pd.to_datetime(work["race_date"], errors="coerce").dt.normalize()
    else:
        fallback_day = pd.to_datetime(history["race_date"], errors="coerce").max() + pd.Timedelta(days=1)
        work["_race_day"] = pd.Timestamp(fallback_day).normalize()

    feature_store = _empty_feature_store(len(work))

    for race_day, day_df in work.groupby("_race_day", sort=True):
        history_subset = history[pd.to_datetime(history["race_date"], errors="coerce").dt.normalize() < race_day]
        lookups = _build_final_lookups(history_subset)

        for row in day_df.itertuples():
            pos = int(row.Index)
            horse_key = _safe_key(getattr(row, "horse_id", None))
            jockey_key = _safe_key(getattr(row, "jockey", None))
            trainer_key = _safe_key(getattr(row, "trainer", None))
            course_key = _safe_key(getattr(row, "course", None))
            surface_key = _safe_key(getattr(row, "surface", None))
            condition_key = _safe_key(getattr(row, "track_condition", None))
            dist_band = get_distance_band(getattr(row, "distance", None))

            _assign_stats(feature_store, pos, "horse_overall", lookups.horse_overall.get(horse_key))
            if horse_key is not None and surface_key is not None:
                _assign_stats(feature_store, pos, "horse_surface", lookups.horse_surface.get((horse_key, surface_key)))
            else:
                _assign_stats(feature_store, pos, "horse_surface", None)
            _assign_stats(feature_store, pos, "horse_long_turf", lookups.horse_long.get(horse_key))
            if horse_key is not None and course_key is not None:
                _assign_stats(feature_store, pos, "horse_course", lookups.horse_course.get((horse_key, course_key)))
            else:
                _assign_stats(feature_store, pos, "horse_course", None)
            if horse_key is not None and dist_band is not None:
                _assign_stats(feature_store, pos, "horse_dist_band", lookups.horse_dist_band.get((horse_key, dist_band)))
            else:
                _assign_stats(feature_store, pos, "horse_dist_band", None)
            _assign_stats(feature_store, pos, "horse_graded", lookups.horse_graded.get(horse_key))
            if horse_key is not None and condition_key is not None:
                _assign_stats(feature_store, pos, "horse_condition", lookups.horse_condition.get((horse_key, condition_key)))
            else:
                _assign_stats(feature_store, pos, "horse_condition", None)
            _assign_stats(feature_store, pos, "jockey_overall", lookups.jockey_overall.get(jockey_key))
            _assign_stats(feature_store, pos, "jockey_long_turf", lookups.jockey_long.get(jockey_key))
            if jockey_key is not None and dist_band is not None:
                _assign_stats(feature_store, pos, "jockey_dist_band", lookups.jockey_dist_band.get((jockey_key, dist_band)))
            else:
                _assign_stats(feature_store, pos, "jockey_dist_band", None)
            if jockey_key is not None and course_key is not None:
                _assign_stats(feature_store, pos, "jockey_course", lookups.jockey_course.get((jockey_key, course_key)))
            else:
                _assign_stats(feature_store, pos, "jockey_course", None)
            _assign_stats(feature_store, pos, "jockey_graded", lookups.jockey_graded.get(jockey_key))
            if jockey_key is not None and trainer_key is not None:
                _assign_stats(feature_store, pos, "jockey_trainer", lookups.jockey_trainer.get((jockey_key, trainer_key)))
            else:
                _assign_stats(feature_store, pos, "jockey_trainer", None)
            _assign_stats(feature_store, pos, "trainer_overall", lookups.trainer_overall.get(trainer_key))
            _assign_stats(feature_store, pos, "trainer_long_turf", lookups.trainer_long.get(trainer_key))
            if trainer_key is not None and dist_band is not None:
                _assign_stats(feature_store, pos, "trainer_dist_band", lookups.trainer_dist_band.get((trainer_key, dist_band)))
            else:
                _assign_stats(feature_store, pos, "trainer_dist_band", None)
            _assign_recent_stats(feature_store, pos, lookups.horse_recent.get(horse_key))

    for col, values in feature_store.items():
        work[col] = values
    return work.drop(columns=["_race_day"])


def add_aggregate_features(history: pd.DataFrame, entry: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "race_date" not in history.columns:
        raise ValueError("aggregate feature 作成には history の race_date が必要です。")
    required_target_cols = {"finish_percentile", "is_top3"}
    if not required_target_cols.issubset(history.columns):
        raise ValueError("aggregate feature 作成前に add_targets(history) を実行してください。")

    history_feat = _add_sparse_context_flags(_build_history_aggregate_features(history))
    entry_feat = _add_sparse_context_flags(_build_entry_aggregate_features(entry, history_feat))
    return history_feat, entry_feat


def infer_feature_columns(history: pd.DataFrame, entry: pd.DataFrame, use_odds: bool) -> List[str]:
    candidates = list(RAW_MODEL_FEATURES) + list(AGGREGATE_FEATURE_COLUMNS) + list(SPARSE_CONTEXT_FEATURE_COLUMNS)
    if not use_odds:
        candidates = [c for c in candidates if c not in ODDS_RELATED_FEATURES]

    feature_cols = [
        c
        for c in candidates
        if c in history.columns
        and c in entry.columns
        and history[c].notna().any()
        and entry[c].notna().any()
    ]
    if not feature_cols:
        raise ValueError("利用可能な特徴量がありません。CSV列を確認してください。")
    assert_no_result_leakage(feature_cols)
    return feature_cols
