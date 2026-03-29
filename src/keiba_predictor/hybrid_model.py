from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

try:
    from .model import (
        CalibratedBinaryModel,
        WeightedRegressionEnsemble,
        fit_calibrated_classifier,
        fit_regression_ensemble,
        get_feature_importance as get_regression_feature_importance,
    )
    from .simulation import (
        gaussian_group_rank_distribution,
        gaussian_group_topk_probabilities,
        normalize_scores_by_group,
        plackett_luce_group_rank_distribution,
        plackett_luce_group_topk_probabilities,
    )
except ImportError:  # pragma: no cover - direct script fallback
    from model import (
        CalibratedBinaryModel,
        WeightedRegressionEnsemble,
        fit_calibrated_classifier,
        fit_regression_ensemble,
        get_feature_importance as get_regression_feature_importance,
    )
    from simulation import (
        gaussian_group_rank_distribution,
        gaussian_group_topk_probabilities,
        normalize_scores_by_group,
        plackett_luce_group_rank_distribution,
        plackett_luce_group_topk_probabilities,
    )


EPSILON = 1e-6
CATEGORICAL_HINTS = {
    "sex",
    "jockey",
    "trainer",
    "last_class",
    "surface",
    "course",
    "grade",
    "class",
    "turn",
    "direction",
    "course_direction",
}
DEFAULT_TEMPERATURE_GRID = [0.35, 0.5, 0.65, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2]


@dataclass
class CatBoostRaceRankerModel:
    model: CatBoostRanker
    feature_cols: List[str]
    categorical_cols: List[str]

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        x = prepare_catboost_frame(df, self.feature_cols, self.categorical_cols)
        return np.asarray(self.model.predict(x), dtype=float)

    def feature_importance(self) -> pd.DataFrame:
        raw = self.model.get_feature_importance(type="PredictionValuesChange", prettified=True)
        if raw is None or len(raw) == 0:
            return pd.DataFrame(columns=["feature", "importance"])
        out = raw.rename(columns={"Feature Id": "feature", "Importances": "importance"})
        out = out[["feature", "importance"]].copy()
        total = float(out["importance"].sum())
        if total > 0:
            out["importance"] = out["importance"] / total
        return out.sort_values("importance", ascending=False).reset_index(drop=True)


@dataclass
class CatBoostTop3ClassifierModel:
    model: CatBoostClassifier
    feature_cols: List[str]
    categorical_cols: List[str]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x = prepare_catboost_frame(df, self.feature_cols, self.categorical_cols)
        prob = self.model.predict_proba(x)[:, 1]
        return np.clip(np.asarray(prob, dtype=float), EPSILON, 1.0 - EPSILON)

    def feature_importance(self) -> pd.DataFrame:
        raw = self.model.get_feature_importance(type="PredictionValuesChange", prettified=True)
        if raw is None or len(raw) == 0:
            return pd.DataFrame(columns=["feature", "importance"])
        out = raw.rename(columns={"Feature Id": "feature", "Importances": "importance"})
        out = out[["feature", "importance"]].copy()
        total = float(out["importance"].sum())
        if total > 0:
            out["importance"] = out["importance"] / total
        return out.sort_values("importance", ascending=False).reset_index(drop=True)


@dataclass
class ProbabilityBlender:
    feature_cols: List[str]
    model: LogisticRegression | None
    weights: np.ndarray | None = None

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        clipped = df[self.feature_cols].apply(pd.to_numeric, errors="coerce").clip(EPSILON, 1.0 - EPSILON)
        return np.log(clipped / (1.0 - clipped)).to_numpy(dtype=float)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.feature_cols:
            return np.full(len(df), 0.5, dtype=float)
        if self.model is None:
            numeric = df[self.feature_cols].apply(pd.to_numeric, errors="coerce")
            if self.weights is not None and len(self.weights) == len(self.feature_cols):
                averaged = np.average(numeric.to_numpy(dtype=float), axis=1, weights=self.weights)
                return np.clip(np.asarray(averaged, dtype=float), EPSILON, 1.0 - EPSILON)
            averaged = numeric.mean(axis=1)
            return np.clip(averaged.to_numpy(dtype=float), EPSILON, 1.0 - EPSILON)
        return np.clip(self.model.predict_proba(self._matrix(df))[:, 1], EPSILON, 1.0 - EPSILON)


@dataclass
class HybridRaceModel:
    rank_model: CatBoostRaceRankerModel
    classifier_model: CatBoostTop3ClassifierModel
    shadow_rank_model: CatBoostRaceRankerModel | None
    regression_model: WeightedRegressionEnsemble
    auxiliary_classifier_model: CalibratedBinaryModel
    odds_temperature: float
    shadow_temperature: float
    blender: ProbabilityBlender
    feature_cols: List[str]
    no_odds_feature_cols: List[str]


def infer_column_types(df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for col in feature_cols:
        if col in CATEGORICAL_HINTS:
            categorical_cols.append(col)
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)
    return numeric_cols, categorical_cols


def prepare_catboost_frame(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> pd.DataFrame:
    out = df[list(feature_cols)].copy()
    categorical_set = set(categorical_cols)
    for col in out.columns:
        if col in categorical_set:
            out[col] = out[col].astype("string").fillna("MISSING").astype(str)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def time_series_race_splits(
    df: pd.DataFrame,
    n_splits: int,
    min_train_races: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    races = (
        df[["race_id", "race_date"]]
        .drop_duplicates()
        .sort_values(["race_date", "race_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    n_races = len(races)
    if n_races < 3:
        raise ValueError(f"時系列CVに必要なレース数が不足しています。n_races={n_races}")

    n_splits = max(1, min(int(n_splits), n_races - 1))
    min_train_races = max(1, min(int(min_train_races), n_races - n_splits))

    remaining = races.iloc[min_train_races:].copy()
    val_chunks = np.array_split(remaining.index.to_numpy(), n_splits)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for chunk in val_chunks:
        if len(chunk) == 0:
            continue
        train_race_ids = set(races.iloc[: chunk[0]]["race_id"])
        val_race_ids = set(races.iloc[chunk]["race_id"])
        train_idx = df.index[df["race_id"].isin(train_race_ids)].to_numpy()
        val_idx = df.index[df["race_id"].isin(val_race_ids)].to_numpy()
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        splits.append((train_idx, val_idx))

    if not splits:
        raise ValueError("有効な時系列CV split を作成できませんでした。")
    return splits


def _safe_log_loss(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    y_arr = np.asarray(y_true, dtype=int)
    if len(np.unique(y_arr)) < 2:
        return float("nan")
    return float(log_loss(y_arr, np.clip(np.asarray(y_prob, dtype=float), EPSILON, 1.0 - EPSILON)))


def _build_ranker(random_state: int, config: Dict[str, object]) -> CatBoostRanker:
    cfg = config.get("model", {}).get("ranker", {})
    return CatBoostRanker(
        loss_function=str(cfg.get("loss_function", "QueryRMSE")),
        eval_metric=str(cfg.get("eval_metric", "NDCG:top=3")),
        iterations=int(cfg.get("iterations", 500)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        depth=int(cfg.get("depth", 6)),
        l2_leaf_reg=float(cfg.get("l2_leaf_reg", 8.0)),
        random_strength=float(cfg.get("random_strength", 0.5)),
        min_data_in_leaf=int(cfg.get("min_data_in_leaf", 5)),
        bootstrap_type=str(cfg.get("bootstrap_type", "Bernoulli")),
        subsample=float(cfg.get("subsample", 0.8)),
        random_seed=random_state,
        thread_count=int(cfg.get("thread_count", -1)),
        allow_writing_files=False,
        verbose=False,
    )


def _build_classifier(random_state: int, config: Dict[str, object]) -> CatBoostClassifier:
    cfg = config.get("model", {}).get("classifier", {})
    params: Dict[str, object] = {
        "loss_function": str(cfg.get("loss_function", "Logloss")),
        "eval_metric": str(cfg.get("eval_metric", "Logloss")),
        "iterations": int(cfg.get("iterations", 400)),
        "learning_rate": float(cfg.get("learning_rate", 0.05)),
        "depth": int(cfg.get("depth", 6)),
        "l2_leaf_reg": float(cfg.get("l2_leaf_reg", 8.0)),
        "random_strength": float(cfg.get("random_strength", 0.5)),
        "min_data_in_leaf": int(cfg.get("min_data_in_leaf", 5)),
        "bootstrap_type": str(cfg.get("bootstrap_type", "Bernoulli")),
        "subsample": float(cfg.get("subsample", 0.8)),
        "random_seed": random_state,
        "thread_count": int(cfg.get("thread_count", -1)),
        "allow_writing_files": False,
        "verbose": False,
    }
    auto_class_weights = cfg.get("auto_class_weights", "Balanced")
    if auto_class_weights not in [None, "none", "None"]:
        params["auto_class_weights"] = auto_class_weights
    return CatBoostClassifier(**params)


def _resolve_group_holdout_split(group_ids: Sequence[object], train_fraction: float = 0.85) -> int | None:
    groups = pd.Series(group_ids, dtype="string").reset_index(drop=True)
    if len(groups) == 0:
        return None

    race_end_rows = np.flatnonzero(groups.ne(groups.shift(-1)).fillna(True).to_numpy()) + 1
    if len(race_end_rows) < 2:
        return None

    target_row = max(1, min(len(groups) - 1, int(np.ceil(len(groups) * train_fraction))))
    boundary_idx = int(np.searchsorted(race_end_rows, target_row, side="left"))
    if boundary_idx >= len(race_end_rows) - 1:
        boundary_idx = len(race_end_rows) - 2

    split_row = int(race_end_rows[boundary_idx])
    if split_row <= 0 or split_row >= len(groups):
        return None
    return split_row


def fit_ranker_model(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    sample_weight: Sequence[float],
    random_state: int,
    config: Dict[str, object],
) -> CatBoostRaceRankerModel:
    _, categorical_cols = infer_column_types(train_df, feature_cols)
    order = train_df.sort_values(["race_date", "race_id"], kind="mergesort").index
    sorted_train = train_df.loc[order].reset_index(drop=True)
    sorted_weight = pd.Series(sample_weight, index=train_df.index).loc[order].to_numpy(dtype=float)
    x = prepare_catboost_frame(sorted_train, feature_cols, categorical_cols)
    y = sorted_train["finish_percentile"].to_numpy(dtype=float)
    group = sorted_train["race_id"].astype(str)

    early_stopping = int(config.get("model", {}).get("ranker", {}).get("early_stopping_rounds", 0))
    model = _build_ranker(random_state=random_state, config=config)
    full_pool = Pool(
        x,
        y,
        group_id=group,
        cat_features=categorical_cols,
        weight=sorted_weight,
    )

    split_row = None
    if early_stopping > 0 and len(sorted_train) >= 40:
        split_row = _resolve_group_holdout_split(group, train_fraction=0.85)

    if split_row is not None:
        train_pool = Pool(
            x.iloc[:split_row],
            y[:split_row],
            group_id=group.iloc[:split_row],
            cat_features=categorical_cols,
            weight=sorted_weight[:split_row],
        )
        eval_pool = Pool(
            x.iloc[split_row:],
            y[split_row:],
            group_id=group.iloc[split_row:],
            cat_features=categorical_cols,
            weight=sorted_weight[split_row:],
        )
        model.fit(
            train_pool,
            eval_set=eval_pool,
            early_stopping_rounds=early_stopping,
            verbose=False,
        )
    else:
        model.fit(
            full_pool,
            verbose=False,
        )
    return CatBoostRaceRankerModel(
        model=model,
        feature_cols=list(feature_cols),
        categorical_cols=list(categorical_cols),
    )


def fit_classifier_model(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    sample_weight: Sequence[float],
    random_state: int,
    config: Dict[str, object],
) -> CatBoostTop3ClassifierModel:
    _, categorical_cols = infer_column_types(train_df, feature_cols)
    x = prepare_catboost_frame(train_df, feature_cols, categorical_cols)
    y = train_df["is_top3"].to_numpy(dtype=int)
    sw = np.asarray(sample_weight, dtype=float)

    early_stopping = int(config.get("model", {}).get("classifier", {}).get("early_stopping_rounds", 0))
    model = _build_classifier(random_state=random_state, config=config)

    if early_stopping > 0 and len(train_df) >= 40:
        n_train = int(len(train_df) * 0.85)
        model.fit(
            x.iloc[:n_train], y[:n_train],
            eval_set=(x.iloc[n_train:], y[n_train:]),
            cat_features=categorical_cols,
            sample_weight=sw[:n_train],
            early_stopping_rounds=early_stopping,
            verbose=False,
        )
    else:
        model.fit(
            x, y,
            cat_features=categorical_cols,
            sample_weight=sw,
            verbose=False,
        )
    return CatBoostTop3ClassifierModel(
        model=model,
        feature_cols=list(feature_cols),
        categorical_cols=list(categorical_cols),
    )


def _has_shadow_model(feature_cols: Sequence[str], no_odds_feature_cols: Sequence[str]) -> bool:
    return list(feature_cols) != list(no_odds_feature_cols)


def _temperature_grid(config: Dict[str, object]) -> List[float]:
    grid = config.get("simulation", {}).get("temperature_grid")
    if grid is None:
        return DEFAULT_TEMPERATURE_GRID
    return [float(value) for value in grid]


def fit_score_temperature(
    race_ids: Sequence[object],
    normalized_scores: Sequence[float],
    labels: Sequence[int],
    config: Dict[str, object],
    seed: int,
) -> float:
    y_arr = np.asarray(labels, dtype=int)
    if len(np.unique(y_arr)) < 2:
        return 1.0

    sim_cfg = config.get("simulation", {})
    n_trials = int(sim_cfg.get("calibration_trials", 8000))
    block_size = int(sim_cfg.get("calibration_block_size", min(n_trials, 4000)))

    best_temp = 1.0
    best_loss = float("inf")

    for idx, temp in enumerate(_temperature_grid(config)):
        prob = plackett_luce_group_topk_probabilities(
            group_ids=race_ids,
            strengths=normalized_scores,
            topk=3,
            temperature=float(temp),
            n_trials=n_trials,
            seed=seed + (idx * 97),
            block_size=block_size,
        )
        brier = brier_score_loss(y_arr, prob)
        if len(y_arr) < 100:
            combined_loss = brier
        else:
            combined_loss = (0.7 * log_loss(y_arr, np.clip(prob, EPSILON, 1.0 - EPSILON))) + (0.3 * brier)
        if combined_loss < best_loss:
            best_loss = float(combined_loss)
            best_temp = float(temp)

    return best_temp


def fit_probability_blender(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    labels: Sequence[int],
    random_state: int,
) -> ProbabilityBlender:
    feature_list = list(feature_cols)
    if not feature_list:
        return ProbabilityBlender(feature_cols=feature_list, model=None, weights=None)

    y_arr = np.asarray(labels, dtype=int)
    if len(np.unique(y_arr)) < 2:
        return ProbabilityBlender(feature_cols=feature_list, model=None, weights=None)

    numeric = df[feature_list].apply(pd.to_numeric, errors="coerce").clip(EPSILON, 1.0 - EPSILON)

    def combined_loss(prob: np.ndarray) -> float:
        clipped = np.clip(np.asarray(prob, dtype=float), EPSILON, 1.0 - EPSILON)
        return float((0.7 * log_loss(y_arr, clipped)) + (0.3 * brier_score_loss(y_arr, clipped)))

    component_losses = {col: combined_loss(numeric[col].to_numpy(dtype=float)) for col in feature_list}
    best_feature = min(component_losses, key=component_losses.get)
    best_single_loss = float(component_losses[best_feature])

    loss_arr = np.asarray([component_losses[col] for col in feature_list], dtype=float)
    inverse_loss_weights = 1.0 / np.clip(loss_arr, EPSILON, None)
    inverse_loss_weights = inverse_loss_weights / inverse_loss_weights.sum()
    weighted_prob = np.average(numeric.to_numpy(dtype=float), axis=1, weights=inverse_loss_weights)
    weighted_loss = combined_loss(weighted_prob)

    model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=random_state)
    logistic_blender = ProbabilityBlender(feature_cols=feature_list, model=model, weights=None)
    model.fit(logistic_blender._matrix(df), y_arr)
    logistic_loss = combined_loss(logistic_blender.predict_proba(df))

    if logistic_loss <= min(weighted_loss, best_single_loss):
        return logistic_blender
    if weighted_loss <= best_single_loss:
        return ProbabilityBlender(feature_cols=feature_list, model=None, weights=inverse_loss_weights)
    return ProbabilityBlender(feature_cols=[best_feature], model=None, weights=np.asarray([1.0], dtype=float))


def _regression_posterior_stats(
    regression_model: WeightedRegressionEnsemble,
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.clip(regression_model.predict(df), 0.0, 1.0)
    pred_matrix = regression_model.predict_model_matrix(df)
    ensemble_std = (
        pred_matrix.std(axis=1, ddof=1)
        if pred_matrix.shape[1] > 1
        else np.zeros(pred_matrix.shape[0], dtype=float)
    )
    sigma_floor = max(0.03, float(regression_model.sigma_floor))
    sigma = np.sqrt((ensemble_std ** 2) + (sigma_floor ** 2))
    sigma = np.maximum(sigma, sigma_floor)
    return mu, sigma


def _predict_regression_topk_probabilities(
    regression_model: WeightedRegressionEnsemble,
    df: pd.DataFrame,
    topk: int,
    config: Dict[str, object],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sim_cfg = config.get("simulation", {})
    mu, sigma = _regression_posterior_stats(regression_model, df)
    prob = gaussian_group_topk_probabilities(
        group_ids=df["race_id"].astype(str),
        mu=mu,
        sigma=sigma,
        topk=topk,
        n_trials=int(sim_cfg.get("calibration_trials", 8000)),
        seed=seed,
        block_size=int(sim_cfg.get("calibration_block_size", 4000)),
    )
    return prob, mu, sigma


def _predict_regression_rank_distribution(
    regression_model: WeightedRegressionEnsemble,
    df: pd.DataFrame,
    config: Dict[str, object],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sim_cfg = config.get("simulation", {})
    entity_ids = np.arange(len(df), dtype=int)
    race_ids = df["race_id"].astype(str).to_numpy()
    mu, sigma = _regression_posterior_stats(regression_model, df)
    sim_df, diag_df = gaussian_group_rank_distribution(
        entity_ids=entity_ids,
        group_ids=race_ids,
        mu=mu,
        sigma=sigma,
        stages=sim_cfg.get("stages", [50_000, 150_000, 400_000]),
        threshold=float(sim_cfg.get("convergence_threshold", 0.002)),
        seed=seed,
        block_size=int(sim_cfg.get("block_size", 50_000)),
    )
    sim_df = sim_df.sort_values("entity_id").reset_index(drop=True)
    sim_df = sim_df.rename(
        columns={
            "group_id": "race_id",
            "mu": "regression_mu",
            "sigma": "regression_sigma",
            "win_prob": "regression_win_prob",
            "top2_prob": "regression_top2_prob",
            "top3_prob": "regression_top3_prob",
            "mean_rank": "regression_mean_rank",
            "top3_ci_low": "regression_top3_ci_low",
            "top3_ci_high": "regression_top3_ci_high",
            "top3_ci_width": "regression_top3_ci_width",
            "trials_used": "regression_trials_used",
        }
    )
    return sim_df, diag_df


def _generate_internal_oof(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    no_odds_feature_cols: Sequence[str],
    sample_weight: Sequence[float],
    random_state: int,
    config: Dict[str, object],
) -> pd.DataFrame:
    cv_cfg = config.get("cv", {})
    splits = time_series_race_splits(
        train_df,
        n_splits=int(cv_cfg.get("stacking_inner_splits", cv_cfg.get("classifier_inner_splits", 3))),
        min_train_races=int(cv_cfg.get("stacking_inner_min_train_races", cv_cfg.get("classifier_inner_min_train_races", 20))),
    )

    has_shadow = _has_shadow_model(feature_cols, no_odds_feature_cols)
    sw_series = pd.Series(sample_weight, index=train_df.index)
    frames: List[pd.DataFrame] = []

    min_rows_for_complex_models = 30

    for fold_no, (tr_idx, va_idx) in enumerate(splits, start=1):
        inner_train = train_df.iloc[tr_idx].reset_index(drop=True)
        inner_val = train_df.iloc[va_idx].reset_index(drop=True)
        inner_weight = sw_series.iloc[tr_idx].to_numpy(dtype=float)
        use_complex = len(inner_train) >= min_rows_for_complex_models

        classifier_model = fit_classifier_model(
            inner_train,
            feature_cols=feature_cols,
            sample_weight=inner_weight,
            random_state=random_state + (fold_no * 100) + 1,
            config=config,
        )

        fold_frame = inner_val[["race_id", "is_top3", "finish_rank"]].copy()
        if "odds" in inner_val.columns:
            fold_frame["odds"] = pd.to_numeric(inner_val["odds"], errors="coerce")
        fold_frame["fold"] = fold_no
        fold_frame["classifier_top3_prob"] = classifier_model.predict_proba(inner_val)

        if use_complex:
            rank_model = fit_ranker_model(
                inner_train,
                feature_cols=feature_cols,
                sample_weight=inner_weight,
                random_state=random_state + (fold_no * 100),
                config=config,
            )
            regression_model = fit_regression_ensemble(
                inner_train,
                feature_cols=feature_cols,
                y=inner_train["finish_percentile"].to_numpy(dtype=float),
                sample_weight=inner_weight,
                random_state=random_state + (fold_no * 100) + 2,
                config=config,
            )
            auxiliary_classifier_model = fit_calibrated_classifier(
                inner_train,
                feature_cols=feature_cols,
                y=inner_train["is_top3"].to_numpy(dtype=int),
                sample_weight=inner_weight,
                random_state=random_state + (fold_no * 100) + 3,
                config=config,
            )
            fold_frame["rank_score_raw"] = rank_model.predict_scores(inner_val)
            fold_frame["aux_classifier_top3_prob"] = auxiliary_classifier_model.predict_proba(inner_val)
            regression_mu, regression_sigma = _regression_posterior_stats(regression_model, inner_val)
            fold_frame["regression_mu"] = regression_mu
            fold_frame["regression_sigma"] = regression_sigma

            if has_shadow:
                shadow_model = fit_ranker_model(
                    inner_train,
                    feature_cols=no_odds_feature_cols,
                    sample_weight=inner_weight,
                    random_state=random_state + (fold_no * 100) + 5,
                    config=config,
                )
                fold_frame["shadow_rank_score_raw"] = shadow_model.predict_scores(inner_val)
        else:
            fold_frame["rank_score_raw"] = 0.0
            fold_frame["aux_classifier_top3_prob"] = fold_frame["classifier_top3_prob"]
            fold_frame["regression_mu"] = 0.5
            fold_frame["regression_sigma"] = 0.2
            if has_shadow:
                fold_frame["shadow_rank_score_raw"] = 0.0

        frames.append(fold_frame)

    if not frames:
        raise ValueError("内部OOFを生成できませんでした。")

    return pd.concat(frames, ignore_index=True)


def fit_hybrid_race_model(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    no_odds_feature_cols: Sequence[str],
    sample_weight: Sequence[float],
    random_state: int,
    config: Dict[str, object],
) -> HybridRaceModel:
    has_shadow = _has_shadow_model(feature_cols, no_odds_feature_cols)
    internal_oof = _generate_internal_oof(
        train_df=train_df,
        feature_cols=feature_cols,
        no_odds_feature_cols=no_odds_feature_cols,
        sample_weight=sample_weight,
        random_state=random_state,
        config=config,
    )

    internal_oof["rank_strength"] = normalize_scores_by_group(internal_oof["race_id"], internal_oof["rank_score_raw"])
    odds_temperature = fit_score_temperature(
        race_ids=internal_oof["race_id"],
        normalized_scores=internal_oof["rank_strength"],
        labels=internal_oof["is_top3"],
        config=config,
        seed=random_state + 700,
    )
    sim_cfg = config.get("simulation", {})
    calib_trials = int(sim_cfg.get("calibration_trials", 8000))
    calib_block = int(sim_cfg.get("calibration_block_size", min(calib_trials, 4000)))
    internal_oof["rank_top3_prob"] = plackett_luce_group_topk_probabilities(
        group_ids=internal_oof["race_id"],
        strengths=internal_oof["rank_strength"],
        topk=3,
        temperature=odds_temperature,
        n_trials=calib_trials,
        seed=random_state + 800,
        block_size=calib_block,
    )

    shadow_temperature = odds_temperature
    if has_shadow:
        internal_oof["shadow_rank_strength"] = normalize_scores_by_group(
            internal_oof["race_id"],
            internal_oof["shadow_rank_score_raw"],
        )
        shadow_temperature = fit_score_temperature(
            race_ids=internal_oof["race_id"],
            normalized_scores=internal_oof["shadow_rank_strength"],
            labels=internal_oof["is_top3"],
            config=config,
            seed=random_state + 900,
        )
        internal_oof["shadow_no_odds_top3_prob"] = plackett_luce_group_topk_probabilities(
            group_ids=internal_oof["race_id"],
            strengths=internal_oof["shadow_rank_strength"],
            topk=3,
            temperature=shadow_temperature,
            n_trials=calib_trials,
            seed=random_state + 1000,
            block_size=calib_block,
        )
    else:
        internal_oof["shadow_no_odds_top3_prob"] = internal_oof["rank_top3_prob"]

    internal_oof["regression_top3_prob"] = gaussian_group_topk_probabilities(
        group_ids=internal_oof["race_id"],
        mu=internal_oof["regression_mu"],
        sigma=internal_oof["regression_sigma"],
        topk=3,
        n_trials=calib_trials,
        seed=random_state + 1050,
        block_size=calib_block,
    )

    blender_cols = [
        "rank_top3_prob",
        "classifier_top3_prob",
        "shadow_no_odds_top3_prob",
        "regression_top3_prob",
        "aux_classifier_top3_prob",
    ]
    blender = fit_probability_blender(
        df=internal_oof,
        feature_cols=blender_cols,
        labels=internal_oof["is_top3"],
        random_state=random_state + 1100,
    )

    rank_model = fit_ranker_model(
        train_df,
        feature_cols=feature_cols,
        sample_weight=sample_weight,
        random_state=random_state + 1200,
        config=config,
    )
    classifier_model = fit_classifier_model(
        train_df,
        feature_cols=feature_cols,
        sample_weight=sample_weight,
        random_state=random_state + 1300,
        config=config,
    )
    regression_model = fit_regression_ensemble(
        train_df,
        feature_cols=feature_cols,
        y=train_df["finish_percentile"].to_numpy(dtype=float),
        sample_weight=np.asarray(sample_weight, dtype=float),
        random_state=random_state + 1350,
        config=config,
    )
    auxiliary_classifier_model = fit_calibrated_classifier(
        train_df,
        feature_cols=feature_cols,
        y=train_df["is_top3"].to_numpy(dtype=int),
        sample_weight=np.asarray(sample_weight, dtype=float),
        random_state=random_state + 1380,
        config=config,
    )
    shadow_rank_model = None
    if has_shadow:
        shadow_rank_model = fit_ranker_model(
            train_df,
            feature_cols=no_odds_feature_cols,
            sample_weight=sample_weight,
            random_state=random_state + 1400,
            config=config,
        )

    return HybridRaceModel(
        rank_model=rank_model,
        classifier_model=classifier_model,
        shadow_rank_model=shadow_rank_model,
        regression_model=regression_model,
        auxiliary_classifier_model=auxiliary_classifier_model,
        odds_temperature=float(odds_temperature),
        shadow_temperature=float(shadow_temperature),
        blender=blender,
        feature_cols=list(feature_cols),
        no_odds_feature_cols=list(no_odds_feature_cols),
    )


def _adaptive_blend_weights(blender: ProbabilityBlender) -> Tuple[float, float]:
    """Compute ranker vs regression blend weights from the blender's component quality.

    If the blender uses a single best feature or only regression-based features,
    the ranker gets zero weight. Otherwise, extract approximate weights from the
    blender's inverse-loss weights or logistic model coefficients.
    Falls back to 0.65/0.35 when the blender provides no information.
    """
    if blender.weights is not None and len(blender.feature_cols) > 1:
        rank_idx = None
        reg_idx = None
        for i, col in enumerate(blender.feature_cols):
            if "rank" in col and "shadow" not in col:
                rank_idx = i
            elif "regression" in col:
                reg_idx = i
        if rank_idx is not None and reg_idx is not None:
            rw = float(blender.weights[rank_idx])
            gw = float(blender.weights[reg_idx])
            total = rw + gw
            if total > 0:
                return rw / total, gw / total
    if len(blender.feature_cols) == 1:
        col = blender.feature_cols[0]
        if "regression" in col:
            return 0.0, 1.0
        if "rank" in col:
            return 1.0, 0.0
        return 0.5, 0.5
    return 0.65, 0.35


def predict_hybrid_model(
    model: HybridRaceModel,
    entry_df: pd.DataFrame,
    config: Dict[str, object],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sim_cfg = config.get("simulation", {})
    entity_ids = np.arange(len(entry_df), dtype=int)
    race_ids = entry_df["race_id"].astype(str).to_numpy()

    rank_strength = normalize_scores_by_group(race_ids, model.rank_model.predict_scores(entry_df))
    sim_df, diag_df = plackett_luce_group_rank_distribution(
        entity_ids=entity_ids,
        group_ids=race_ids,
        strengths=rank_strength,
        temperature=model.odds_temperature,
        stages=sim_cfg.get("stages", [50_000, 150_000, 400_000]),
        threshold=float(sim_cfg.get("convergence_threshold", 0.002)),
        seed=seed,
        block_size=int(sim_cfg.get("block_size", 50_000)),
    )
    sim_df = sim_df.sort_values("entity_id").reset_index(drop=True)
    sim_df = sim_df.rename(columns={"group_id": "race_id", "strength": "rank_strength", "temperature": "rank_temperature"})
    diag_df["source"] = "catboost_ranker"

    regression_sim_df, regression_diag_df = _predict_regression_rank_distribution(
        regression_model=model.regression_model,
        df=entry_df,
        config=config,
        seed=seed + 250,
    )
    if not regression_diag_df.empty:
        regression_diag_df["source"] = "regression_ensemble"

    classifier_top3_prob = model.classifier_model.predict_proba(entry_df)
    auxiliary_classifier_top3_prob = model.auxiliary_classifier_model.predict_proba(entry_df)
    component_df = sim_df.merge(
        regression_sim_df[
            [
                "entity_id",
                "race_id",
                "regression_mu",
                "regression_sigma",
                "regression_win_prob",
                "regression_top2_prob",
                "regression_top3_prob",
                "regression_mean_rank",
                "regression_top3_ci_low",
                "regression_top3_ci_high",
                "regression_top3_ci_width",
                "regression_trials_used",
            ]
        ],
        on=["entity_id", "race_id"],
        how="left",
    )
    component_df["rank_top3_prob"] = component_df["top3_prob"]
    component_df["classifier_top3_prob"] = classifier_top3_prob
    component_df["aux_classifier_top3_prob"] = auxiliary_classifier_top3_prob

    if model.shadow_rank_model is not None:
        shadow_strength = normalize_scores_by_group(race_ids, model.shadow_rank_model.predict_scores(entry_df))
        shadow_prob = plackett_luce_group_topk_probabilities(
            group_ids=race_ids,
            strengths=shadow_strength,
            topk=3,
            temperature=model.shadow_temperature,
            n_trials=int(sim_cfg.get("calibration_trials", 8000)),
            seed=seed + 500,
            block_size=int(sim_cfg.get("calibration_block_size", 4000)),
        )
        component_df["shadow_rank_strength"] = shadow_strength
        component_df["shadow_no_odds_top3_prob"] = shadow_prob
    else:
        component_df["shadow_rank_strength"] = rank_strength
        component_df["shadow_no_odds_top3_prob"] = component_df["top3_prob"]

    component_df["component_model_std"] = component_df[
        [
            "rank_top3_prob",
            "classifier_top3_prob",
            "shadow_no_odds_top3_prob",
            "regression_top3_prob",
            "aux_classifier_top3_prob",
        ]
    ].std(axis=1)
    component_df["ensemble_top3_prob"] = model.blender.predict_proba(component_df)
    component_df["calibrated_top3_prob"] = component_df["ensemble_top3_prob"]
    if not regression_sim_df.empty:
        ranker_w, regression_w = _adaptive_blend_weights(model.blender)
        component_df["win_prob"] = (ranker_w * component_df["win_prob"]) + (regression_w * component_df["regression_win_prob"])
        component_df["top2_prob"] = (ranker_w * component_df["top2_prob"]) + (regression_w * component_df["regression_top2_prob"])
        component_df["mean_rank"] = (ranker_w * component_df["mean_rank"]) + (regression_w * component_df["regression_mean_rank"])
        component_df["top3_ci_low"] = np.minimum(component_df["top3_ci_low"], component_df["regression_top3_ci_low"])
        component_df["top3_ci_high"] = np.maximum(component_df["top3_ci_high"], component_df["regression_top3_ci_high"])
        component_df["top3_ci_width"] = component_df["top3_ci_high"] - component_df["top3_ci_low"]

    diagnostics = [frame for frame in [diag_df, regression_diag_df] if frame is not None and not frame.empty]
    merged_diag_df = pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame()
    return component_df, merged_diag_df


def precision_at_k_by_race(
    df: pd.DataFrame,
    prob_col: str,
    label_col: str,
    race_col: str,
    k: int,
) -> float:
    scores: List[float] = []
    for _, g in df.groupby(race_col):
        kk = min(k, len(g))
        scores.append(float(g.nlargest(kk, prob_col)[label_col].mean()))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def win_hit_rate_at_1(
    df: pd.DataFrame,
    prob_col: str,
    finish_rank_col: str,
    race_col: str,
) -> float:
    hits: List[float] = []
    for _, g in df.groupby(race_col):
        top_row = g.nlargest(1, prob_col).iloc[0]
        hits.append(float(int(top_row[finish_rank_col]) == 1))
    if not hits:
        return float("nan")
    return float(np.mean(hits))


def win_roi_at_1(
    df: pd.DataFrame,
    prob_col: str,
    finish_rank_col: str,
    odds_col: str,
    race_col: str,
) -> float:
    profits: List[float] = []
    if odds_col not in df.columns:
        return float("nan")
    for _, g in df.groupby(race_col):
        top_row = g.nlargest(1, prob_col).iloc[0]
        odds = pd.to_numeric(top_row.get(odds_col), errors="coerce")
        if pd.isna(odds):
            continue
        profit = float(odds - 1.0) if int(top_row[finish_rank_col]) == 1 else -1.0
        profits.append(profit)
    if not profits:
        return float("nan")
    return float(np.mean(profits))


def evaluate_time_series_cv(
    history_df: pd.DataFrame,
    feature_cols: Sequence[str],
    no_odds_feature_cols: Sequence[str],
    sample_weight: pd.Series,
    config: Dict[str, object],
) -> Dict[str, object]:
    cv_cfg = config.get("cv", {})
    splits = time_series_race_splits(
        history_df,
        n_splits=int(cv_cfg.get("n_splits", 4)),
        min_train_races=int(cv_cfg.get("min_train_races", 50)),
    )

    fold_metrics: List[Dict[str, float]] = []
    oof_frames: List[pd.DataFrame] = []

    for fold_no, (train_idx, val_idx) in enumerate(splits, start=1):
        train_df = history_df.iloc[train_idx].reset_index(drop=True)
        val_df = history_df.iloc[val_idx].reset_index(drop=True)
        sw_train = sample_weight.iloc[train_idx].to_numpy(dtype=float)

        hybrid_model = fit_hybrid_race_model(
            train_df=train_df,
            feature_cols=feature_cols,
            no_odds_feature_cols=no_odds_feature_cols,
            sample_weight=sw_train,
            random_state=int(config.get("seed", 42)) + (fold_no * 1000),
            config=config,
        )
        pred_df, _ = predict_hybrid_model(
            model=hybrid_model,
            entry_df=val_df,
            config=config,
            seed=int(config.get("seed", 42)) + 9000 + fold_no,
        )

        fold_eval = val_df[["race_id", "finish_rank", "is_top3"]].copy()
        if "odds" in val_df.columns:
            fold_eval["odds"] = pd.to_numeric(val_df["odds"], errors="coerce")
        pred_use_cols = [c for c in pred_df.columns if c not in {"entity_id", "race_id"}]
        fold_eval = pd.concat([fold_eval.reset_index(drop=True), pred_df[pred_use_cols].reset_index(drop=True)], axis=1)
        fold_eval["fold"] = fold_no
        oof_frames.append(fold_eval)

        fold_metrics.append(
            {
                "fold": float(fold_no),
                "brier_score": float(brier_score_loss(fold_eval["is_top3"], fold_eval["ensemble_top3_prob"])),
                "log_loss": _safe_log_loss(fold_eval["is_top3"], fold_eval["ensemble_top3_prob"]),
                "ranker_brier_score": float(brier_score_loss(fold_eval["is_top3"], fold_eval["top3_prob"])),
                "classifier_brier_score": float(brier_score_loss(fold_eval["is_top3"], fold_eval["classifier_top3_prob"])),
                "shadow_brier_score": float(brier_score_loss(fold_eval["is_top3"], fold_eval["shadow_no_odds_top3_prob"])),
                "regression_brier_score": float(brier_score_loss(fold_eval["is_top3"], fold_eval["regression_top3_prob"])),
                "aux_classifier_brier_score": float(
                    brier_score_loss(fold_eval["is_top3"], fold_eval["aux_classifier_top3_prob"])
                ),
                "top3_precision": precision_at_k_by_race(
                    fold_eval,
                    prob_col="ensemble_top3_prob",
                    label_col="is_top3",
                    race_col="race_id",
                    k=3,
                ),
                "component_model_std_mean": float(pd.to_numeric(fold_eval["component_model_std"], errors="coerce").mean()),
                "win_hit_rate_at1": win_hit_rate_at_1(
                    fold_eval,
                    prob_col="win_prob",
                    finish_rank_col="finish_rank",
                    race_col="race_id",
                ),
                "win_roi_at1": win_roi_at_1(
                    fold_eval,
                    prob_col="win_prob",
                    finish_rank_col="finish_rank",
                    odds_col="odds",
                    race_col="race_id",
                ),
                "odds_temperature": float(hybrid_model.odds_temperature),
                "shadow_temperature": float(hybrid_model.shadow_temperature),
            }
        )

    oof_df = pd.concat(oof_frames, ignore_index=True)
    summary = {
        "brier_score": float(brier_score_loss(oof_df["is_top3"], oof_df["ensemble_top3_prob"])),
        "log_loss": _safe_log_loss(oof_df["is_top3"], oof_df["ensemble_top3_prob"]),
        "ranker_brier_score": float(brier_score_loss(oof_df["is_top3"], oof_df["top3_prob"])),
        "classifier_brier_score": float(brier_score_loss(oof_df["is_top3"], oof_df["classifier_top3_prob"])),
        "shadow_brier_score": float(brier_score_loss(oof_df["is_top3"], oof_df["shadow_no_odds_top3_prob"])),
        "regression_brier_score": float(brier_score_loss(oof_df["is_top3"], oof_df["regression_top3_prob"])),
        "aux_classifier_brier_score": float(brier_score_loss(oof_df["is_top3"], oof_df["aux_classifier_top3_prob"])),
        "top3_precision": precision_at_k_by_race(
            oof_df,
            prob_col="ensemble_top3_prob",
            label_col="is_top3",
            race_col="race_id",
            k=3,
        ),
        "component_model_std_mean": float(pd.to_numeric(oof_df["component_model_std"], errors="coerce").mean()),
        "win_hit_rate_at1": win_hit_rate_at_1(
            oof_df,
            prob_col="win_prob",
            finish_rank_col="finish_rank",
            race_col="race_id",
        ),
        "win_roi_at1": win_roi_at_1(
            oof_df,
            prob_col="win_prob",
            finish_rank_col="finish_rank",
            odds_col="odds",
            race_col="race_id",
        ),
    }

    prob_true, prob_pred = calibration_curve(
        oof_df["is_top3"],
        np.clip(oof_df["ensemble_top3_prob"], EPSILON, 1.0 - EPSILON),
        n_bins=int(config.get("plot", {}).get("calibration_bins", 10)),
        strategy="quantile",
    )
    calibration_df = pd.DataFrame({"pred_mean": prob_pred, "true_rate": prob_true})

    return {
        "fold_metrics": pd.DataFrame(fold_metrics),
        "oof_predictions": oof_df,
        "summary": summary,
        "calibration_curve": calibration_df,
    }


def get_feature_importance(model: HybridRaceModel) -> pd.DataFrame:
    rank_imp = model.rank_model.feature_importance().rename(columns={"importance": "rank_importance"})
    cls_imp = model.classifier_model.feature_importance().rename(columns={"importance": "classifier_importance"})
    reg_imp = get_regression_feature_importance(model.regression_model).rename(columns={"importance": "regression_importance"})
    frames = [rank_imp, cls_imp, reg_imp]

    if model.shadow_rank_model is not None:
        shadow_imp = model.shadow_rank_model.feature_importance().rename(columns={"importance": "shadow_rank_importance"})
        frames.append(shadow_imp)

    merged = None
    for frame in frames:
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="feature", how="outer")

    if merged is None or merged.empty:
        return pd.DataFrame(columns=["feature", "importance"])

    merged = merged.fillna(0.0)
    component_cols = [
        col
        for col in ["rank_importance", "classifier_importance", "regression_importance", "shadow_rank_importance"]
        if col in merged.columns
    ]
    weights = {
        "rank_importance": 0.45,
        "classifier_importance": 0.25,
        "regression_importance": 0.20,
        "shadow_rank_importance": 0.10,
    }
    total_weight = sum(weights[col] for col in component_cols)
    merged["importance"] = 0.0
    for col in component_cols:
        merged["importance"] += merged[col] * (weights[col] / total_weight)

    return merged.sort_values("importance", ascending=False).reset_index(drop=True)
