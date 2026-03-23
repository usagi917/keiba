from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


class IdentityCalibrator:
    def predict(self, prob: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0 - 1e-6)


class PlattCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=500, solver="lbfgs")
        self.fitted = False

    @staticmethod
    def _to_logit(prob: np.ndarray) -> np.ndarray:
        p = np.clip(prob, 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p)).reshape(-1, 1)

    def fit(self, prob: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.model.fit(self._to_logit(np.asarray(prob, dtype=float)), np.asarray(y, dtype=int))
        self.fitted = True
        return self

    def predict(self, prob: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0 - 1e-6)
        return self.model.predict_proba(self._to_logit(np.asarray(prob, dtype=float)))[:, 1]


class IsotonicLikeCalibrator:
    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip")
        self.fitted = False

    def fit(self, prob: np.ndarray, y: np.ndarray) -> "IsotonicLikeCalibrator":
        self.model.fit(np.asarray(prob, dtype=float), np.asarray(y, dtype=float))
        self.fitted = True
        return self

    def predict(self, prob: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0 - 1e-6)
        if not self.fitted:
            return p
        return np.asarray(self.model.predict(p), dtype=float)


@dataclass
class WeightedRegressionEnsemble:
    base_models: List[Tuple[str, Pipeline]]
    weights: np.ndarray
    feature_cols: List[str]
    sigma_floor: float
    oof_errors: Dict[str, float]

    def predict_model_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [
                np.clip(model.predict(df[self.feature_cols]), 0.0, 1.0)
                for _, model in self.base_models
            ]
        )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        matrix = self.predict_model_matrix(df)
        return np.average(matrix, axis=1, weights=self.weights)


@dataclass
class CalibratedBinaryModel:
    base_models: List[Tuple[str, Pipeline]]
    weights: np.ndarray
    calibrator: object
    feature_cols: List[str]
    oof_losses: Dict[str, float]

    def predict_model_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [
                np.clip(model.predict_proba(df[self.feature_cols])[:, 1], 1e-6, 1.0 - 1e-6)
                for _, model in self.base_models
            ]
        )

    def predict_raw_proba(self, df: pd.DataFrame) -> np.ndarray:
        matrix = self.predict_model_matrix(df)
        raw = np.average(matrix, axis=1, weights=self.weights)
        return np.clip(raw, 1e-6, 1.0 - 1e-6)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw_proba(df)
        cal = self.calibrator.predict(raw)
        return np.clip(cal, 1e-6, 1.0 - 1e-6)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


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


def build_preprocessor(df: pd.DataFrame, feature_cols: Sequence[str]) -> ColumnTransformer:
    numeric_cols, categorical_cols = infer_column_types(df, feature_cols)

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("前処理対象の特徴量がありません。")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _none_or_int(value: object) -> int | None:
    if value in [None, "null"]:
        return None
    return int(value)


def _blend_min_weight(config: Dict[str, object]) -> float:
    return float(config.get("model", {}).get("blending", {}).get("min_weight", 0.08))


def _residual_sigma_scale(config: Dict[str, object]) -> float:
    return float(config.get("model", {}).get("blending", {}).get("residual_sigma_scale", 0.35))


def get_regression_base_names(config: Dict[str, object]) -> List[str]:
    names = config.get("model", {}).get("regressor", {}).get("bases", ["ridge", "random_forest", "extra_trees"])
    return [str(name) for name in names]


def get_classifier_base_names(config: Dict[str, object]) -> List[str]:
    names = config.get("model", {}).get("classifier", {}).get("bases", ["logistic", "random_forest", "extra_trees"])
    return [str(name) for name in names]


def build_regression_estimator(name: str, random_state: int, config: Dict[str, object]):
    reg_cfg = config.get("model", {}).get("regressor", {})

    if name == "ridge":
        cfg = reg_cfg.get("ridge", {})
        return Ridge(alpha=float(cfg.get("alpha", 1.0)))

    fallback = reg_cfg.get("random_forest", reg_cfg)
    if name == "random_forest":
        cfg = fallback
        return RandomForestRegressor(
            n_estimators=int(cfg.get("n_estimators", 160)),
            max_depth=_none_or_int(cfg.get("max_depth", 10)),
            min_samples_leaf=int(cfg.get("min_samples_leaf", 4)),
            max_features=cfg.get("max_features", "sqrt"),
            n_jobs=int(cfg.get("n_jobs", -1)),
            random_state=random_state,
        )

    if name == "extra_trees":
        cfg = reg_cfg.get("extra_trees", {})
        return ExtraTreesRegressor(
            n_estimators=int(cfg.get("n_estimators", 240)),
            max_depth=_none_or_int(cfg.get("max_depth", 10)),
            min_samples_leaf=int(cfg.get("min_samples_leaf", 3)),
            max_features=cfg.get("max_features", "sqrt"),
            n_jobs=int(cfg.get("n_jobs", -1)),
            random_state=random_state,
        )

    raise ValueError(f"未知の回帰 base model です: {name}")


def build_classifier_estimator(name: str, random_state: int, config: Dict[str, object]):
    cls_cfg = config.get("model", {}).get("classifier", {})

    if name == "logistic":
        cfg = cls_cfg.get("logistic", cls_cfg)
        return LogisticRegression(
            C=float(cfg.get("C", 0.5)),
            max_iter=int(cfg.get("max_iter", 1000)),
            solver=str(cfg.get("solver", "lbfgs")),
            class_weight=cfg.get("class_weight"),
            random_state=random_state,
        )

    fallback = cls_cfg.get("random_forest", {})
    if name == "random_forest":
        cfg = fallback
        return RandomForestClassifier(
            n_estimators=int(cfg.get("n_estimators", 160)),
            max_depth=_none_or_int(cfg.get("max_depth", 8)),
            min_samples_leaf=int(cfg.get("min_samples_leaf", 4)),
            max_features=cfg.get("max_features", "sqrt"),
            class_weight=cfg.get("class_weight"),
            n_jobs=int(cfg.get("n_jobs", -1)),
            random_state=random_state,
        )

    if name == "extra_trees":
        cfg = cls_cfg.get("extra_trees", {})
        return ExtraTreesClassifier(
            n_estimators=int(cfg.get("n_estimators", 240)),
            max_depth=_none_or_int(cfg.get("max_depth", 8)),
            min_samples_leaf=int(cfg.get("min_samples_leaf", 3)),
            max_features=cfg.get("max_features", "sqrt"),
            class_weight=cfg.get("class_weight"),
            n_jobs=int(cfg.get("n_jobs", -1)),
            random_state=random_state,
        )

    raise ValueError(f"未知の分類 base model です: {name}")


def build_regression_pipeline(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    random_state: int,
    config: Dict[str, object],
    model_name: str,
) -> Pipeline:
    preprocessor = build_preprocessor(train_df, feature_cols)
    model = build_regression_estimator(model_name, random_state=random_state, config=config)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_classifier_pipeline(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    random_state: int,
    config: Dict[str, object],
    model_name: str,
) -> Pipeline:
    preprocessor = build_preprocessor(train_df, feature_cols)
    model = build_classifier_estimator(model_name, random_state=random_state, config=config)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def fit_regression_base(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    y: np.ndarray,
    sample_weight: np.ndarray,
    random_state: int,
    config: Dict[str, object],
    model_name: str,
) -> Pipeline:
    pipeline = build_regression_pipeline(
        train_df,
        feature_cols,
        random_state=random_state,
        config=config,
        model_name=model_name,
    )
    pipeline.fit(train_df[list(feature_cols)], y, model__sample_weight=sample_weight)
    return pipeline


def fit_classifier_base(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    y: np.ndarray,
    sample_weight: np.ndarray,
    random_state: int,
    config: Dict[str, object],
    model_name: str,
) -> Pipeline:
    pipeline = build_classifier_pipeline(
        train_df,
        feature_cols,
        random_state=random_state,
        config=config,
        model_name=model_name,
    )
    pipeline.fit(train_df[list(feature_cols)], y, model__sample_weight=sample_weight)
    return pipeline


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

    train_end = min_train_races
    remaining = races.iloc[train_end:].copy()
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


def _normalize_weights(values: Sequence[float], min_weight: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    if arr.sum() <= 0:
        arr = np.ones_like(arr, dtype=float)
    arr = arr / arr.sum()
    arr = np.maximum(arr, min_weight)
    return arr / arr.sum()


def _losses_to_weights(losses: Sequence[float], min_weight: float) -> np.ndarray:
    arr = np.asarray(losses, dtype=float)
    scores = 1.0 / np.clip(arr, 1e-6, None)
    return _normalize_weights(scores, min_weight=min_weight)


def _fit_probability_calibrator(prob: np.ndarray, y: np.ndarray, method: str) -> object:
    if len(np.unique(y)) < 2 or len(np.unique(np.round(prob, 6))) < 3:
        return IdentityCalibrator()
    if method == "isotonic":
        return IsotonicLikeCalibrator().fit(prob, y)
    return PlattCalibrator().fit(prob, y)


def fit_regression_ensemble(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    y: np.ndarray,
    sample_weight: np.ndarray,
    random_state: int,
    config: Dict[str, object],
    fixed_weights: np.ndarray | None = None,
    fixed_sigma_floor: float | None = None,
) -> WeightedRegressionEnsemble:
    base_names = get_regression_base_names(config)
    min_weight = _blend_min_weight(config)
    y_arr = np.asarray(y, dtype=float)
    sw_arr = np.asarray(sample_weight, dtype=float)
    weights = np.asarray(fixed_weights, dtype=float) if fixed_weights is not None else None
    sigma_floor = float(fixed_sigma_floor) if fixed_sigma_floor is not None else 0.05
    oof_errors: Dict[str, float] = {}

    if weights is None:
        cv_cfg = config.get("cv", {})
        inner_splits = int(cv_cfg.get("regression_inner_splits", cv_cfg.get("classifier_inner_splits", 3)))
        inner_min_train_races = int(
            cv_cfg.get("regression_inner_min_train_races", cv_cfg.get("classifier_inner_min_train_races", 20))
        )
        try:
            splits = time_series_race_splits(train_df, n_splits=inner_splits, min_train_races=inner_min_train_races)
            oof_matrix = np.full((len(train_df), len(base_names)), np.nan, dtype=float)
            for fold_no, (tr_idx, va_idx) in enumerate(splits, start=1):
                inner_train = train_df.iloc[tr_idx]
                inner_val = train_df.iloc[va_idx]
                for model_no, model_name in enumerate(base_names):
                    model = fit_regression_base(
                        inner_train,
                        feature_cols,
                        y=y_arr[tr_idx],
                        sample_weight=sw_arr[tr_idx],
                        random_state=random_state + (fold_no * 100) + model_no,
                        config=config,
                        model_name=model_name,
                    )
                    oof_matrix[va_idx, model_no] = np.clip(model.predict(inner_val[list(feature_cols)]), 0.0, 1.0)

            valid_mask = np.isfinite(oof_matrix).all(axis=1)
            if valid_mask.sum() >= max(50, len(base_names) * 8):
                losses = []
                for idx, model_name in enumerate(base_names):
                    mae = float(mean_absolute_error(y_arr[valid_mask], oof_matrix[valid_mask, idx]))
                    oof_errors[model_name] = mae
                    losses.append(mae)
                weights = _losses_to_weights(losses, min_weight=min_weight)
                blended_oof = np.average(oof_matrix[valid_mask], axis=1, weights=weights)
                rmse = float(np.sqrt(np.mean((y_arr[valid_mask] - blended_oof) ** 2)))
                sigma_floor = max(0.03, rmse * _residual_sigma_scale(config))
            else:
                weights = _normalize_weights(np.ones(len(base_names)), min_weight=min_weight)
        except Exception:
            weights = _normalize_weights(np.ones(len(base_names)), min_weight=min_weight)
    else:
        weights = _normalize_weights(weights, min_weight=min_weight)

    full_models: List[Tuple[str, Pipeline]] = []
    for model_no, model_name in enumerate(base_names):
        full_models.append(
            (
                model_name,
                fit_regression_base(
                    train_df,
                    feature_cols,
                    y=y_arr,
                    sample_weight=sw_arr,
                    random_state=random_state + 1000 + model_no,
                    config=config,
                    model_name=model_name,
                ),
            )
        )

    return WeightedRegressionEnsemble(
        base_models=full_models,
        weights=weights,
        feature_cols=list(feature_cols),
        sigma_floor=float(sigma_floor),
        oof_errors=oof_errors,
    )


def fit_calibrated_classifier(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    y: np.ndarray,
    sample_weight: np.ndarray,
    random_state: int,
    config: Dict[str, object],
) -> CalibratedBinaryModel:
    base_names = get_classifier_base_names(config)
    min_weight = _blend_min_weight(config)
    y_arr = np.asarray(y, dtype=int)
    sw_arr = np.asarray(sample_weight, dtype=float)
    method = str(config.get("model", {}).get("calibration", {}).get("method", "isotonic")).lower()
    weights = _normalize_weights(np.ones(len(base_names)), min_weight=min_weight)
    oof_losses: Dict[str, float] = {}
    calibrator: object = IdentityCalibrator()

    cv_cfg = config.get("cv", {})
    inner_splits = int(cv_cfg.get("classifier_inner_splits", 3))
    inner_min_train_races = int(cv_cfg.get("classifier_inner_min_train_races", 20))
    try:
        splits = time_series_race_splits(train_df, n_splits=inner_splits, min_train_races=inner_min_train_races)
        oof_matrix = np.full((len(train_df), len(base_names)), np.nan, dtype=float)

        for fold_no, (tr_idx, va_idx) in enumerate(splits, start=1):
            if len(np.unique(y_arr[tr_idx])) < 2:
                continue
            inner_train = train_df.iloc[tr_idx]
            inner_val = train_df.iloc[va_idx]
            for model_no, model_name in enumerate(base_names):
                model = fit_classifier_base(
                    inner_train,
                    feature_cols,
                    y=y_arr[tr_idx],
                    sample_weight=sw_arr[tr_idx],
                    random_state=random_state + (fold_no * 100) + model_no,
                    config=config,
                    model_name=model_name,
                )
                oof_matrix[va_idx, model_no] = np.clip(
                    model.predict_proba(inner_val[list(feature_cols)])[:, 1],
                    1e-6,
                    1.0 - 1e-6,
                )

        valid_mask = np.isfinite(oof_matrix).all(axis=1)
        if valid_mask.sum() >= max(50, len(base_names) * 8) and len(np.unique(y_arr[valid_mask])) >= 2:
            losses = []
            for idx, model_name in enumerate(base_names):
                prob = np.clip(oof_matrix[valid_mask, idx], 1e-6, 1.0 - 1e-6)
                combined_loss = (0.7 * log_loss(y_arr[valid_mask], prob)) + (0.3 * brier_score_loss(y_arr[valid_mask], prob))
                oof_losses[model_name] = float(combined_loss)
                losses.append(combined_loss)
            weights = _losses_to_weights(losses, min_weight=min_weight)
            blended_oof = np.average(oof_matrix[valid_mask], axis=1, weights=weights)
            calibrator = _fit_probability_calibrator(blended_oof, y_arr[valid_mask], method=method)
    except Exception:
        pass

    full_models: List[Tuple[str, Pipeline]] = []
    for model_no, model_name in enumerate(base_names):
        full_models.append(
            (
                model_name,
                fit_classifier_base(
                    train_df,
                    feature_cols,
                    y=y_arr,
                    sample_weight=sw_arr,
                    random_state=random_state + 1000 + model_no,
                    config=config,
                    model_name=model_name,
                ),
            )
        )

    return CalibratedBinaryModel(
        base_models=full_models,
        weights=weights,
        calibrator=calibrator,
        feature_cols=list(feature_cols),
        oof_losses=oof_losses,
    )


def precision_at_k_by_race(
    df: pd.DataFrame,
    prob_col: str = "pred_top3",
    label_col: str = "is_top3",
    race_col: str = "race_id",
    k: int = 3,
) -> float:
    scores: List[float] = []
    for _, g in df.groupby(race_col):
        kk = min(k, len(g))
        top_g = g.nlargest(kk, prob_col)
        scores.append(float(top_g[label_col].mean()))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def evaluate_time_series_cv(
    history_df: pd.DataFrame,
    feature_cols: Sequence[str],
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

        y_reg_train = train_df["finish_percentile"].to_numpy(dtype=float)
        y_cls_train = train_df["is_top3"].to_numpy(dtype=int)
        sw_train = sample_weight.iloc[train_idx].to_numpy(dtype=float)

        reg_model = fit_regression_ensemble(
            train_df,
            feature_cols,
            y=y_reg_train,
            sample_weight=sw_train,
            random_state=int(config.get("seed", 42)) + fold_no,
            config=config,
        )
        cls_model = fit_calibrated_classifier(
            train_df,
            feature_cols,
            y=y_cls_train,
            sample_weight=sw_train,
            random_state=int(config.get("seed", 42)) + 100 + fold_no,
            config=config,
        )

        pred_reg = np.clip(reg_model.predict(val_df), 0.0, 1.0)
        pred_top3 = cls_model.predict_proba(val_df)

        fold_eval = val_df[["race_id", "finish_percentile", "is_top3"]].copy()
        fold_eval["pred_reg"] = pred_reg
        fold_eval["pred_top3"] = pred_top3
        fold_eval["fold"] = fold_no

        oof_frames.append(fold_eval)
        fold_metrics.append(
            {
                "fold": float(fold_no),
                "brier_score": float(brier_score_loss(val_df["is_top3"], pred_top3)),
                "log_loss": float(log_loss(val_df["is_top3"], np.clip(pred_top3, 1e-6, 1.0 - 1e-6))),
                "top3_precision": float(
                    precision_at_k_by_race(fold_eval, prob_col="pred_top3", label_col="is_top3", race_col="race_id", k=3)
                ),
                "mae_finish_percentile": float(mean_absolute_error(val_df["finish_percentile"], pred_reg)),
            }
        )

    oof_df = pd.concat(oof_frames, ignore_index=True)
    summary = {
        "brier_score": float(brier_score_loss(oof_df["is_top3"], oof_df["pred_top3"])),
        "log_loss": float(log_loss(oof_df["is_top3"], np.clip(oof_df["pred_top3"], 1e-6, 1.0 - 1e-6))),
        "top3_precision": float(
            precision_at_k_by_race(oof_df, prob_col="pred_top3", label_col="is_top3", race_col="race_id", k=3)
        ),
        "mae_finish_percentile": float(mean_absolute_error(oof_df["finish_percentile"], oof_df["pred_reg"])),
    }

    prob_true, prob_pred = calibration_curve(
        oof_df["is_top3"],
        np.clip(oof_df["pred_top3"], 1e-6, 1.0 - 1e-6),
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


def fit_bootstrap_regression_ensemble(
    history_df: pd.DataFrame,
    feature_cols: Sequence[str],
    sample_weight: pd.Series,
    config: Dict[str, object],
    n_models: int | None = None,
    seed_offset: int = 0,
) -> Tuple[List[WeightedRegressionEnsemble], WeightedRegressionEnsemble]:
    model_cfg = config.get("model", {}).get("bootstrap", {})
    n_boot = int(n_models or model_cfg.get("n_models", 12))
    seed = int(config.get("seed", 42)) + seed_offset
    rng = np.random.default_rng(seed)

    y = history_df["finish_percentile"].to_numpy(dtype=float)
    sw = sample_weight.to_numpy(dtype=float)

    main_model = fit_regression_ensemble(
        history_df,
        feature_cols,
        y=y,
        sample_weight=sw,
        random_state=seed + 999,
        config=config,
    )

    models: List[WeightedRegressionEnsemble] = []
    for i in range(n_boot):
        idx = rng.choice(len(history_df), size=len(history_df), replace=True)
        boot_df = history_df.iloc[idx].reset_index(drop=True)
        boot_y = y[idx]
        boot_sw = sw[idx]
        models.append(
            fit_regression_ensemble(
                boot_df,
                feature_cols,
                y=boot_y,
                sample_weight=boot_sw,
                random_state=seed + i,
                config=config,
                fixed_weights=main_model.weights,
                fixed_sigma_floor=main_model.sigma_floor,
            )
        )

    return models, main_model


def predict_regression_ensemble(
    models: Sequence[WeightedRegressionEnsemble],
    entry_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_matrix = np.column_stack([np.clip(model.predict(entry_df), 0.0, 1.0) for model in models])
    mu = pred_matrix.mean(axis=1)
    ensemble_std = pred_matrix.std(axis=1, ddof=1) if pred_matrix.shape[1] > 1 else np.zeros(pred_matrix.shape[0], dtype=float)
    sigma_floor = float(np.mean([model.sigma_floor for model in models])) if models else 0.05
    sigma = np.sqrt((ensemble_std ** 2) + (sigma_floor ** 2))
    sigma = np.maximum(sigma, max(0.03, sigma_floor))
    return mu, sigma, pred_matrix


def _extract_pipeline_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        importance = np.abs(coef)
        if importance.ndim > 1:
            importance = importance.mean(axis=0)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    total = float(np.sum(np.abs(importance)))
    if total > 0:
        importance = np.abs(importance) / total
    return pd.DataFrame({"feature": feature_names, "importance": importance})


def get_feature_importance(main_model: WeightedRegressionEnsemble) -> pd.DataFrame:
    weighted_frames: List[pd.DataFrame] = []
    for weight, (_, pipeline) in zip(main_model.weights, main_model.base_models):
        imp_df = _extract_pipeline_feature_importance(pipeline)
        if imp_df.empty:
            continue
        imp_df = imp_df.copy()
        imp_df["importance"] = imp_df["importance"] * float(weight)
        weighted_frames.append(imp_df)

    if not weighted_frames:
        return pd.DataFrame(columns=["feature", "importance"])

    combined = pd.concat(weighted_frames, ignore_index=True)
    out = combined.groupby("feature", as_index=False)["importance"].sum()
    return out.sort_values("importance", ascending=False).reset_index(drop=True)
