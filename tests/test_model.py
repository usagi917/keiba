from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from keiba_predictor.features import (
    add_aggregate_features,
    add_basic_features,
    add_targets,
    infer_feature_columns,
)
from keiba_predictor.model import (
    build_preprocessor,
    fit_calibrated_classifier,
    fit_regression_ensemble,
)


def _prepare_train_data(synthetic_history, synthetic_entry, base_config):
    """Helper to prepare training data with targets and features."""
    history = add_targets(synthetic_history)
    history, entry = add_basic_features(history, synthetic_entry)
    history, entry = add_aggregate_features(history, entry)
    feat_cols = infer_feature_columns(history, entry, use_odds=True)
    return history, feat_cols


class TestFitRegressionEnsemble:
    def test_fits_and_predicts(self, synthetic_history, synthetic_entry, base_config):
        history, feat_cols = _prepare_train_data(synthetic_history, synthetic_entry, base_config)
        model = fit_regression_ensemble(
            train_df=history,
            feature_cols=feat_cols,
            y=history["finish_percentile"].to_numpy(dtype=float),
            sample_weight=np.ones(len(history)),
            random_state=42,
            config=base_config,
        )
        predictions = model.predict(history)
        assert len(predictions) == len(history)
        assert np.all(np.isfinite(predictions))

    def test_predictions_in_range(self, synthetic_history, synthetic_entry, base_config):
        history, feat_cols = _prepare_train_data(synthetic_history, synthetic_entry, base_config)
        model = fit_regression_ensemble(
            train_df=history,
            feature_cols=feat_cols,
            y=history["finish_percentile"].to_numpy(dtype=float),
            sample_weight=np.ones(len(history)),
            random_state=42,
            config=base_config,
        )
        predictions = model.predict(history)
        assert predictions.min() >= -0.5
        assert predictions.max() <= 1.5

    def test_build_preprocessor_drops_all_missing_numeric_columns(self):
        train_df = pd.DataFrame(
            {
                "age": [3, 4, 5],
                "odds": [np.nan, np.nan, np.nan],
                "jockey": ["J01", "J02", "J03"],
            }
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            preprocessor = build_preprocessor(train_df, ["age", "odds", "jockey"])
            transformed = preprocessor.fit_transform(train_df)

        assert transformed.shape[0] == len(train_df)
        assert all("Skipping features without any observed values" not in str(w.message) for w in caught)
        assert "num__age" in preprocessor.get_feature_names_out()
        # odds is all-NaN, should be dropped regardless of imputer
        assert not any("__odds" in n for n in preprocessor.get_feature_names_out())


class TestOddsImputation:
    """オッズ系特徴量は mean imputation を使用すること (Median では偏った補完になる)"""

    def test_odds_nan_imputed_to_mean_not_median(self):
        """偏ったオッズ分布では mean ≠ median なので、mean 補完なら NaN が平均に揃う"""
        # median=1.0, mean=25.75 の歪んだ分布
        df = pd.DataFrame({"odds": [np.nan, 1.0, 1.0, 1.0, 100.0]})
        preprocessor = build_preprocessor(df, ["odds"])
        transformed = preprocessor.fit_transform(df)
        # mean imputation: NaN → mean → StandardScaler で 0.0 になる
        # median imputation: NaN → 1.0 → scaled で負の値になる
        assert transformed[0, 0] == pytest.approx(0.0, abs=0.05)

    def test_non_odds_numeric_unchanged_strategy(self):
        """オッズ以外の数値特徴量 (age 等) は median imputation を維持すること"""
        # age の median=4, mean=4 (対称) → どちらでも同じだが、transformer 名で確認
        df = pd.DataFrame({"age": [3, 4, 5], "odds": [np.nan, 5.0, 3.0]})
        preprocessor = build_preprocessor(df, ["age", "odds"])
        preprocessor.fit(df)
        feature_names = list(preprocessor.get_feature_names_out())
        # odds は odds_num__ プレフィックスを持つこと
        assert any("odds_num__odds" in n for n in feature_names)
        # age は通常の num__ プレフィックスを持つこと
        assert any("num__age" in n for n in feature_names)


class TestFitCalibratedClassifier:
    def test_fits_and_predicts(self, synthetic_history, synthetic_entry, base_config):
        history, feat_cols = _prepare_train_data(synthetic_history, synthetic_entry, base_config)
        model = fit_calibrated_classifier(
            train_df=history,
            feature_cols=feat_cols,
            y=history["is_top3"].to_numpy(dtype=int),
            sample_weight=np.ones(len(history)),
            random_state=42,
            config=base_config,
        )
        proba = model.predict_proba(history)
        assert len(proba) == len(history)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)
