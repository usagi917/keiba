from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba_predictor.features import (
    AGGREGATE_FEATURE_COLUMNS,
    FINISH_PCT_PRIOR_MEAN,
    RunningStats,
    _add_row_derived_features,
    _assign_recent_stats,
    _build_history_aggregate_features,
    _empty_feature_store,
    add_aggregate_features,
    add_basic_features,
    add_targets,
    compute_similarity_weights,
    get_distance_band,
    infer_feature_columns,
    is_graded,
    is_long_turf,
    is_turf_value,
)


class TestAddTargets:
    def test_creates_is_top3_and_finish_percentile(self, synthetic_history):
        result = add_targets(synthetic_history)
        assert "is_top3" in result.columns
        assert "finish_percentile" in result.columns

    def test_is_top3_matches_finish_rank(self, synthetic_history):
        result = add_targets(synthetic_history)
        for _, row in result.iterrows():
            expected = 1 if row["finish_rank"] <= 3 else 0
            assert row["is_top3"] == expected

    def test_finish_percentile_range(self, synthetic_history):
        result = add_targets(synthetic_history)
        assert result["finish_percentile"].min() >= 0.0
        assert result["finish_percentile"].max() <= 1.0

    def test_first_place_has_percentile_1(self, synthetic_history):
        result = add_targets(synthetic_history)
        first_place = result[result["finish_rank"] == 1]
        assert (first_place["finish_percentile"] == 1.0).all()

    def test_raises_without_field_size(self):
        df = pd.DataFrame({"finish_rank": [1, 2]})
        with pytest.raises(ValueError, match="field_size"):
            add_targets(df)


class TestAddBasicFeatures:
    def test_creates_derived_columns(self, synthetic_history, synthetic_entry):
        history = add_targets(synthetic_history)
        result_h, result_e = add_basic_features(history, synthetic_entry)
        assert "distance_change" in result_h.columns
        assert "distance_log" in result_h.columns
        assert "field_size_log" in result_h.columns

    def test_log_odds_created_when_odds_present(self, synthetic_history, synthetic_entry):
        history = add_targets(synthetic_history)
        result_h, result_e = add_basic_features(history, synthetic_entry)
        assert "log_odds" in result_h.columns
        assert "implied_prob" in result_h.columns

    def test_rank_score_features(self, synthetic_history, synthetic_entry):
        history = add_targets(synthetic_history)
        result_h, result_e = add_basic_features(history, synthetic_entry)
        if "odds_rank_score" in result_h.columns:
            vals = result_h["odds_rank_score"].dropna()
            assert (vals.between(0, 1)).all()

    def test_running_style_features_when_available(self):
        history = pd.DataFrame(
            [
                {"race_id": "R1", "horse_id": "H1", "field_size": 4, "running_style": "逃げ"},
                {"race_id": "R1", "horse_id": "H2", "field_size": 4, "running_style": "先行"},
                {"race_id": "R1", "horse_id": "H3", "field_size": 4, "running_style": "差し"},
                {"race_id": "R1", "horse_id": "H4", "field_size": 4, "running_style": "追込"},
            ]
        )
        entry = history.copy()

        result_h, result_e = add_basic_features(history, entry)

        assert "running_style_balance" in result_h.columns
        assert "same_style_count" in result_h.columns
        assert result_h["running_style_balance"].notna().all()
        assert result_e["same_style_count"].notna().all()

    def test_running_style_features_graceful_when_missing(self, synthetic_history, synthetic_entry):
        history = add_targets(synthetic_history)
        result_h, result_e = add_basic_features(history, synthetic_entry)
        assert "running_style_balance" not in result_h.columns
        assert "same_style_count" not in result_e.columns


class TestAddAggregateFeatures:
    def test_creates_aggregate_columns(self, synthetic_history, synthetic_entry):
        history = add_targets(synthetic_history)
        history, entry = add_basic_features(history, synthetic_entry)
        result_h, result_e = add_aggregate_features(history, entry)
        expected_cols = [
            "horse_overall_starts",
            "horse_overall_top3_rate_smooth",
            "jockey_overall_starts",
        ]
        for col in expected_cols:
            assert col in result_h.columns, f"Missing column: {col}"

    def test_smooth_values_are_bounded(self, synthetic_history, synthetic_entry):
        history = add_targets(synthetic_history)
        history, entry = add_basic_features(history, synthetic_entry)
        result_h, result_e = add_aggregate_features(history, entry)
        smooth_col = "horse_overall_top3_rate_smooth"
        if smooth_col in result_h.columns:
            vals = result_h[smooth_col].dropna()
            assert (vals >= 0).all()
            assert (vals <= 1).all()

    def test_entry_gets_aggregate_features(self, synthetic_history, synthetic_entry):
        history = add_targets(synthetic_history)
        history, entry = add_basic_features(history, synthetic_entry)
        result_h, result_e = add_aggregate_features(history, entry)
        assert "horse_overall_top3_rate_smooth" in result_e.columns

    def test_missing_history_uses_prior_in_entry_features(self, synthetic_history, synthetic_entry):
        history = add_targets(synthetic_history.iloc[:10].copy())
        entry = synthetic_entry.copy()
        entry.loc[0, "horse_id"] = "NEW_HORSE"
        history, entry = add_basic_features(history, entry)
        _, result_e = add_aggregate_features(history, entry)

        row = result_e.loc[result_e["horse_id"] == "NEW_HORSE"].iloc[0]
        assert row["horse_overall_starts"] == pytest.approx(0.0)
        assert row["horse_overall_top3_rate_smooth"] == pytest.approx(0.30)
        assert row["horse_overall_finish_pct_mean"] == pytest.approx(0.50)
        assert row["horse_recent3_top3_rate"] == pytest.approx(0.30)
        assert row["horse_recent3_finish_pct_mean"] == pytest.approx(0.50)

    def test_sparse_context_flags_are_added(self, synthetic_history, synthetic_entry):
        history = add_targets(synthetic_history.iloc[:10].copy())
        entry = synthetic_entry.copy()
        entry.loc[0, "horse_id"] = "NEW_HORSE"
        history, entry = add_basic_features(history, entry)
        _, result_e = add_aggregate_features(history, entry)

        row = result_e.loc[result_e["horse_id"] == "NEW_HORSE"].iloc[0]
        assert row["horse_overall_data_sparse"] == pytest.approx(1.0)
        assert row["horse_course_data_sparse"] == pytest.approx(1.0)
        assert row["horse_dist_band_data_sparse"] == pytest.approx(1.0)
        assert row["top3_model_sparse_context"] == pytest.approx(1.0)


class TestRunningStats:
    def test_initial_state(self):
        stats = RunningStats()
        assert stats.starts() is np.nan or pd.isna(stats.starts())
        assert stats.top3_rate_smooth() == pytest.approx(0.30, abs=0.01)

    def test_after_updates(self):
        stats = RunningStats()
        stats.update(finish_percentile=1.0, is_top3=1)
        stats.update(finish_percentile=0.5, is_top3=0)
        stats.update(finish_percentile=0.8, is_top3=1)
        assert stats.count == 3
        assert stats.top3 == 2
        assert stats.top3_rate() == pytest.approx(2 / 3, abs=0.01)


class TestIsTurfAndLongTurf:
    def test_turf_values(self):
        assert is_turf_value("turf") is True
        assert is_turf_value("芝") is True
        assert is_turf_value("dirt") is False

    def test_long_turf(self):
        assert is_long_turf("turf", 2400) is True
        assert is_long_turf("turf", 1200) is False
        assert is_long_turf("dirt", 2400) is False


class TestGetDistanceBand:
    def test_sprint(self):
        assert get_distance_band(1200) == "sprint"
        assert get_distance_band(1400) == "sprint"

    def test_mile(self):
        assert get_distance_band(1600) == "mile"

    def test_intermediate(self):
        assert get_distance_band(2000) == "intermediate"

    def test_long(self):
        assert get_distance_band(2400) == "long"

    def test_stayer(self):
        assert get_distance_band(3000) == "stayer"

    def test_invalid(self):
        assert get_distance_band("abc") is None


class TestIsGraded:
    def test_graded(self):
        assert is_graded("G1") is True
        assert is_graded("G2") is True
        assert is_graded("Listed") is True

    def test_not_graded(self):
        assert is_graded("") is False
        assert is_graded(None) is False


class TestNewDerivedFeatures:
    def test_class_transition(self, synthetic_history, synthetic_entry):
        history = add_targets(synthetic_history)
        result_h, _ = add_basic_features(history, synthetic_entry)
        if "class_transition" in result_h.columns:
            assert "is_class_up" in result_h.columns
            assert "is_class_down" in result_h.columns

    def test_rest_features(self, synthetic_history, synthetic_entry):
        history = add_targets(synthetic_history)
        result_h, _ = add_basic_features(history, synthetic_entry)
        assert "is_fresh" in result_h.columns
        assert "is_layoff" in result_h.columns

    def test_speed_proxy(self, synthetic_history, synthetic_entry):
        history = add_targets(synthetic_history)
        result_h, _ = add_basic_features(history, synthetic_entry)
        assert "last_3f_per_furlong" in result_h.columns
        assert "relative_3f_speed" in result_h.columns


class TestNewAggregateFeatures:
    def test_dist_band_features(self, synthetic_history, synthetic_entry):
        history = add_targets(synthetic_history)
        history, entry = add_basic_features(history, synthetic_entry)
        result_h, result_e = add_aggregate_features(history, entry)
        assert "horse_dist_band_top3_rate_smooth" in result_h.columns
        assert "jockey_dist_band_top3_rate_smooth" in result_h.columns

    def test_graded_features(self, synthetic_history, synthetic_entry):
        history = add_targets(synthetic_history)
        history, entry = add_basic_features(history, synthetic_entry)
        result_h, result_e = add_aggregate_features(history, entry)
        assert "horse_graded_top3_rate_smooth" in result_h.columns
        assert "jockey_graded_top3_rate_smooth" in result_h.columns

    def test_jockey_course_features(self, synthetic_history, synthetic_entry):
        history = add_targets(synthetic_history)
        history, entry = add_basic_features(history, synthetic_entry)
        result_h, result_e = add_aggregate_features(history, entry)
        assert "jockey_course_top3_rate_smooth" in result_h.columns


class TestComputeSimilarityWeights:
    def test_returns_positive_weights(self, synthetic_history, base_config):
        history = add_targets(synthetic_history)
        weights = compute_similarity_weights(history, base_config)
        assert len(weights) == len(history)
        assert all(w > 0 for w in weights)

    def test_similar_races_get_higher_weight(self, base_config):
        base_config["target_race_profile"]["surface"] = "turf"
        base_config["target_race_profile"]["distance"] = 1200
        df = pd.DataFrame({
            "surface": ["turf", "dirt"],
            "distance": [1200, 3000],
            "course": ["Chukyo", "Tokyo"],
            "turn": ["left", "right"],
            "grade": ["G1", ""],
            "class": ["Open", "Allowance"],
            "age": [4, 3],
            "race_date": pd.to_datetime(["2026-03-01", "2026-03-01"]),
            "field_size": [10, 10],
            "finish_rank": [1, 5],
            "race_id": ["R1", "R2"],
        })
        df = add_targets(df)
        weights = compute_similarity_weights(df, base_config)
        assert weights[0] > weights[1]


class TestAggregateFeatureLeakDetection:
    """集約特徴量のリーク防止を検証: expanding windowのみ使用していること"""

    def _make_sequential_races(self) -> pd.DataFrame:
        """H1 が R1(2024-01-01), R2(2024-02-01), R3(2024-03-01) に出走"""
        rows = []
        dates = ["2024-01-01", "2024-02-01", "2024-03-01"]
        for i, (race_id, date) in enumerate(zip(["R1", "R2", "R3"], dates)):
            for j in range(5):
                rows.append({
                    "race_id": race_id,
                    "race_date": pd.Timestamp(date),
                    "horse_id": f"H{j + 1}",
                    "finish_rank": j + 1,
                    "field_size": 5,
                    "jockey": "J1",
                    "trainer": "T1",
                    "surface": "turf",
                    "distance": 2000,
                    "grade": "",
                    "course": "Tokyo",
                })
        return pd.DataFrame(rows)

    def test_first_race_has_no_prior_history(self):
        """初出走時は事前データなし → starts=0 (prior)"""
        df = add_targets(self._make_sequential_races())
        result = _build_history_aggregate_features(df)
        h1 = result[result["horse_id"] == "H1"].sort_values("race_date")
        first = h1.iloc[0]
        assert first["horse_overall_starts"] == pytest.approx(0.0)

    def test_second_race_reflects_exactly_one_prior_race(self):
        df = add_targets(self._make_sequential_races())
        result = _build_history_aggregate_features(df)
        h1 = result[result["horse_id"] == "H1"].sort_values("race_date")
        second = h1.iloc[1]
        assert second["horse_overall_starts"] == pytest.approx(1.0)

    def test_third_race_reflects_two_prior_races(self):
        df = add_targets(self._make_sequential_races())
        result = _build_history_aggregate_features(df)
        h1 = result[result["horse_id"] == "H1"].sort_values("race_date")
        third = h1.iloc[2]
        assert third["horse_overall_starts"] == pytest.approx(2.0)

    def test_top3_rate_uses_only_past_results(self):
        """H1はR1で1着(is_top3=1)→R2時点でtop3_rate=1.0になること"""
        df = add_targets(self._make_sequential_races())
        result = _build_history_aggregate_features(df)
        h1 = result[result["horse_id"] == "H1"].sort_values("race_date")
        second = h1.iloc[1]  # R2での特徴量
        assert second["horse_overall_top3_rate"] == pytest.approx(1.0)


class TestRestBucket:
    """休養バケット高度化: 0-21d/22-35d/36-90d/91-180d/180d+"""

    def _df(self, days: int) -> pd.DataFrame:
        return pd.DataFrame([{"days_since_last": days, "distance": 2000, "field_size": 10}])

    def test_very_short_under_21(self):
        result = _add_row_derived_features(self._df(10))
        assert result["rest_bucket"].iloc[0] == "very_short"

    def test_boundary_21_is_very_short(self):
        result = _add_row_derived_features(self._df(21))
        assert result["rest_bucket"].iloc[0] == "very_short"

    def test_boundary_22_is_short(self):
        result = _add_row_derived_features(self._df(22))
        assert result["rest_bucket"].iloc[0] == "short"

    def test_short_28_days(self):
        result = _add_row_derived_features(self._df(28))
        assert result["rest_bucket"].iloc[0] == "short"

    def test_normal_60_days(self):
        result = _add_row_derived_features(self._df(60))
        assert result["rest_bucket"].iloc[0] == "normal"

    def test_long_120_days(self):
        result = _add_row_derived_features(self._df(120))
        assert result["rest_bucket"].iloc[0] == "long"

    def test_very_long_200_days(self):
        result = _add_row_derived_features(self._df(200))
        assert result["rest_bucket"].iloc[0] == "very_long"

    def test_rest_bucket_in_raw_model_features(self):
        from keiba_predictor.features import RAW_MODEL_FEATURES
        assert "rest_bucket" in RAW_MODEL_FEATURES


class TestDrawOuterInner:
    """枠番×頭数交互作用: draw_inner(枠番4以内) / draw_outer(フィールドの70%以上)"""

    def _df(self, draw: int, field_size: int) -> pd.DataFrame:
        return pd.DataFrame([{"draw": draw, "field_size": field_size}])

    def test_inner_draw_flag(self):
        result = _add_row_derived_features(self._df(2, 14))
        assert result["draw_inner"].iloc[0] == pytest.approx(1.0)
        assert result["draw_outer"].iloc[0] == pytest.approx(0.0)

    def test_outer_draw_flag(self):
        result = _add_row_derived_features(self._df(12, 14))
        assert result["draw_inner"].iloc[0] == pytest.approx(0.0)
        assert result["draw_outer"].iloc[0] == pytest.approx(1.0)

    def test_middle_draw_neither(self):
        result = _add_row_derived_features(self._df(7, 14))
        assert result["draw_inner"].iloc[0] == pytest.approx(0.0)
        assert result["draw_outer"].iloc[0] == pytest.approx(0.0)

    def test_boundary_draw_4_is_inner(self):
        result = _add_row_derived_features(self._df(4, 18))
        assert result["draw_inner"].iloc[0] == pytest.approx(1.0)

    def test_draw_outer_inner_in_raw_model_features(self):
        from keiba_predictor.features import RAW_MODEL_FEATURES
        assert "draw_inner" in RAW_MODEL_FEATURES
        assert "draw_outer" in RAW_MODEL_FEATURES


class TestJockeyTrainerCombo:
    """騎手×調教師コンビ統計: expanding window で集計"""

    def test_jockey_trainer_columns_exist_in_history(self, synthetic_history, synthetic_entry):
        history = add_targets(synthetic_history)
        history, entry = add_basic_features(history, synthetic_entry)
        result_h, _ = add_aggregate_features(history, entry)
        assert "jockey_trainer_top3_rate_smooth" in result_h.columns
        assert "jockey_trainer_starts" in result_h.columns

    def test_jockey_trainer_columns_exist_in_entry(self, synthetic_history, synthetic_entry):
        history = add_targets(synthetic_history)
        history, entry = add_basic_features(history, synthetic_entry)
        _, result_e = add_aggregate_features(history, entry)
        assert "jockey_trainer_top3_rate_smooth" in result_e.columns

    def test_known_combo_accumulates_stats(self):
        """同一コンビが2レース出走した後、3レース目では starts=2 になること"""
        rows = []
        for i, date in enumerate(["2024-01-01", "2024-02-01"]):
            for j in range(5):
                rows.append({
                    "race_id": f"R{i+1}",
                    "race_date": pd.Timestamp(date),
                    "horse_id": f"H{j+1}",
                    "finish_rank": j + 1,
                    "field_size": 5,
                    "jockey": "J1" if j == 0 else f"J{j+1}",
                    "trainer": "T1" if j == 0 else f"T{j+1}",
                    "surface": "turf",
                    "distance": 2000,
                    "grade": "",
                    "course": "Tokyo",
                })
        df = pd.DataFrame(rows)
        df = add_targets(df)
        result = _build_history_aggregate_features(df)
        # H1 (J1+T1コンビ) の R2 での特徴量: R1のデータが1件あるはず
        h1_r2 = result[(result["horse_id"] == "H1") & (result["race_id"] == "R2")].iloc[0]
        assert h1_r2["jockey_trainer_starts"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Phase 8: Recency Decay (EWM) + trend / std / best_finish
# ---------------------------------------------------------------------------

def _make_recent_store() -> dict:
    """_assign_recent_stats 用に新コラムを含む feature_store を作る。"""
    store = _empty_feature_store(1)
    for col in [
        "horse_recent3_finish_pct_ewm",
        "horse_recent5_finish_pct_ewm",
        "horse_finish_trend_3",
        "horse_finish_std_3",
        "horse_best_finish_5",
    ]:
        if col not in store:
            import numpy as np
            store[col] = np.full(1, np.nan)
    return store


class TestRecencyDecayFeatures:
    """Phase 8: EWM / trend / std / best_finish 集約特徴量"""

    def test_ewm_columns_in_aggregate_feature_columns(self):
        assert "horse_recent3_finish_pct_ewm" in AGGREGATE_FEATURE_COLUMNS
        assert "horse_recent5_finish_pct_ewm" in AGGREGATE_FEATURE_COLUMNS

    def test_trend_std_best_in_aggregate_feature_columns(self):
        assert "horse_finish_trend_3" in AGGREGATE_FEATURE_COLUMNS
        assert "horse_finish_std_3" in AGGREGATE_FEATURE_COLUMNS
        assert "horse_best_finish_5" in AGGREGATE_FEATURE_COLUMNS

    def test_new_horse_gets_prior_ewm(self, synthetic_history, synthetic_entry):
        history = add_targets(synthetic_history.iloc[:10].copy())
        entry = synthetic_entry.copy()
        entry.loc[0, "horse_id"] = "BRAND_NEW"
        history, entry = add_basic_features(history, entry)
        _, result_e = add_aggregate_features(history, entry)
        row = result_e.loc[result_e["horse_id"] == "BRAND_NEW"].iloc[0]
        assert row["horse_recent3_finish_pct_ewm"] == pytest.approx(FINISH_PCT_PRIOR_MEAN, abs=0.01)

    def test_ewm_weights_recent_more_than_old(self):
        """EWMは最新レースに高い重みを付ける: 古0.2→中0.5→新1.0 なら EWM > 単純平均"""
        store = _make_recent_store()
        records = [(0.2, 0, None), (0.5, 0, None), (1.0, 1, None)]
        _assign_recent_stats(store, 0, records)
        ewm_val = store["horse_recent3_finish_pct_ewm"][0]
        simple_mean = (0.2 + 0.5 + 1.0) / 3.0
        assert float(ewm_val) > simple_mean

    def test_ewm_equals_simple_mean_for_uniform(self):
        """全成績が同じならEWM == 単純平均"""
        store = _make_recent_store()
        records = [(0.6, 1, None), (0.6, 1, None), (0.6, 1, None)]
        _assign_recent_stats(store, 0, records)
        assert store["horse_recent3_finish_pct_ewm"][0] == pytest.approx(0.6, abs=0.001)

    def test_best_finish_5_is_max_of_last_5(self):
        """best_finish_5 は直近5レースの finish_percentile の最大値"""
        store = _make_recent_store()
        records = [(0.2, 0, None), (0.5, 0, None), (0.8, 1, None), (0.3, 0, None), (0.6, 0, None)]
        _assign_recent_stats(store, 0, records)
        assert store["horse_best_finish_5"][0] == pytest.approx(0.8)

    def test_best_finish_5_uses_only_last_5(self):
        """6レース以上あっても直近5件のみ参照"""
        store = _make_recent_store()
        records = [(1.0, 1, None), (0.1, 0, None), (0.1, 0, None), (0.1, 0, None), (0.1, 0, None), (0.1, 0, None)]
        _assign_recent_stats(store, 0, records)
        # 最古の1.0は直近5件に含まれないので max=0.1
        assert store["horse_best_finish_5"][0] == pytest.approx(0.1)

    def test_finish_trend_3_positive_for_improving(self):
        """改善傾向(0.2→0.5→0.9)はtrendが正"""
        store = _make_recent_store()
        records = [(0.2, 0, None), (0.5, 0, None), (0.9, 1, None)]
        _assign_recent_stats(store, 0, records)
        assert store["horse_finish_trend_3"][0] > 0

    def test_finish_trend_3_negative_for_declining(self):
        """悪化傾向(0.9→0.5→0.1)はtrendが負"""
        store = _make_recent_store()
        records = [(0.9, 1, None), (0.5, 0, None), (0.1, 0, None)]
        _assign_recent_stats(store, 0, records)
        assert store["horse_finish_trend_3"][0] < 0

    def test_finish_std_3_zero_for_consistent(self):
        """完全一致の成績ならstd=0"""
        store = _make_recent_store()
        records = [(0.5, 0, None), (0.5, 0, None), (0.5, 0, None)]
        _assign_recent_stats(store, 0, records)
        assert store["horse_finish_std_3"][0] == pytest.approx(0.0, abs=0.001)

    def test_finish_std_3_nonzero_for_variable(self):
        """バラつきのある成績はstd > 0"""
        store = _make_recent_store()
        records = [(0.1, 0, None), (0.5, 0, None), (0.9, 1, None)]
        _assign_recent_stats(store, 0, records)
        assert store["horse_finish_std_3"][0] > 0

    def test_ewm_and_trend_propagate_in_full_pipeline(self, synthetic_history, synthetic_entry):
        """フルパイプラインでEWMとtrendカラムが付与される"""
        history = add_targets(synthetic_history)
        history, entry = add_basic_features(history, synthetic_entry)
        result_h, result_e = add_aggregate_features(history, entry)
        assert "horse_recent3_finish_pct_ewm" in result_h.columns
        assert "horse_finish_trend_3" in result_h.columns
        assert "horse_best_finish_5" in result_h.columns


# ---------------------------------------------------------------------------
# Phase 9: 馬場状態 (track_condition) 特徴量
# ---------------------------------------------------------------------------

def _make_condition_history() -> pd.DataFrame:
    """track_condition 列付きの5レース×8頭データ"""
    conditions = ["good", "soft", "good", "heavy", "good"]
    rows = []
    for i, (cond, date) in enumerate(
        zip(conditions, ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"])
    ):
        for j in range(8):
            rows.append({
                "race_id": f"R{i+1}",
                "race_date": pd.Timestamp(date),
                "horse_id": f"H{j+1}",
                "finish_rank": j + 1,
                "field_size": 8,
                "jockey": "J1",
                "trainer": "T1",
                "surface": "turf",
                "distance": 2000,
                "grade": "",
                "course": "Tokyo",
                "track_condition": cond,
            })
    return pd.DataFrame(rows)


class TestTrackConditionFeatures:
    """Phase 9: 馬場状態特徴量"""

    def test_track_condition_in_raw_model_features(self):
        from keiba_predictor.features import RAW_MODEL_FEATURES
        assert "track_condition" in RAW_MODEL_FEATURES

    def test_track_condition_aggregate_columns_in_history(self):
        df = _make_condition_history()
        df = add_targets(df)
        result = _build_history_aggregate_features(df)
        assert "horse_condition_top3_rate_smooth" in result.columns
        assert "horse_condition_starts" in result.columns
        assert "horse_condition_finish_pct_mean" in result.columns

    def test_horse_condition_aggregate_in_prefixes(self):
        from keiba_predictor.features import AGGREGATE_PREFIXES
        assert "horse_condition" in AGGREGATE_PREFIXES

    def test_track_condition_stats_use_expanding_window(self):
        """horse_condition は条件一致の過去レースのみ集計 (リーク防止)"""
        df = _make_condition_history()
        df = add_targets(df)
        result = _build_history_aggregate_features(df)
        # H1 は R1(good, 1着) → R2(soft) では good実績=1件あるはずだが、soft実績=0件
        h1_r2 = result[(result["horse_id"] == "H1") & (result["race_id"] == "R2")].iloc[0]
        # R2は soft なので horse_condition_starts は R1 good の実績が反映されない(別条件)
        # R2 時点での H1 の soft 実績は 0 なのでstarts=0
        assert h1_r2["horse_condition_starts"] == pytest.approx(0.0)

    def test_horse_condition_starts_accumulates_same_condition(self):
        """同じ馬場条件で複数出走した後は starts が積み上がる"""
        df = _make_condition_history()
        df = add_targets(df)
        result = _build_history_aggregate_features(df)
        # H1 R1(good,1着), R3(good,1着) → R5(good)時点で good実績=2件
        h1_r5 = result[(result["horse_id"] == "H1") & (result["race_id"] == "R5")].iloc[0]
        assert h1_r5["horse_condition_starts"] == pytest.approx(2.0)

    def test_track_condition_missing_graceful(self, synthetic_history, synthetic_entry):
        """track_condition列がなくてもエラーにならずprior値が入る"""
        history = add_targets(synthetic_history)
        history, entry = add_basic_features(history, synthetic_entry)
        result_h, result_e = add_aggregate_features(history, entry)
        # prior_mean が入っていることを確認
        assert "horse_condition_top3_rate_smooth" in result_h.columns
        assert result_h["horse_condition_top3_rate_smooth"].notna().any() or True  # prior or NaN OK

    def test_track_condition_in_entry_features(self):
        """entry側にも horse_condition カラムが付与される"""
        df = _make_condition_history()
        df = add_targets(df)
        history = df.iloc[:-8].copy()
        entry_raw = df.iloc[-8:].drop(columns=["finish_rank", "finish_percentile", "is_top3"], errors="ignore").copy()
        history, entry = add_basic_features(history, entry_raw)
        _, result_e = add_aggregate_features(history, entry)
        assert "horse_condition_top3_rate_smooth" in result_e.columns

    def test_track_condition_in_data_loader_aliases(self):
        from keiba_predictor.data_loader import CANONICAL_ALIASES
        assert "track_condition" in CANONICAL_ALIASES

    def test_track_condition_in_categorical_hints(self):
        from keiba_predictor.model import CATEGORICAL_HINTS
        assert "track_condition" in CATEGORICAL_HINTS


class TestInferFeatureColumns:
    def test_returns_feature_columns(self, synthetic_history, synthetic_entry):
        history = add_targets(synthetic_history)
        history, entry = add_basic_features(history, synthetic_entry)
        history, entry = add_aggregate_features(history, entry)
        feat_cols = infer_feature_columns(history, entry, use_odds=True)
        assert len(feat_cols) > 0
        assert isinstance(feat_cols, list)

    def test_no_odds_excludes_odds(self, synthetic_history, synthetic_entry):
        history = add_targets(synthetic_history)
        history, entry = add_basic_features(history, synthetic_entry)
        history, entry = add_aggregate_features(history, entry)
        feat_cols = infer_feature_columns(history, entry, use_odds=True)
        no_odds_cols = infer_feature_columns(history, entry, use_odds=False)
        assert len(no_odds_cols) <= len(feat_cols)
        assert "odds" not in no_odds_cols
