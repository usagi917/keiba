from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba_predictor.data_loader import (
    DataValidationError,
    auto_map_columns,
    coerce_dtypes,
    ensure_entity_identifier,
    infer_field_size,
    normalize_colname,
    prepare_datasets,
    validate_schema,
)


class TestNormalizeColname:
    def test_basic_english(self):
        assert normalize_colname("race_id") == "raceid"

    def test_strips_spaces_and_dashes(self):
        assert normalize_colname("horse - name") == "horsename"

    def test_japanese(self):
        assert normalize_colname("馬名") == "馬名"

    def test_case_insensitive(self):
        assert normalize_colname("RaceID") == "raceid"


class TestAutoMapColumns:
    def test_maps_known_aliases(self):
        df = pd.DataFrame({"馬名": ["A"], "着順": [1]})
        renamed, mapping, _ = auto_map_columns(df)
        assert "horse_name" in renamed.columns
        assert "finish_rank" in renamed.columns

    def test_leaves_canonical_names_unchanged(self):
        df = pd.DataFrame({"race_id": ["R001"], "horse_id": ["H001"]})
        renamed, mapping, _ = auto_map_columns(df)
        assert "race_id" in renamed.columns
        assert "horse_id" in renamed.columns
        assert len(mapping) == 0

    def test_suggests_close_matches(self):
        df = pd.DataFrame({"race_idd": ["R001"]})
        _, _, suggestions = auto_map_columns(df)
        assert "race_id" in suggestions or len(suggestions) == 0

    def test_maps_phase3_extended_aliases(self):
        df = pd.DataFrame(
            {
                "脚質": ["先行"],
                "通過順": ["3-3-2-1"],
                "上がり順位": [2],
                "馬場バイアス": ["inside"],
            }
        )
        renamed, _, _ = auto_map_columns(df)
        assert "running_style" in renamed.columns
        assert "passing_position" in renamed.columns
        assert "last_3f_rank" in renamed.columns
        assert "track_bias" in renamed.columns


class TestCoerceDtypes:
    def test_converts_race_date(self):
        df = pd.DataFrame({"race_date": ["2026-03-29"]})
        result = coerce_dtypes(df)
        assert pd.api.types.is_datetime64_any_dtype(result["race_date"])

    def test_converts_numeric_hints(self):
        df = pd.DataFrame({"distance": ["1200"], "odds": ["3.5"]})
        result = coerce_dtypes(df)
        assert np.issubdtype(result["distance"].dtype, np.number)
        assert np.issubdtype(result["odds"].dtype, np.number)

    def test_handles_bad_values(self):
        df = pd.DataFrame({"distance": ["abc"]})
        result = coerce_dtypes(df)
        assert pd.isna(result["distance"].iloc[0])


class TestEnsureEntityIdentifier:
    def test_uses_horse_id_when_available(self):
        df = pd.DataFrame({"horse_id": ["H001"], "horse_name": ["Foo"]})
        result, source = ensure_entity_identifier(df)
        assert source == "horse_id"

    def test_falls_back_to_horse_name(self):
        df = pd.DataFrame({"horse_name": ["Foo"]})
        result, source = ensure_entity_identifier(df)
        assert source == "horse_name_fallback"
        assert "horse_id" in result.columns

    def test_raises_without_identifiers(self):
        df = pd.DataFrame({"col": [1]})
        with pytest.raises(DataValidationError):
            ensure_entity_identifier(df)


class TestInferFieldSize:
    def test_uses_existing_field_size(self):
        df = pd.DataFrame({"field_size": [10, 10], "race_id": ["R1", "R1"]})
        result = infer_field_size(df, is_entry=False)
        assert (result["field_size"] == 10).all()

    def test_infers_from_race_id_grouping(self):
        df = pd.DataFrame({"race_id": ["R1"] * 5, "field_size": [np.nan] * 5})
        result = infer_field_size(df, is_entry=False)
        assert (result["field_size"] == 5).all()

    def test_entry_uses_len(self):
        df = pd.DataFrame({"col": range(8)})
        result = infer_field_size(df, is_entry=True)
        assert (result["field_size"] == 8).all()


class TestValidateSchema:
    def test_accepts_valid_data(self):
        history = pd.DataFrame({
            "race_id": ["R1"], "race_date": ["2026-01-01"],
            "finish_rank": [1], "horse_id": ["H1"], "draw": [1],
            "surface": ["turf"], "distance": [1200], "course": ["Tokyo"],
        })
        entry = pd.DataFrame({
            "horse_id": ["H1"], "draw": [1],
            "surface": ["turf"], "distance": [1200], "course": ["Tokyo"],
        })
        result = validate_schema(history, entry, use_odds=True)
        assert "common_feature_candidates" in result

    def test_raises_on_missing_required(self):
        history = pd.DataFrame({"race_id": ["R1"]})
        entry = pd.DataFrame({"horse_id": ["H1"]})
        with pytest.raises(DataValidationError):
            validate_schema(history, entry, use_odds=True)

    def test_excludes_odds_when_disabled(self):
        history = pd.DataFrame({
            "race_id": ["R1"], "race_date": ["2026-01-01"],
            "finish_rank": [1], "horse_id": ["H1"],
            "odds": [3.0], "popularity": [1], "draw": [1],
        })
        entry = pd.DataFrame({
            "horse_id": ["H1"], "odds": [3.0],
            "popularity": [1], "draw": [1],
        })
        result = validate_schema(history, entry, use_odds=False)
        assert "odds" not in result["common_feature_candidates"]
        assert "popularity" not in result["common_feature_candidates"]


class TestPrepareDatasets:
    def test_round_trip(self, synthetic_history, synthetic_entry):
        history, entry, info = prepare_datasets(
            synthetic_history, synthetic_entry, use_odds=True,
        )
        assert "race_id" in history.columns
        assert "horse_id" in history.columns
        assert "field_size" in history.columns
        assert len(info["common_feature_candidates"]) > 0
