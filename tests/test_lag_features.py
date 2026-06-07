from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba_predictor.features import build_lag_features


def _history() -> pd.DataFrame:
    # horse A: 3走 / horse B: 2走 (時系列順がシャッフルされていても紐づくこと)
    rows = [
        {"race_id": "R3", "race_date": "2020-03-01", "horse_id": "A", "finish_rank": 2, "distance": 2000, "class": "Open"},
        {"race_id": "R1", "race_date": "2020-01-01", "horse_id": "A", "finish_rank": 5, "distance": 1600, "class": "3Win"},
        {"race_id": "R2", "race_date": "2020-02-01", "horse_id": "A", "finish_rank": 3, "distance": 1800, "class": "Open"},
        {"race_id": "R1", "race_date": "2020-01-01", "horse_id": "B", "finish_rank": 1, "distance": 1600, "class": "3Win"},
        {"race_id": "R2", "race_date": "2020-02-01", "horse_id": "B", "finish_rank": 4, "distance": 1800, "class": "Open"},
    ]
    return pd.DataFrame(rows)


class TestBuildLagFeatures:
    def test_links_each_runner_to_its_own_prior_race(self):
        out = build_lag_features(_history())
        a_r2 = out[(out["horse_id"] == "A") & (out["race_id"] == "R2")].iloc[0]
        assert a_r2["last_finish"] == 5
        assert a_r2["last_distance"] == 1600
        assert a_r2["days_since_last"] == 31  # 01-01 -> 02-01
        a_r3 = out[(out["horse_id"] == "A") & (out["race_id"] == "R3")].iloc[0]
        assert a_r3["last_finish"] == 3
        assert a_r3["last_distance"] == 1800
        assert a_r3["days_since_last"] == 29  # 02-01 -> 03-01 (うるう年)

    def test_first_appearance_has_no_prior(self):
        out = build_lag_features(_history())
        a_r1 = out[(out["horse_id"] == "A") & (out["race_id"] == "R1")].iloc[0]
        assert pd.isna(a_r1["last_finish"])
        assert pd.isna(a_r1["days_since_last"])

    def test_preserves_row_count_and_does_not_reorder_caller_index(self):
        df = _history()
        out = build_lag_features(df)
        assert len(out) == len(df)
        # 元の race_id/horse_id の組が保持されている
        assert set(zip(out["horse_id"], out["race_id"])) == set(zip(df["horse_id"], df["race_id"]))

    def test_does_not_overwrite_existing_values_by_default(self):
        df = _history()
        df["last_finish"] = 99  # 既存値
        out = build_lag_features(df, overwrite=False)
        # 既存の非NaN値は維持される
        assert (out["last_finish"] == 99).all()
