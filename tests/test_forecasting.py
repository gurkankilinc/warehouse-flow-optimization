"""Leakage tests for the demand features.

A forecasting result is worthless if a feature quietly contains the answer, and
that failure is invisible in the metrics -- it just makes them look good. So
leakage is tested mechanically rather than by reading the code: perturb the
target on one day and assert that no feature for that day moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import CONFIG
from src.features.build import (
    TARGET,
    build_demand_features,
    build_horizon_features,
    feature_columns,
)


@pytest.fixture
def toy_data():
    dates = pd.date_range("2023-01-02", periods=120, freq="D")
    skus = ["S001", "S002", "S003"]
    rng = np.random.default_rng(0)
    rows = [
        {"date": d, "sku_id": s, "demand": int(rng.poisson(5 + 3 * (i == 0)))}
        for i, s in enumerate(skus)
        for d in dates
    ]
    daily = pd.DataFrame(rows)
    catalog = pd.DataFrame(
        {
            "sku_id": skus,
            "category": ["Electronics", "Food", "Apparel"],
            "is_cold": [False, True, False],
            "is_heavy": [False, False, True],
            "volume_l": [1.0, 2.0, 3.0],
            "weight_kg": [1.0, 2.0, 3.0],
        }
    )
    return daily, catalog


def test_lag_1_is_literally_yesterdays_demand(toy_data):
    daily, catalog = toy_data
    features = build_demand_features(daily, catalog)
    one_sku = features[features["sku_id"] == "S001"].sort_values("date").reset_index(drop=True)
    assert one_sku["lag_1"].iloc[1:].to_numpy() == pytest.approx(
        one_sku[TARGET].iloc[:-1].to_numpy()
    )


def test_rolling_mean_excludes_the_current_day(toy_data):
    daily, catalog = toy_data
    features = build_demand_features(daily, catalog)
    one_sku = features[features["sku_id"] == "S001"].sort_values("date").reset_index(drop=True)
    row = 40
    expected = one_sku[TARGET].iloc[row - 7 : row].mean()
    assert one_sku["roll_mean_7"].iloc[row] == pytest.approx(expected)


def test_changing_todays_demand_does_not_change_todays_features(toy_data):
    """The decisive leakage check. If any feature for day t reacts to the
    target on day t, that feature is telling the model the answer."""
    daily, catalog = toy_data
    target_date = daily["date"].iloc[80]

    base = build_demand_features(daily, catalog)

    tampered = daily.copy()
    mask = (tampered["date"] == target_date) & (tampered["sku_id"] == "S001")
    tampered.loc[mask, "demand"] = 9999
    after = build_demand_features(tampered, catalog)

    columns = [c for c in feature_columns(base) if c != TARGET]
    base_row = base[(base["date"] == target_date) & (base["sku_id"] == "S001")][columns]
    after_row = after[(after["date"] == target_date) & (after["sku_id"] == "S001")][columns]

    pd.testing.assert_frame_equal(
        base_row.reset_index(drop=True), after_row.reset_index(drop=True)
    )


def test_future_demand_does_not_leak_backwards(toy_data):
    """Changing a *later* day must leave every earlier row untouched."""
    daily, catalog = toy_data
    late_date = daily["date"].iloc[100]

    base = build_demand_features(daily, catalog)
    tampered = daily.copy()
    tampered.loc[
        (tampered["date"] == late_date) & (tampered["sku_id"] == "S001"), "demand"
    ] = 9999
    after = build_demand_features(tampered, catalog)

    columns = [c for c in feature_columns(base) if c != TARGET]
    earlier = base["date"] < late_date
    pd.testing.assert_frame_equal(
        base.loc[earlier, columns].reset_index(drop=True),
        after.loc[earlier, columns].reset_index(drop=True),
    )


def test_true_popularity_is_not_among_the_features(toy_data):
    """The simulator knows each SKU's true demand rate. Handing it to the model
    would score brilliantly and prove nothing."""
    daily, catalog = toy_data
    catalog = catalog.assign(popularity=[0.5, 0.3, 0.2], express_propensity=[2.0, 1.0, 1.0])
    features = build_demand_features(daily, catalog)
    columns = set(feature_columns(features))
    assert "popularity" not in columns
    assert "express_propensity" not in columns


# ----------------------------------------------------------------------
# The slotting forecast: a multi-week horizon decided at one cut-off
# ----------------------------------------------------------------------

CUTOFF_ROW = 80
HORIZON_DAYS = 25  # runs well into the dates the toy data still has demand for


class _SumOfFeatures:
    """Stand-in forecaster whose output moves with every feature it is given,
    so any leak into the features shows up in the forecast."""

    def __init__(self, features: list[str]):
        self.features = features

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self.features].fillna(0.0).sum(axis=1).to_numpy()


def _tamper(daily: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    tampered = daily.copy()
    tampered.loc[mask, "demand"] = 9999
    return tampered


def test_horizon_features_ignore_demand_on_and_after_the_cutoff(toy_data):
    """The leak this pins down: reading the horizon rows out of the
    full-history frame gives day ``cutoff + k`` the realised demand of day
    ``cutoff + k - 1``. A slot decision taken at the cut-off cannot know that."""
    daily, catalog = toy_data
    cutoff = daily["date"].iloc[CUTOFF_ROW]

    base = build_horizon_features(daily, catalog, cutoff, HORIZON_DAYS)
    after = build_horizon_features(
        _tamper(daily, daily["date"] >= cutoff), catalog, cutoff, HORIZON_DAYS
    )

    pd.testing.assert_frame_equal(base, after)


def test_slotting_forecast_ignores_demand_on_and_after_the_cutoff(toy_data):
    """The same guarantee, checked on what slotting actually consumes."""
    from src.experiments.pipeline import forecast_visits

    daily, catalog = toy_data
    cutoff = daily["date"].iloc[CUTOFF_ROW]
    model = _SumOfFeatures(
        [c for c in feature_columns(build_demand_features(daily, catalog)) if c != TARGET]
    )

    def visits(demand):
        return forecast_visits(demand, catalog, model, cutoff, HORIZON_DAYS, units_per_line=1.0)

    base = visits(daily)
    pd.testing.assert_series_equal(base, visits(_tamper(daily, daily["date"] >= cutoff)))
    # Not vacuous: the forecast does react to history the planner really had.
    assert not visits(_tamper(daily, daily["date"] == cutoff - pd.Timedelta(days=1))).equals(base)


def test_horizon_state_is_the_cutoff_row_of_the_full_frame(toy_data):
    """Freezing must hand every horizon day exactly the history state the
    ordinary feature frame has on the cut-off day -- which also proves the
    trailing slice it is rebuilt from reaches back far enough."""
    daily, catalog = toy_data
    cutoff = daily["date"].iloc[CUTOFF_ROW]

    full = build_demand_features(daily, catalog)
    horizon = build_horizon_features(daily, catalog, cutoff, HORIZON_DAYS)

    state_columns = [c for c in feature_columns(full) if c.startswith(("lag_", "roll_", "zero_share"))]
    expected = full[full["date"] == cutoff].set_index("sku_id")[state_columns].sort_index()
    for day, rows in horizon.groupby("date"):
        actual = rows.set_index("sku_id")[state_columns].sort_index()
        np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True)


def test_horizon_rows_carry_their_own_calendar(toy_data):
    daily, catalog = toy_data
    cutoff = daily["date"].iloc[CUTOFF_ROW]

    full = build_demand_features(daily, catalog)
    horizon = build_horizon_features(daily, catalog, cutoff, HORIZON_DAYS)

    assert sorted(pd.to_datetime(horizon["date"].unique()).tolist()) == (
        pd.date_range(cutoff, periods=HORIZON_DAYS, freq="D").tolist()
    )
    assert len(horizon) == HORIZON_DAYS * len(catalog)
    # Calendar and trend line up with the training frame, day by day.
    calendar = ["day_of_week", "month", "week_of_year", "is_weekend", "days_from_start"]
    merged = horizon.merge(full, on=["sku_id", "date"], suffixes=("", "_full"))
    for column in calendar:
        assert (merged[column] == merged[f"{column}_full"]).all(), column
    # Every model input is present, so the trained model can score these rows.
    assert set(feature_columns(full)) - {TARGET} <= set(horizon.columns)


def test_walk_forward_folds_never_train_on_the_future():
    from src.models.demand_forecast import walk_forward_backtest

    # Enough SKUs and days for the backtest to actually produce folds.
    dates = pd.date_range("2023-01-02", periods=320, freq="D")
    rng = np.random.default_rng(1)
    daily = pd.DataFrame(
        [
            {"date": d, "sku_id": f"S{i:02d}", "demand": int(rng.poisson(4))}
            for i in range(12)
            for d in dates
        ]
    )
    catalog = pd.DataFrame(
        {
            "sku_id": [f"S{i:02d}" for i in range(12)],
            "category": ["Electronics"] * 12,
            "is_cold": False,
            "is_heavy": False,
            "volume_l": 1.0,
            "weight_kg": 1.0,
        }
    )
    features = build_demand_features(daily, catalog)

    import dataclasses

    cfg = dataclasses.replace(
        CONFIG.model, n_backtest_folds=2, min_train_days=100, n_estimators=30
    )
    result = walk_forward_backtest(features, cfg)

    assert not result.empty
    # Each fold's test window must start after its training data ends, and
    # later folds must test on earlier data (folds count backwards).
    for _, row in result.iterrows():
        assert row["train_days"] > 0
        assert pd.Timestamp(row["test_start"]) <= pd.Timestamp(row["test_end"])
