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
from src.features.build import TARGET, build_demand_features, feature_columns


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
