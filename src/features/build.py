"""Feature engineering for the demand model.

Two rules govern everything here.

**No look-ahead.** Every lag and rolling statistic is shifted by at least one
day before it is attached to a row, so a feature for day *t* only ever contains
information that existed at the end of day *t-1*. ``assert_no_leakage`` in the
test suite checks this mechanically rather than trusting the code to be right.

**No generator internals.** The simulator knows each SKU's true popularity
parameter, and feeding it to the model would be a spectacular way to score
well while learning nothing. Only quantities a real planner could observe --
past sales, the calendar, and static product attributes -- are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CONFIG, ModelConfig

__all__ = ["build_demand_features", "feature_columns", "TARGET"]

TARGET = "demand"

_CALENDAR_FEATURES = [
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
    "days_from_start",
]

_STATIC_FEATURES = ["category_code", "is_cold", "is_heavy", "volume_l", "weight_kg"]


def build_demand_features(
    daily_demand: pd.DataFrame,
    catalog: pd.DataFrame,
    model_cfg: ModelConfig | None = None,
) -> pd.DataFrame:
    """Return a modelling frame: one row per (sku_id, date).

    The frame keeps ``date`` and ``sku_id`` so that splits can be made on time
    and predictions can be joined back to products.
    """
    model_cfg = model_cfg or CONFIG.model

    df = daily_demand.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)

    grouped = df.groupby("sku_id", sort=False)["demand"]

    for lag in model_cfg.demand_lags:
        df[f"lag_{lag}"] = grouped.shift(lag)

    # Rolling windows are computed on the already-shifted series, so window
    # `w` at day t covers days t-w .. t-1 and never touches day t itself.
    shifted = grouped.shift(1)
    by_sku = shifted.groupby(df["sku_id"], sort=False)
    for window in model_cfg.demand_rolling_windows:
        roll = by_sku.rolling(window, min_periods=1)
        df[f"roll_mean_{window}"] = roll.mean().reset_index(level=0, drop=True)
        df[f"roll_std_{window}"] = roll.std().reset_index(level=0, drop=True)
        df[f"roll_max_{window}"] = roll.max().reset_index(level=0, drop=True)

    # Share of recent days with no demand: the single most informative feature
    # for intermittent items, and the thing a plain average hides.
    df["zero_share_28"] = (
        shifted.eq(0).groupby(df["sku_id"], sort=False).rolling(28, min_periods=1)
        .mean().reset_index(level=0, drop=True)
    )

    # Same weekday last week, and the week-on-week ratio.
    df["lag_7_ratio"] = df["lag_7"] / df["roll_mean_28"].replace(0, np.nan)

    # --- calendar ------------------------------------------------------
    dates = df["date"].dt
    df["day_of_week"] = dates.dayofweek
    df["day_of_month"] = dates.day
    df["month"] = dates.month
    df["week_of_year"] = dates.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["days_from_start"] = (df["date"] - df["date"].min()).dt.days

    # --- static product attributes -------------------------------------
    static = catalog[["sku_id", "category", "is_cold", "is_heavy", "volume_l", "weight_kg"]].copy()
    static["category_code"] = static["category"].astype("category").cat.codes
    static = static.drop(columns=["category"])
    df = df.merge(static, on="sku_id", how="left")
    df["is_cold"] = df["is_cold"].astype(int)
    df["is_heavy"] = df["is_heavy"].astype(int)

    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs: everything except identifiers and the target."""
    exclude = {"date", "sku_id", TARGET}
    return [c for c in df.columns if c not in exclude]
