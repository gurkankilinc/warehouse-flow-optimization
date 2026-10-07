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

__all__ = ["build_demand_features", "build_horizon_features", "feature_columns", "TARGET"]

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

_ZERO_SHARE_WINDOW = 28

# Everything in a feature row that is *not* derived from past demand.
_NON_STATE_COLUMNS = {"date", "sku_id", TARGET, *_CALENDAR_FEATURES, *_STATIC_FEATURES}


def build_demand_features(
    daily_demand: pd.DataFrame,
    catalog: pd.DataFrame,
    model_cfg: ModelConfig | None = None,
    *,
    origin: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return a modelling frame: one row per (sku_id, date).

    The frame keeps ``date`` and ``sku_id`` so that splits can be made on time
    and predictions can be joined back to products.

    ``origin`` anchors ``days_from_start``. It defaults to the earliest date in
    the frame, which is right for the full history; a caller that passes only a
    trailing slice must pass the history's real start, or the trend feature
    would restart at zero.
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
        shifted.eq(0).groupby(df["sku_id"], sort=False).rolling(_ZERO_SHARE_WINDOW, min_periods=1)
        .mean().reset_index(level=0, drop=True)
    )

    # Same weekday last week, and the week-on-week ratio.
    df["lag_7_ratio"] = df["lag_7"] / df["roll_mean_28"].replace(0, np.nan)

    _add_calendar_features(df, df["date"].min() if origin is None else pd.Timestamp(origin))
    return _add_static_features(df, catalog)


def build_horizon_features(
    daily_demand: pd.DataFrame,
    catalog: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    horizon_days: int,
    model_cfg: ModelConfig | None = None,
) -> pd.DataFrame:
    """Feature rows for ``[cutoff_date, cutoff_date + horizon_days)``, built
    from nothing on or after ``cutoff_date``.

    Each horizon row is the feature row of day *t* in ``build_demand_features``
    shifted to the future, except that the demand-history features
    (lags, rolling statistics, zero share) are **frozen at the cut-off**: every
    horizon day carries the history state as it stood at the end of the day
    before the cut-off. The calendar and static product features are those of
    the actual horizon day.

    That is the information a planner really has when the slots are assigned.
    Taking the horizon rows from a frame built over the full history would
    instead give day ``cutoff + 50`` a ``lag_1`` equal to the *realised* demand
    of day ``cutoff + 49`` -- a one-day-ahead forecast dressed up as a
    multi-week one, using demand the simulation has not produced yet.

    The demand input is truncated before any feature is computed, so the
    guarantee holds by construction rather than by the shift logic alone.
    """
    model_cfg = model_cfg or CONFIG.model
    cutoff_date = pd.Timestamp(cutoff_date)

    demand = daily_demand.copy()
    demand["date"] = pd.to_datetime(demand["date"])
    origin = demand["date"].min()
    history = demand[demand["date"] < cutoff_date]
    if history.empty:
        raise ValueError(f"no demand history before {cutoff_date.date()}")

    # Only the trailing window the longest feature reaches back over is needed;
    # rebuilding over the whole history would give the same state, slower.
    lookback = max(
        max(model_cfg.demand_lags), max(model_cfg.demand_rolling_windows), _ZERO_SHARE_WINDOW
    )
    history = history[history["date"] >= cutoff_date - pd.Timedelta(days=lookback)]

    # A placeholder row on the cut-off day itself, with unknown demand. Every
    # history feature is shifted by at least one day, so its row holds exactly
    # the state at the end of the previous day.
    skus = catalog["sku_id"].to_numpy()
    placeholder = pd.DataFrame({"date": cutoff_date, "sku_id": skus, TARGET: np.nan})
    frame = build_demand_features(
        pd.concat([history, placeholder], ignore_index=True), catalog, model_cfg, origin=origin
    )
    state = frame[frame["date"] == cutoff_date]
    state = state[["sku_id", *[c for c in state.columns if c not in _NON_STATE_COLUMNS]]]

    horizon = pd.MultiIndex.from_product(
        [skus, pd.date_range(cutoff_date, periods=horizon_days, freq="D")],
        names=["sku_id", "date"],
    ).to_frame(index=False)
    horizon = horizon.merge(state, on="sku_id", how="left")
    _add_calendar_features(horizon, origin)
    return _add_static_features(horizon, catalog)


def _add_calendar_features(df: pd.DataFrame, origin: pd.Timestamp) -> None:
    dates = df["date"].dt
    df["day_of_week"] = dates.dayofweek
    df["day_of_month"] = dates.day
    df["month"] = dates.month
    df["week_of_year"] = dates.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["days_from_start"] = (df["date"] - origin).dt.days


def _add_static_features(df: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
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
