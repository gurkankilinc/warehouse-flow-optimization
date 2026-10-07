"""ML-1: SKU-level demand forecasting.

One **global** model is trained across all SKUs rather than one model per SKU.
With 2,400 products, most of which sell intermittently, per-SKU models would be
fitting noise; a global model borrows strength across products and lets the
static attributes (category, size) carry information for the thin ones. This is
also what production forecasting systems do.

The model is judged against three naive baselines, not against nothing:

    naive_last        yesterday's demand
    seasonal_naive    the same weekday last week
    moving_average    the trailing 28-day mean

The headline metric is WMAPE rather than MAPE. Daily SKU demand is full of
zeros, and MAPE is undefined the moment the actual is zero -- reporting it
would be arithmetically meaningless on exactly the items that are hardest.

Validation is walk-forward: train on everything up to a cut-off, score the
next `forecast_horizon_days`, roll the cut-off forward, repeat. A random
train/test split would leak the future into the past and flatter the model.

What the backtest measures is **one-day-ahead** accuracy, averaged over a
block of `forecast_horizon_days` days -- not a 7-day-ahead forecast. Test rows
come from the ordinary feature frame, so the row for the fifth test day has a
``lag_1`` equal to the realised demand of the fourth. That is a fair contest
(the naive baselines read the same lags), but it is not the multi-week,
frozen-at-cut-off forecast that slotting consumes (``build_horizon_features``),
which is necessarily less accurate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import CONFIG, ModelConfig
from src.features.build import TARGET, build_demand_features, feature_columns

__all__ = ["DemandForecaster", "walk_forward_backtest", "evaluate_naive_baselines", "wmape"]


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted MAPE: total absolute error divided by total actual volume."""
    denominator = np.abs(y_true).sum()
    if denominator == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denominator)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed error: positive means the model over-forecasts."""
    return float((y_pred - y_true).mean())


@dataclass
class DemandForecaster:
    """Thin wrapper so the rest of the project does not import LightGBM."""

    model: object = None
    features: list[str] | None = None
    model_cfg: ModelConfig | None = None

    def fit(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame | None = None,
        model_cfg: ModelConfig | None = None,
    ) -> "DemandForecaster":
        import lightgbm as lgb

        self.model_cfg = model_cfg or self.model_cfg or CONFIG.model
        self.features = feature_columns(train)

        params = dict(
            objective="tweedie",  # non-negative, zero-inflated counts
            tweedie_variance_power=1.2,
            n_estimators=self.model_cfg.n_estimators,
            learning_rate=self.model_cfg.learning_rate,
            max_depth=self.model_cfg.max_depth,
            min_child_samples=self.model_cfg.min_samples_leaf,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            verbose=-1,
            n_jobs=-1,
        )
        self.model = lgb.LGBMRegressor(**params)

        fit_kwargs = {}
        if valid is not None and not valid.empty:
            fit_kwargs["eval_X"] = valid[self.features]
            fit_kwargs["eval_y"] = valid[TARGET]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self.model_cfg.early_stopping_rounds, verbose=False)
            ]

        self.model.fit(train[self.features], train[TARGET], **fit_kwargs)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(frame[self.features])
        return np.clip(preds, 0.0, None)

    def feature_importance(self) -> pd.DataFrame:
        return (
            pd.DataFrame(
                {"feature": self.features, "gain": self.model.booster_.feature_importance("gain")}
            )
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------


def evaluate_naive_baselines(test: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Score the rules a warehouse would use without any model at all."""
    y = test[TARGET].to_numpy(dtype=float)
    candidates = {
        "naive_last": test["lag_1"],
        "seasonal_naive": test["lag_7"],
        "moving_average_28": test["roll_mean_28"],
    }
    results = {}
    for name, series in candidates.items():
        pred = series.fillna(0.0).to_numpy(dtype=float)
        results[name] = {"wmape": wmape(y, pred), "mae": mae(y, pred), "bias": bias(y, pred)}
    return results


# ----------------------------------------------------------------------
# Walk-forward validation
# ----------------------------------------------------------------------


def walk_forward_backtest(
    features: pd.DataFrame,
    model_cfg: ModelConfig | None = None,
    *,
    cutoff_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Roll a train/test boundary forward through time.

    Each fold scores a block of ``forecast_horizon_days`` consecutive days, but
    every test row carries lags of realised demand up to the day before it, so
    the scores are one-day-ahead accuracy (see the module docstring).

    ``cutoff_date`` bounds the whole exercise: no fold is allowed to see data
    on or after it. It is set to the first day of the simulation window, so the
    forecast that drives slot assignment cannot have peeked at the period it is
    evaluated on.
    """
    model_cfg = model_cfg or CONFIG.model

    df = features.dropna(subset=["lag_28"]).copy()
    if cutoff_date is not None:
        df = df[df["date"] < cutoff_date]

    dates = np.sort(df["date"].unique())
    horizon = model_cfg.forecast_horizon_days
    rows = []

    for fold in range(model_cfg.n_backtest_folds):
        # Folds are laid out backwards from the cut-off, so the last fold ends
        # right where the simulation window begins.
        test_end = len(dates) - fold * horizon
        test_start = test_end - horizon
        if test_start <= model_cfg.min_train_days:
            break

        train_dates = dates[:test_start]
        test_dates = dates[test_start:test_end]

        train = df[df["date"].isin(train_dates)]
        test = df[df["date"].isin(test_dates)]

        # Hold out the tail of training as a validation set for early stopping.
        valid_cut = train_dates[-horizon]
        fit_part = train[train["date"] < valid_cut]
        valid_part = train[train["date"] >= valid_cut]

        forecaster = DemandForecaster().fit(fit_part, valid_part, model_cfg)
        pred = forecaster.predict(test)
        y = test[TARGET].to_numpy(dtype=float)

        row = {
            "fold": fold,
            "train_days": len(train_dates),
            "test_start": pd.Timestamp(test_dates[0]).date(),
            "test_end": pd.Timestamp(test_dates[-1]).date(),
            "n_test_rows": len(test),
            "model_wmape": wmape(y, pred),
            "model_mae": mae(y, pred),
            "model_bias": bias(y, pred),
        }
        for name, scores in evaluate_naive_baselines(test).items():
            row[f"{name}_wmape"] = scores["wmape"]
            row[f"{name}_mae"] = scores["mae"]
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    if not result.empty:
        best_naive = result[
            ["naive_last_wmape", "seasonal_naive_wmape", "moving_average_28_wmape"]
        ].min(axis=1)
        result["improvement_vs_best_naive"] = 1.0 - result["model_wmape"] / best_naive
    return result


def train_final_model(
    features: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    model_cfg: ModelConfig | None = None,
) -> DemandForecaster:
    """Fit the model that the optimiser will actually use.

    Trained strictly on data before ``cutoff_date`` -- the first day of the
    simulation window -- so the slotting decisions it feeds are made with
    information that genuinely existed at the time.
    """
    model_cfg = model_cfg or CONFIG.model
    df = features.dropna(subset=["lag_28"])
    df = df[df["date"] < cutoff_date]

    dates = np.sort(df["date"].unique())
    valid_cut = dates[-model_cfg.forecast_horizon_days]
    return DemandForecaster().fit(
        df[df["date"] < valid_cut], df[df["date"] >= valid_cut], model_cfg
    )
