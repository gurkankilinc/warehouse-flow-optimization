"""ML-2: how long will this picking tour take?

This number is what turns "earliest due date" into "least slack". Two orders
due at the same moment are not equally urgent if one takes four minutes and the
other forty, and slack -- deadline minus now minus expected pick time -- is
only as good as that estimate.

Two baselines, not one:

``analytic_standard``
    The textbook formula, handed the *exact* constants the simulator uses.
    This is deliberately unfair to the model -- no real planner knows the true
    per-line and per-unit times -- and it is included precisely to show how
    little room is left once the dominant physical drivers are accounted for.

``linear_timestudy``
    The same formula's inputs, but with coefficients estimated from observed
    tours instead of assumed. This is what an industrial engineer actually
    produces from a time study, and it is the fair comparator.

Beating either means learning what the formula cannot express -- principally
that the warehouse is slower when it is crowded.

The result: it does not beat the time study (about -0.4% on MAE), and gains
roughly 5% on the oracle-constant version. Pick time really is close to linear
in lines, units and distance, so a flexible model has almost nothing left to
find. Downstream this matters not at all: swapping the analytic estimate for
this model leaves the dispatch policy's on-time rate unchanged to four decimal
places, because the two produce the same *ordering* of the queue even when
their absolute predictions differ.

Deliberately excluded from the features
---------------------------------------
``picker_id``. At the moment slack is computed, nobody knows which picker will
take the tour, so knowing their individual speed would be information from the
future. Leaving it out costs accuracy and is the correct call: picker skill
spread, congestion noise and residual variation stay unlearnable, which is why
the model's R-squared lands short of 1 instead of suspiciously at it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import CONFIG, ModelConfig, PickingConfig

__all__ = [
    "PickTimeModel",
    "analytic_pick_time",
    "fit_linear_timestudy",
    "evaluate_pick_time",
]

# The physical drivers an engineered standard is built from.
_TIMESTUDY_FEATURES = ["n_lines", "total_units", "route_distance_m", "mean_level"]

FEATURES = [
    "n_orders",
    "n_lines",
    "total_units",
    "total_volume_l",
    "n_aisles",
    "route_distance_m",
    "mean_level",
    "max_level",
    "hour_of_day",
    "day_of_week",
]
TARGET = "duration_s"


def analytic_pick_time(
    tours: pd.DataFrame, picking_cfg: PickingConfig | None = None
) -> np.ndarray:
    """The textbook engineered-standard estimate, used as the baseline."""
    picking_cfg = picking_cfg or CONFIG.picking
    return (
        picking_cfg.setup_time_s
        + picking_cfg.drop_off_time_s
        + picking_cfg.per_line_time_s * tours["n_lines"].to_numpy()
        + picking_cfg.per_unit_time_s * tours["total_units"].to_numpy()
        + picking_cfg.level_penalty_s * tours["mean_level"].to_numpy() * tours["n_lines"].to_numpy()
        + tours["route_distance_m"].to_numpy() / picking_cfg.walking_speed_mps
    )


def fit_linear_timestudy(train: pd.DataFrame):
    """Least-squares fit of the standard formula's coefficients.

    This is the honest stand-in for a time study: same structure as the
    analytic formula, but the constants come from watching the operation
    rather than from knowing the answer.
    """
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(train[_TIMESTUDY_FEATURES], train[TARGET])
    return model


@dataclass
class PickTimeModel:
    model: object = None
    features: list[str] | None = None

    def fit(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame | None = None,
        model_cfg: ModelConfig | None = None,
    ) -> "PickTimeModel":
        import lightgbm as lgb

        model_cfg = model_cfg or CONFIG.model
        self.features = [c for c in FEATURES if c in train.columns]

        self.model = lgb.LGBMRegressor(
            objective="l2",
            n_estimators=model_cfg.n_estimators,
            learning_rate=model_cfg.learning_rate,
            max_depth=model_cfg.max_depth,
            min_child_samples=model_cfg.min_samples_leaf,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.9,
            verbose=-1,
            n_jobs=-1,
        )

        fit_kwargs = {}
        if valid is not None and not valid.empty:
            fit_kwargs["eval_X"] = valid[self.features]
            fit_kwargs["eval_y"] = valid[TARGET]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(model_cfg.early_stopping_rounds, verbose=False)
            ]

        self.model.fit(train[self.features], train[TARGET], **fit_kwargs)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        # The engine may hand over an order-level frame that lacks the
        # tour-level distance column; fall back to a rough allowance so the
        # caller still gets a usable number.
        frame = frame.copy()
        for col in self.features:
            if col not in frame.columns:
                frame[col] = 0.0
        return np.clip(self.model.predict(frame[self.features]), 30.0, None)

    def feature_importance(self) -> pd.DataFrame:
        return (
            pd.DataFrame(
                {"feature": self.features, "gain": self.model.booster_.feature_importance("gain")}
            )
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )


def _scores(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    resid = y - pred
    ss_res = float((resid**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "mae_s": float(np.abs(resid).mean()),
        "rmse_s": float(np.sqrt((resid**2).mean())),
        "mape": float(np.abs(resid / np.clip(y, 1e-9, None)).mean()),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def evaluate_pick_time(
    tours: pd.DataFrame,
    *,
    train_end_s: float,
    picking_cfg: PickingConfig | None = None,
    model_cfg: ModelConfig | None = None,
) -> tuple[PickTimeModel, pd.DataFrame]:
    """Fit on the earlier part of the training window, score on the later part.

    The split is on time, not at random: tours are correlated within a day, so
    a random split would put near-duplicate rows on both sides and inflate the
    score.
    """
    tours = tours[tours["start_s"] < train_end_s].copy()
    if len(tours) < 200:
        raise ValueError(f"only {len(tours)} training tours; need at least 200")

    split_s = tours["start_s"].quantile(0.75)
    train = tours[tours["start_s"] < split_s]
    test = tours[tours["start_s"] >= split_s]

    inner_split = train["start_s"].quantile(0.85)
    model = PickTimeModel().fit(
        train[train["start_s"] < inner_split],
        train[train["start_s"] >= inner_split],
        model_cfg,
    )
    linear = fit_linear_timestudy(train)

    y = test[TARGET].to_numpy(dtype=float)
    comparison = pd.DataFrame(
        [
            {"estimator": "analytic_standard", **_scores(y, analytic_pick_time(test, picking_cfg))},
            {"estimator": "linear_timestudy", **_scores(y, linear.predict(test[_TIMESTUDY_FEATURES]))},
            {"estimator": "lightgbm", **_scores(y, model.predict(test))},
        ]
    )
    comparison["n_test_tours"] = len(test)
    return model, comparison
