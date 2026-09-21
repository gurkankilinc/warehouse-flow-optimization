"""Estimate demand parameters from the real transaction data.

This is the bridge between the real world and the simulator. Four things are
estimated here, and each one feeds a specific part of the simulation:

  popularity curve  -> how skewed demand is across SKUs (drives ABC classes and
                       therefore the whole slotting problem)
  lines per order   -> how many distinct SKUs a picker collects per order
                       (drives tour length, and hence routing)
  units per line    -> how much is picked at each stop (drives pick time)
  calendar factors  -> weekday and month multipliers (drives the seasonality the
                       forecasting model has to learn)

Distributions are stored as quantile grids rather than fitted parametric
families. Real basket sizes are lumpy and long-tailed; sampling from the
empirical inverse CDF reproduces that faithfully without pretending the data is
negative-binomial when it is not.

If the real dataset is unavailable, ``fallback_params`` produces the same
structure from documented defaults and marks ``source = "fallback"`` so that
every report downstream states which mode it ran in.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config import CALIBRATION_FILE, CONFIG, PROCESSED_DIR, CalibrationConfig

__all__ = ["CalibrationParams", "fit_from_transactions", "fallback_params", "load_calibration", "build"]

_QUANTILE_GRID = 500
_POPULARITY_POINTS = 400


@dataclass
class CalibrationParams:
    """Everything the simulator needs to know about real-world demand shape."""

    source: str                      # "real" or "fallback"
    source_name: str
    n_transactions: int
    n_products: int
    date_min: str
    date_max: str

    # Descending normalised demand share, resampled to a fixed-length grid.
    popularity_curve: list[float] = field(default_factory=list)
    zipf_exponent: float = 1.0
    top20_demand_share: float = 0.0

    lines_per_order_quantiles: list[float] = field(default_factory=list)
    units_per_line_quantiles: list[float] = field(default_factory=list)
    lines_per_order_mean: float = 0.0
    units_per_line_mean: float = 0.0

    weekday_factors: list[float] = field(default_factory=list)   # Mon..Sun, mean 1
    month_factors: list[float] = field(default_factory=list)     # Jan..Dec, mean 1

    # ------------------------------------------------------------------
    # Sampling helpers used by the simulator
    # ------------------------------------------------------------------

    def popularity_weights(self, n_skus: int) -> np.ndarray:
        """Resample the popularity curve onto ``n_skus`` products.

        Interpolating the *cumulative* curve (rather than the shares directly)
        keeps the total at 1 and preserves the head/tail balance regardless of
        how many SKUs the simulated catalog has.
        """
        curve = np.asarray(self.popularity_curve, dtype=float)
        if curve.size == 0:
            raise ValueError("calibration has no popularity curve")

        cumulative = np.concatenate([[0.0], np.cumsum(curve)])
        cumulative /= cumulative[-1]
        source_x = np.linspace(0.0, 1.0, cumulative.size)
        target_x = np.linspace(0.0, 1.0, n_skus + 1)
        weights = np.diff(np.interp(target_x, source_x, cumulative))
        return weights / weights.sum()

    def sample_lines_per_order(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return self._sample(self.lines_per_order_quantiles, rng, size)

    def sample_units_per_line(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return self._sample(self.units_per_line_quantiles, rng, size)

    @staticmethod
    def _sample(grid: list[float], rng: np.random.Generator, size: int) -> np.ndarray:
        arr = np.asarray(grid, dtype=float)
        idx = rng.integers(0, arr.size, size=size)
        return np.maximum(1, np.rint(arr[idx])).astype(int)

    # ------------------------------------------------------------------

    def save(self, path: Path = CALIBRATION_FILE) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path = CALIBRATION_FILE) -> "CalibrationParams":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def summary(self) -> str:
        return (
            f"source              : {self.source} ({self.source_name})\n"
            f"transactions        : {self.n_transactions:,}\n"
            f"distinct products   : {self.n_products:,}\n"
            f"date range          : {self.date_min} .. {self.date_max}\n"
            f"zipf exponent       : {self.zipf_exponent:.3f}\n"
            f"top-20% demand share: {self.top20_demand_share:.1%}\n"
            f"lines per order     : mean {self.lines_per_order_mean:.1f}\n"
            f"units per line      : mean {self.units_per_line_mean:.1f}\n"
            f"weekday factors     : {', '.join(f'{v:.2f}' for v in self.weekday_factors)}\n"
        )


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------


def _fit_zipf_exponent(shares: np.ndarray) -> float:
    """Slope of log(share) against log(rank); a pure Zipf law gives -alpha."""
    ranks = np.arange(1, shares.size + 1, dtype=float)
    mask = shares > 0
    if mask.sum() < 10:
        return 1.0
    slope, _ = np.polyfit(np.log(ranks[mask]), np.log(shares[mask]), 1)
    return float(-slope)


def _quantile_grid(values: np.ndarray, n_points: int = _QUANTILE_GRID) -> list[float]:
    probs = (np.arange(n_points) + 0.5) / n_points
    return [float(v) for v in np.quantile(values, probs)]


def fit_from_transactions(
    transactions: pd.DataFrame, cfg: CalibrationConfig | None = None
) -> CalibrationParams:
    """Estimate all demand parameters from cleaned transaction lines."""
    cfg = cfg or CONFIG.calibration

    # --- popularity ---------------------------------------------------
    per_sku = (
        transactions.groupby("stock_code")["quantity"].sum().sort_values(ascending=False)
    )
    shares = (per_sku / per_sku.sum()).to_numpy()
    zipf_exponent = _fit_zipf_exponent(shares)
    top20 = float(shares[: max(1, int(0.2 * shares.size))].sum())

    # Resample onto a fixed grid so the JSON stays small and catalog size free.
    cumulative = np.concatenate([[0.0], np.cumsum(shares)])
    source_x = np.linspace(0.0, 1.0, cumulative.size)
    target_x = np.linspace(0.0, 1.0, _POPULARITY_POINTS + 1)
    popularity_curve = np.diff(np.interp(target_x, source_x, cumulative))

    # --- basket structure ---------------------------------------------
    per_order = transactions.groupby("invoice").agg(
        lines=("stock_code", "nunique"), units=("quantity", "sum")
    )
    lines_per_order = per_order["lines"].to_numpy(dtype=float)
    units_per_line = transactions["quantity"].to_numpy(dtype=float)

    # --- calendar effects ---------------------------------------------
    daily = transactions.groupby("invoice_date")["quantity"].sum()
    daily.index = pd.to_datetime(daily.index)

    weekday_mean = daily.groupby(daily.index.dayofweek).mean()
    weekday_factors = (weekday_mean / weekday_mean.mean()).reindex(range(7), fill_value=1.0)

    month_mean = daily.groupby(daily.index.month).mean()
    month_factors = (month_mean / month_mean.mean()).reindex(range(1, 13), fill_value=1.0)

    return CalibrationParams(
        source="real",
        source_name=cfg.source_name,
        n_transactions=int(len(transactions)),
        n_products=int(per_sku.size),
        date_min=str(daily.index.min().date()),
        date_max=str(daily.index.max().date()),
        popularity_curve=[float(v) for v in popularity_curve],
        zipf_exponent=zipf_exponent,
        top20_demand_share=top20,
        lines_per_order_quantiles=_quantile_grid(lines_per_order),
        units_per_line_quantiles=_quantile_grid(units_per_line),
        lines_per_order_mean=float(lines_per_order.mean()),
        units_per_line_mean=float(units_per_line.mean()),
        weekday_factors=[float(v) for v in weekday_factors.to_numpy()],
        month_factors=[float(v) for v in month_factors.to_numpy()],
    )


def fallback_params(cfg: CalibrationConfig | None = None) -> CalibrationParams:
    """Documented parametric defaults, used when the real dataset is missing.

    Shapes are chosen to be plausible rather than convenient: a Zipf-like
    popularity curve, and geometric-ish basket sizes with a long right tail.
    """
    cfg = cfg or CONFIG.calibration
    rng = np.random.default_rng(CONFIG.simulation.seed)

    ranks = np.arange(1, _POPULARITY_POINTS + 1, dtype=float)
    curve = ranks ** (-cfg.fallback_zipf_exponent)
    curve /= curve.sum()

    lines = rng.geometric(1.0 / cfg.fallback_lines_per_order_mean, size=20_000).astype(float)
    units = rng.geometric(1.0 / cfg.fallback_units_per_line_mean, size=20_000).astype(float)

    # Mild weekday and seasonal structure so the forecast model still has signal.
    weekday = np.array([1.12, 1.06, 1.02, 1.04, 1.15, 0.79, 0.82])
    month = 1.0 + 0.25 * np.sin((np.arange(12) - 2) / 12 * 2 * np.pi)

    return CalibrationParams(
        source="fallback",
        source_name=f"parametric defaults (could not obtain {cfg.source_name})",
        n_transactions=0,
        n_products=_POPULARITY_POINTS,
        date_min="",
        date_max="",
        popularity_curve=[float(v) for v in curve],
        zipf_exponent=cfg.fallback_zipf_exponent,
        top20_demand_share=float(curve[: int(0.2 * curve.size)].sum()),
        lines_per_order_quantiles=_quantile_grid(lines),
        units_per_line_quantiles=_quantile_grid(units),
        lines_per_order_mean=float(lines.mean()),
        units_per_line_mean=float(units.mean()),
        weekday_factors=[float(v) for v in weekday / weekday.mean()],
        month_factors=[float(v) for v in month / month.mean()],
    )


def load_calibration(path: Path = CALIBRATION_FILE) -> CalibrationParams:
    """Load calibration, building it first if it has not been created yet."""
    if path.exists():
        return CalibrationParams.load(path)
    return build()[0]


def build(*, force_download: bool = False) -> tuple[CalibrationParams, pd.DataFrame | None]:
    """Run the full calibration pipeline: download -> clean -> fit -> save."""
    from src.calibration.clean import clean_transactions, load_raw
    from src.calibration.download import ensure_raw_dataset

    result = ensure_raw_dataset(force=force_download)
    if not result.ok:
        print(f"[calibration] real dataset unavailable -> using fallback\n  {result.error}")
        params = fallback_params()
        params.save()
        return params, None

    raw = load_raw(result.path)
    clean, audit = clean_transactions(raw)
    params = fit_from_transactions(clean)
    params.save()

    clean.to_parquet(PROCESSED_DIR / "transactions_clean.parquet", index=False)
    audit.to_csv(PROCESSED_DIR / "cleaning_audit.csv", index=False)
    return params, audit


if __name__ == "__main__":  # pragma: no cover - manual entry point
    params, audit = build()
    print(params.summary())
    if audit is not None:
        print("\nCleaning audit:")
        print(audit.to_string(index=False))
