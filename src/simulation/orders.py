"""Demand and order generation.

The order process is generated identically for every day of the 18-month
history, but only the *detail* (individual orders and their lines) is kept for
the discrete-event simulation window. Older days are aggregated to daily
SKU-level demand, which is all the forecasting model needs. This keeps the
history long enough to learn seasonality without materialising millions of
order lines that nothing would read.

Order composition -- how many lines, how many units per line, which SKUs, and
how demand moves across weekdays and months -- comes from the calibrated real
dataset rather than from invented parameters.

Deadlines come from ``trucks.py``: an order is assigned to the first departure
that satisfies its service level's minimum lead time, and its deadline is that
departure minus the staging buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import CONFIG, DemandConfig, SimulationConfig, TruckConfig
from src.calibration.fit_parameters import CalibrationParams

__all__ = ["GeneratedDemand", "generate_demand_and_orders"]


@dataclass
class GeneratedDemand:
    daily_demand: pd.DataFrame   # date x sku_id demand over the full history
    orders: pd.DataFrame         # order header rows, simulation window only
    order_lines: pd.DataFrame    # order detail rows, simulation window only
    receipts: pd.DataFrame       # inbound putaway rows, simulation window only
    window_start_day: int
    start_date: pd.Timestamp


def _sample_categorical(cdf: np.ndarray, rng: np.random.Generator, size: int) -> np.ndarray:
    """Fast weighted sampling via inverse-CDF lookup."""
    return np.searchsorted(cdf, rng.random(size), side="right")


def _expected_orders_per_day(
    dates: pd.DatetimeIndex,
    calibration: CalibrationParams,
    demand_cfg: DemandConfig,
) -> np.ndarray:
    """Deterministic demand level before noise: trend x weekday x month."""
    day_index = np.arange(len(dates), dtype=float)
    trend = (1.0 + demand_cfg.annual_growth) ** (day_index / 365.0)

    weekday = np.asarray(calibration.weekday_factors, dtype=float)[dates.dayofweek.to_numpy()]
    month = np.asarray(calibration.month_factors, dtype=float)[dates.month.to_numpy() - 1]

    return demand_cfg.base_orders_per_day * trend * weekday * month


def generate_demand_and_orders(
    catalog: pd.DataFrame,
    calibration: CalibrationParams,
    rng: np.random.Generator,
    *,
    demand_cfg: DemandConfig | None = None,
    truck_cfg: TruckConfig | None = None,
    sim_cfg: SimulationConfig | None = None,
) -> GeneratedDemand:
    demand_cfg = demand_cfg or CONFIG.demand
    truck_cfg = truck_cfg or CONFIG.truck
    sim_cfg = sim_cfg or CONFIG.simulation

    n_days = demand_cfg.history_days
    start_date = pd.Timestamp(demand_cfg.start_date)
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    n_skus = len(catalog)
    sku_ids = catalog["sku_id"].to_numpy()
    popularity_cdf = np.cumsum(catalog["popularity"].to_numpy())
    popularity_cdf /= popularity_cdf[-1]
    express = catalog["express_propensity"].to_numpy()
    volume = catalog["volume_l"].to_numpy()
    weight = catalog["weight_kg"].to_numpy()

    # The DES window sits at the end of the history so the forecasting model
    # can be trained on everything before it -- no look-ahead.
    window_start_day = n_days - sim_cfg.window_days

    expected = _expected_orders_per_day(dates, calibration, demand_cfg)
    shape = 1.0 / demand_cfg.noise_dispersion

    hourly_w = np.asarray(demand_cfg.hourly_arrival_weights, dtype=float)
    hourly_cdf = np.cumsum(hourly_w / hourly_w.sum())

    service_weights = np.asarray(demand_cfg.service_level_weights, dtype=float)

    daily_demand = np.zeros((n_days, n_skus), dtype=np.int64)

    order_rows: list[dict] = []
    line_frames: list[pd.DataFrame] = []
    order_counter = 0

    for day in range(n_days):
        # --- how many orders today -----------------------------------
        rate = rng.gamma(shape, expected[day] / shape)
        n_orders = int(rng.poisson(rate))
        if n_orders == 0:
            continue

        # --- order composition ---------------------------------------
        lines_per_order = calibration.sample_lines_per_order(rng, n_orders)
        total_lines = int(lines_per_order.sum())

        line_sku = _sample_categorical(popularity_cdf, rng, total_lines)
        line_units = calibration.sample_units_per_line(rng, total_lines)
        line_order = np.repeat(np.arange(n_orders), lines_per_order)

        np.add.at(daily_demand[day], line_sku, line_units)

        if day < window_start_day:
            continue  # history days only contribute aggregate demand

        # --- per-order aggregates ------------------------------------
        order_units = np.bincount(line_order, weights=line_units, minlength=n_orders)
        order_volume = np.bincount(
            line_order, weights=line_units * volume[line_sku], minlength=n_orders
        )
        order_weight = np.bincount(
            line_order, weights=line_units * weight[line_sku], minlength=n_orders
        )
        express_score = (
            np.bincount(line_order, weights=express[line_sku], minlength=n_orders)
            / lines_per_order
        )

        # --- service level -------------------------------------------
        # Urgency correlates with the product mix: orders full of express-prone
        # categories skew towards tighter service levels. This is what makes
        # urgency a SKU-level property worth optimising slots for.
        w = np.empty((n_orders, service_weights.size))
        w[:, 0] = service_weights[0] * express_score
        w[:, 1] = service_weights[1] * np.sqrt(express_score)
        w[:, 2] = service_weights[2]
        w /= w.sum(axis=1, keepdims=True)
        service_idx = (np.cumsum(w, axis=1) < rng.random((n_orders, 1))).sum(axis=1)

        # --- arrival time --------------------------------------------
        hour = _sample_categorical(hourly_cdf, rng, n_orders) + rng.random(n_orders)
        created_h = day * 24.0 + hour

        for i in range(n_orders):
            order_rows.append(
                {
                    "order_id": f"ORD{order_counter + i:07d}",
                    "day": day,
                    "created_h": created_h[i],
                    "service_level": demand_cfg.service_levels[service_idx[i]],
                    "n_lines": int(lines_per_order[i]),
                    "total_units": float(order_units[i]),
                    "total_volume_l": float(order_volume[i]),
                    "total_weight_kg": float(order_weight[i]),
                    "express_score": float(express_score[i]),
                }
            )

        line_frames.append(
            pd.DataFrame(
                {
                    "order_id": np.array(
                        [f"ORD{order_counter + i:07d}" for i in line_order]
                    ),
                    "sku_id": sku_ids[line_sku],
                    "units": line_units,
                }
            )
        )
        order_counter += n_orders

    orders = pd.DataFrame(order_rows)
    order_lines = (
        pd.concat(line_frames, ignore_index=True)
        if line_frames
        else pd.DataFrame(columns=["order_id", "sku_id", "units"])
    )
    # An order can hit the same SKU twice; a picker visits that slot once, so
    # duplicates are merged and the header line count is restated to match.
    order_lines = (
        order_lines.groupby(["order_id", "sku_id"], as_index=False)["units"].sum()
    )
    if not orders.empty:
        true_lines = order_lines.groupby("order_id")["units"].size()
        orders["n_lines"] = orders["order_id"].map(true_lines).astype(int)

    orders = _assign_trucks(orders, n_days, truck_cfg, demand_cfg)

    daily_demand_df = pd.DataFrame(daily_demand, index=dates, columns=sku_ids)
    daily_demand_df = (
        daily_demand_df.stack().rename("demand").reset_index()
        .rename(columns={"level_0": "date", "level_1": "sku_id"})
    )

    receipts = _generate_receipts(
        daily_demand, dates, sku_ids, window_start_day, n_days, rng
    )

    return GeneratedDemand(
        daily_demand=daily_demand_df,
        orders=orders,
        order_lines=order_lines,
        receipts=receipts,
        window_start_day=window_start_day,
        start_date=start_date,
    )


def _assign_trucks(
    orders: pd.DataFrame, n_days: int, truck_cfg: TruckConfig, demand_cfg: DemandConfig
) -> pd.DataFrame:
    """Attach each order to a departing truck and derive its deadline."""
    from src.simulation.trucks import departure_grid

    if orders.empty:
        return orders

    departures = departure_grid(n_days + 3, truck_cfg)
    service_to_lead = dict(zip(demand_cfg.service_levels, truck_cfg.service_min_lead_h))
    lead = orders["service_level"].map(service_to_lead).to_numpy()

    truck_idx = np.searchsorted(departures, orders["created_h"].to_numpy() + lead, side="left")
    truck_idx = np.minimum(truck_idx, departures.size - 1)

    # Capacity overflow: a full truck pushes its latest arrivals to the next
    # departure, which is exactly what a real yard does when a trailer fills up.
    order_pos = np.argsort(orders["created_h"].to_numpy(), kind="stable")
    assigned = truck_idx.copy()
    counts: dict[int, int] = {}
    for pos in order_pos:
        t = assigned[pos]
        while counts.get(t, 0) >= truck_cfg.truck_capacity_orders and t < departures.size - 1:
            t += 1
        assigned[pos] = t
        counts[t] = counts.get(t, 0) + 1

    orders = orders.copy()
    orders["truck_idx"] = assigned
    orders["truck_id"] = [f"TRK{i:04d}" for i in assigned]
    orders["truck_departure_h"] = departures[assigned]
    orders["deadline_h"] = orders["truck_departure_h"] - truck_cfg.staging_buffer_h
    orders["slack_at_creation_h"] = orders["deadline_h"] - orders["created_h"]
    return orders


def _generate_receipts(
    daily_demand: np.ndarray,
    dates: pd.DatetimeIndex,
    sku_ids: np.ndarray,
    window_start_day: int,
    n_days: int,
    rng: np.random.Generator,
    *,
    cover_days: int = 10,
) -> pd.DataFrame:
    """Inbound goods receipts that have to be put away into slots.

    Replenishment is demand-driven: a SKU is more likely to be received when it
    has been selling, and the quantity covers roughly ``cover_days`` of recent
    demand. This is what gives the putaway side of the problem something to
    optimise.
    """
    rows: list[dict] = []
    for day in range(window_start_day, n_days):
        recent = daily_demand[max(0, day - 28) : day].mean(axis=0)
        if recent.sum() <= 0:
            continue
        # Reorder-point logic: a SKU holding `cover_days` of stock is
        # replenished roughly every `cover_days`, whatever its volume. Fast
        # movers get *bigger* receipts, not more frequent ones.
        prob = np.where(recent > 0, 1.0 / cover_days, 0.0)
        received = rng.random(recent.size) < prob
        idx = np.flatnonzero(received)
        if idx.size == 0:
            continue
        qty = np.maximum(1, np.rint(recent[idx] * cover_days * rng.uniform(0.7, 1.3, idx.size)))
        arrival_h = day * 24.0 + rng.uniform(6.0, 16.0, idx.size)
        for j, sku_i in enumerate(idx):
            rows.append(
                {
                    "receipt_id": f"RCV{day:04d}{j:04d}",
                    "day": day,
                    "arrival_h": float(arrival_h[j]),
                    "sku_id": sku_ids[sku_i],
                    "units": int(qty[j]),
                }
            )
    return pd.DataFrame(rows)
