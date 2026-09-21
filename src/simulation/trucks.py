"""Outbound truck schedule -- the source of every deadline in the project.

Deadlines are not invented numbers. Each order is assigned to a departing
truck, and its deadline is

    deadline = planned departure - staging buffer

Trucks then arrive early, on time, or late according to a right-skewed delay
distribution, and compete for a limited number of dock doors. That single
mechanism produces all the urgency dynamics the optimizer has to cope with:
a late truck quietly relaxes its orders, an early one tightens them, and a
dock queue can hold up a truck that arrived on time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CONFIG, TruckConfig

__all__ = ["build_truck_schedule", "departure_grid"]


def departure_grid(n_days: int, truck_cfg: TruckConfig | None = None) -> np.ndarray:
    """Absolute planned departure times, in hours from the start of day 0."""
    truck_cfg = truck_cfg or CONFIG.truck
    days = np.arange(n_days, dtype=float)[:, None]
    hours = np.asarray(truck_cfg.departure_hours, dtype=float)[None, :]
    return np.sort((days * 24.0 + hours).ravel())


def build_truck_schedule(
    n_days: int,
    rng: np.random.Generator,
    truck_cfg: TruckConfig | None = None,
    start_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """One row per truck departure slot, with planned and realised timings.

    Columns
    -------
    truck_id            : ``TRK0000`` style identifier
    planned_departure_h : hours from simulation start (this defines deadlines)
    planned_arrival_h   : when the truck is expected at the yard
    actual_arrival_h    : realised arrival, after delay / early sampling
    delay_h             : signed deviation from plan (negative = early)
    loading_time_h      : how long it occupies a dock
    capacity_orders     : how many orders it can take
    """
    truck_cfg = truck_cfg or CONFIG.truck

    planned_departure = departure_grid(n_days, truck_cfg)
    n = planned_departure.size
    planned_arrival = planned_departure - truck_cfg.lead_time_before_departure_h

    # Delay model: most trucks are punctual; a minority are late with a long
    # tail (gamma), and a few show up early. This asymmetry is what real yard
    # data looks like -- lateness has no upper bound, earliness does.
    delay = np.zeros(n)
    u = rng.random(n)

    late_mask = u < truck_cfg.delay_probability
    n_late = int(late_mask.sum())
    if n_late:
        scale = truck_cfg.delay_mean_h / truck_cfg.delay_shape
        delay[late_mask] = rng.gamma(truck_cfg.delay_shape, scale, size=n_late)

    early_mask = (u >= truck_cfg.delay_probability) & (
        u < truck_cfg.delay_probability + truck_cfg.early_probability
    )
    n_early = int(early_mask.sum())
    if n_early:
        delay[early_mask] = -rng.exponential(truck_cfg.early_mean_h, size=n_early)

    actual_arrival = planned_arrival + delay

    loading_sd = truck_cfg.loading_time_mean_h * truck_cfg.loading_time_cv
    loading_time = np.maximum(
        0.1, rng.normal(truck_cfg.loading_time_mean_h, loading_sd, size=n)
    )

    schedule = pd.DataFrame(
        {
            "truck_id": [f"TRK{i:04d}" for i in range(n)],
            "planned_departure_h": planned_departure,
            "planned_arrival_h": planned_arrival,
            "actual_arrival_h": actual_arrival,
            "delay_h": delay,
            "loading_time_h": loading_time,
            "capacity_orders": truck_cfg.truck_capacity_orders,
        }
    )

    if start_date is not None:
        schedule["planned_departure_at"] = start_date + pd.to_timedelta(
            schedule["planned_departure_h"], unit="h"
        )
        schedule["actual_arrival_at"] = start_date + pd.to_timedelta(
            schedule["actual_arrival_h"], unit="h"
        )

    return schedule
