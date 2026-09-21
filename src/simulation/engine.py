"""Discrete-event simulation of warehouse operations.

What the engine models, and why each piece earns its place:

* **Pickers** are a limited resource working a fixed shift. Without a capacity
  limit, every order would be picked instantly and deadlines would never bind.
* **Tours** are formed by the dispatch + batching policies and routed by the
  routing policy. This is where the optimisation actually acts.
* **Trucks** arrive early, late or on time, queue for a limited number of dock
  doors, and leave with whatever is staged. An order that misses its truck
  rolls to the next departure and is counted late against its original promise.
* **Putaway** consumes its own small crew and is measured separately, because
  "where do we put incoming goods" is the second half of the problem.

Variance reduction
------------------
Congestion, picker skill and pick-time noise are drawn *before* the run from a
stream that does not depend on the policy under test, and every scenario
replays the same draws. This is the common-random-numbers technique: the
measured difference between two policies is then the effect of the policy
rather than of the weather, which is what makes a five-replication comparison
meaningful instead of noise.

The simulator's own timing formula uses factors (instantaneous congestion,
individual picker skill, residual noise) that are never exposed as model
features, so the pick-time model has irreducible error and cannot simply
re-derive the generator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import simpy

from config import (
    CONFIG,
    PickingConfig,
    PolicyConfig,
    SimulationConfig,
    TruckConfig,
    WarehouseConfig,
)
from src.policies.batching import BATCHING_RULES
from src.policies.dispatch import DISPATCH_RULES
from src.policies.routing import route_nn_2opt, route_s_shape

__all__ = ["Policies", "SimulationResult", "StochasticStreams", "run_simulation"]

_POLL_S = 60.0          # how often an idle picker checks for work
_RISK_REFRESH_S = 900.0  # WMS recomputes risk scores every 15 minutes
_CONGESTION_BUCKET_S = 900.0

ROUTING_RULES: dict[str, Callable] = {
    "s_shape": route_s_shape,
    "nn_2opt": route_nn_2opt,
}


# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------


@dataclass
class Policies:
    """The decision layer under test."""

    slot_assignment: pd.DataFrame          # sku_id -> slot_id
    dispatch: str = "edd"
    batching: str = "sequential"
    routing: str = "s_shape"
    #: How the crew picks what to load when it cannot load everything.
    #: "fifo" takes staged order; "triage" mirrors the picking-side rule.
    loading: str = "fifo"
    pick_time_model: object | None = None  # .predict(DataFrame) -> seconds
    risk_model: object | None = None       # .predict_proba(DataFrame) -> P(miss)
    name: str = "baseline"


@dataclass
class StochasticStreams:
    """Pre-drawn randomness, shared identically across scenarios."""

    congestion: np.ndarray      # multiplier per 15-minute bucket
    picker_skill: np.ndarray    # multiplier per picker
    noise: np.ndarray           # residual multiplier per tour

    @classmethod
    def build(
        cls,
        horizon_s: float,
        n_pickers: int,
        seed: int,
        picking_cfg: PickingConfig | None = None,
    ) -> "StochasticStreams":
        picking_cfg = picking_cfg or CONFIG.picking
        rng = np.random.default_rng(seed)

        n_buckets = int(horizon_s / _CONGESTION_BUCKET_S) + 2
        hours = (np.arange(n_buckets) * _CONGESTION_BUCKET_S / 3600.0) % 24.0
        # Aisles are busiest in the middle of the shift, when everyone is out.
        shape = np.sin((hours - 6.0) / 17.0 * np.pi).clip(0.0)
        congestion = picking_cfg.congestion_base + picking_cfg.congestion_amplitude * shape
        congestion *= np.exp(rng.normal(0.0, picking_cfg.congestion_noise_sd, n_buckets))

        picker_skill = np.exp(rng.normal(0.0, picking_cfg.picker_skill_sd, n_pickers))
        noise = np.exp(rng.normal(0.0, picking_cfg.pick_time_noise_cv, 400_000))

        return cls(congestion=congestion, picker_skill=picker_skill, noise=noise)


@dataclass
class PendingOrder:
    idx: int
    order_id: str
    created_s: float
    deadline_s: float            # staging cut-off: when picking must be done
    truck_idx: int               # mutates when the order misses and rolls over
    original_truck_idx: int      # the truck that was promised
    promised_departure_s: float  # that truck's planned departure
    service_level: str
    line_slot: np.ndarray
    line_units: np.ndarray
    line_volume: np.ndarray
    line_level: np.ndarray
    total_lines: int
    remaining: list[int] = field(default_factory=list)
    predicted_pick_s: float = 0.0
    risk_score: float = 0.0
    first_pick_s: float | None = None
    staged_s: float | None = None
    loaded_s: float | None = None
    shipped_s: float | None = None
    shipped_truck_idx: int = -1
    tours_used: int = 0
    #: "picking" if it was still being picked when its truck loaded,
    #: "loading" if it was ready on the staging lanes but the crew never got to
    #: it. Splitting the blame is the point of modelling loading at all.
    miss_cause: str = ""


@dataclass
class SimulationResult:
    orders: pd.DataFrame
    tours: pd.DataFrame
    trucks: pd.DataFrame
    putaway: pd.DataFrame
    kpis: dict
    risk_snapshots: pd.DataFrame = field(default_factory=pd.DataFrame)
    trace: pd.DataFrame = field(default_factory=pd.DataFrame)


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------


def run_simulation(
    *,
    slots: pd.DataFrame,
    catalog: pd.DataFrame,
    orders: pd.DataFrame,
    order_lines: pd.DataFrame,
    receipts: pd.DataFrame,
    truck_schedule: pd.DataFrame,
    policies: Policies,
    streams: StochasticStreams,
    window_start_day: int,
    log_risk_snapshots: bool = False,
    trace_interval_h: float | None = None,
    sim_cfg: SimulationConfig | None = None,
    picking_cfg: PickingConfig | None = None,
    truck_cfg: TruckConfig | None = None,
    policy_cfg: PolicyConfig | None = None,
    warehouse_cfg: WarehouseConfig | None = None,
) -> SimulationResult:
    sim_cfg = sim_cfg or CONFIG.simulation
    picking_cfg = picking_cfg or CONFIG.picking
    truck_cfg = truck_cfg or CONFIG.truck
    policy_cfg = policy_cfg or CONFIG.policy
    warehouse_cfg = warehouse_cfg or CONFIG.warehouse
    cross_aisle_ys = warehouse_cfg.cross_aisle_ys

    origin_h = window_start_day * 24.0
    horizon_s = (sim_cfg.window_days + 2) * 86400.0

    # --- lookup tables -------------------------------------------------
    slot_index = {sid: i for i, sid in enumerate(slots["slot_id"])}
    slot_x = slots["x"].to_numpy()
    slot_y = slots["y"].to_numpy()
    slot_level = slots["level"].to_numpy()
    slot_aisle = slots["aisle"].to_numpy()
    slot_receiving = slots["receiving_distance"].to_numpy()

    sku_to_slot = {
        row.sku_id: slot_index[row.slot_id]
        for row in policies.slot_assignment.itertuples(index=False)
    }
    sku_volume = dict(zip(catalog["sku_id"], catalog["volume_l"]))

    dispatch_fn = DISPATCH_RULES[policies.dispatch]
    batching_fn = BATCHING_RULES[policies.batching]
    routing_fn = ROUTING_RULES[policies.routing]

    # --- build order objects -------------------------------------------
    lines_by_order = {
        oid: grp for oid, grp in order_lines.groupby("order_id", sort=False)
    }

    pending_orders: list[PendingOrder] = []
    all_orders: list[PendingOrder] = []
    for i, row in enumerate(orders.itertuples(index=False)):
        grp = lines_by_order.get(row.order_id)
        if grp is None or grp.empty:
            continue
        skus = grp["sku_id"].to_numpy()
        units = grp["units"].to_numpy(dtype=float)
        line_slot = np.array([sku_to_slot[s] for s in skus], dtype=int)
        line_volume = np.array([sku_volume[s] for s in skus], dtype=float) * units

        order = PendingOrder(
            idx=i,
            order_id=row.order_id,
            created_s=(row.created_h - origin_h) * 3600.0,
            deadline_s=(row.deadline_h - origin_h) * 3600.0,
            truck_idx=int(row.truck_idx),
            original_truck_idx=int(row.truck_idx),
            promised_departure_s=(row.truck_departure_h - origin_h) * 3600.0,
            service_level=row.service_level,
            line_slot=line_slot,
            line_units=units,
            line_volume=line_volume,
            line_level=slot_level[line_slot],
            total_lines=len(line_slot),
            remaining=list(range(len(line_slot))),
        )
        all_orders.append(order)

    # Predicted pick time is scored for every order in one batched call.
    # Calling the model once per order instead cost tens of seconds per run --
    # the per-call DataFrame construction dominated, not the model itself.
    for order, predicted in zip(
        all_orders,
        _predict_order_pick_times(
            all_orders, policies.pick_time_model, picking_cfg, slot_x, slot_y
        ),
    ):
        order.predicted_pick_s = float(predicted)

    all_orders.sort(key=lambda o: o.created_s)

    # --- trucks ---------------------------------------------------------
    truck_schedule = truck_schedule.copy()
    truck_schedule["departure_s"] = (truck_schedule["planned_departure_h"] - origin_h) * 3600.0
    truck_schedule["arrival_s"] = (truck_schedule["actual_arrival_h"] - origin_h) * 3600.0
    truck_schedule["loading_s"] = truck_schedule["loading_time_h"] * 3600.0

    orders_by_truck: dict[int, list[PendingOrder]] = {}
    for o in all_orders:
        orders_by_truck.setdefault(o.truck_idx, []).append(o)

    # Planned departure by truck index, used when rolling a missed order
    # forward. Simply incrementing the index is not safe: trucks are concurrent
    # processes and a delayed one can depart *after* its successor, so `+1`
    # could hand the order to a truck that has already gone. It would then sit
    # in a list nobody reads again -- occupying a staging slot forever, which
    # eventually fills the lanes and blocks every picker.
    departure_by_idx = truck_schedule["departure_s"].to_numpy()

    # --- simulation state ----------------------------------------------
    env = simpy.Environment()
    docks = simpy.Resource(env, capacity=truck_cfg.n_docks)
    putaway_crew = simpy.Resource(env, capacity=picking_cfg.n_putaway_workers)
    loaders = simpy.Resource(env, capacity=truck_cfg.n_load_teams)
    # A Container rather than a counter: `put` blocks when the lanes are full,
    # so a backed-up dock physically stops pickers from dropping off.
    staging_area = simpy.Container(
        env, capacity=truck_cfg.staging_capacity_orders, init=0
    )

    tour_records: list[dict] = []
    truck_records: list[dict] = []
    putaway_records: list[dict] = []
    risk_snapshots: list[pd.DataFrame] = []
    trace_records: list[dict] = []
    tour_counter = {"n": 0}

    def congestion_at(now_s: float) -> float:
        bucket = min(int(now_s / _CONGESTION_BUCKET_S), streams.congestion.size - 1)
        return float(streams.congestion[bucket])

    # ---------------- tour construction ----------------
    release_horizon_s = policy_cfg.release_horizon_h * 3600.0

    def build_tour(now_s: float):
        # Wave release: an order becomes pickable only once its truck is close.
        # Picking further ahead than this fills the staging lanes with work for
        # trucks that have not arrived, and starves the one that has.
        ready = [
            o
            for o in pending_orders
            if o.created_s <= now_s
            and o.remaining
            and o.deadline_s <= now_s + release_horizon_s
        ]
        if not ready:
            return None
        ordered = dispatch_fn(ready, now_s, policy_cfg)
        tour = batching_fn(ordered, picking_cfg, policy_cfg, slot_aisle)
        if not tour:
            return None
        # Reserve the lines immediately so a second picker cannot claim them.
        for order, line_idx in tour:
            taken = set(line_idx)
            order.remaining = [i for i in order.remaining if i not in taken]
            if order.first_pick_s is None:
                order.first_pick_s = now_s
            order.tours_used += 1
        return tour

    def execute_tour(tour, picker_id: int, now_s: float):
        stop_slots = np.concatenate([np.asarray(o.line_slot)[li] for o, li in tour])
        stop_units = np.concatenate([np.asarray(o.line_units)[li] for o, li in tour])
        stop_levels = np.concatenate([np.asarray(o.line_level)[li] for o, li in tour])

        _, distance = routing_fn(
            slot_x[stop_slots], slot_y[stop_slots], warehouse_cfg.dock_x, cross_aisle_ys
        )

        congestion = congestion_at(now_s)
        travel_s = distance / picking_cfg.walking_speed_mps * congestion
        handling_s = (
            picking_cfg.per_line_time_s * len(stop_slots)
            + picking_cfg.per_unit_time_s * float(stop_units.sum())
            + picking_cfg.level_penalty_s * float(stop_levels.sum())
        )
        base = picking_cfg.setup_time_s + travel_s + handling_s + picking_cfg.drop_off_time_s

        n = tour_counter["n"]
        tour_counter["n"] = n + 1
        duration = base * streams.picker_skill[picker_id] * streams.noise[n % streams.noise.size]

        record = {
            "tour_id": f"T{n:06d}",
            "picker_id": picker_id,
            "start_s": now_s,
            "hour_of_day": (now_s / 3600.0) % 24.0,
            "day_of_week": int((now_s / 86400.0) % 7),
            "n_orders": len(tour),
            "n_lines": int(len(stop_slots)),
            "total_units": float(stop_units.sum()),
            "total_volume_l": float(
                sum(np.asarray(o.line_volume)[li].sum() for o, li in tour)
            ),
            "n_aisles": int(np.unique(slot_aisle[stop_slots]).size),
            "route_distance_m": float(distance),
            "mean_level": float(stop_levels.mean()),
            "max_level": int(stop_levels.max()),
            "duration_s": float(duration),
        }
        return duration, record

    def picker_worker(picker_id: int):
        while True:
            wait = _seconds_until_shift(env.now, sim_cfg)
            if wait > 0:
                yield env.timeout(wait)
                continue

            tour = build_tour(env.now)
            if tour is None:
                yield env.timeout(_POLL_S)
                continue

            duration, record = execute_tour(tour, picker_id, env.now)
            yield env.timeout(duration)

            record["end_s"] = env.now
            tour_records.append(record)

            for order, _ in tour:
                if order.remaining or order.staged_s is not None:
                    continue
                # Dropping off needs room on the staging lanes. When the dock
                # has backed up there is none, and the picker waits -- the
                # loading operation reaching back and stalling picking.
                yield staging_area.put(1)
                order.staged_s = env.now
                if order in pending_orders:
                    pending_orders.remove(order)

    def order_arrivals():
        for order in all_orders:
            delay = order.created_s - env.now
            if delay > 0:
                yield env.timeout(delay)
            pending_orders.append(order)

    def risk_refresh():
        """Periodically rescore pending orders, the way a WMS would.

        When ``log_risk_snapshots`` is on, every scoring pass is also recorded.
        Those snapshots -- features as they looked at a decision moment, joined
        later to whether the order actually missed its truck -- are the training
        set for the SLA risk model. Building the training data this way is the
        only honest option: the label only exists once the run has played out.
        """
        while True:
            yield env.timeout(_RISK_REFRESH_S)
            if not pending_orders:
                continue

            need_features = log_risk_snapshots or policies.risk_model is not None
            if not need_features:
                continue

            snapshot = list(pending_orders)
            features = _risk_features(snapshot, env.now, picking_cfg)

            if log_risk_snapshots:
                logged = features.copy()
                logged["order_id"] = [o.order_id for o in snapshot]
                logged["snapshot_s"] = env.now
                risk_snapshots.append(logged)

            if policies.risk_model is None:
                continue
            try:
                scores = policies.risk_model.predict_proba(features)
            except Exception:  # noqa: BLE001 - a broken model must not kill the run
                continue
            for order, score in zip(snapshot, scores):
                order.risk_score = float(score)

    def truck_process(row):
        arrival = float(row.arrival_s)
        if arrival > 0:
            yield env.timeout(arrival)

        # Wait in the yard until it is nearly time to load.
        load_start_target = float(row.departure_s) - float(row.loading_s)
        if env.now < load_start_target:
            yield env.timeout(load_start_target - env.now)

        queue_entry = env.now
        with docks.request() as dock_req:
            yield dock_req
            dock_wait = env.now - queue_entry

            with loaders.request() as crew:
                yield crew
                crew_wait = env.now - queue_entry - dock_wait
                # Everything on the staging lanes at this moment is loadable;
                # anything still being picked has already missed this truck.
                cutoff = env.now

                assigned = orders_by_truck.get(int(row.Index), [])
                loadable = [
                    o for o in assigned
                    if o.shipped_s is None
                    and o.staged_s is not None
                    and o.staged_s <= cutoff
                ]
                ready = _order_for_loading(loadable, policies.loading, truck_cfg)

                # The crew has until departure, and never less than a minimum
                # turn -- a truck that arrived late still gets loaded, it just
                # leaves late.
                window = max(
                    float(row.departure_s) - env.now,
                    truck_cfg.min_loading_window_h * 3600.0,
                )
                spent, loaded = 0.0, []
                for order in ready:
                    cost = _loading_cost(order, truck_cfg)
                    if spent + cost > window or len(loaded) >= int(row.capacity_orders):
                        break
                    spent += cost
                    loaded.append(order)

                yield env.timeout(spent)
                if loaded:
                    yield staging_area.get(len(loaded))
                for order in loaded:
                    order.loaded_s = env.now

        departure = max(env.now, float(row.departure_s))
        if env.now < departure:
            yield env.timeout(departure - env.now)

        for order in loaded:
            order.shipped_s = departure
            order.shipped_truck_idx = int(row.Index)

        rolled_picking = rolled_loading = 0
        for order in assigned:
            if order.shipped_s is not None:
                continue
            # Attribute the miss. `miss_cause` keeps the *first* reason, since
            # an order that rolls twice was really lost at the first hurdle.
            if order.staged_s is not None and order.staged_s <= cutoff:
                rolled_loading += 1
                if not order.miss_cause:
                    order.miss_cause = "loading"
            else:
                rolled_picking += 1
                if not order.miss_cause:
                    order.miss_cause = "picking"

            # Roll to the next truck that has not left yet, never merely to the
            # next index.
            nxt = int(np.searchsorted(departure_by_idx, env.now, side="right"))
            target = max(order.truck_idx + 1, nxt)
            if target < departure_by_idx.size:
                order.truck_idx = target
                orders_by_truck.setdefault(target, []).append(order)

        truck_records.append(
            {
                "truck_idx": int(row.Index),
                "truck_id": row.truck_id,
                "planned_departure_s": float(row.departure_s),
                "arrival_s": arrival,
                "delay_h": float(row.delay_h),
                "dock_wait_s": dock_wait,
                "crew_wait_s": crew_wait,
                "loading_time_s": spent,
                "loading_window_s": window,
                "window_used_pct": 100.0 * spent / max(window, 1e-9),
                "departure_s": departure,
                "orders_ready": len(ready),
                "orders_loaded": len(loaded),
                "orders_rolled": rolled_picking + rolled_loading,
                "rolled_picking": rolled_picking,
                "rolled_loading": rolled_loading,
            }
        )

    def putaway_arrivals():
        for row in receipts.itertuples(index=False):
            arrival_s = (row.arrival_h - origin_h) * 3600.0
            delay = arrival_s - env.now
            if delay > 0:
                yield env.timeout(delay)
            env.process(putaway_task(row, arrival_s))

    def putaway_task(row, arrival_s: float):
        with putaway_crew.request() as req:
            yield req
            slot = sku_to_slot[row.sku_id]
            distance = 2.0 * float(slot_receiving[slot])
            duration = (
                picking_cfg.setup_time_s
                + distance / picking_cfg.walking_speed_mps * congestion_at(env.now)
                + picking_cfg.per_unit_time_s * float(row.units)
            )
            yield env.timeout(duration)
            putaway_records.append(
                {
                    "receipt_id": row.receipt_id,
                    "sku_id": row.sku_id,
                    "units": int(row.units),
                    "arrival_s": arrival_s,
                    "completed_s": env.now,
                    "distance_m": distance,
                    "duration_s": duration,
                }
            )

    def trace():
        """Periodic snapshot of the whole system.

        A discrete-event model that quietly stops doing work looks identical to
        one that finished early, so being able to watch the queues is not a
        debugging luxury -- it is the only way to tell the two apart.
        """
        while True:
            yield env.timeout(trace_interval_h * 3600.0)
            releasable = sum(
                1
                for o in pending_orders
                if o.created_s <= env.now and o.deadline_s <= env.now + release_horizon_s
            )
            trace_records.append(
                {
                    "day": env.now / 86400.0,
                    "pending": len(pending_orders),
                    "releasable": releasable,
                    "staging_level": staging_area.level,
                    "staging_put_queue": len(staging_area.put_queue),
                    "staging_get_queue": len(staging_area.get_queue),
                    "dock_queue": len(docks.queue),
                    "loader_queue": len(loaders.queue),
                    "tours": len(tour_records),
                }
            )

    # --- wire up and run -------------------------------------------------
    if trace_interval_h:
        env.process(trace())
    env.process(order_arrivals())
    env.process(putaway_arrivals())
    env.process(risk_refresh())
    for pid in range(picking_cfg.n_pickers):
        env.process(picker_worker(pid))

    window_trucks = truck_schedule[
        (truck_schedule["departure_s"] >= -86400) & (truck_schedule["departure_s"] <= horizon_s)
    ]
    for row in window_trucks.itertuples():
        env.process(truck_process(row))

    env.run(until=horizon_s)

    snapshots = (
        pd.concat(risk_snapshots, ignore_index=True) if risk_snapshots else pd.DataFrame()
    )
    result = _collect_results(
        all_orders, tour_records, truck_records, putaway_records, snapshots,
        sim_cfg, picking_cfg,
    )
    result.trace = pd.DataFrame(trace_records)
    return result


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _loading_cost(order: PendingOrder, truck_cfg: TruckConfig) -> float:
    return (
        truck_cfg.load_setup_per_order_s
        + truck_cfg.load_per_unit_s * float(order.line_units.sum())
    )


def _order_for_loading(
    loadable: list[PendingOrder], rule: str, truck_cfg: TruckConfig
) -> list[PendingOrder]:
    """Decide what goes on the deck first when it cannot all fit.

    ``fifo`` -- whatever reached the staging lanes first. The natural default,
    and what happens when nobody has thought about it.

    ``triage`` -- the loading-side mirror of the picking-side rule, and it falls
    out of how the KPI is defined. An order that has *already* missed a truck
    cannot become on-time again, because on-time means leaving on the truck it
    was promised; loading it can no longer move that number. An order that has
    not missed yet still can. So the crew loads the not-yet-missed orders first,
    and within those the cheapest to handle, since fitting more of them into the
    same window saves more deadlines.
    """
    if rule == "triage":
        return sorted(
            loadable,
            key=lambda o: (
                1 if o.miss_cause else 0,          # rescuable first
                _loading_cost(o, truck_cfg),       # then quickest to load
            ),
        )
    return sorted(loadable, key=lambda o: o.staged_s)


def _seconds_until_shift(now_s: float, sim_cfg: SimulationConfig) -> float:
    """0 if the shift is running, otherwise the wait until it starts."""
    day = math.floor(now_s / 86400.0)
    start = day * 86400.0 + sim_cfg.shift_start_hour * 3600.0
    end = day * 86400.0 + sim_cfg.shift_end_hour * 3600.0
    if now_s < start:
        return start - now_s
    if now_s < end:
        return 0.0
    return (day + 1) * 86400.0 + sim_cfg.shift_start_hour * 3600.0 - now_s


def _predict_order_pick_times(
    orders: list[PendingOrder],
    model,
    picking_cfg: PickingConfig,
    slot_x: np.ndarray,
    slot_y: np.ndarray,
) -> np.ndarray:
    """Estimate how long each order will take, for slack computation.

    Without a model this is the analytic estimate a planner would make by hand:
    handling time plus a rough allowance for walking. That is the honest
    baseline the ML estimator has to beat.

    Note the features describe the *order*, not the tour it eventually rides
    in: at the moment slack is computed, nobody knows what it will be batched
    with. Estimating from what is actually known is the point.
    """
    if not orders:
        return np.zeros(0)

    n_lines = np.array([o.total_lines for o in orders], dtype=float)
    units = np.array([float(o.line_units.sum()) for o in orders])
    handling = picking_cfg.per_line_time_s * n_lines + picking_cfg.per_unit_time_s * units
    fixed = picking_cfg.setup_time_s + picking_cfg.drop_off_time_s

    if model is None:
        rough_travel = np.array(
            [
                2.0 * float(np.mean(slot_y[o.line_slot])) + float(np.ptp(slot_x[o.line_slot]))
                for o in orders
            ]
        )
        return fixed + handling + rough_travel / picking_cfg.walking_speed_mps

    features = pd.DataFrame(
        {
            "n_lines": n_lines,
            "total_units": units,
            "total_volume_l": [float(o.line_volume.sum()) for o in orders],
            "n_aisles": [int(np.unique(slot_x[o.line_slot]).size) for o in orders],
            "mean_level": [float(o.line_level.mean()) for o in orders],
            "max_level": [int(o.line_level.max()) for o in orders],
            "hour_of_day": [(o.created_s / 3600.0) % 24.0 for o in orders],
            "day_of_week": [int((o.created_s / 86400.0) % 7) for o in orders],
            "n_orders": 1,
            "route_distance_m": rough_route_distance(orders, slot_x, slot_y),
        }
    )
    try:
        return np.asarray(model.predict(features)).ravel()
    except Exception:  # noqa: BLE001 - fall back rather than abort the run
        return fixed + handling


def rough_route_distance(
    orders: list[PendingOrder], slot_x: np.ndarray, slot_y: np.ndarray
) -> np.ndarray:
    """Cheap stand-in for the tour distance the model was trained on.

    The real route is unknown before batching, so this uses the span of the
    order's own stops. It is an approximation, and the pick-time model is
    correspondingly less accurate here than on the tours it was fitted to --
    a limitation worth naming rather than hiding.
    """
    return np.array(
        [
            2.0 * float(np.mean(slot_y[o.line_slot])) + float(np.ptp(slot_x[o.line_slot]))
            for o in orders
        ]
    )


def _risk_features(
    pending: list[PendingOrder], now_s: float, picking_cfg: PickingConfig
) -> pd.DataFrame:
    queue_lines = sum(len(o.remaining) for o in pending)
    return pd.DataFrame(
        {
            "slack_s": [o.deadline_s - now_s - o.predicted_pick_s for o in pending],
            "predicted_pick_s": [o.predicted_pick_s for o in pending],
            "lines_remaining": [len(o.remaining) for o in pending],
            "total_lines": [o.total_lines for o in pending],
            "age_s": [now_s - o.created_s for o in pending],
            "queue_orders": len(pending),
            "queue_lines": queue_lines,
            "hour_of_day": (now_s / 3600.0) % 24.0,
            "day_of_week": int((now_s / 86400.0) % 7),
        }
    )


def _collect_results(
    all_orders: list[PendingOrder],
    tour_records: list[dict],
    truck_records: list[dict],
    putaway_records: list[dict],
    risk_snapshots: pd.DataFrame,
    sim_cfg: SimulationConfig,
    picking_cfg: PickingConfig,
) -> SimulationResult:
    orders_df = pd.DataFrame(
        [
            {
                "order_id": o.order_id,
                "service_level": o.service_level,
                "created_s": o.created_s,
                "deadline_s": o.deadline_s,
                "promised_departure_s": o.promised_departure_s,
                "first_pick_s": o.first_pick_s,
                "staged_s": o.staged_s,
                "loaded_s": o.loaded_s,
                "shipped_s": o.shipped_s,
                "miss_cause": o.miss_cause,
                "total_lines": o.total_lines,
                "tours_used": o.tours_used,
                "predicted_pick_s": o.predicted_pick_s,
                "original_truck_idx": o.original_truck_idx,
                "shipped_truck_idx": o.shipped_truck_idx,
            }
            for o in all_orders
        ]
    )

    orders_df["in_measurement_window"] = orders_df["created_s"].between(
        sim_cfg.measure_start_s, sim_cfg.measure_end_s
    )
    orders_df["in_training_window"] = orders_df["created_s"] < sim_cfg.training_end_s

    # Two different questions, kept deliberately separate:
    #
    #   on_time      did the order leave on the truck it was promised? This is
    #                the warehouse's own KPI -- the part pickers control.
    #   tardiness_h  how late was it against the promised departure *clock*?
    #                This includes the carrier's own lateness, which the
    #                warehouse cannot control.
    #
    # An order can therefore be on time and still carry tardiness, because it
    # caught the right truck and that truck left the yard late. Collapsing the
    # two would either blame the warehouse for carrier delays or hide them.
    #
    # Note the staging deadline (`deadline_s`) is the operational target that
    # makes the promise achievable, not the promise itself: every truck departs
    # after its own staging cut-off, so scoring ship time against the staging
    # deadline would mark literally every order late.
    orders_df["shipped"] = orders_df["shipped_s"].notna()
    orders_df["on_time"] = orders_df["shipped"] & (
        orders_df["shipped_truck_idx"] == orders_df["original_truck_idx"]
    )
    # Secondary view: did the picking operation itself hit the staging target?
    orders_df["staged_on_time"] = orders_df["staged_s"].notna() & (
        orders_df["staged_s"] <= orders_df["deadline_s"] + 1e-6
    )
    orders_df["tardiness_h"] = (
        (orders_df["shipped_s"] - orders_df["promised_departure_s"]).clip(lower=0.0) / 3600.0
    )
    # How long an order sat on the staging lanes waiting for a loader. This is
    # dead time the picking side cannot see and the loading side owns.
    orders_df["staging_dwell_h"] = (
        (orders_df["loaded_s"] - orders_df["staged_s"]) / 3600.0
    )

    tours_df = pd.DataFrame(tour_records)
    trucks_df = pd.DataFrame(truck_records)
    putaway_df = pd.DataFrame(putaway_records)

    kpis = _compute_kpis(orders_df, tours_df, trucks_df, putaway_df, sim_cfg, picking_cfg)
    return SimulationResult(
        orders_df, tours_df, trucks_df, putaway_df, kpis, risk_snapshots
    )


def _compute_kpis(
    orders_df: pd.DataFrame,
    tours_df: pd.DataFrame,
    trucks_df: pd.DataFrame,
    putaway_df: pd.DataFrame,
    sim_cfg: SimulationConfig,
    picking_cfg: PickingConfig,
) -> dict:
    m = orders_df[orders_df["in_measurement_window"]]
    warmup_s = sim_cfg.measure_start_s
    end_s = sim_cfg.measure_end_s

    kpis: dict[str, float] = {
        "orders_measured": int(len(m)),
        "on_time_rate": float(m["on_time"].mean()) if len(m) else float("nan"),
        "staged_on_time_rate": float(m["staged_on_time"].mean()) if len(m) else float("nan"),
        "shipped_rate": float(m["shipped"].mean()) if len(m) else float("nan"),
        "mean_tardiness_h": float(m.loc[m["shipped"], "tardiness_h"].mean()) if len(m) else float("nan"),
        "p95_tardiness_h": float(m.loc[m["shipped"], "tardiness_h"].quantile(0.95)) if len(m) else float("nan"),
        # Splitting total lateness by cause: an order that caught its truck is
        # only late because the carrier was, and no picking policy can fix that.
        "mean_tardiness_h__made_truck": (
            float(m.loc[m["on_time"], "tardiness_h"].mean()) if m["on_time"].any() else 0.0
        ),
        "mean_tardiness_h__missed_truck": (
            float(m.loc[m["shipped"] & ~m["on_time"], "tardiness_h"].mean())
            if (m["shipped"] & ~m["on_time"]).any()
            else 0.0
        ),
    }

    for level, grp in m.groupby("service_level"):
        kpis[f"on_time_rate__{level}"] = float(grp["on_time"].mean())

    if not tours_df.empty:
        t = tours_df[tours_df["start_s"].between(warmup_s, end_s)]
        if not t.empty:
            kpis["tours"] = int(len(t))
            kpis["total_distance_km"] = float(t["route_distance_m"].sum() / 1000.0)
            kpis["distance_per_line_m"] = float(
                t["route_distance_m"].sum() / max(t["n_lines"].sum(), 1)
            )
            kpis["mean_tour_distance_m"] = float(t["route_distance_m"].mean())
            kpis["mean_orders_per_tour"] = float(t["n_orders"].mean())
            kpis["mean_aisles_per_tour"] = float(t["n_aisles"].mean())
            picker_seconds = float(t["duration_s"].sum())
            available = (
                sim_cfg.evaluation_days
                * (sim_cfg.shift_end_hour - sim_cfg.shift_start_hour)
                * 3600.0
                * picking_cfg.n_pickers
            )
            kpis["picker_utilisation"] = picker_seconds / available
            travel_s = t["route_distance_m"].sum() / picking_cfg.walking_speed_mps
            kpis["travel_share_of_pick_time"] = travel_s / max(picker_seconds, 1.0)

    # --- the two halves of shipping, kept separate ---------------------
    if len(m):
        missed = m[~m["on_time"]]
        kpis["missed_orders"] = int(len(missed))
        for cause in ("picking", "loading"):
            share = (missed["miss_cause"] == cause).mean() if len(missed) else 0.0
            kpis[f"miss_share__{cause}"] = float(share)
        dwell = m["staging_dwell_h"].dropna()
        if len(dwell):
            kpis["mean_staging_dwell_h"] = float(dwell.mean())
            kpis["p95_staging_dwell_h"] = float(dwell.quantile(0.95))

    if not trucks_df.empty:
        tr = trucks_df[trucks_df["planned_departure_s"].between(warmup_s, end_s)]
        if not tr.empty:
            kpis["mean_dock_wait_min"] = float(tr["dock_wait_s"].mean() / 60.0)
            kpis["mean_crew_wait_min"] = float(tr["crew_wait_s"].mean() / 60.0)
            kpis["orders_rolled"] = int(tr["orders_rolled"].sum())
            kpis["rolled_picking"] = int(tr["rolled_picking"].sum())
            kpis["rolled_loading"] = int(tr["rolled_loading"].sum())
            kpis["mean_orders_per_truck"] = float(tr["orders_loaded"].mean())
            kpis["mean_loading_min"] = float(tr["loading_time_s"].mean() / 60.0)
            kpis["loading_window_used_pct"] = float(tr["window_used_pct"].mean())
            loader_seconds = float(tr["loading_time_s"].sum())
            available = (
                sim_cfg.evaluation_days
                * (sim_cfg.shift_end_hour - sim_cfg.shift_start_hour)
                * 3600.0
                * CONFIG.truck.n_load_teams
            )
            kpis["loader_utilisation"] = loader_seconds / max(available, 1.0)

    if not putaway_df.empty:
        p = putaway_df[putaway_df["arrival_s"].between(warmup_s, end_s)]
        if not p.empty:
            kpis["putaway_distance_per_receipt_m"] = float(p["distance_m"].mean())
            kpis["putaway_total_distance_km"] = float(p["distance_m"].sum() / 1000.0)

    return kpis
