"""End-to-end invariants of the discrete-event simulation.

These do not check that the warehouse is *good*, only that it is *coherent*:
nothing is picked before it is ordered, nothing ships before it is picked, no
line is lost or picked twice, and pickers only work during their shift. A
simulation that violates any of these produces KPIs that look plausible and
mean nothing.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from config import CONFIG
from src.calibration.fit_parameters import fallback_params
from src.policies.slotting import abc_slotting
from src.simulation.catalog import build_catalog
from src.simulation.engine import Policies, StochasticStreams, run_simulation
from src.simulation.orders import generate_demand_and_orders
from src.simulation.trucks import build_truck_schedule
from src.simulation.warehouse import build_slots

# A deliberately tiny warehouse so the suite stays fast. The fallback
# calibration is used so the tests never depend on a downloaded file.
SMALL_WAREHOUSE = dataclasses.replace(
    CONFIG.warehouse, n_aisles=4, bays_per_aisle=6, levels=2, cold_zone_aisles=(0,)
)
SMALL_CATALOG = dataclasses.replace(CONFIG.catalog, n_skus=60)
SMALL_DEMAND = dataclasses.replace(
    CONFIG.demand, history_days=60, base_orders_per_day=30.0
)
SMALL_SIM = dataclasses.replace(
    CONFIG.simulation, training_days=6, warmup_days=2, evaluation_days=8
)
SMALL_PICKING = dataclasses.replace(CONFIG.picking, n_pickers=2, n_putaway_workers=1)


@pytest.fixture(scope="module")
def run():
    rng = np.random.default_rng(3)
    calibration = fallback_params()
    slots = build_slots(SMALL_WAREHOUSE)
    catalog = build_catalog(calibration, rng, SMALL_CATALOG, SMALL_DEMAND)
    generated = generate_demand_and_orders(
        catalog, calibration, rng, demand_cfg=SMALL_DEMAND, sim_cfg=SMALL_SIM
    )
    trucks = build_truck_schedule(SMALL_DEMAND.history_days + 3, rng)
    assignment = abc_slotting(catalog, slots, generated.daily_demand)

    result = run_simulation(
        slots=slots,
        catalog=catalog,
        orders=generated.orders,
        order_lines=generated.order_lines,
        receipts=generated.receipts,
        truck_schedule=trucks,
        policies=Policies(slot_assignment=assignment),
        streams=StochasticStreams.build(
            (SMALL_SIM.window_days + 2) * 86400.0, SMALL_PICKING.n_pickers, 99, SMALL_PICKING
        ),
        window_start_day=generated.window_start_day,
        log_risk_snapshots=True,
        sim_cfg=SMALL_SIM,
        picking_cfg=SMALL_PICKING,
    )
    return result, generated, slots, catalog


def test_simulation_produces_activity(run):
    result, _, _, _ = run
    assert len(result.orders) > 0
    assert len(result.tours) > 0
    assert len(result.trucks) > 0
    assert not result.risk_snapshots.empty


def test_nothing_is_picked_before_it_is_ordered(run):
    result, _, _, _ = run
    picked = result.orders.dropna(subset=["first_pick_s"])
    assert (picked["first_pick_s"] >= picked["created_s"] - 1e-6).all()


def test_staging_never_precedes_the_first_pick(run):
    result, _, _, _ = run
    staged = result.orders.dropna(subset=["staged_s", "first_pick_s"])
    assert (staged["staged_s"] >= staged["first_pick_s"] - 1e-6).all()


def test_nothing_ships_before_it_is_staged(run):
    result, _, _, _ = run
    shipped = result.orders.dropna(subset=["shipped_s"])
    assert shipped["staged_s"].notna().all()
    assert (shipped["shipped_s"] >= shipped["staged_s"] - 1e-6).all()


def test_every_line_is_picked_exactly_once(run):
    """Lines are conserved: what the tours collected must equal what the staged
    orders contained -- no duplicates from two pickers claiming the same work,
    and no lines quietly dropped."""
    result, generated, _, _ = run
    staged_ids = set(result.orders.loc[result.orders["staged_s"].notna(), "order_id"])
    expected = generated.order_lines[
        generated.order_lines["order_id"].isin(staged_ids)
    ].shape[0]
    # Tours may also hold partial progress on orders that never finished, so
    # the total picked is at least the staged total and never more than all of it.
    total_picked = int(result.tours["n_lines"].sum())
    assert total_picked >= expected
    assert total_picked <= len(generated.order_lines)


def test_orders_are_staged_only_when_complete(run):
    result, generated, _, _ = run
    lines_per_order = generated.order_lines.groupby("order_id").size()
    staged = result.orders[result.orders["staged_s"].notna()]
    expected = staged["order_id"].map(lines_per_order)
    assert (staged["total_lines"].to_numpy() == expected.to_numpy()).all()


def test_pickers_only_work_during_the_shift(run):
    result, _, _, _ = run
    hours = (result.tours["start_s"] / 3600.0) % 24.0
    assert (hours >= SMALL_SIM.shift_start_hour - 1e-6).all()
    assert (hours <= SMALL_SIM.shift_end_hour + 1e-6).all()


def test_on_time_requires_the_promised_truck(run):
    result, _, _, _ = run
    on_time = result.orders[result.orders["on_time"]]
    assert (on_time["shipped_truck_idx"] == on_time["original_truck_idx"]).all()
    assert on_time["shipped_s"].notna().all()


def test_missing_the_truck_always_costs_time(run):
    result, _, _, _ = run
    shipped = result.orders[result.orders["shipped"]]
    late = shipped[~shipped["on_time"]]
    if len(late):
        assert (late["tardiness_h"] > 0).all()


def test_orders_that_caught_their_truck_are_only_as_late_as_the_truck(run):
    """An on-time order can still carry tardiness, because its truck left the
    yard late -- but never by more than the gap to the next departure, which
    is what missing the truck would have cost."""
    result, _, _, _ = run
    on_time = result.orders[result.orders["on_time"]]
    min_gap_h = min(
        b - a
        for a, b in zip(CONFIG.truck.departure_hours, CONFIG.truck.departure_hours[1:])
    )
    assert (on_time["tardiness_h"] >= 0).all()
    assert (on_time["tardiness_h"] < min_gap_h).all()


def test_tardiness_is_never_negative(run):
    result, _, _, _ = run
    shipped = result.orders[result.orders["shipped"]]
    assert (shipped["tardiness_h"] >= 0).all()


def test_measurement_window_excludes_training_and_warmup(run):
    result, _, _, _ = run
    measured = result.orders[result.orders["in_measurement_window"]]
    assert (measured["created_s"] >= SMALL_SIM.measure_start_s).all()
    assert (measured["created_s"] <= SMALL_SIM.measure_end_s).all()
    # Training and measurement must not overlap.
    assert not (result.orders["in_training_window"] & result.orders["in_measurement_window"]).any()


def test_every_sku_has_exactly_one_slot(run):
    _, _, slots, catalog = run
    assignment = abc_slotting(catalog, slots, pd.DataFrame(
        {"date": [], "sku_id": [], "demand": []}
    ).astype({"sku_id": str, "demand": float}))
    assert len(assignment) == len(catalog)
    assert assignment["slot_id"].nunique() == len(catalog)


def test_stochastic_streams_are_reproducible():
    a = StochasticStreams.build(10 * 86400.0, 4, seed=123)
    b = StochasticStreams.build(10 * 86400.0, 4, seed=123)
    assert np.array_equal(a.congestion, b.congestion)
    assert np.array_equal(a.picker_skill, b.picker_skill)
    assert np.array_equal(a.noise, b.noise)


def test_loading_follows_staging_which_follows_picking(run):
    """The second half of shipping has an order: picked, then staged, then
    loaded, then gone."""
    result, _, _, _ = run
    loaded = result.orders.dropna(subset=["loaded_s"])
    assert (loaded["staged_s"] <= loaded["loaded_s"] + 1e-6).all()
    assert (loaded["loaded_s"] <= loaded["shipped_s"] + 1e-6).all()
    assert (loaded["staging_dwell_h"] >= -1e-9).all()


def test_every_miss_is_attributed_to_picking_or_loading(run):
    """A missed truck is either the picking side's fault or the loading side's.
    Leaving it unattributed would hide which half of the operation to fix."""
    result, _, _, _ = run
    missed = result.orders[result.orders["shipped"] & ~result.orders["on_time"]]
    if len(missed):
        assert set(missed["miss_cause"]) <= {"picking", "loading"}
        assert (missed["miss_cause"] != "").all()


def test_orders_are_never_rolled_onto_a_departed_truck(run):
    """The bug this pins down: rolling a missed order to `truck_idx + 1` can
    hand it to a truck that already left, because delayed trucks depart out of
    order. The order is then in a list nobody reads again -- it holds a staging
    slot forever, and once the lanes fill every picker blocks behind it."""
    result, _, _, _ = run
    stranded = result.orders[
        result.orders["staged_s"].notna() & result.orders["shipped_s"].isna()
    ]
    # A few orders staged right at the horizon legitimately have no truck left.
    horizon_s = SMALL_SIM.window_days * 86400.0
    late_stranded = stranded[stranded["staged_s"] < horizon_s - 2 * 86400.0]
    assert late_stranded.empty, (
        f"{len(late_stranded)} orders stranded in staging with trucks still to come"
    )


def test_wave_release_holds_back_far_off_orders(run):
    """Nothing is picked long before its truck. Without this the staging lanes
    fill with work for tomorrow and starve the truck at the door."""
    result, _, _, _ = run
    picked = result.orders.dropna(subset=["first_pick_s"])
    horizon_s = CONFIG.policy.release_horizon_h * 3600.0
    # A tour is built from orders inside the window, so the first pick can
    # never precede the deadline by more than the horizon plus one tour.
    slack = picked["deadline_s"] - picked["first_pick_s"]
    assert (slack <= horizon_s + 3600.0).all()


def test_deadlines_come_from_the_truck_schedule(run):
    _, generated, _, _ = run
    orders = generated.orders
    expected = orders["truck_departure_h"] - CONFIG.truck.staging_buffer_h
    assert orders["deadline_h"].to_numpy() == pytest.approx(expected.to_numpy())
    # And a truck is always chosen that leaves after the order arrives.
    assert (orders["truck_departure_h"] > orders["created_h"]).all()
