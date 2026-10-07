"""Wiring: turn raw tables into trained models and ready-to-run policies.

This module exists so that ``models.evaluate`` and ``experiments.run_comparison``
share one definition of how the pieces fit together, rather than each assembling
their own slightly different version.

The order of operations enforces the no-leakage rule:

1. Demand features are built over the whole history.
2. The demand model is trained strictly before the simulation window starts,
   and the forecast that drives slotting is built from that same history
   only -- its demand features are frozen at the cut-off.
3. A baseline simulation runs over the window, logging tour timings and
   risk snapshots.
4. The pick-time and SLA models are fitted on the *training* phase of that run
   only -- they never see the evaluation phase.
5. Scenarios are then evaluated on the phase none of the models has seen.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import CONFIG, PROCESSED_DIR
from src.calibration.fit_parameters import load_calibration
from src.features.build import build_demand_features, build_horizon_features
from src.models.demand_forecast import DemandForecaster, train_final_model
from src.models.pick_time import PickTimeModel, evaluate_pick_time
from src.models.sla_risk import SLARiskModel, build_training_set, evaluate_sla_risk
from src.policies.slotting import abc_slotting, urgency_slotting
from src.simulation.engine import Policies, StochasticStreams, run_simulation
from src.simulation.generate import load_dataset

__all__ = [
    "Context",
    "build_context",
    "build_policies",
    "forecast_visits",
    "run_scenario",
    "load_or_build_context",
    "SCENARIOS",
]

#: Preparing a context trains three models and runs a full simulation, so it is
#: cached for consumers -- notably the dashboard -- that only want to read it.
#: The name is versioned: bump it whenever the way a context is built changes,
#: so no consumer silently keeps reading artefacts from the old code. v2: the
#: slotting forecast no longer sees demand from inside the simulation window.
CONTEXT_CACHE = PROCESSED_DIR / "context_v2.joblib"


SCENARIOS: dict[str, dict[str, str]] = {
    # Shipping splits into two halves, and every scenario states what each half
    # is doing. Placement is `slotting`; evacuation is dispatch + batching +
    # routing on the picking side and `loading` on the vehicle side.
    # name              slotting    dispatch      batching      routing      loading
    "baseline":       dict(slotting="abc",     dispatch="edd",    batching="sequential", routing="s_shape", loading="fifo"),
    "slotting_only":  dict(slotting="urgency", dispatch="edd",    batching="sequential", routing="s_shape", loading="fifo"),
    "dispatch_only":  dict(slotting="abc",     dispatch="triage", batching="sequential", routing="s_shape", loading="fifo"),
    "routing_only":   dict(slotting="abc",     dispatch="edd",    batching="savings",    routing="nn_2opt", loading="fifo"),
    "loading_only":   dict(slotting="abc",     dispatch="edd",    batching="sequential", routing="s_shape", loading="triage"),
    "optimized":      dict(slotting="urgency", dispatch="triage", batching="savings",    routing="nn_2opt", loading="triage"),
    # Reported rather than quietly dropped: adding SLA-risk escalation on top
    # of the optimised bundle makes it *worse*, at every threshold tried. The
    # classifier ranks the bulk of the queue well but is badly calibrated at
    # the top, and escalation spends exactly that top. See the README and
    # `python -m src.experiments.dispatch_study`.
    "risk_escalation": dict(slotting="urgency", dispatch="risk_aware", batching="savings", routing="nn_2opt", loading="triage"),
}

#: Dispatch rules that additionally need a pick-time estimate. ``triage`` works
#: from the analytic estimate alone, which is why ``dispatch_only`` is a pure
#: policy change with no model attached -- the ablation then attributes its
#: gain to the scheduling rule rather than to machine learning.
_NEEDS_PICK_MODEL = {"risk_aware"}
_NEEDS_RISK_MODEL = {"risk_aware"}


@dataclass
class Context:
    """Everything a scenario run needs, prepared once and reused."""

    tables: dict[str, pd.DataFrame]
    window_start_day: int
    start_date: pd.Timestamp
    cutoff_date: pd.Timestamp
    demand_features: pd.DataFrame
    demand_model: DemandForecaster
    pick_time_model: PickTimeModel
    risk_model: SLARiskModel
    pick_time_report: pd.DataFrame
    risk_report: pd.DataFrame
    forecast_visits: pd.Series
    urgency_share: pd.Series
    abc_assignment: pd.DataFrame
    urgency_assignment: pd.DataFrame


def _make_streams(seed: int, n_pickers: int | None = None) -> StochasticStreams:
    sim = CONFIG.simulation
    return StochasticStreams.build(
        horizon_s=(sim.window_days + 2) * 86400.0,
        n_pickers=n_pickers or CONFIG.picking.n_pickers,
        seed=seed,
    )


def forecast_visits(
    daily_demand: pd.DataFrame,
    catalog: pd.DataFrame,
    model: DemandForecaster,
    cutoff_date: pd.Timestamp,
    horizon_days: int,
    units_per_line: float,
) -> pd.Series:
    """Expected pick visits per SKU over the period the slots must serve.

    Built only from demand before ``cutoff_date``: the history features are
    frozen at the cut-off (see ``build_horizon_features``), so the forecast
    knows exactly what the ABC baseline knows -- the past -- and the
    comparison between the two slotting rules is a comparison of how they use
    it. Earlier versions read the horizon rows out of the full-history feature
    frame, whose lags carry the realised demand of the very period being
    slotted for.

    The model predicts units; slotting cares about *visits*, since a picker
    walks to a slot once whether the line is for one unit or fifty. Dividing by
    the calibrated units-per-line converts between the two. It is a constant
    factor and so does not change the ranking, but it keeps the score in the
    unit the objective is written in.
    """
    horizon = build_horizon_features(daily_demand, catalog, cutoff_date, horizon_days)
    predictions = model.predict(horizon)
    units = pd.Series(predictions, index=horizon["sku_id"].to_numpy()).groupby(level=0).sum()
    return units / max(units_per_line, 1e-9)


def _urgency_share(
    orders: pd.DataFrame, order_lines: pd.DataFrame, origin_h: float, training_end_s: float
) -> pd.Series:
    """Fraction of each SKU's picks that sit in same-day orders.

    Computed only over the training phase, so the weight applied to the
    evaluation period is not derived from it.
    """
    created_s = (orders["created_h"] - origin_h) * 3600.0
    train_orders = orders[created_s < training_end_s]

    lines = order_lines.merge(
        train_orders[["order_id", "service_level"]], on="order_id", how="inner"
    )
    if lines.empty:
        return pd.Series(dtype=float)

    total = lines.groupby("sku_id").size()
    same_day = lines[lines["service_level"] == "SAME_DAY"].groupby("sku_id").size()
    return (same_day.reindex(total.index, fill_value=0) / total).astype(float)


def build_context(*, verbose: bool = True) -> Context:
    """Run every preparation step, in the order that keeps the split honest."""
    sim = CONFIG.simulation
    tables = load_dataset()
    meta = tables["_meta"].iloc[0]
    window_start_day = int(meta.window_start_day)
    start_date = pd.Timestamp(meta.start_date)
    cutoff_date = start_date + pd.Timedelta(days=window_start_day)
    origin_h = window_start_day * 24.0

    if verbose:
        print(f"[1/5] building demand features ({len(tables['daily_demand']):,} rows)")
    features = build_demand_features(tables["daily_demand"], tables["sku_catalog"])

    if verbose:
        print(f"[2/5] training demand model on data before {cutoff_date.date()}")
    demand_model = train_final_model(features, cutoff_date)

    if verbose:
        print("[3/5] baseline simulation to generate operational training data")
    abc_assignment = abc_slotting(
        tables["sku_catalog"], tables["warehouse_slots"], tables["daily_demand"], as_of=cutoff_date
    )
    training_run = run_simulation(
        slots=tables["warehouse_slots"],
        catalog=tables["sku_catalog"],
        orders=tables["orders"],
        order_lines=tables["order_lines"],
        receipts=tables["putaway_receipts"],
        truck_schedule=tables["trucks"],
        policies=Policies(slot_assignment=abc_assignment, name="training"),
        streams=_make_streams(CONFIG.simulation.seed + 777),
        window_start_day=window_start_day,
        log_risk_snapshots=True,
    )

    if verbose:
        print(f"[4/5] fitting operational models "
              f"({len(training_run.tours):,} tours, {len(training_run.risk_snapshots):,} snapshots)")
    pick_time_model, pick_time_report = evaluate_pick_time(
        training_run.tours, train_end_s=sim.training_end_s
    )
    risk_data = build_training_set(
        training_run.risk_snapshots, training_run.orders, train_end_s=sim.training_end_s
    )
    risk_model, risk_report = evaluate_sla_risk(risk_data)

    if verbose:
        print("[5/5] computing forecast-driven slot assignment")
    visits = forecast_visits(
        tables["daily_demand"],
        tables["sku_catalog"],
        demand_model,
        cutoff_date,
        sim.window_days,
        load_calibration().units_per_line_mean,
    )
    urgency_share = _urgency_share(
        tables["orders"], tables["order_lines"], origin_h, sim.training_end_s
    )
    urgency_assignment = urgency_slotting(
        tables["sku_catalog"], tables["warehouse_slots"], visits, urgency_share
    )

    return Context(
        tables=tables,
        window_start_day=window_start_day,
        start_date=start_date,
        cutoff_date=cutoff_date,
        demand_features=features,
        demand_model=demand_model,
        pick_time_model=pick_time_model,
        risk_model=risk_model,
        pick_time_report=pick_time_report,
        risk_report=risk_report,
        forecast_visits=visits,
        urgency_share=urgency_share,
        abc_assignment=abc_assignment,
        urgency_assignment=urgency_assignment,
    )


def save_context(context: Context, path=CONTEXT_CACHE) -> None:
    """Cache the expensive parts: the trained models and the slot assignments.

    The raw tables and the demand feature frame are deliberately not stored --
    they already live in ``data/processed`` or are rebuilt in seconds, and
    including them made the cache sixty times larger for no benefit.
    """
    import joblib

    joblib.dump(
        dataclasses.replace(context, tables={}, demand_features=pd.DataFrame()), path
    )


def load_or_build_context(*, refresh: bool = False, verbose: bool = True) -> Context:
    """Load the cached context, or build and cache it if there is none."""
    import joblib

    if not refresh and CONTEXT_CACHE.exists():
        try:
            cached = joblib.load(CONTEXT_CACHE)
            if not cached.tables:
                cached.tables = load_dataset()
            return cached
        except Exception:  # noqa: BLE001 - a stale cache should never be fatal
            pass

    context = build_context(verbose=verbose)
    save_context(context)
    return context


def build_policies(context: Context, scenario: str) -> Policies:
    spec = SCENARIOS[scenario]
    assignment = (
        context.urgency_assignment if spec["slotting"] == "urgency" else context.abc_assignment
    )
    return Policies(
        slot_assignment=assignment,
        dispatch=spec["dispatch"],
        batching=spec["batching"],
        routing=spec["routing"],
        loading=spec.get("loading", "fifo"),
        pick_time_model=(
            context.pick_time_model if spec["dispatch"] in _NEEDS_PICK_MODEL else None
        ),
        risk_model=context.risk_model if spec["dispatch"] in _NEEDS_RISK_MODEL else None,
        name=scenario,
    )


def run_scenario(
    context: Context,
    scenario: str,
    *,
    seed_offset: int = 0,
    picking_cfg=None,
    truck_schedule: pd.DataFrame | None = None,
    orders: pd.DataFrame | None = None,
    order_lines: pd.DataFrame | None = None,
):
    """Run one scenario. Replications differ only by the stochastic streams.

    The optional overrides exist for the stress tests, which perturb the world
    (heavier demand, later trucks, fewer pickers) while keeping the *models and
    slot assignment fitted under normal conditions*. That mismatch is the point:
    it is what happens to a real deployment when conditions move.
    """
    policies = build_policies(context, scenario)
    picking_cfg = picking_cfg or CONFIG.picking
    return run_simulation(
        slots=context.tables["warehouse_slots"],
        catalog=context.tables["sku_catalog"],
        orders=context.tables["orders"] if orders is None else orders,
        order_lines=(
            context.tables["order_lines"] if order_lines is None else order_lines
        ),
        receipts=context.tables["putaway_receipts"],
        truck_schedule=context.tables["trucks"] if truck_schedule is None else truck_schedule,
        policies=policies,
        streams=_make_streams(CONFIG.simulation.seed + 777 + seed_offset, picking_cfg.n_pickers),
        window_start_day=context.window_start_day,
        picking_cfg=picking_cfg,
    )
