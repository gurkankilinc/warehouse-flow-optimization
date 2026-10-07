"""Slot assignment -- deciding which SKU lives where.

The objective is the standard storage-location one: minimise

    sum over SKUs of  (expected visits) x (walking distance of its slot)

Dock distance is used as the per-visit cost proxy. That is an approximation --
in a multi-stop tour the marginal cost of one extra stop is not exactly twice
its dock distance -- but it is the approximation the classic COI / ABC slotting
rules are built on, and the discrete-event simulation then measures what it
actually costs. So the heuristic proposes and the simulator judges.

Two policies:

``abc_slotting``      baseline. Rank SKUs by observed demand over a lookback
                      window, best slots to the fastest movers. This is what
                      real warehouses do, and it is a strong baseline.

``urgency_slotting``  the optimised policy. Rank by *forecast* demand weighted
                      by how often the SKU shows up in urgent orders, so a
                      slow mover that always ships same-day can outrank a fast
                      mover that never does.

Both respect the physical constraints (cold chain, ground level for heavy
SKUs), and both are solved by the same assignment routine so the comparison
isolates the scoring rule rather than the solver.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CONFIG, PolicyConfig

__all__ = [
    "assign_slots",
    "abc_slotting",
    "urgency_slotting",
    "abc_xyz_classes",
    "assignment_cost",
]

_INFEASIBLE = 1e9


# ----------------------------------------------------------------------
# Core assignment
# ----------------------------------------------------------------------


def _feasibility_mask(catalog: pd.DataFrame, slots: pd.DataFrame) -> np.ndarray:
    """``mask[i, j]`` is True when SKU i may be stored in slot j.

    The chilled zone is *exclusive*: cold SKUs must be in it and ambient SKUs
    must be out of it. Refrigerated space is expensive and you do not store
    t-shirts in it -- and, less obviously, without exclusivity the ambient
    fast movers would occupy the chilled aisles (which happen to sit nearest
    the dock) and leave the cold SKUs with nowhere to go.

    Heavy SKUs are restricted to ground level, which is a strict subset of
    what everything else can use.
    """
    sku_cold = catalog["is_cold"].to_numpy()[:, None]
    sku_heavy = catalog["is_heavy"].to_numpy()[:, None]
    slot_cold = slots["is_cold"].to_numpy()[None, :]
    slot_ground = slots["is_ground"].to_numpy()[None, :]

    ok_cold = sku_cold == slot_cold
    ok_heavy = ~sku_heavy | slot_ground
    return ok_cold & ok_heavy


def assign_slots(
    catalog: pd.DataFrame,
    slots: pd.DataFrame,
    scores: np.ndarray,
    *,
    exact: bool | None = None,
) -> pd.DataFrame:
    """Assign each SKU to one slot, minimising ``sum(score * dock_distance)``.

    Parameters
    ----------
    scores : per-SKU cost weight (expected visits, possibly urgency-adjusted).
    exact  : solve with the Hungarian algorithm instead of the greedy rule.
             Greedy is optimal when every SKU may use every slot (the
             rearrangement inequality: pairing the largest weights with the
             smallest distances minimises the sum) and near-optimal under the
             zone constraints. The exact solver is kept so tests can quantify
             that gap rather than assume it away.

    Returns a frame with columns ``sku_id``, ``slot_id``, ``score``.
    """
    exact = CONFIG.policy.use_exact_assignment if exact is None else exact
    if len(catalog) > len(slots):
        raise ValueError(f"{len(catalog)} SKUs do not fit in {len(slots)} slots")

    feasible = _feasibility_mask(catalog, slots)
    distance = slots["dock_distance"].to_numpy()

    if exact:
        from scipy.optimize import linear_sum_assignment

        cost = scores[:, None] * distance[None, :]
        cost = np.where(feasible, cost, _INFEASIBLE)
        rows, cols = linear_sum_assignment(cost)
        chosen = np.empty(len(catalog), dtype=int)
        chosen[rows] = cols
    else:
        chosen = _assign_greedy(catalog, slots, scores)

    return pd.DataFrame(
        {
            "sku_id": catalog["sku_id"].to_numpy(),
            "slot_id": slots["slot_id"].to_numpy()[chosen],
            "score": scores,
        }
    )


def _assign_greedy(
    catalog: pd.DataFrame, slots: pd.DataFrame, scores: np.ndarray
) -> np.ndarray:
    """Highest-scoring SKU takes the closest slot it is allowed to occupy.

    Because the chilled zone is exclusive, the problem splits into two
    independent subproblems (cold and ambient) that are solved separately.

    Within a zone the only remaining constraint is that heavy SKUs need ground
    level. Plain greedy could starve them -- light SKUs would take every ground
    slot before the solver reaches a heavy one -- so a light SKU is allowed a
    ground slot only while free ground slots still outnumber the heavy SKUs
    that have yet to be placed. That keeps the rule greedy and optimal in the
    unconstrained direction while making starvation impossible.
    """
    sku_cold = catalog["is_cold"].to_numpy()
    sku_heavy = catalog["is_heavy"].to_numpy()
    slot_cold = slots["is_cold"].to_numpy()
    slot_ground = slots["is_ground"].to_numpy()
    distance = slots["dock_distance"].to_numpy()

    chosen = np.full(len(catalog), -1, dtype=int)

    for zone_is_cold in (True, False):
        zone_skus = np.flatnonzero(sku_cold == zone_is_cold)
        zone_slots = np.flatnonzero(slot_cold == zone_is_cold)
        if zone_skus.size == 0:
            continue
        if zone_skus.size > zone_slots.size:
            zone = "chilled" if zone_is_cold else "ambient"
            raise ValueError(
                f"{zone} zone holds {zone_slots.size} slots but needs {zone_skus.size}"
            )

        slot_order = zone_slots[np.argsort(distance[zone_slots], kind="stable")]
        sku_order = zone_skus[np.argsort(-scores[zone_skus], kind="stable")]

        heavy_remaining = int(sku_heavy[zone_skus].sum())
        free_ground = int(slot_ground[slot_order].sum())
        taken = np.zeros(slot_order.size, dtype=bool)
        cursor = 0

        for sku in sku_order:
            needs_ground = bool(sku_heavy[sku])
            j = cursor
            slot_pos = -1
            while j < slot_order.size:
                if not taken[j]:
                    slot = slot_order[j]
                    is_ground = bool(slot_ground[slot])
                    if needs_ground:
                        if is_ground:
                            slot_pos = j
                            break
                    elif not is_ground or free_ground > heavy_remaining:
                        slot_pos = j
                        break
                j += 1

            if slot_pos < 0:
                raise ValueError("no feasible slot remains; check zone capacities")

            slot = slot_order[slot_pos]
            taken[slot_pos] = True
            chosen[sku] = slot
            if slot_ground[slot]:
                free_ground -= 1
            if needs_ground:
                heavy_remaining -= 1
            while cursor < slot_order.size and taken[cursor]:
                cursor += 1

    return chosen


def assignment_cost(
    assignment: pd.DataFrame, slots: pd.DataFrame, scores: np.ndarray | None = None
) -> float:
    """Objective value of an assignment, for comparing solvers."""
    dist = slots.set_index("slot_id")["dock_distance"]
    weights = assignment["score"].to_numpy() if scores is None else scores
    return float((weights * assignment["slot_id"].map(dist).to_numpy()).sum())


# ----------------------------------------------------------------------
# Scoring rules
# ----------------------------------------------------------------------


def abc_slotting(
    catalog: pd.DataFrame,
    slots: pd.DataFrame,
    daily_demand: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
    policy_cfg: PolicyConfig | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Baseline: rank purely by demand observed over a trailing window.

    This is deliberately backward-looking. It is what a warehouse without a
    forecasting capability does, and it is the thing the optimised policy has
    to beat -- not a random layout.
    """
    policy_cfg = policy_cfg or CONFIG.policy

    history = daily_demand
    if as_of is not None:
        dates = pd.to_datetime(history["date"])
        start = as_of - pd.Timedelta(days=policy_cfg.abc_lookback_days)
        history = history[(dates >= start) & (dates < as_of)]

    totals = history.groupby("sku_id")["demand"].sum()
    scores = totals.reindex(catalog["sku_id"], fill_value=0).to_numpy(dtype=float)
    return assign_slots(catalog, slots, scores, **kwargs)


def urgency_slotting(
    catalog: pd.DataFrame,
    slots: pd.DataFrame,
    forecast_visits: pd.Series,
    urgency_share: pd.Series,
    policy_cfg: PolicyConfig | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Optimised: forecast pick frequency, weighted by urgency exposure.

    ``forecast_visits`` comes from the demand model rather than from a plain
    trailing total, so the slot layout is aimed at the period the slots will
    serve rather than the weeks behind. It is built from the same information
    the ABC baseline has -- demand before the cut-off -- so the two rules
    differ in how they use the past, not in what they are allowed to see.
    ``urgency_share`` is the fraction of a SKU's picks that sit in same-day
    orders.
    """
    policy_cfg = policy_cfg or CONFIG.policy

    visits = forecast_visits.reindex(catalog["sku_id"], fill_value=0.0).to_numpy(float)
    share = urgency_share.reindex(catalog["sku_id"], fill_value=0.0).to_numpy(float)

    scores = visits * (1.0 + policy_cfg.urgency_beta * share)
    return assign_slots(catalog, slots, scores, **kwargs)


def abc_xyz_classes(
    daily_demand: pd.DataFrame, catalog: pd.DataFrame
) -> pd.DataFrame:
    """Classic ABC (volume) x XYZ (predictability) segmentation.

    ABC splits SKUs by cumulative share of demand (80 / 95 percent breaks).
    XYZ splits them by coefficient of variation: X is steady and easy to
    forecast, Z is erratic. The combination is what a planner actually reasons
    with -- an AZ item is high volume but unpredictable, and needs different
    treatment from an AX item.

    The variability is measured on **weekly** buckets, not daily ones. Daily
    SKU demand is intermittent almost by definition, so a daily CV puts nearly
    every SKU in Z and the classification says nothing. Weekly is the bucket a
    replenishment planner actually works in.
    """
    per_sku = daily_demand.groupby("sku_id")["demand"].agg(["sum"])
    per_sku = per_sku.reindex(catalog["sku_id"]).fillna(0.0)

    ranked = per_sku.sort_values("sum", ascending=False)
    cumulative = ranked["sum"].cumsum() / max(ranked["sum"].sum(), 1e-9)
    abc = pd.Series(
        np.where(cumulative <= 0.80, "A", np.where(cumulative <= 0.95, "B", "C")),
        index=ranked.index,
    )

    weekly = (
        daily_demand.assign(week=pd.to_datetime(daily_demand["date"]).dt.to_period("W"))
        .groupby(["sku_id", "week"])["demand"]
        .sum()
        .groupby("sku_id")
        .agg(["mean", "std"])
        .reindex(catalog["sku_id"])
        .fillna(0.0)
    )
    cv = (weekly["std"] / weekly["mean"].replace(0, np.nan)).fillna(np.inf)
    xyz = pd.Series(
        np.where(cv <= 0.5, "X", np.where(cv <= 1.0, "Y", "Z")), index=per_sku.index
    )

    return pd.DataFrame(
        {
            "sku_id": per_sku.index,
            "total_demand": per_sku["sum"].to_numpy(),
            "weekly_cv": cv.reindex(per_sku.index).to_numpy(),
            "abc_class": abc.reindex(per_sku.index).to_numpy(),
            "xyz_class": xyz.reindex(per_sku.index).to_numpy(),
        }
    ).reset_index(drop=True)
