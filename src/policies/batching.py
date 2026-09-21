"""Order batching: which orders share a picking tour.

A picker walking to the far end of aisle 9 for order A may as well collect
order B's line in the same aisle. That saving is the whole point of batching.
Two strategies:

``batch_sequential``
    Fill the cart with orders in dispatch order until a limit binds. This is
    the baseline, and it is not a strawman -- since dispatch already sorts by
    urgency, sequential filling naturally groups orders with similar deadlines.

``batch_savings``
    Keep the first (most urgent) order as the seed, then repeatedly add
    whichever candidate shares the most aisles with what is already on the
    cart. This is the warehouse analogue of the Clarke-Wright savings idea:
    value an addition by the travel it avoids, not by its own size.

Both respect a deadline-gap limit, so a relaxed order is never allowed to ride
along with an urgent one and slow it down.

Large orders are handled by *splitting*: an order whose lines exceed a cart is
picked over several tours and is only staged when its last line is collected.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from config import CONFIG, PickingConfig, PolicyConfig

__all__ = ["BATCHING_RULES", "batch_sequential", "batch_savings"]


def _take_lines(order, capacity_lines: int, capacity_volume: float) -> list[int]:
    """Indices of the order's remaining lines that fit in the space left."""
    taken: list[int] = []
    volume = 0.0
    for i, line_idx in enumerate(order.remaining):
        if len(taken) >= capacity_lines:
            break
        line_volume = order.line_volume[line_idx]
        if taken and volume + line_volume > capacity_volume:
            break
        taken.append(line_idx)
        volume += line_volume
    return taken


def batch_sequential(
    ordered: Sequence,
    picking_cfg: PickingConfig | None = None,
    policy_cfg: PolicyConfig | None = None,
    slot_aisle: np.ndarray | None = None,
) -> list[tuple]:
    """Fill the cart with orders in the sequence dispatch handed over."""
    picking_cfg = picking_cfg or CONFIG.picking
    policy_cfg = policy_cfg or CONFIG.policy

    if not ordered:
        return []

    seed = ordered[0]
    lines_left = picking_cfg.max_lines_per_tour
    volume_left = picking_cfg.max_volume_per_tour_l

    tour: list[tuple] = []
    for order in ordered:
        if lines_left <= 0:
            break
        if abs(order.deadline_s - seed.deadline_s) > policy_cfg.batching_max_deadline_gap_h * 3600:
            continue
        picked = _take_lines(order, lines_left, volume_left)
        if not picked:
            continue
        tour.append((order, picked))
        lines_left -= len(picked)
        volume_left -= sum(order.line_volume[i] for i in picked)

    return tour


def batch_savings(
    ordered: Sequence,
    picking_cfg: PickingConfig | None = None,
    policy_cfg: PolicyConfig | None = None,
    slot_aisle: np.ndarray | None = None,
) -> list[tuple]:
    """Grow the tour by aisle overlap rather than by queue position.

    ``slot_aisle`` maps a slot index to its aisle, so overlap can be measured
    in the unit that actually costs walking: an aisle entered.
    """
    picking_cfg = picking_cfg or CONFIG.picking
    policy_cfg = policy_cfg or CONFIG.policy

    if not ordered:
        return []
    if slot_aisle is None:
        return batch_sequential(ordered, picking_cfg, policy_cfg)

    seed = ordered[0]
    lines_left = picking_cfg.max_lines_per_tour
    volume_left = picking_cfg.max_volume_per_tour_l

    picked = _take_lines(seed, lines_left, volume_left)
    tour = [(seed, picked)]
    lines_left -= len(picked)
    volume_left -= sum(seed.line_volume[i] for i in picked)
    aisles_on_cart = {slot_aisle[seed.line_slot[i]] for i in picked}

    candidates = [
        o
        for o in ordered[1 : 1 + policy_cfg.batching_candidate_pool]
        if abs(o.deadline_s - seed.deadline_s) <= policy_cfg.batching_max_deadline_gap_h * 3600
    ]

    while lines_left > 0 and candidates:
        best, best_lines, best_score = None, None, -1.0
        for cand in candidates:
            cand_lines = _take_lines(cand, lines_left, volume_left)
            if not cand_lines:
                continue
            cand_aisles = {slot_aisle[cand.line_slot[i]] for i in cand_lines}
            shared = len(cand_aisles & aisles_on_cart)
            new = len(cand_aisles - aisles_on_cart)
            # Reward overlap, penalise every fresh aisle the picker must enter.
            score = (shared - new) / len(cand_aisles)
            if score > best_score:
                best, best_lines, best_score = cand, cand_lines, score

        if best is None or best_score <= -1.0:
            break

        tour.append((best, best_lines))
        lines_left -= len(best_lines)
        volume_left -= sum(best.line_volume[i] for i in best_lines)
        aisles_on_cart |= {slot_aisle[best.line_slot[i]] for i in best_lines}
        candidates.remove(best)

    return tour


BATCHING_RULES: dict[str, Callable] = {
    "sequential": batch_sequential,
    "savings": batch_savings,
}
