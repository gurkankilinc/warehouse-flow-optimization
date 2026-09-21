"""Tests for the two optimisation layers: slot assignment and tour routing.

The slotting tests do more than check it runs. They verify the claim the
project rests on -- that the greedy rule is optimal without constraints and
close to optimal with them -- by solving the same instances exactly and
comparing. A heuristic that is asserted to be good and never checked is just a
guess.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.policies.routing import build_distance_matrix, route_nn_2opt, route_s_shape, two_opt
from src.policies.slotting import assign_slots, assignment_cost
from src.utils.geometry import tour_length


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def make_slots(n_aisles=4, n_bays=5, levels=2, cold_aisles=(0,)):
    rows = []
    for a in range(n_aisles):
        for b in range(n_bays):
            for level in range(levels):
                x, y = a * 3.0, 4.0 + b * 1.2
                rows.append(
                    {
                        "slot_id": f"A{a}-B{b}-L{level}",
                        "aisle": a,
                        "x": x,
                        "y": y,
                        "level": level,
                        "dock_distance": x + y,
                        "receiving_distance": x + y,
                        "is_cold": a in cold_aisles,
                        "is_ground": level == 0,
                    }
                )
    return pd.DataFrame(rows)


def make_catalog(n, cold_frac=0.0, heavy_frac=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "sku_id": [f"S{i:03d}" for i in range(n)],
            "is_cold": np.arange(n) < int(n * cold_frac),
            "is_heavy": rng.permutation(np.arange(n) < int(n * heavy_frac)),
        }
    )


# ----------------------------------------------------------------------
# Slot assignment
# ----------------------------------------------------------------------


def test_unconstrained_greedy_matches_the_exact_optimum():
    """The rearrangement inequality says pairing the largest weights with the
    smallest distances is optimal. With no zone constraints, greedy does
    exactly that, so it should tie the Hungarian solver every time."""
    rng = np.random.default_rng(7)
    slots = make_slots(cold_aisles=())
    for trial in range(15):
        catalog = make_catalog(20, seed=trial)
        scores = rng.uniform(0, 100, len(catalog))
        greedy = assign_slots(catalog, slots, scores, exact=False)
        exact = assign_slots(catalog, slots, scores, exact=True)
        assert assignment_cost(greedy, slots) == pytest.approx(
            assignment_cost(exact, slots), rel=1e-9
        )


def test_constrained_greedy_stays_within_one_percent_of_optimal():
    rng = np.random.default_rng(11)
    slots = make_slots(n_aisles=6, n_bays=6, levels=3, cold_aisles=(0, 1))
    gaps = []
    for trial in range(10):
        catalog = make_catalog(60, cold_frac=0.3, heavy_frac=0.2, seed=trial)
        scores = rng.uniform(0, 100, len(catalog))
        greedy = assignment_cost(assign_slots(catalog, slots, scores, exact=False), slots)
        exact = assignment_cost(assign_slots(catalog, slots, scores, exact=True), slots)
        gaps.append((greedy - exact) / exact)
    assert max(gaps) < 0.01, f"worst optimality gap {max(gaps):.4%}"


def test_assignment_is_one_to_one():
    slots = make_slots()
    catalog = make_catalog(20, cold_frac=0.25, heavy_frac=0.25)
    scores = np.arange(len(catalog), dtype=float)
    result = assign_slots(catalog, slots, scores)
    assert result["slot_id"].nunique() == len(catalog)
    assert set(result["sku_id"]) == set(catalog["sku_id"])


def test_zone_constraints_are_respected():
    slots = make_slots(n_aisles=6, n_bays=6, levels=3, cold_aisles=(0, 1))
    catalog = make_catalog(60, cold_frac=0.3, heavy_frac=0.25, seed=3)
    scores = np.random.default_rng(5).uniform(0, 100, len(catalog))
    merged = (
        assign_slots(catalog, slots, scores)
        .merge(catalog, on="sku_id")
        .merge(slots[["slot_id", "is_cold", "is_ground"]], on="slot_id", suffixes=("_sku", "_slot"))
    )
    # Chilled storage is exclusive in both directions.
    assert (merged["is_cold_sku"] == merged["is_cold_slot"]).all()
    # Heavy goods never end up above floor level.
    assert (~merged["is_heavy"] | merged["is_ground"]).all()


def test_heavy_skus_are_not_starved_of_ground_slots():
    """The failure mode plain greedy would hit: light SKUs with high scores
    take every ground slot, and the heavy ones then have nowhere to go."""
    slots = make_slots(n_aisles=2, n_bays=4, levels=2, cold_aisles=())  # 8 ground, 8 upper
    catalog = pd.DataFrame(
        {
            "sku_id": [f"S{i}" for i in range(16)],
            "is_cold": False,
            # The heavy ones are the *least* popular, so a naive greedy would
            # reach them only after the ground level is gone.
            "is_heavy": [False] * 8 + [True] * 8,
        }
    )
    scores = np.arange(16, 0, -1, dtype=float)
    result = assign_slots(catalog, slots, scores).merge(
        slots[["slot_id", "is_ground"]], on="slot_id"
    )
    heavy_rows = result[result["sku_id"].isin(catalog.loc[catalog["is_heavy"], "sku_id"])]
    assert heavy_rows["is_ground"].all()


def test_more_skus_than_slots_is_rejected():
    slots = make_slots(n_aisles=1, n_bays=2, levels=1, cold_aisles=())
    catalog = make_catalog(10)
    with pytest.raises(ValueError):
        assign_slots(catalog, slots, np.ones(len(catalog)))


# ----------------------------------------------------------------------
# Routing
# ----------------------------------------------------------------------


def _random_stops(rng, n, n_aisles=8):
    x = rng.choice(np.arange(n_aisles) * 3.0, n)
    y = rng.uniform(4.0, 30.0, n)
    return x, y


def test_two_opt_never_lengthens_the_tour():
    rng = np.random.default_rng(21)
    for _ in range(40):
        n = int(rng.integers(3, 25))
        x, y = _random_stops(rng, n)
        dist = build_distance_matrix(x, y)
        initial = np.arange(1, n + 1)
        improved = two_opt(initial.copy(), dist)
        assert tour_length(improved, dist) <= tour_length(initial, dist) + 1e-9


def test_nn_2opt_beats_or_matches_s_shape_on_average():
    """S-shape is a genuinely decent heuristic, so the claim is a better mean
    over many tours -- not that it wins every single one."""
    rng = np.random.default_rng(33)
    cross = (0.0, 16.4, 32.8)
    wins = []
    for _ in range(60):
        x, y = _random_stops(rng, int(rng.integers(5, 30)))
        _, s_shape = route_s_shape(x, y, 0.0, cross)
        _, optimised = route_nn_2opt(x, y, 0.0, cross)
        wins.append(optimised <= s_shape + 1e-9)
    assert np.mean(wins) > 0.9


def test_routing_cannot_help_with_a_single_cross_aisle():
    """A structural property worth locking down, because it looks like a bug.

    With one cross aisle every stop must be entered and exited from the same
    end, so tour cost is ``2 * sum(y) + (a 1-D traversal in x)``. The first
    term ignores the visiting order entirely and the second is already solved
    by sorting on aisle, which is exactly what S-shape does. No routing
    heuristic can win, and measuring a gain here would mean the distance model
    is wrong.
    """
    rng = np.random.default_rng(77)
    for _ in range(25):
        x, y = _random_stops(rng, int(rng.integers(8, 30)))
        _, s_shape = route_s_shape(x, y, 0.0, (0.0,))
        _, optimised = route_nn_2opt(x, y, 0.0, (0.0,))
        assert optimised == pytest.approx(s_shape, rel=1e-9)


def test_extra_cross_aisles_shorten_tours():
    """Adding transverse aisles gives the picker shortcuts, so the same stops
    must cost less -- and give routing something to optimise."""
    rng = np.random.default_rng(88)
    single, triple = [], []
    for _ in range(40):
        x, y = _random_stops(rng, 25)
        single.append(route_nn_2opt(x, y, 0.0, (0.0,))[1])
        triple.append(route_nn_2opt(x, y, 0.0, (0.0, 16.4, 32.8))[1])
    assert np.mean(triple) < np.mean(single)


@pytest.mark.parametrize("router", [route_s_shape, route_nn_2opt])
def test_router_visits_every_stop_exactly_once(router):
    rng = np.random.default_rng(44)
    x, y = _random_stops(rng, 17)
    order, _ = router(x, y)
    assert sorted(order.tolist()) == list(range(17))


@pytest.mark.parametrize("router", [route_s_shape, route_nn_2opt])
@pytest.mark.parametrize("n", [0, 1, 2])
def test_router_handles_degenerate_tours(router, n):
    x = np.array([3.0] * n)
    y = np.linspace(5.0, 9.0, n) if n else np.array([])
    order, distance = router(x, y)
    assert len(order) == n
    assert distance >= 0.0
