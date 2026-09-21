"""Travel-distance model for a multi-block picker-to-parts warehouse.

Pickers cannot walk through racks. They move along *cross aisles* -- transverse
corridors at fixed y positions -- and turn into a picking aisle to reach a
slot. To get from one picking aisle to another they must leave via some cross
aisle, walk across, and come back in:

    same aisle      -> |y1 - y2|
    different aisle -> min over cross aisles c of
                           |y1 - c| + |x1 - x2| + |y2 - c|

Why the number of cross aisles matters
--------------------------------------
With a single cross aisle at y = 0 the second formula collapses to
``y1 + |dx| + y2``, and a whole tour then costs

    2 * sum(y over stops)  +  (one-dimensional traversal in x)

The first term does not depend on the visiting order at all, and the second is
minimised by simply sorting the stops by aisle. In that world S-shape routing
is already optimal and no amount of 2-opt can improve it -- a result that is
easy to verify empirically and easy to mistake for a bug.

Add a second or third cross aisle and the picker gains a choice of exit for
every stop. The cost stops decomposing, the visiting order starts to matter,
and routing becomes a genuine combinatorial problem. Real distribution centres
are built this way, which is precisely why routing heuristics are worth
anything there.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "travel_distance",
    "travel_distance_matrix",
    "distance_to_point",
    "tour_length",
    "DEFAULT_CROSS_AISLES",
]

_EPS = 1e-9

#: A single cross aisle at the dock line -- the degenerate single-block case.
DEFAULT_CROSS_AISLES: tuple[float, ...] = (0.0,)


def _as_cross(cross_aisle_ys) -> np.ndarray:
    if cross_aisle_ys is None:
        cross_aisle_ys = DEFAULT_CROSS_AISLES
    return np.asarray(cross_aisle_ys, dtype=float).ravel()


def travel_distance(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    cross_aisle_ys=None,
    *,
    same_aisle_tol: float = _EPS,
) -> float:
    """Walking distance between two points under the aisle constraint."""
    if abs(x1 - x2) <= same_aisle_tol:
        return abs(y1 - y2)
    cross = _as_cross(cross_aisle_ys)
    return float(np.min(np.abs(y1 - cross) + abs(x1 - x2) + np.abs(y2 - cross)))


def distance_to_point(
    xs: np.ndarray,
    ys: np.ndarray,
    px: float,
    py: float,
    cross_aisle_ys=None,
    *,
    same_aisle_tol: float = _EPS,
) -> np.ndarray:
    """Vectorised distance from many points ``(xs, ys)`` to a single point.

    Used for the dock- and receiving-distance columns, which are computed once
    per slot.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    cross = _as_cross(cross_aisle_ys)

    dx = np.abs(xs - px)
    across = np.min(
        np.abs(ys[:, None] - cross[None, :]) + np.abs(py - cross)[None, :], axis=1
    ) + dx
    within = np.abs(ys - py)
    return np.where(dx <= same_aisle_tol, within, across)


def travel_distance_matrix(
    xs: np.ndarray, ys: np.ndarray, cross_aisle_ys=None, *, same_aisle_tol: float = _EPS
) -> np.ndarray:
    """Full pairwise distance matrix for a set of points.

    Tours in this project are short (a cart holds at most ~45 lines), so an
    O(n^2) matrix is cheap and lets 2-opt run on plain lookups.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    cross = _as_cross(cross_aisle_ys)

    dx = np.abs(xs[:, None] - xs[None, :])
    # For every candidate cross aisle: walk out to it, across, and back in.
    legs = np.abs(ys[:, None] - cross[None, :])            # (n, n_cross)
    across = (legs[:, None, :] + legs[None, :, :]).min(axis=2) + dx
    within = np.abs(ys[:, None] - ys[None, :])
    return np.where(dx <= same_aisle_tol, within, across)


def tour_length(order: np.ndarray, dist: np.ndarray) -> float:
    """Length of a closed tour visiting ``order`` in sequence.

    Index 0 of ``dist`` is assumed to be the depot (dock); ``order`` contains
    the stop indices to visit between leaving and returning to it.
    """
    if len(order) == 0:
        return 0.0
    total = dist[0, order[0]] + dist[order[-1], 0]
    if len(order) > 1:
        total += dist[order[:-1], order[1:]].sum()
    return float(total)
