"""Pick-tour routing.

Two routes are implemented because the project's whole claim rests on beating a
*competent* baseline, not a strawman:

``route_s_shape``
    The serpentine traversal used in real warehouses: visit aisles in order,
    alternating direction so the picker never doubles back across the block.
    It is simple, needs no computation, and is genuinely hard to beat by much.

``route_nn_2opt``
    Nearest-neighbour construction followed by 2-opt improvement. Tours here
    hold at most ~45 stops, so an exact-ish local search is affordable and
    there is no need for heavier machinery.

Both return the visiting sequence *and* the exact distance under the aisle
travel model, so the comparison between them is apples to apples.
"""

from __future__ import annotations

import numpy as np

from src.utils.geometry import tour_length, travel_distance_matrix

__all__ = ["build_distance_matrix", "route_s_shape", "route_nn_2opt", "two_opt"]


def build_distance_matrix(
    stop_x: np.ndarray, stop_y: np.ndarray, dock_x: float = 0.0, cross_aisle_ys=None
) -> np.ndarray:
    """Distance matrix with the dock as index 0 and stops as 1..n."""
    xs = np.concatenate([[dock_x], np.asarray(stop_x, dtype=float)])
    ys = np.concatenate([[0.0], np.asarray(stop_y, dtype=float)])
    return travel_distance_matrix(xs, ys, cross_aisle_ys)


def route_s_shape(
    stop_x: np.ndarray, stop_y: np.ndarray, dock_x: float = 0.0, cross_aisle_ys=None
) -> tuple[np.ndarray, float]:
    """Serpentine traversal: aisles in ascending order, alternating direction."""
    n = len(stop_x)
    if n == 0:
        return np.array([], dtype=int), 0.0

    dist = build_distance_matrix(stop_x, stop_y, dock_x, cross_aisle_ys)
    stop_x = np.asarray(stop_x, dtype=float)
    stop_y = np.asarray(stop_y, dtype=float)

    order: list[int] = []
    for pass_index, aisle_x in enumerate(np.unique(stop_x)):
        in_aisle = np.flatnonzero(stop_x == aisle_x)
        # Alternate up/down so consecutive aisles are entered from the end the
        # picker already stands at.
        ascending = pass_index % 2 == 0
        in_aisle = in_aisle[np.argsort(stop_y[in_aisle], kind="stable")]
        if not ascending:
            in_aisle = in_aisle[::-1]
        order.extend(int(i) + 1 for i in in_aisle)  # +1: index 0 is the dock

    seq = np.asarray(order, dtype=int)
    return seq - 1, tour_length(seq, dist)


def route_nn_2opt(
    stop_x: np.ndarray, stop_y: np.ndarray, dock_x: float = 0.0, cross_aisle_ys=None
) -> tuple[np.ndarray, float]:
    """Nearest-neighbour construction improved by 2-opt."""
    n = len(stop_x)
    if n == 0:
        return np.array([], dtype=int), 0.0
    if n == 1:
        dist = build_distance_matrix(stop_x, stop_y, dock_x, cross_aisle_ys)
        return np.array([0]), tour_length(np.array([1]), dist)

    dist = build_distance_matrix(stop_x, stop_y, dock_x, cross_aisle_ys)

    # --- nearest neighbour from the dock ------------------------------
    unvisited = set(range(1, n + 1))
    current = 0
    seq: list[int] = []
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist[current, j])
        seq.append(nxt)
        unvisited.remove(nxt)
        current = nxt

    improved_seq = two_opt(np.asarray(seq, dtype=int), dist)
    return improved_seq - 1, tour_length(improved_seq, dist)


def two_opt(seq: np.ndarray, dist: np.ndarray, *, max_moves: int | None = None) -> np.ndarray:
    """Reverse tour segments while doing so shortens the route.

    Index 0 of ``dist`` is the depot and is never moved; only the order of the
    stops between leaving and returning to it changes.

    Every candidate reversal is scored at once as an array operation and the
    single best one is applied per iteration. The textbook nested-loop version
    is O(n^2) *Python* operations per pass, which at ~10,000 tours per
    simulation run dominated the entire runtime; scoring the same candidates
    with NumPy makes the search effectively free while exploring strictly more
    of the neighbourhood (best-improvement rather than first-improvement).
    """
    seq = seq.copy()
    n = seq.size
    if n < 3:
        return seq

    max_moves = max_moves if max_moves is not None else 4 * n + 50
    # `upper` restricts candidates to j > i; reversing a single stop is a no-op.
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)

    for _ in range(max_moves):
        path = np.concatenate([[0], seq, [0]])
        before = path[0:n]        # node preceding the segment start
        start = path[1 : n + 1]   # segment start
        end = path[1 : n + 1]     # segment end
        after = path[2 : n + 2]   # node following the segment end

        delta = (
            dist[np.ix_(before, end)]
            + dist[np.ix_(start, after)]
            - dist[before, start][:, None]
            - dist[end, after][None, :]
        )
        delta = np.where(upper, delta, np.inf)

        flat = int(np.argmin(delta))
        i, j = divmod(flat, n)
        if delta[i, j] >= -1e-9:
            break
        seq[i : j + 1] = seq[i : j + 1][::-1]

    return seq
