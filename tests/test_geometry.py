"""The travel model underpins every distance number the project reports.

If these are wrong, every KPI is wrong in a way no downstream test would catch,
so they are checked against hand-computed values rather than against the
implementation's own output.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.geometry import (
    distance_to_point,
    tour_length,
    travel_distance,
    travel_distance_matrix,
)


def test_same_aisle_is_vertical_distance_only():
    # Two slots in the same aisle: the picker walks straight along it.
    assert travel_distance(6.0, 4.0, 6.0, 19.0) == pytest.approx(15.0)


def test_different_aisle_forces_exit_and_re_entry():
    # From (3, 10) to (9, 4): walk out 10, across 6, back in 4 = 20.
    assert travel_distance(3.0, 10.0, 9.0, 4.0) == pytest.approx(20.0)


def test_never_shorter_than_euclidean():
    # Racks block direct movement, so the aisle metric can never beat straight
    # line distance. A cheaper result would mean walking through steel.
    rng = np.random.default_rng(0)
    for _ in range(200):
        x1, x2 = rng.uniform(0, 60, 2)
        y1, y2 = rng.uniform(0, 30, 2)
        aisle = travel_distance(x1, y1, x2, y2)
        euclid = np.hypot(x1 - x2, y1 - y2)
        assert aisle >= euclid - 1e-9


def test_dock_distance_matches_manual_formula():
    xs = np.array([0.0, 6.0, 12.0])
    ys = np.array([4.0, 10.0, 7.0])
    got = distance_to_point(xs, ys, 0.0, 0.0)
    # Aisle 0 shares the dock's x, so it is a straight walk; the others must
    # cross first.
    assert got == pytest.approx([4.0, 16.0, 19.0])


def test_matrix_is_symmetric_with_zero_diagonal():
    rng = np.random.default_rng(1)
    xs = rng.choice([0.0, 3.0, 6.0, 9.0], 12)
    ys = rng.uniform(4.0, 25.0, 12)
    matrix = travel_distance_matrix(xs, ys)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)


def test_matrix_agrees_with_scalar_function():
    rng = np.random.default_rng(2)
    xs = rng.choice([0.0, 3.0, 6.0], 8)
    ys = rng.uniform(4.0, 20.0, 8)
    matrix = travel_distance_matrix(xs, ys)
    for i in range(len(xs)):
        for j in range(len(xs)):
            assert matrix[i, j] == pytest.approx(travel_distance(xs[i], ys[i], xs[j], ys[j]))


def test_tour_length_counts_both_legs_to_the_depot():
    xs = np.array([0.0, 3.0, 3.0])   # index 0 is the depot
    ys = np.array([0.0, 10.0, 14.0])
    dist = travel_distance_matrix(xs, ys)
    # depot -> (3,10) = 13, -> (3,14) = 4, -> depot = 17. Total 34.
    assert tour_length(np.array([1, 2]), dist) == pytest.approx(34.0)


def test_empty_tour_costs_nothing():
    dist = travel_distance_matrix(np.array([0.0]), np.array([0.0]))
    assert tour_length(np.array([], dtype=int), dist) == 0.0
