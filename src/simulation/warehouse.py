"""Warehouse layout generation.

Produces the physical slot grid: every storage location with its coordinates,
its walking distance from the shipping dock and the goods-in door, and the
storage constraints that apply to it (cold chain, ground-level-only).

The layout is deterministic -- given a WarehouseConfig it is always identical.
Randomness enters the project only through demand and operations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CONFIG, WarehouseConfig
from src.utils.geometry import distance_to_point

__all__ = ["build_slots", "slot_travel_distance_matrix"]


def build_slots(cfg: WarehouseConfig | None = None) -> pd.DataFrame:
    """Return one row per storage slot.

    Columns
    -------
    slot_id      : stable identifier, e.g. ``A03-B07-R-L1``
    aisle, bay, side, level : integer grid coordinates
    x, y, z      : physical coordinates in meters
    dock_distance     : one-way walking distance to the shipping dock
    receiving_distance: one-way walking distance from the goods-in door
    is_cold      : slot sits in the temperature-controlled zone
    is_ground    : slot is at floor level (required for heavy SKUs)
    """
    cfg = cfg or CONFIG.warehouse

    aisle, bay, side, level = (
        arr.ravel()
        for arr in np.meshgrid(
            np.arange(cfg.n_aisles),
            np.arange(cfg.bays_per_aisle),
            np.arange(cfg.sides),
            np.arange(cfg.levels),
            indexing="ij",
        )
    )

    x = aisle * cfg.aisle_width
    y = cfg.cross_aisle_offset + bay * cfg.bay_depth
    z = level * cfg.level_height

    side_label = np.where(side == 0, "L", "R")
    slot_id = np.array(
        [
            f"A{a:02d}-B{b:02d}-{s}-L{l}"
            for a, b, s, l in zip(aisle, bay, side_label, level)
        ]
    )

    slots = pd.DataFrame(
        {
            "slot_id": slot_id,
            "aisle": aisle,
            "bay": bay,
            "side": side_label,
            "level": level,
            "x": x,
            "y": y,
            "z": z,
        }
    )

    cross = cfg.cross_aisle_ys
    slots["dock_distance"] = distance_to_point(
        slots["x"].to_numpy(), slots["y"].to_numpy(), cfg.dock_x, 0.0, cross
    )
    slots["receiving_distance"] = distance_to_point(
        slots["x"].to_numpy(), slots["y"].to_numpy(), cfg.receiving_x, 0.0, cross
    )

    slots["is_cold"] = slots["aisle"].isin(cfg.cold_zone_aisles)
    slots["is_ground"] = slots["level"] <= cfg.heavy_max_level

    return slots.sort_values("slot_id").reset_index(drop=True)


def slot_travel_distance_matrix(
    slots: pd.DataFrame, cfg: WarehouseConfig | None = None
) -> np.ndarray:
    """Pairwise walking distances between the given slots.

    Only used for small subsets (the stops of a single picking tour), never for
    the full slot set -- that would be a needlessly large matrix.
    """
    from src.utils.geometry import travel_distance_matrix

    cfg = cfg or CONFIG.warehouse
    return travel_distance_matrix(
        slots["x"].to_numpy(), slots["y"].to_numpy(), cfg.cross_aisle_ys
    )
