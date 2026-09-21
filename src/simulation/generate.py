"""Build the full synthetic dataset and write it to ``data/processed``.

Run with::

    python -m src.simulation.generate

Everything is derived from ``CONFIG`` and a single seed, so the output is
reproducible: two people running this command get byte-identical files.

A small extract of each table is also written to ``data/sample`` and committed,
so a visitor to the repository can see the data without running anything.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import CONFIG, PROCESSED_DIR, SAMPLE_DIR
from src.calibration.fit_parameters import load_calibration
from src.simulation.catalog import build_catalog
from src.simulation.orders import generate_demand_and_orders
from src.simulation.trucks import build_truck_schedule
from src.simulation.warehouse import build_slots

__all__ = ["build_dataset", "load_dataset"]

_SAMPLE_ROWS = 500

_TABLES = (
    "warehouse_slots",
    "sku_catalog",
    "daily_demand",
    "orders",
    "order_lines",
    "trucks",
    "putaway_receipts",
)


def build_dataset(seed: int | None = None) -> dict[str, pd.DataFrame]:
    """Generate every table the rest of the project consumes."""
    seed = CONFIG.simulation.seed if seed is None else seed
    rng = np.random.default_rng(seed)

    calibration = load_calibration()
    slots = build_slots()
    catalog = build_catalog(calibration, rng)
    generated = generate_demand_and_orders(catalog, calibration, rng)

    # The truck grid must span the same horizon the order assignment used,
    # otherwise truck ids would not line up with the ones on the orders.
    trucks = build_truck_schedule(
        CONFIG.demand.history_days + 3, rng, start_date=generated.start_date
    )

    return {
        "warehouse_slots": slots,
        "sku_catalog": catalog,
        "daily_demand": generated.daily_demand,
        "orders": generated.orders,
        "order_lines": generated.order_lines,
        "trucks": trucks,
        "putaway_receipts": generated.receipts,
        "_meta": pd.DataFrame(
            [
                {
                    "seed": seed,
                    "window_start_day": generated.window_start_day,
                    "start_date": generated.start_date,
                    "calibration_source": calibration.source,
                }
            ]
        ),
    }


def save_dataset(tables: dict[str, pd.DataFrame]) -> None:
    for name, frame in tables.items():
        frame.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)

    for name in _TABLES:
        frame = tables[name]
        head = frame.head(_SAMPLE_ROWS)
        head.to_csv(SAMPLE_DIR / f"{name}.csv", index=False)


def load_dataset() -> dict[str, pd.DataFrame]:
    """Read the generated tables, building them first if they are missing."""
    meta_path = PROCESSED_DIR / "_meta.parquet"
    if not meta_path.exists():
        tables = build_dataset()
        save_dataset(tables)
        return tables

    tables = {
        name: pd.read_parquet(PROCESSED_DIR / f"{name}.parquet") for name in _TABLES
    }
    tables["_meta"] = pd.read_parquet(meta_path)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    tables = build_dataset(args.seed)
    save_dataset(tables)

    print(f"Generated with seed {tables['_meta']['seed'].iloc[0]} "
          f"(calibration: {tables['_meta']['calibration_source'].iloc[0]})\n")
    for name in _TABLES:
        frame = tables[name]
        print(f"  {name:<20} {len(frame):>8,} rows x {frame.shape[1]:>2} cols")
    print(f"\nFull tables -> {PROCESSED_DIR}")
    print(f"Committed samples ({_SAMPLE_ROWS} rows each) -> {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
