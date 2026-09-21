"""Where the warehouse tips over.

Run with::

    python -m src.experiments.load_curve

Sweeps the order arrival rate and records what happens to the on-time rate under
both the baseline and the optimised policy. Two things this is for.

**Justifying the operating point.** The whole comparison is run at one demand
level, and that choice needs defending. Too far below capacity and every policy
scores 100%, so the experiment measures nothing; too far above and the queue
collapses regardless of policy, so it measures nothing again. The curve shows
where that window is.

**Showing what optimisation actually buys.** Near capacity the interesting
question is not "how much better is the on-time rate" but "how much more demand
can the same four pickers absorb before service falls over". The horizontal gap
between the two curves answers that, and it is the form a warehouse manager can
act on.

The catalog and slot assignment are held fixed across the sweep, so only arrival
volume changes. Each level regenerates the order stream, which is why this is a
separate script rather than part of the main comparison.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from config import CONFIG, REPORTS_DIR
from src.calibration.fit_parameters import load_calibration
from src.experiments.pipeline import load_or_build_context, run_scenario
from src.simulation.catalog import build_catalog
from src.simulation.orders import generate_demand_and_orders

__all__ = ["run_load_curve"]

DEFAULT_LEVELS = (150, 175, 200, 225, 250, 275)
SCENARIOS = ("baseline", "optimized")


def _orders_at(base_per_day: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regenerate the order stream at a given arrival rate, same catalog."""
    calibration = load_calibration()
    rng = np.random.default_rng(CONFIG.simulation.seed)
    catalog = build_catalog(calibration, rng)
    demand_cfg = dataclasses.replace(CONFIG.demand, base_orders_per_day=float(base_per_day))
    generated = generate_demand_and_orders(catalog, calibration, rng, demand_cfg=demand_cfg)
    return generated.orders, generated.order_lines


def run_load_curve(context, levels=DEFAULT_LEVELS, n_replications: int = 2) -> pd.DataFrame:
    rows = []
    for level in levels:
        orders, order_lines = _orders_at(level)
        for scenario in SCENARIOS:
            for rep in range(n_replications):
                kpis = run_scenario(
                    context,
                    scenario,
                    seed_offset=rep * CONFIG.experiment.replication_seed_offset,
                    orders=orders,
                    order_lines=order_lines,
                ).kpis
                rows.append(
                    {
                        "orders_per_day": level,
                        "scenario": scenario,
                        "replication": rep,
                        "on_time_rate": kpis["on_time_rate"],
                        "picker_utilisation": kpis.get("picker_utilisation", float("nan")),
                        "orders_measured": kpis["orders_measured"],
                    }
                )
        summary = (
            pd.DataFrame(rows)
            .query("orders_per_day == @level")
            .groupby("scenario")["on_time_rate"]
            .mean()
        )
        print(
            f"  {level:>4} orders/day   "
            + "   ".join(f"{s} {v:.4f}" for s, v in summary.items())
        )
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 160)

    context = load_or_build_context()
    print("\nLoad sweep (catalog and slotting fixed; only arrival volume changes)\n")
    raw = run_load_curve(context)

    table = (
        raw.groupby(["orders_per_day", "scenario"])
        .agg(on_time_rate=("on_time_rate", "mean"),
             picker_utilisation=("picker_utilisation", "mean"))
        .round(4)
        .reset_index()
        .pivot(index="orders_per_day", columns="scenario")
    )

    print("\n" + "=" * 70)
    print("LOAD CURVE")
    print("=" * 70)
    print(table.to_string())

    on_time = raw.pivot_table(
        index="orders_per_day", columns="scenario", values="on_time_rate"
    )
    for threshold in (0.99, 0.95):
        capacities = {}
        for scenario in SCENARIOS:
            ok = on_time.index[on_time[scenario] >= threshold]
            capacities[scenario] = int(ok.max()) if len(ok) else None
        print(
            f"\nHighest swept load still holding {threshold:.0%} on-time: "
            + ", ".join(f"{s} = {v}" for s, v in capacities.items())
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(REPORTS_DIR / "load_curve_raw.csv", index=False)
    table.round(4).to_csv(REPORTS_DIR / "load_curve.csv")
    print(f"\nReports written to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
