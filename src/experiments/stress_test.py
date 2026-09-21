"""Does the optimisation still help when conditions are not the ones it learned?

Run with::

    python -m src.experiments.stress_test

An improvement that only exists under the exact conditions the models were
fitted on is not an improvement, it is an overfit. Three perturbations are
applied, and in every one the models, the forecast and the slot assignment stay
exactly as they were fitted under normal conditions -- nothing is refitted. That
mismatch is deliberate: it is what a deployed system actually faces.

    demand_surge        40% more orders than the models ever saw
    truck_delay_heavy   carriers arrive far later and far more often
    picker_shortage     a quarter of the workforce is missing

The interesting question is not whether the KPIs get worse -- they will -- but
whether the *gap* between baseline and optimised survives, narrows, or inverts.
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
from src.simulation.trucks import build_truck_schedule

__all__ = ["run_stress_tests"]

SCENARIOS_UNDER_TEST = ("baseline", "optimized")


def _surge_orders(multiplier: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regenerate the order stream at higher volume, same catalog.

    The seed and the catalog construction are unchanged, so the products and
    their slots stay identical -- only the arrival volume moves.
    """
    calibration = load_calibration()
    rng = np.random.default_rng(CONFIG.simulation.seed)
    catalog = build_catalog(calibration, rng)
    demand_cfg = dataclasses.replace(
        CONFIG.demand,
        base_orders_per_day=CONFIG.demand.base_orders_per_day * multiplier,
    )
    generated = generate_demand_and_orders(catalog, calibration, rng, demand_cfg=demand_cfg)
    return generated.orders, generated.order_lines


def _delayed_trucks(multiplier: float) -> pd.DataFrame:
    """Same planned schedule, much worse punctuality.

    Planned departures are untouched, so every order's promised truck and
    deadline stay valid -- only the arrivals slip.
    """
    truck_cfg = dataclasses.replace(
        CONFIG.truck,
        delay_probability=min(0.95, CONFIG.truck.delay_probability * 1.8),
        delay_mean_h=CONFIG.truck.delay_mean_h * multiplier,
        early_probability=CONFIG.truck.early_probability / 2.0,
    )
    rng = np.random.default_rng(CONFIG.simulation.seed + 4242)
    return build_truck_schedule(CONFIG.demand.history_days + 3, rng, truck_cfg)


def run_stress_tests(context, n_replications: int | None = None) -> pd.DataFrame:
    experiment = CONFIG.experiment
    # The baseline's own run-to-run spread at this load is large, so a small
    # number of replications leaves the interval too wide to conclude anything.
    n_replications = n_replications or experiment.n_replications
    offset = experiment.replication_seed_offset

    if experiment.picker_shortage_count >= CONFIG.picking.n_pickers:
        raise ValueError(
            f"picker_shortage_count ({experiment.picker_shortage_count}) must be "
            f"below n_pickers ({CONFIG.picking.n_pickers}) to be a shortage at all"
        )

    surge_orders, surge_lines = _surge_orders(experiment.demand_surge_multiplier)
    delayed = _delayed_trucks(experiment.truck_delay_multiplier)
    short_crew = dataclasses.replace(
        CONFIG.picking, n_pickers=experiment.picker_shortage_count
    )

    conditions = {
        "normal": {},
        "demand_surge": {"orders": surge_orders, "order_lines": surge_lines},
        "truck_delay_heavy": {"truck_schedule": delayed},
        "picker_shortage": {"picking_cfg": short_crew},
    }

    rows = []
    for condition, overrides in conditions.items():
        for rep in range(n_replications):
            for scenario in SCENARIOS_UNDER_TEST:
                result = run_scenario(
                    context, scenario, seed_offset=rep * offset, **overrides
                )
                rows.append(
                    {
                        "condition": condition,
                        "scenario": scenario,
                        "replication": rep,
                        **result.kpis,
                    }
                )
            print(
                f"  {condition:<18} rep {rep}  "
                f"baseline {rows[-2]['on_time_rate']:.4f}  "
                f"optimized {rows[-1]['on_time_rate']:.4f}"
            )
    return pd.DataFrame(rows)


def summarise(raw: pd.DataFrame, kpi: str = "on_time_rate") -> pd.DataFrame:
    """Paired baseline-vs-optimised gap under each condition."""
    wide = raw.pivot_table(
        index=["condition", "replication"], columns="scenario", values=kpi
    )
    rows = []
    for condition, group in wide.groupby(level="condition"):
        delta = group["optimized"] - group["baseline"]
        n = len(delta)
        sem = delta.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        rows.append(
            {
                "condition": condition,
                "baseline": group["baseline"].mean(),
                "optimized": group["optimized"].mean(),
                "delta": delta.mean(),
                "ci95_low": delta.mean() - 1.96 * sem,
                "ci95_high": delta.mean() + 1.96 * sem,
                "holds_up": delta.mean() - 1.96 * sem > 0,
            }
        )
    order = ["normal", "demand_surge", "truck_delay_heavy", "picker_shortage"]
    return (
        pd.DataFrame(rows)
        .set_index("condition")
        .reindex([c for c in order if c in {r["condition"] for r in rows}])
        .reset_index()
    )


def main() -> None:
    pd.set_option("display.width", 200)

    # Reuses the cached artefacts on purpose: the whole point is that nothing
    # is refitted for the stressed conditions.
    context = load_or_build_context()
    print("\nStress testing (models and slotting stay as fitted under normal conditions)\n")
    raw = run_stress_tests(context)

    for kpi in ("on_time_rate", "distance_per_line_m", "mean_tardiness_h"):
        print("\n" + "=" * 92)
        print(f"STRESS SUMMARY  --  {kpi}   (optimized minus baseline)")
        print("=" * 92)
        print(summarise(raw, kpi).round(4).to_string(index=False))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(REPORTS_DIR / "stress_raw.csv", index=False)
    summarise(raw).round(6).to_csv(REPORTS_DIR / "stress_summary.csv", index=False)
    print(f"\nReports written to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
