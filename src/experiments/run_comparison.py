"""Baseline vs optimised, with a per-component ablation.

Run with::

    python -m src.experiments.run_comparison

Design choices that make the numbers mean something:

**Replications with common random numbers.** Each scenario is run several times
with different stochastic streams, but every scenario sees the *same* set of
streams. Differences are therefore paired, and the reported spread is the
spread of the difference rather than the (much larger) spread of the levels.

**Ablation.** Running the optimised bundle alone would say "it is better"
without saying why. Each component is also run on its own against the baseline,
so the gain can be attributed instead of asserted.

**A real baseline.** ABC slotting, earliest-due-date dispatch and S-shape
routing are what competent warehouses actually do. Beating a random layout
would prove nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CONFIG, REPORTS_DIR
from src.experiments.pipeline import SCENARIOS, build_context, run_scenario, save_context

__all__ = ["run_all_scenarios", "summarise"]

HEADLINE_KPIS = [
    "on_time_rate",
    "mean_tardiness_h",
    "p95_tardiness_h",
    "distance_per_line_m",
    "total_distance_km",
    "mean_orders_per_tour",
    "mean_aisles_per_tour",
    "picker_utilisation",
    "putaway_distance_per_receipt_m",
    "orders_rolled",
]


def run_all_scenarios(context, n_replications: int | None = None) -> pd.DataFrame:
    n_replications = n_replications or CONFIG.experiment.n_replications
    offset = CONFIG.experiment.replication_seed_offset

    rows = []
    for rep in range(n_replications):
        for scenario in SCENARIOS:
            result = run_scenario(context, scenario, seed_offset=rep * offset)
            row = {"scenario": scenario, "replication": rep}
            row.update(result.kpis)
            rows.append(row)
            print(
                f"  rep {rep}  {scenario:<14} "
                f"on-time {result.kpis['on_time_rate']:.4f}  "
                f"dist/line {result.kpis['distance_per_line_m']:.2f} m"
            )
    return pd.DataFrame(rows)


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each KPI, by scenario."""
    available = [k for k in HEADLINE_KPIS if k in raw.columns]
    summary = raw.groupby("scenario")[available].agg(["mean", "std"])
    order = [s for s in SCENARIOS if s in summary.index]
    return summary.loc[order]


def paired_deltas(raw: pd.DataFrame, kpi: str = "on_time_rate") -> pd.DataFrame:
    """Per-replication difference against the baseline, with a 95% interval.

    Pairing matters: replication 3 of ``optimized`` and replication 3 of
    ``baseline`` faced identical congestion, picker skill and truck delays, so
    subtracting them cancels the noise that would otherwise swamp the effect.
    """
    wide = raw.pivot(index="replication", columns="scenario", values=kpi)
    rows = []
    for scenario in SCENARIOS:
        if scenario == "baseline" or scenario not in wide.columns:
            continue
        delta = wide[scenario] - wide["baseline"]
        n = len(delta)
        sem = delta.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        rows.append(
            {
                "scenario": scenario,
                "kpi": kpi,
                "baseline_mean": wide["baseline"].mean(),
                "scenario_mean": wide[scenario].mean(),
                "delta_mean": delta.mean(),
                "delta_ci95_low": delta.mean() - 1.96 * sem,
                "delta_ci95_high": delta.mean() + 1.96 * sem,
                "relative_pct": 100.0 * delta.mean() / wide["baseline"].mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 220)

    context = build_context()
    save_context(context)  # so the dashboard does not have to retrain anything
    print(f"\nRunning {len(SCENARIOS)} scenarios x "
          f"{CONFIG.experiment.n_replications} replications\n")
    raw = run_all_scenarios(context)

    print("\n" + "=" * 100)
    print("SCENARIO SUMMARY  (mean over replications)")
    print("=" * 100)
    summary = summarise(raw)
    print(summary.xs("mean", axis=1, level=1).round(4).to_string())

    for kpi in ("on_time_rate", "distance_per_line_m", "mean_tardiness_h",
                "putaway_distance_per_receipt_m"):
        print("\n" + "-" * 100)
        print(f"PAIRED DELTA vs BASELINE  --  {kpi}")
        print("-" * 100)
        print(paired_deltas(raw, kpi).round(4).to_string(index=False))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(REPORTS_DIR / "scenario_raw.csv", index=False)
    summary.round(6).to_csv(REPORTS_DIR / "scenario_summary.csv")
    pd.concat(
        [paired_deltas(raw, k) for k in
         ("on_time_rate", "distance_per_line_m", "mean_tardiness_h",
          "putaway_distance_per_receipt_m")]
    ).to_csv(REPORTS_DIR / "scenario_deltas.csv", index=False)
    print(f"\nReports written to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
