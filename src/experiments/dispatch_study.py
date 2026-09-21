"""Why the obvious dispatch upgrades fail, and which one works.

Run with::

    python -m src.experiments.dispatch_study

This exists because two of the three "smarter" dispatch rules in this project
turned out to be worse than the plain earliest-due-date baseline, and a claim
like that should be reproducible rather than asserted in a README.

The sweep isolates each idea:

* **FIFO** -- the naive floor, to show EDD is not a strawman.
* **Least slack** -- the natural next step after EDD, and a regression. It
  minimises maximum lateness, whereas the KPI here is the *number* of orders
  that miss their truck. Among orders due together it starts the longest one
  first, trading several small saves for one large one, and it pushes big
  orders to the head of the queue where they fill a cart alone and wreck
  batching. The margin sweep shows this is structural, not a tuning artefact.
* **Triage** -- the online form of the Moore-Hodgson rule, which is the
  classical answer for minimising late *counts*: stop letting orders that can
  no longer make their deadline block the ones that still can, take earliest
  due date among the rest, break ties by shortest job.
* **Risk escalation** -- the SLA classifier promoting orders it thinks will
  miss. Swept across the full threshold range, it never beats EDD, let alone
  triage. Section 3 of the README explains the mechanism.
"""

from __future__ import annotations

import dataclasses

import pandas as pd

from config import CONFIG, REPORTS_DIR
from src.experiments.pipeline import load_or_build_context
from src.simulation.engine import Policies, StochasticStreams, run_simulation

__all__ = ["run_dispatch_study"]


def _run(context, dispatch, *, use_pick_model=False, use_risk_model=False,
         policy_cfg=None, rep=0):
    sim = CONFIG.simulation
    policies = Policies(
        slot_assignment=context.abc_assignment,
        dispatch=dispatch,
        batching="sequential",
        routing="s_shape",
        pick_time_model=context.pick_time_model if use_pick_model else None,
        risk_model=context.risk_model if use_risk_model else None,
    )
    streams = StochasticStreams.build(
        (sim.window_days + 2) * 86400.0,
        CONFIG.picking.n_pickers,
        CONFIG.simulation.seed + 777 + rep * CONFIG.experiment.replication_seed_offset,
    )
    return run_simulation(
        slots=context.tables["warehouse_slots"],
        catalog=context.tables["sku_catalog"],
        orders=context.tables["orders"],
        order_lines=context.tables["order_lines"],
        receipts=context.tables["putaway_receipts"],
        truck_schedule=context.tables["trucks"],
        policies=policies,
        streams=streams,
        window_start_day=context.window_start_day,
        policy_cfg=policy_cfg,
    ).kpis


def run_dispatch_study(context, n_replications: int = 2) -> pd.DataFrame:
    variants: list[tuple[str, dict]] = [
        ("fifo", dict(dispatch="fifo")),
        ("edd (baseline)", dict(dispatch="edd")),
        ("least_slack", dict(dispatch="least_slack")),
        ("least_slack + ML pick time", dict(dispatch="least_slack", use_pick_model=True)),
        ("triage", dict(dispatch="triage")),
        ("triage + ML pick time", dict(dispatch="triage", use_pick_model=True)),
    ]
    for threshold in (0.2, 0.35, 0.5, 0.8):
        variants.append(
            (
                f"risk escalation (thr {threshold:.2f})",
                dict(
                    dispatch="risk_aware",
                    use_pick_model=True,
                    use_risk_model=True,
                    policy_cfg=dataclasses.replace(
                        CONFIG.policy, risk_escalation_threshold=threshold
                    ),
                ),
            )
        )

    rows = []
    for label, kwargs in variants:
        for rep in range(n_replications):
            kpis = _run(context, rep=rep, **kwargs)
            rows.append(
                {
                    "variant": label,
                    "replication": rep,
                    "on_time_rate": kpis["on_time_rate"],
                    "missed_pct": 100 * (1 - kpis["on_time_rate"]),
                    "p95_tardiness_h": kpis["p95_tardiness_h"],
                    "mean_orders_per_tour": kpis["mean_orders_per_tour"],
                }
            )
        print(f"  {label:<34} on-time {rows[-1]['on_time_rate']:.4f}")
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 200)

    context = load_or_build_context()
    print("\nDispatch rule study (ABC slotting, sequential batching, S-shape routing)\n")
    raw = run_dispatch_study(context)

    summary = (
        raw.groupby("variant", sort=False)
        .agg(
            on_time_rate=("on_time_rate", "mean"),
            missed_pct=("missed_pct", "mean"),
            p95_tardiness_h=("p95_tardiness_h", "mean"),
            orders_per_tour=("mean_orders_per_tour", "mean"),
        )
        .round(4)
    )
    baseline = summary.loc["edd (baseline)", "missed_pct"]
    summary["missed_vs_edd_pct"] = (
        100 * (summary["missed_pct"] - baseline) / baseline
    ).round(1)

    print("\n" + "=" * 96)
    print("DISPATCH RULE COMPARISON")
    print("=" * 96)
    print(summary.to_string())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(REPORTS_DIR / "dispatch_study_raw.csv", index=False)
    summary.to_csv(REPORTS_DIR / "dispatch_study.csv")
    print(f"\nReports written to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
