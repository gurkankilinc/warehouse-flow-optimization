"""Evaluate all three models against their baselines.

Run with::

    python -m src.models.evaluate

Every model is reported next to the rule it has to beat, because an error
metric on its own says nothing about whether the model was worth building.
"""

from __future__ import annotations

import pandas as pd

from config import CONFIG, REPORTS_DIR
from src.experiments.pipeline import build_context
from src.models.demand_forecast import walk_forward_backtest


def main() -> None:
    pd.set_option("display.width", 200)

    context = build_context()

    print("\n" + "=" * 78)
    print("ML-1  DEMAND FORECAST  (walk-forward backtest)")
    print("=" * 78)
    backtest = walk_forward_backtest(context.demand_features, cutoff_date=context.cutoff_date)
    columns = [
        "fold", "train_days", "test_start", "test_end",
        "model_wmape", "naive_last_wmape", "seasonal_naive_wmape",
        "moving_average_28_wmape", "improvement_vs_best_naive",
    ]
    print(backtest[columns].round(4).to_string(index=False))
    best_naive = backtest[
        ["naive_last_wmape", "seasonal_naive_wmape", "moving_average_28_wmape"]
    ].min(axis=1)
    print(
        f"\n  mean WMAPE  model {backtest['model_wmape'].mean():.4f}"
        f"   best naive {best_naive.mean():.4f}"
        f"   -> {100 * backtest['improvement_vs_best_naive'].mean():.1f}% better"
    )
    print("\n  top features by gain:")
    print(context.demand_model.feature_importance().head(8).to_string(index=False))

    print("\n" + "=" * 78)
    print("ML-2  PICK TIME  (vs engineered-standard formula)")
    print("=" * 78)
    print(context.pick_time_report.round(4).to_string(index=False))
    mae_by = context.pick_time_report.set_index("estimator")["mae_s"]
    print(
        f"\n  MAE improvement vs linear time study : "
        f"{100 * (1 - mae_by['lightgbm'] / mae_by['linear_timestudy']):.1f}%"
    )
    print(
        f"  MAE improvement vs oracle-constant formula: "
        f"{100 * (1 - mae_by['lightgbm'] / mae_by['analytic_standard']):.1f}%"
    )
    print("\n  top features by gain:")
    print(context.pick_time_model.feature_importance().head(6).to_string(index=False))

    print("\n" + "=" * 78)
    print("ML-3  SLA RISK  (vs slack-only ranking)")
    print("=" * 78)
    print(context.risk_report.round(4).to_string(index=False))
    print("\n  top features by gain:")
    print(context.risk_model.feature_importance().head(6).to_string(index=False))
    print(
        "\n  Read these four columns together -- they disagree, and the\n"
        "  disagreement is the finding. Pooled AUC rewards separating busy\n"
        "  periods from quiet ones. Within-decision AUC asks the question a\n"
        "  dispatcher actually faces: can the model rank the orders competing\n"
        "  for the next picker right now? The model wins there. But average\n"
        "  precision and Brier describe the *top* of the ranking and its\n"
        "  calibration, and there the model loses badly -- which is precisely\n"
        "  the region an escalation policy consumes. Running the policy\n"
        "  end-to-end (src.experiments.dispatch_study) is what caught it."
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    backtest.to_csv(REPORTS_DIR / "model_demand_backtest.csv", index=False)
    context.pick_time_report.to_csv(REPORTS_DIR / "model_pick_time.csv", index=False)
    context.risk_report.to_csv(REPORTS_DIR / "model_sla_risk.csv", index=False)
    print(f"\nReports written to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
