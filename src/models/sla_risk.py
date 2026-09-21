"""ML-3: will this order miss its truck?

Slack answers "how much room is left"; this model answers "given everything
happening right now, how likely is that room to evaporate". They disagree
usefully: an order with an hour of slack at 07:00 on a quiet Tuesday is safe,
and the same order with the same slack at 16:00 with 400 lines queued ahead of
it is not. Slack cannot see the queue. The model can.

Training data comes from snapshots logged during a baseline run: the features
as they looked at each 15-minute scoring pass, labelled afterwards with whether
that order did in fact miss its promised truck.

How this one turned out
-----------------------
It does not help, and the metrics say why only if you read all of them. The
model beats slack at ranking the orders inside a single decision moment
(within-decision AUC 0.82 against 0.62) -- which sounds like exactly what a
dispatcher needs. But its average precision is a third of slack's and its
Brier score three times worse: it orders the bulk of the queue well while
being unreliable and over-confident at the very top. An escalation policy
spends nothing *but* the top, so it promotes the wrong orders and the on-time
rate falls at every threshold tried.

Two lessons are worth keeping. A single headline metric would have hidden
this: pooled AUC, within-decision AUC, average precision and calibration point
in different directions here, and which one matters is decided by how the
score gets used, not by convention. And no offline metric settled it -- the
end-to-end simulation did.

There is also a structural caveat. The model is trained under one policy and
then used to change it, so the moment it starts escalating orders the
distribution it was fitted on shifts underneath it. That is the standard
feedback-loop problem in operational ML, not an artefact of simulation, and it
is a further reason to judge the policy on the on-time rate rather than on
classifier metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import CONFIG, ModelConfig

__all__ = [
    "SLARiskModel",
    "build_training_set",
    "evaluate_sla_risk",
    "within_decision_auc",
]

FEATURES = [
    "slack_s",
    "predicted_pick_s",
    "lines_remaining",
    "total_lines",
    "age_s",
    "queue_orders",
    "queue_lines",
    "hour_of_day",
    "day_of_week",
]
TARGET = "missed"


def build_training_set(
    snapshots: pd.DataFrame, orders: pd.DataFrame, *, train_end_s: float
) -> pd.DataFrame:
    """Join decision-time snapshots to the outcome each order ended up with."""
    if snapshots.empty:
        raise ValueError("no risk snapshots were logged; run with log_risk_snapshots=True")

    outcome = orders[["order_id", "on_time"]].copy()
    outcome[TARGET] = (~outcome["on_time"]).astype(int)

    data = snapshots.merge(outcome[["order_id", TARGET]], on="order_id", how="inner")
    return data[data["snapshot_s"] < train_end_s].reset_index(drop=True)


@dataclass
class SLARiskModel:
    model: object = None
    features: list[str] | None = None
    threshold: float = 0.5

    def fit(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame | None = None,
        model_cfg: ModelConfig | None = None,
    ) -> "SLARiskModel":
        import lightgbm as lgb

        model_cfg = model_cfg or CONFIG.model
        self.features = [c for c in FEATURES if c in train.columns]

        self.model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=model_cfg.n_estimators,
            learning_rate=model_cfg.learning_rate,
            max_depth=model_cfg.max_depth,
            min_child_samples=model_cfg.min_samples_leaf,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.9,
            verbose=-1,
            n_jobs=-1,
        )

        fit_kwargs = {}
        if valid is not None and not valid.empty:
            fit_kwargs["eval_X"] = valid[self.features]
            fit_kwargs["eval_y"] = valid[TARGET]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(model_cfg.early_stopping_rounds, verbose=False)
            ]

        self.model.fit(train[self.features], train[TARGET], **fit_kwargs)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        frame = frame.copy()
        for col in self.features:
            if col not in frame.columns:
                frame[col] = 0.0
        return self.model.predict_proba(frame[self.features])[:, 1]

    def feature_importance(self) -> pd.DataFrame:
        return (
            pd.DataFrame(
                {"feature": self.features, "gain": self.model.booster_.feature_importance("gain")}
            )
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )


def within_decision_auc(
    y: np.ndarray, score: np.ndarray, groups: np.ndarray, *, min_group: int = 20
) -> float:
    """AUC computed *inside* each decision moment, then averaged.

    Plain AUC pools every snapshot together, so a feature that only says "the
    warehouse is busy right now" scores well: it separates busy hours from
    quiet ones. At the instant a picker asks what to do next, that feature has
    the *same value for every order in the queue* and cannot rank any of them.

    Grouping by snapshot removes that shortcut and asks the question the policy
    actually poses: can the model tell these particular waiting orders apart?
    It is a necessary check, not a sufficient one -- a model can win here and
    still be useless for escalation if its top-ranked predictions are
    unreliable, which is what average precision and the Brier score expose.
    """
    from sklearn.metrics import roc_auc_score

    frame = pd.DataFrame({"y": y, "score": score, "group": groups})
    scores = []
    for _, grp in frame.groupby("group"):
        if len(grp) < min_group or grp["y"].nunique() < 2:
            continue
        scores.append(roc_auc_score(grp["y"], grp["score"]))
    return float(np.mean(scores)) if scores else float("nan")


def _slack_only_baseline(frame: pd.DataFrame) -> np.ndarray:
    """Rank by negated slack: the rule the model has to prove itself against.

    Scaled into [0, 1] so it can be scored with the same metrics; only the
    ordering matters for AUC.
    """
    slack = frame["slack_s"].to_numpy(dtype=float)
    return 1.0 / (1.0 + np.exp(slack / 3600.0))


def evaluate_sla_risk(
    data: pd.DataFrame, model_cfg: ModelConfig | None = None
) -> tuple[SLARiskModel, pd.DataFrame]:
    """Time-split fit and evaluation against the slack-only rule."""
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )

    data = data.sort_values("snapshot_s").reset_index(drop=True)
    split_s = data["snapshot_s"].quantile(0.75)
    train = data[data["snapshot_s"] < split_s]
    test = data[data["snapshot_s"] >= split_s]

    inner = train["snapshot_s"].quantile(0.85)
    model = SLARiskModel().fit(
        train[train["snapshot_s"] < inner], train[train["snapshot_s"] >= inner], model_cfg
    )

    y = test[TARGET].to_numpy()
    groups = test["snapshot_s"].to_numpy()
    rows = []
    for name, pred in (
        ("slack_only", _slack_only_baseline(test)),
        ("lightgbm", model.predict_proba(test)),
    ):
        rows.append(
            {
                "model": name,
                "roc_auc_pooled": float(roc_auc_score(y, pred)),
                "roc_auc_within_decision": within_decision_auc(y, pred, groups),
                "average_precision": float(average_precision_score(y, pred)),
                "brier": float(brier_score_loss(y, np.clip(pred, 0, 1))),
            }
        )

    comparison = pd.DataFrame(rows)
    comparison["n_test_snapshots"] = len(test)
    comparison["positive_rate"] = float(y.mean())
    return model, comparison
