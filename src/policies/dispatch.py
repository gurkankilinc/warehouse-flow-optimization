"""Dispatch rules: which waiting order gets picked next.

This is the operational half of the project. Four rules, in increasing order of
information used:

``fifo``        first come, first served. The naive rule.
``edd``         earliest due date. The classic scheduling rule and the
                baseline here -- it already knows about deadlines, so beating
                it requires more than just "we looked at the deadline".
``least_slack`` slack = deadline - now - predicted pick time.
``triage``      savable orders first, EDD within them, shortest as tie-break.
``risk_aware``  triage, with orders the SLA model flags as at risk escalated
                ahead of the queue.

Why least-slack loses to EDD here
---------------------------------
It is the natural next step after EDD and it measurably *hurts* (96.6% on-time
against EDD's 97.0% in this warehouse). The reason is that the two rules
optimise different things. Least-slack minimises maximum lateness, so among
orders due at the same time it starts the **longest** one first. The KPI here
is the *number* of orders that miss their truck, and by that measure spending
forty minutes rescuing one large order while five small ones expire is a bad
trade. Least-slack also pushes big orders to the front of the queue, where they
fill a cart on their own and destroy the batching that shorter orders would
have shared.

For minimising the count of late jobs the relevant classical result is the
Moore-Hodgson rule: schedule by due date, and once a job can no longer make its
deadline, stop letting it block the ones that still can. ``triage`` is the
online form of that idea, and unlike least-slack it beats EDD.

Every rule takes the same arguments and returns pending orders in the sequence
they should be picked, so the engine can swap them without knowing which is in
play.
"""

from __future__ import annotations

from typing import Callable, Sequence

from config import CONFIG, PolicyConfig

__all__ = ["DISPATCH_RULES", "dispatch_fifo", "dispatch_edd", "dispatch_least_slack", "dispatch_risk_aware"]

# ``PendingOrder`` is defined in the engine; typing it structurally here keeps
# the policy layer free of a circular import.
DispatchRule = Callable[[Sequence, float], list]


def dispatch_fifo(pending: Sequence, now_s: float, policy_cfg: PolicyConfig | None = None) -> list:
    return sorted(pending, key=lambda o: o.created_s)


def dispatch_edd(pending: Sequence, now_s: float, policy_cfg: PolicyConfig | None = None) -> list:
    return sorted(pending, key=lambda o: o.deadline_s)


def dispatch_least_slack(
    pending: Sequence, now_s: float, policy_cfg: PolicyConfig | None = None
) -> list:
    """Order by how little room is left, not by when it is due.

    Two orders due at the same time are not equally urgent if one takes twenty
    minutes to pick and the other takes two. Slack captures that; EDD cannot.
    """
    policy_cfg = policy_cfg or CONFIG.policy
    margin = policy_cfg.slack_safety_margin_s

    def slack(o) -> float:
        return o.deadline_s - now_s - o.predicted_pick_s - margin

    return sorted(pending, key=slack)


def dispatch_triage(
    pending: Sequence, now_s: float, policy_cfg: PolicyConfig | None = None
) -> list:
    """Save what can still be saved, in due-date order, shortest first.

    Three keys, in priority order:

    1. **Is it still savable?** An order whose deadline has already passed
       cannot be rescued, and letting it sit at the front of the queue costs
       the orders behind it. It goes to the back -- it still ships, just on a
       later truck, which it was going to do anyway.
    2. **Due date.** Among savable orders, EDD.
    3. **Predicted pick time, shortest first.** The tie-break that turns this
       from EDD into something better: when two orders are due together,
       clearing the quick one first saves strictly more deadlines.
    """
    policy_cfg = policy_cfg or CONFIG.policy
    margin = policy_cfg.slack_safety_margin_s

    def key(o):
        savable = (o.deadline_s - now_s - o.predicted_pick_s - margin) >= 0
        return (0 if savable else 1, o.deadline_s, o.predicted_pick_s)

    return sorted(pending, key=key)


def dispatch_risk_aware(
    pending: Sequence, now_s: float, policy_cfg: PolicyConfig | None = None
) -> list:
    """Triage, with model-flagged at-risk orders escalated.

    Slack alone is blind to *why* an order is tight. The risk model has seen
    which combinations of queue state, order size and time of day historically
    ended up late, so it can catch orders whose arithmetic still looks
    comfortable but whose situation does not.

    Escalation is restricted to orders that are still savable, for the same
    reason triage demotes the hopeless ones: promoting an order that cannot
    make its truck spends capacity to change nothing.
    """
    policy_cfg = policy_cfg or CONFIG.policy
    margin = policy_cfg.slack_safety_margin_s
    threshold = policy_cfg.risk_escalation_threshold

    def key(o):
        savable = (o.deadline_s - now_s - o.predicted_pick_s - margin) >= 0
        escalated = savable and o.risk_score >= threshold
        return (0 if escalated else 1, 0 if savable else 1, o.deadline_s, o.predicted_pick_s)

    return sorted(pending, key=key)


DISPATCH_RULES: dict[str, DispatchRule] = {
    "fifo": dispatch_fifo,
    "edd": dispatch_edd,
    "least_slack": dispatch_least_slack,
    "triage": dispatch_triage,
    "risk_aware": dispatch_risk_aware,
}
