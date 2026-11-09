"""Oracle gate: an upper bound on any per-context routing rule over a fixed set of paths.

Quality of a path on a context = its head-to-head result vs a fixed reference system (win = 1, loss = 0;
the reference itself scores 0.5). The judge is noisy, so picking the per-context maximum and scoring it
on the same judgments is optimistically biased (winner's curse). We therefore cross-fit over the two
presentation orders: choose with one order's verdicts, evaluate with the other's, and average both splits.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

ORDERS = ("sys_first", "ref_first")


@dataclass
class Scores:
    """score[order][system][ctx] in {0, 1}; the reference system has 0.5 everywhere."""

    by_order: dict[str, dict[str, dict[str, float]]]
    contexts: list[str]


def scores_from_judgments(records: Iterable[dict], reference: str) -> Scores:
    by_order: dict[str, dict[str, dict[str, float]]] = {o: defaultdict(dict) for o in ORDERS}
    ctxs: set[str] = set()
    for r in records:
        if r.get("reference") != reference or "winner" not in r:
            continue
        order = "sys_first" if r["first"] == r["system"] else "ref_first"
        by_order[order][r["system"]][r["ctx"]] = float(r["winner"] == r["system"])
        ctxs.add(r["ctx"])
    complete = sorted(c for c in ctxs if all(c in s for o in ORDERS for s in by_order[o].values()))
    for o in ORDERS:
        by_order[o][reference] = {c: 0.5 for c in complete}
    return Scores({o: dict(v) for o, v in by_order.items()}, complete)


@dataclass(frozen=True)
class RouteResult:
    quality: float
    cost: float
    share: dict[str, float]


def _choose(sel: dict[str, dict[str, float]], options: Sequence[str], cost: dict[str, float], ctx: str) -> str:
    # highest selection score; ties -> cheaper path
    return max(options, key=lambda s: (sel[s][ctx], -cost[s]))


def oracle(scores: Scores, options: Sequence[str], cost: dict[str, float], cross_fit: bool = True) -> RouteResult:
    q, c, picks = [], [], defaultdict(int)
    splits = [(ORDERS[0], ORDERS[1]), (ORDERS[1], ORDERS[0])] if cross_fit else [(o, o) for o in ORDERS]
    for sel_o, eval_o in splits:
        sel, ev = scores.by_order[sel_o], scores.by_order[eval_o]
        for ctx in scores.contexts:
            s = _choose(sel, options, cost, ctx)
            q.append(ev[s][ctx])
            c.append(cost[s])
            picks[s] += 1
    n = sum(picks.values())
    return RouteResult(float(np.mean(q)), float(np.mean(c)), {s: picks[s] / n for s in options})


def fixed(scores: Scores, system: str, cost: dict[str, float]) -> RouteResult:
    q = [scores.by_order[o][system][c] for o in ORDERS for c in scores.contexts]
    return RouteResult(float(np.mean(q)), cost[system], {system: 1.0})


def budget_frontier(scores: Scores, cheap: str, options: Sequence[str], cost: dict[str, float],
                    budgets: Sequence[float]) -> list[RouteResult]:
    """Cross-fitted oracle under a mean-cost budget: start everyone on `cheap`, then upgrade the contexts with
    the largest selection-score gain per extra cost until the budget is spent."""
    out = []
    for b in budgets:
        q, c = [], []
        for sel_o, eval_o in ((ORDERS[0], ORDERS[1]), (ORDERS[1], ORDERS[0])):
            sel, ev = scores.by_order[sel_o], scores.by_order[eval_o]
            choice = {ctx: cheap for ctx in scores.contexts}
            ups = []
            for ctx in scores.contexts:
                best = _choose(sel, options, cost, ctx)
                gain = sel[best][ctx] - sel[cheap][ctx]
                extra = cost[best] - cost[cheap]
                if gain > 0 and extra > 0:
                    ups.append((gain / extra, ctx, best, extra))
            spent = cost[cheap] * len(scores.contexts)
            for _, ctx, best, extra in sorted(ups, reverse=True):
                if (spent + extra) / len(scores.contexts) <= b:
                    choice[ctx], spent = best, spent + extra
            q += [ev[choice[ctx]][ctx] for ctx in scores.contexts]
            c += [cost[choice[ctx]] for ctx in scores.contexts]
        out.append(RouteResult(float(np.mean(q)), float(np.mean(c)), {}))
    return out
