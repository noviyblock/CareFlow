"""Why do learned routers not capture the oracle headroom? Two diagnostics on saved judgments (no LLM calls).

1. Where the headroom lives. Each context is classified by the judge's verdicts for the non-reference paths in the
   two presentation orders:
   - decisive: some path beats the reference in BOTH orders (a stable, order-independent gain);
   - split:    no decisive win, but some path wins in exactly one order (the judge disagrees with itself);
   - none:     every path loses (or ties) in both orders.
   The cross-fitted oracle's gain over always-reference is summed per class. Gain that sits in `split` contexts is
   judge noise by construction; only `decisive` gain is something a router could, in principle, aim for.
   (Analogous to the "selection effect" of Tikhomirov & Savchenko, submission #2935: the slice a gate forwards is
   where human annotators disagree; here, where the judge disagrees with itself.)

2. Is the decisive headroom identifiable before generation? A logistic classifier on pre-generation context
   features predicts "this context has a decisive gain" (trained on one split, AUROC on another). AUROC ~ 0.5 means
   the useful routing decision is not recoverable from the context with these features.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from careflow.routers.oracle import ORDERS, Scores, oracle


def context_classes(scores: Scores, paths: Sequence[str], reference: str) -> dict[str, str]:
    a, b = (scores.by_order[o] for o in ORDERS)
    out = {}
    for c in scores.contexts:
        alts = [p for p in paths if p != reference]
        if any(a[p][c] == 1 and b[p][c] == 1 for p in alts):
            out[c] = "decisive"
        elif any(a[p][c] == 1 or b[p][c] == 1 for p in alts):
            out[c] = "split"
        else:
            out[c] = "none"
    return out


def headroom_by_class(scores: Scores, paths: Sequence[str], reference: str,
                      cost: dict[str, float]) -> dict[str, dict[str, float]]:
    """Per class: share of contexts and the cross-fitted oracle's gain over always-reference (in win-rate points
    of the whole sample, so the gains add up to the total oracle gain)."""
    cls = context_classes(scores, paths, reference)
    n = len(scores.contexts)
    out = {}
    for k in ("decisive", "split", "none"):
        sub = [c for c in scores.contexts if cls[c] == k]
        if not sub:
            out[k] = {"share": 0.0, "gain": 0.0}
            continue
        q = oracle(Scores(scores.by_order, sub), paths, cost).quality
        out[k] = {"share": len(sub) / n, "gain": (q - 0.5) * len(sub) / n}
    return out


def order_consistency(scores: Scores, paths: Sequence[str], reference: str) -> dict[str, float]:
    a, b = (scores.by_order[o] for o in ORDERS)
    return {p: float(np.mean([a[p][c] == b[p][c] for c in scores.contexts])) for p in paths if p != reference}


def identifiability_auroc(x_train: np.ndarray, y_train: Sequence[int], x_test: np.ndarray, y_test: Sequence[int],
                          c: float, n_boot: int, seed: int) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    y_tr, y_te = np.asarray(y_train), np.asarray(y_test)
    clf = LogisticRegression(C=c, max_iter=5000, class_weight="balanced").fit(x_train, y_tr)
    p = clf.predict_proba(x_test)[:, 1]
    auc = float(roc_auc_score(y_te, p))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_te), len(y_te))
        if len(set(y_te[i])) == 2:
            boots.append(roc_auc_score(y_te[i], p[i]))
    return {"auroc": auc, "auroc_ci_low": float(np.percentile(boots, 2.5)),
            "auroc_ci_high": float(np.percentile(boots, 97.5)), "pos_rate_test": float(y_te.mean())}
