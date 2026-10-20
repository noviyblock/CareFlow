"""Bradley-Terry strengths from pairwise outcomes, reported on an Elo-like scale (HEART uses BT/Elo)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

ELO_SCALE = 400 / np.log(10)


def fit(outcomes: Sequence[tuple[str, str, int]], l2: float = 1e-2, base: float = 1500.0) -> dict[str, float]:
    """`outcomes`: (player_i, player_j, 1 if i won else 0). Returns Elo-scaled strengths centred on `base`."""
    players = sorted({p for i, j, _ in outcomes for p in (i, j)})
    if len(players) < 2:
        raise ValueError("need at least two players")
    idx = {p: k for k, p in enumerate(players)}
    x = np.zeros((len(outcomes), len(players)))
    y = np.zeros(len(outcomes))
    for row, (i, j, won) in enumerate(outcomes):
        x[row, idx[i]], x[row, idx[j]], y[row] = 1.0, -1.0, won
    # Symmetrize so the intercept-free model is not biased by which side is listed first.
    x, y = np.vstack([x, -x]), np.concatenate([y, 1 - y])
    model = LogisticRegression(fit_intercept=False, C=1 / l2, max_iter=1000).fit(x, y)
    beta = model.coef_[0] - model.coef_[0].mean()
    return {p: float(base + ELO_SCALE * beta[idx[p]]) for p in players}
