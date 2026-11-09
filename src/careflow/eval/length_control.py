"""Length-controlled win-rates, following Length-Controlled AlpacaEval (Dubois et al., 2024, arXiv:2404.04475).

Our judge strongly prefers longer replies (docs/results.md). We fit one logistic GLM on all head-to-head
judgments vs the reference:  P(system wins) = sigmoid(alpha_system + beta * tanh(dlen / sigma) + gamma * shown_first)
and report sigmoid(alpha_system), i.e. the win-rate the system would have at equal length, averaged over the
presentation order (gamma is centred by coding shown_first as +-0.5). Following the paper, the length term is
shared across systems and squashed with tanh so very long answers cannot dominate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class Judgment:
    ctx: str
    system: str
    won: int        # 1 if `system` beat the reference
    dlen: float     # words(system reply) - words(reference reply)
    shown_first: int


def fit_lc(judgments: Sequence[Judgment], systems: Sequence[str], sigma: float) -> tuple[dict[str, float], float]:
    """Returns (length-controlled win-rate per system, length coefficient beta)."""
    idx = {s: i for i, s in enumerate(systems)}
    x = np.zeros((len(judgments), len(systems) + 2))
    y = np.zeros(len(judgments))
    for r, j in enumerate(judgments):
        x[r, idx[j.system]] = 1.0
        x[r, -2] = np.tanh(j.dlen / sigma)
        x[r, -1] = j.shown_first - 0.5
        y[r] = j.won
    m = LogisticRegression(fit_intercept=False, C=1e4, max_iter=2000).fit(x, y)
    coef = m.coef_[0]
    return {s: float(1 / (1 + np.exp(-coef[idx[s]]))) for s in systems}, float(coef[-2])


def bootstrap_lc(judgments: Sequence[Judgment], systems: Sequence[str], sigma: float, n: int,
                 seed: int) -> dict[str, tuple[float, float]]:
    """95% CI by resampling contexts (both orders of a context move together)."""
    draws = bootstrap_lc_draws(judgments, systems, sigma, n, seed)
    return {s: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for s, v in draws.items()}


def bootstrap_lc_draws(judgments: Sequence[Judgment], systems: Sequence[str], sigma: float, n: int,
                       seed: int) -> dict[str, np.ndarray]:
    """Bootstrap distribution of the LC win-rate per system (contexts resampled)."""
    by_ctx: dict[str, list[Judgment]] = {}
    for j in judgments:
        by_ctx.setdefault(j.ctx, []).append(j)
    ctxs = list(by_ctx)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {s: [] for s in systems}
    for _ in range(n):
        sample = [j for c in rng.choice(ctxs, len(ctxs)) for j in by_ctx[c]]
        present = [s for s in systems if any(j.system == s for j in sample)]
        lc, _ = fit_lc(sample, present, sigma)
        for s in present:
            draws[s].append(lc[s])
    return {s: np.asarray(v) for s, v in draws.items()}


def length_adjusted(judgments: Sequence[Judgment], systems: Sequence[str], sigma: float) -> dict[tuple[str, str], float]:
    """Per (ctx, system): mean over orders of won - [P(win | dlen) - P(win | dlen=0)] under the fitted GLM,
    i.e. the score with the estimated length advantage removed."""
    idx = {s: i for i, s in enumerate(systems)}
    x = np.zeros((len(judgments), len(systems) + 2))
    y = np.array([j.won for j in judgments], float)
    for r, j in enumerate(judgments):
        x[r, idx[j.system]], x[r, -2], x[r, -1] = 1.0, np.tanh(j.dlen / sigma), j.shown_first - 0.5
    coef = LogisticRegression(fit_intercept=False, C=1e4, max_iter=2000).fit(x, y).coef_[0]
    out: dict[tuple[str, str], list[float]] = {}
    for r, j in enumerate(judgments):
        base = coef[idx[j.system]] + coef[-1] * x[r, -1]
        effect = 1 / (1 + np.exp(-(base + coef[-2] * x[r, -2]))) - 1 / (1 + np.exp(-base))
        out.setdefault((j.ctx, j.system), []).append(y[r] - effect)
    return {k: float(np.mean(v)) for k, v in out.items()}
