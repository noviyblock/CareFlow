"""Equivalence testing and power for head-to-head win-rates.

"Not significantly different" is not evidence of no difference. We use two one-sided tests (TOST; Schuirmann,
1987; Lakens, 2017, doi:10.1177/1948550617697177): a system is *equivalent* to the reference within a margin
delta if the (1 - 2*alpha) bootstrap CI of its win-rate lies inside [0.5 - delta, 0.5 + delta]. The unit of
resampling is the context (both presentation orders move together), so the per-context score is in {0, .5, 1}.

Verdicts (per margin, alpha already Bonferroni-adjusted by the caller if needed):
- better / worse: the usual two-sided (1 - alpha) CI excludes 0.5;
- equivalent: the (1 - 2*alpha) CI lies inside the margin (a system can be both "worse" and "equivalent":
  a real but negligible gap);
- inconclusive: neither — the data cannot rule out a relevant effect.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import norm


def bootstrap_mean(scores: Sequence[float], n: int, seed: int) -> np.ndarray:
    s = np.asarray(scores, float)
    rng = np.random.default_rng(seed)
    return s[rng.integers(0, len(s), (n, len(s)))].mean(axis=1)


def verdict(diff_ci: tuple[float, float], eq_ci: tuple[float, float], delta: float) -> str:
    eq = 0.5 - delta < eq_ci[0] and eq_ci[1] < 0.5 + delta
    if diff_ci[1] < 0.5:
        return "worse, equivalent" if eq else "worse"
    if diff_ci[0] > 0.5:
        return "better, equivalent" if eq else "better"
    return "equivalent" if eq else "inconclusive"


def tost_p(draws: np.ndarray, delta: float) -> float:
    """Bootstrap TOST p-value: max of the two one-sided p-values for H0: |wr - 0.5| >= delta."""
    p_low = float(np.mean(draws <= 0.5 - delta))
    p_high = float(np.mean(draws >= 0.5 + delta))
    return max(p_low, p_high)


def required_n(sd: float, delta: float, alpha: float = 0.05, power: float = 0.8, true_diff: float = 0.0) -> int:
    """Contexts needed for TOST to conclude equivalence with the given power (normal approximation).
    true_diff = 0 uses z_{1-beta/2} (both bounds bind), otherwise z_{1-beta} against the nearer bound."""
    margin = delta - abs(true_diff)
    if margin <= 0:
        return -1  # equivalence cannot be shown if the true effect is outside the margin
    zb = norm.ppf(1 - (1 - power) / 2) if true_diff == 0 else norm.ppf(power)
    return int(np.ceil(((norm.ppf(1 - alpha) + zb) * sd / margin) ** 2))


def min_detectable_effect(sd: float, n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest |wr - 0.5| a two-sided test at level alpha detects with the given power."""
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * sd / np.sqrt(n))
