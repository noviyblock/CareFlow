"""Turn-level signals of the seeker's emotional state and their validity against self-reports.

Windows skip greetings/closings and very short turns: in ESConv the first seeker
turn is often "Hello" and the last one "thank you, bye", so endpoint metrics
mostly measure that ritual (see docs/data.md).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import spearmanr

_CLOSING = re.compile(
    r"\b(?:thanks?|thank you|bye|goodbye|good night|take care|appreciate|have a (?:good|great|nice))\b", re.IGNORECASE
)


def substantive(texts: Sequence[str], min_words: int) -> list[int]:
    """Indices of turns that are long enough and not a closing formula."""
    return [i for i, t in enumerate(texts) if len(t.split()) >= min_words and not _CLOSING.search(t)]


def windows(texts: Sequence[str], k: int, min_words: int) -> tuple[list[int], list[int]] | None:
    """First and last k substantive turns (non-overlapping), or None if the dialogue is too short."""
    idx = substantive(texts, min_words)
    if len(idx) < 2 * k:
        return None
    return idx[:k], idx[-k:]


def spearman_ci(x: Sequence[float], y: Sequence[float], n_boot: int, seed: int) -> dict[str, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    res = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            boots.append(spearmanr(x[i], y[i]).statistic)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"rho": float(res.statistic), "p": float(res.pvalue), "ci_low": float(lo), "ci_high": float(hi), "n": float(len(x))}


def label_mass(probs: np.ndarray, labels: Sequence[str], subset: Sequence[str]) -> np.ndarray:
    """Sum of probabilities over `subset` labels (continuous negativity score)."""
    cols = [labels.index(lab) for lab in subset]
    return probs[:, cols].sum(axis=1)


_DIGIT = re.compile(r"[1-5]")

LLM_PROMPT = (
    "Below are messages written by a person who is seeking emotional support.\n"
    "Rate how intense their negative emotional distress is in these messages, on a scale from "
    "1 (very mild) to 5 (very intense). Answer with a single digit only.\n\n{messages}"
)


def llm_intensity(chat: Callable[[str], str], turns: Sequence[str]) -> int | None:
    text = chat(LLM_PROMPT.format(messages="\n".join(f"- {t}" for t in turns)))
    m = _DIGIT.search(text)
    return int(m.group()) if m else None


def paired_rho_diff(
    a: Sequence[float], b: Sequence[float], y: Sequence[float], n_boot: int, seed: int
) -> dict[str, float]:
    """rho(a, y) - rho(b, y) on the same items, with a paired bootstrap 95% CI."""
    a, b, y = (np.asarray(v, float) for v in (a, b, y))
    diff = spearmanr(a, y).statistic - spearmanr(b, y).statistic
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if np.ptp(a[i]) > 0 and np.ptp(b[i]) > 0 and np.ptp(y[i]) > 0:
            boots.append(spearmanr(a[i], y[i]).statistic - spearmanr(b[i], y[i]).statistic)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(diff), "ci_low": float(lo), "ci_high": float(hi), "p_le_0": float(np.mean(np.array(boots) <= 0))}
