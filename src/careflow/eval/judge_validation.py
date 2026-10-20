"""Sampling and scoring for the judge-validation experiment (HEART completions, no human labels)."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import kendalltau, spearmanr

from careflow.eval import bradley_terry

HUMAN = "human"


@dataclass(frozen=True)
class Pair:
    context: int
    a: str  # shown as A0493
    b: str  # shown as A0488
    swapped_of: int | None = None  # index of the pair this one mirrors


def players_for_row(row: dict[str, Any], reference: dict[str, dict[str, Any]], include_human: bool) -> dict[str, str]:
    out = {}
    for name, ref in reference.items():
        text = next((row[c] for c in ref["columns"] if str(row.get(c) or "").strip()), None)
        if text:
            out[name] = text.strip()
    if include_human and str(row.get("human_completion_1") or "").strip():
        out[HUMAN] = row["human_completion_1"].strip()
    return out


def sample_pairs(
    available: Sequence[Sequence[str]], n_contexts: int, pairs_per_context: int, swap_fraction: float, seed: int
) -> list[Pair]:
    """`available[i]` lists players present in context i. Presentation order is randomized."""
    rng = random.Random(seed)
    contexts = rng.sample(range(len(available)), min(n_contexts, len(available)))
    pairs: list[Pair] = []
    for c in sorted(contexts):
        names = sorted(available[c])
        all_pairs = [(x, y) for i, x in enumerate(names) for y in names[i + 1 :]]
        for x, y in rng.sample(all_pairs, min(pairs_per_context, len(all_pairs))):
            pairs.append(Pair(c, x, y) if rng.random() < 0.5 else Pair(c, y, x))
    for k in rng.sample(range(len(pairs)), round(swap_fraction * len(pairs))):
        p = pairs[k]
        pairs.append(Pair(p.context, p.b, p.a, swapped_of=k))
    return pairs


def outcomes(pairs: Sequence[Pair], winners: Sequence[str | None]) -> list[tuple[str, str, int]]:
    """Winner per pair is a player name (or None if the judgment failed). Mirrored pairs count too."""
    return [(p.a, p.b, int(w == p.a)) for p, w in zip(pairs, winners, strict=True) if w is not None]


def position_consistency(pairs: Sequence[Pair], winners: Sequence[str | None]) -> float:
    agree = [
        winners[i] == winners[p.swapped_of]
        for i, p in enumerate(pairs)
        if p.swapped_of is not None and winners[i] is not None and winners[p.swapped_of] is not None
    ]
    return float(np.mean(agree)) if agree else float("nan")


def rank_agreement(scores: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    common = sorted(set(scores) & set(reference))
    x, y = [scores[m] for m in common], [reference[m] for m in common]
    tau = kendalltau(x, y)
    rho = spearmanr(x, y)
    return {"kendall_tau": float(tau.statistic), "kendall_p": float(tau.pvalue),
            "spearman": float(rho.statistic), "n_models": float(len(common))}


def bootstrap_tau(
    pairs: Sequence[Pair], winners: Sequence[str | None], reference: dict[str, float], n: int, seed: int
) -> tuple[float, float]:
    """95% CI of Kendall tau, resampling contexts (pairs within a context move together)."""
    rng = np.random.default_rng(seed)
    by_ctx: dict[int, list[int]] = {}
    for i, p in enumerate(pairs):
        by_ctx.setdefault(p.context, []).append(i)
    ctxs = list(by_ctx)
    taus = []
    for _ in range(n):
        idx = [i for c in rng.choice(ctxs, size=len(ctxs), replace=True) for i in by_ctx[c]]
        res = outcomes([pairs[i] for i in idx], [winners[i] for i in idx])
        players = {p for a, b, _ in res for p in (a, b)} & set(reference)
        if len(players) < 3:
            continue
        taus.append(rank_agreement(bradley_terry.fit(res), reference)["kendall_tau"])
    return float(np.percentile(taus, 2.5)), float(np.percentile(taus, 97.5))
