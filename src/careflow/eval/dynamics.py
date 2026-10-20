"""Emotion-trajectory metrics over a dialogue.

Micro-dynamics follow Empath (arXiv:2609.29056): transition matrices,
persistence, polarity shifts. The two sycophancy-related flags ("too fast"
recovery, "stuck" in negative) are our hypothesis, not part of Empath; their
thresholds live in the experiment config and are calibrated on data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

POLARITY_VALUE = {"neg": -1, "neu": 0, "pos": 1}


def transition_matrix(seqs: Sequence[Sequence[str]], labels: Sequence[str]) -> np.ndarray:
    """Row-normalized P(next = j | current = i), pooled over sequences. Rows without data stay 0."""
    index = {lab: i for i, lab in enumerate(labels)}
    counts = np.zeros((len(labels), len(labels)))
    for seq in seqs:
        for a, b in pairwise(seq):
            counts[index[a], index[b]] += 1
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)


def run_lengths(seq: Sequence[str], state: str) -> list[int]:
    runs, cur = [], 0
    for s in seq:
        if s == state:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


@dataclass(frozen=True)
class Trajectory:
    """Per-dialogue summary of a seeker polarity sequence (values in neg/neu/pos)."""

    n: int
    polarity_shifts: int
    start: int
    end: int
    deescalated: bool
    escalated: bool
    first_pos_turn: int | None
    longest_neg_run: int
    too_fast: bool
    stuck: bool


def trajectory(polarity: Sequence[str], fast_k: int, stuck_k: int) -> Trajectory:
    """`fast_k`: neg -> pos within that many seeker turns from the first neg; `stuck_k`: last k all neg."""
    vals = [POLARITY_VALUE[p] for p in polarity]
    if not vals:
        raise ValueError("Empty polarity sequence")
    shifts = sum(a != b for a, b in pairwise(vals))
    first_neg = next((i for i, v in enumerate(vals) if v < 0), None)
    first_pos = next((i for i, v in enumerate(vals) if v > 0 and (first_neg is None or i > first_neg)), None)
    too_fast = first_neg is not None and first_pos is not None and first_pos - first_neg <= fast_k
    stuck = len(vals) >= stuck_k and all(v < 0 for v in vals[-stuck_k:])
    return Trajectory(
        n=len(vals),
        polarity_shifts=shifts,
        start=vals[0],
        end=vals[-1],
        deescalated=vals[-1] > vals[0],
        escalated=vals[-1] < vals[0],
        first_pos_turn=first_pos,
        longest_neg_run=max(run_lengths(polarity, "neg"), default=0),
        too_fast=too_fast,
        stuck=stuck,
    )


def summarize(trajs: Sequence[Trajectory]) -> dict[str, float]:
    if not trajs:
        return {}
    return {
        "n_dialogues": float(len(trajs)),
        "deescalation_rate": float(np.mean([t.deescalated for t in trajs])),
        "escalation_rate": float(np.mean([t.escalated for t in trajs])),
        "polarity_shifts_per_turn": float(np.mean([t.polarity_shifts / max(t.n - 1, 1) for t in trajs])),
        "longest_neg_run_mean": float(np.mean([t.longest_neg_run for t in trajs])),
        "too_fast_rate": float(np.mean([t.too_fast for t in trajs])),
        "stuck_rate": float(np.mean([t.stuck for t in trajs])),
    }
