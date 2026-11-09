"""Positive controls: replies that are known to be worse than the source system's, built from its outputs
without new LLM calls. If the protocol (judge + statistics) does not detect them, a "no difference" result for
real systems is uninterpretable (docs/robustness_plan.md, H3).

- generic:   one fixed, content-free supportive sentence for every context.
- truncated: only the first sentence of the source reply (shorter; tests the length-controlled metric too).
- shuffled:  the source reply written for a *different* context (deterministic derangement). Same style and
             length distribution as the source, but not responsive — the strongest length-matched control.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

_SENT_END = re.compile(r"(?<=[.!?])\s+")


def first_sentence(text: str) -> str:
    return _SENT_END.split(text.strip(), maxsplit=1)[0]


def derangement(n: int, seed: int) -> list[int]:
    """Permutation without fixed points (n >= 2)."""
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return [int(i) for i in p]


def make_control(kind: str, source_replies: Sequence[str], seed: int, text: str | None = None) -> list[str]:
    if kind == "generic":
        if not text:
            raise ValueError("generic control needs `text`")
        return [text] * len(source_replies)
    if kind == "truncated":
        return [first_sentence(r) for r in source_replies]
    if kind == "shuffled":
        return [source_replies[j] for j in derangement(len(source_replies), seed)]
    raise ValueError(f"unknown control {kind!r}")
