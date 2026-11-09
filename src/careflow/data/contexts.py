"""Evaluation contexts: a dialogue prefix ending right before a supporter turn, plus the human reply."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from careflow.data.schema import Dialogue


@dataclass
class Context:
    dialogue_id: str
    history: list[str]
    speakers: list[str]
    emotion: str | None
    situation: str | None
    human: str  # reference supporter reply
    responses: dict[str, str] = field(default_factory=dict)


def sample_contexts(dialogues: Sequence[Dialogue], n: int, min_history: int, seed: int) -> list[Context]:
    """One target supporter turn per dialogue (uniform among turns with >= min_history prior turns)."""
    rng = random.Random(seed)
    out = []
    for d in dialogues:
        targets = [t.idx for t in d.turns if t.role == "supporter" and t.idx >= min_history]
        if not targets:
            continue
        j = rng.choice(targets)
        out.append(Context(d.dialogue_id, [t.text for t in d.turns[:j]], [t.role for t in d.turns[:j]],
                           d.emotion, d.situation, d.turns[j].text))
    return rng.sample(out, min(n, len(out)))
