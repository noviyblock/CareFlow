"""Common multi-turn dialogue schema shared by all datasets.

Roles are normalized to seeker (person needing support) / supporter. Fields a
dataset does not provide stay None; dataset-specific extras go into `meta`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["seeker", "supporter"]
Split = Literal["train", "valid", "test"]


class Turn(BaseModel):
    idx: int
    role: Role
    text: str
    # Supporter strategy label (ESConv: 8 Helping-Skills strategies).
    strategy: str | None = None
    # Seeker's in-dialogue rating of the support so far (ESConv: 1-5).
    feedback: int | None = None


class Dialogue(BaseModel):
    dialogue_id: str
    source: str
    split: Split
    turns: list[Turn]
    situation: str | None = None
    emotion: str | None = None
    problem_type: str | None = None
    # Seeker self-reported emotion intensity before / after the conversation (ESConv: 1-5).
    initial_intensity: int | None = None
    final_intensity: int | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
