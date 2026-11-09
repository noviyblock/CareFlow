"""Common interface for support systems: respond(context) -> final text + every LLM call made."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from careflow.data.contexts import Context
from careflow.llm.base import Generation


@dataclass
class SystemOutput:
    text: str
    calls: list[Generation] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def latency_s(self) -> float:
        """Calls run sequentially, so end-to-end latency is their sum."""
        return sum(g.latency_s for g in self.calls)

    @property
    def cached(self) -> bool:
        return any(g.cached for g in self.calls)

    @property
    def completion_tokens(self) -> int:
        return sum(g.completion_tokens for g in self.calls)

    @property
    def prompt_tokens(self) -> int:
        return sum(g.prompt_tokens for g in self.calls)


class Chat(Protocol):
    name: str

    def chat(self, messages: list[dict[str, str]], max_tokens: int, temperature: float = 0.0, seed: int = 0) -> Generation: ...


class System(Protocol):
    name: str

    def respond(self, ctx: Context) -> SystemOutput: ...


class Recorder:
    """Wraps a chat model and records every call into the current SystemOutput."""

    def __init__(self, model: Chat, seed: int) -> None:
        self.model, self.seed = model, seed

    def ask(self, out: SystemOutput, step: str, user: str, max_tokens: int, system: str | None = None,
            temperature: float = 0.0, seed_offset: int = 0) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        g = self.model.chat(msgs, max_tokens, temperature, self.seed + seed_offset)
        out.calls.append(g)
        out.trace.append({"step": step, "text": g.text})
        return g.text
