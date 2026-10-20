from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

Message = dict[str, str]


@dataclass(frozen=True)
class Generation:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    finish_reason: str | None = None
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChatModel(Protocol):
    name: str

    def chat(
        self, messages: list[Message], max_tokens: int, temperature: float = 0.0, seed: int = 0
    ) -> Generation: ...

    def close(self) -> None: ...
