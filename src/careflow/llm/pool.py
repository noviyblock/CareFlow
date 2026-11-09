"""Role-based model pool (LRU, bounded) and a cache wrapper.

Cache hits return the stored Generation with `cached=True`; its latency is the
original measurement. Latency experiments must run with `cache.mode=off`.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from careflow.cache import LLMCache
from careflow.llm.base import ChatModel, Generation, Message


def _build(spec: dict[str, Any]) -> ChatModel:
    if spec["backend"] == "mlx":
        from careflow.llm.mlx_backend import MLXChat

        return MLXChat(spec["id"], spec["revision"], spec.get("chat_template_kwargs"))
    if spec["backend"] == "openai_compat":
        from careflow.llm.openai_backend import OpenAICompatChat

        return OpenAICompatChat(spec["id"], spec["revision"], spec["base_url"], spec.get("served_name"),
                                spec.get("api_key_env", "OPENAI_API_KEY"), spec.get("chat_template_kwargs"))
    raise ValueError(f"Unknown backend {spec['backend']!r}")


class CachedChat:
    """`model` may be a ChatModel or a zero-argument loader; a loader runs only on the first cache miss, so a run
    replayed entirely from the cache never loads weights (no GPU memory, no load time)."""

    def __init__(self, model: ChatModel | Callable[[], ChatModel], spec: dict[str, Any],
                 cache: LLMCache | None) -> None:
        self.spec, self.cache = spec, cache
        self._model, self._loader = (model, None) if hasattr(model, "chat") else (None, model)
        self._lock = threading.Lock()
        self.name = self._model.name if self._model is not None else f"{spec['id']}@{spec['revision'][:12]}"

    @property
    def model(self) -> ChatModel:
        if self._model is None:
            with self._lock:  # several worker threads may hit the first cache miss at once
                if self._model is None:
                    self._model = self._loader()
        return self._model

    def chat(
        self, messages: list[Message], max_tokens: int, temperature: float = 0.0, seed: int = 0
    ) -> Generation:
        request = {
            "model": self.spec["id"],
            "revision": self.spec["revision"],
            "template_kwargs": self.spec.get("chat_template_kwargs") or {},
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
        }
        if self.cache is not None and (hit := self.cache.get(request)) is not None:
            return Generation(**{**hit, "cached": True})
        gen = self.model.chat(messages, max_tokens, temperature, seed)
        if self.cache is not None:
            self.cache.put(request, gen.to_dict())
        return gen


class ModelPool:
    def __init__(
        self,
        specs: dict[str, dict[str, Any]],
        max_loaded: int,
        cache: LLMCache | None = None,
        builder: Callable[[dict[str, Any]], ChatModel] = _build,
    ) -> None:
        self.specs, self.max_loaded, self.cache, self.builder = specs, max_loaded, cache, builder
        self._loaded: OrderedDict[str, ChatModel] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, role: str) -> CachedChat:
        spec = self.specs[role]
        return CachedChat(lambda: self._load(spec), spec, self.cache)

    def _load(self, spec: dict[str, Any]) -> ChatModel:
        with self._lock:
            return self._load_locked(spec)

    def _load_locked(self, spec: dict[str, Any]) -> ChatModel:
        key = f"{spec['id']}@{spec['revision']}"
        if key in self._loaded:
            self._loaded.move_to_end(key)
        else:
            while len(self._loaded) >= self.max_loaded:
                _, old = self._loaded.popitem(last=False)
                old.close()
            self._loaded[key] = self.builder(spec)
        return self._loaded[key]

    def close(self) -> None:
        while self._loaded:
            _, m = self._loaded.popitem()
            m.close()
