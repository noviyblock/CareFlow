"""MLX (Apple Silicon) backend. Imported lazily so the rest of the package works without mlx."""

from __future__ import annotations

import gc
import time
from typing import Any

from careflow.llm.base import Generation, Message


class MLXChat:
    def __init__(self, model_id: str, revision: str, chat_template_kwargs: dict[str, Any] | None = None):
        from mlx_lm import load

        self.name = f"{model_id}@{revision[:12]}"
        self.model_id, self.revision = model_id, revision
        self.template_kwargs = chat_template_kwargs or {}
        self.model, self.tokenizer = load(model_id, revision=revision)

    def chat(
        self, messages: list[Message], max_tokens: int, temperature: float = 0.0, seed: int = 0
    ) -> Generation:
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **self.template_kwargs
        )
        mx.random.seed(seed)
        sampler = make_sampler(temp=temperature)
        parts, last = [], None
        start = time.perf_counter()
        for last in stream_generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens, sampler=sampler):
            parts.append(last.text)
        latency = time.perf_counter() - start
        return Generation(
            text="".join(parts).strip(),
            model=self.name,
            prompt_tokens=last.prompt_tokens if last else 0,
            completion_tokens=last.generation_tokens if last else 0,
            latency_s=latency,
            finish_reason=getattr(last, "finish_reason", None),
        )

    def close(self) -> None:
        import mlx.core as mx

        del self.model, self.tokenizer
        gc.collect()
        mx.clear_cache()
