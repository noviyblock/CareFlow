"""OpenAI-compatible backend: a vLLM server (Colab A100, `scripts/colab_serve.sh`) or any hosted endpoint.

Throughput comes from concurrency, not from this class: callers issue requests from a thread pool
(`compute.concurrency`) and the server batches them continuously. The client is thread-safe.

Reproducibility caveats (documented in docs/colab.md): weights are the HF checkpoints at a pinned revision, not the
MLX 4-bit builds, so outputs differ from local runs and the judge must be re-validated on this backend; vLLM with
continuous batching is not bit-deterministic even at temperature 0. The LLM cache keeps every response, so any run
can still be replayed exactly. Latency measured here is server wall-clock under load and is not comparable with
the M5 Pro numbers — never mix the two in one table.
"""

from __future__ import annotations

import os
import time
from typing import Any

from careflow.llm.base import Generation, Message


class OpenAICompatChat:
    def __init__(self, model_id: str, revision: str, base_url: str, served_name: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", chat_template_kwargs: dict[str, Any] | None = None,
                 timeout_s: float = 600.0) -> None:
        from openai import OpenAI

        self.name = f"{model_id}@{revision[:12]}"
        self.served_name = served_name or model_id
        self.template_kwargs = chat_template_kwargs or {}
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, "EMPTY"), timeout=timeout_s)

    def chat(
        self, messages: list[Message], max_tokens: int, temperature: float = 0.0, seed: int = 0
    ) -> Generation:
        extra = {"chat_template_kwargs": self.template_kwargs} if self.template_kwargs else None
        start = time.perf_counter()
        r = self.client.chat.completions.create(
            model=self.served_name, messages=messages, max_tokens=max_tokens, temperature=temperature, seed=seed,
            extra_body=extra)
        latency = time.perf_counter() - start
        choice = r.choices[0]
        return Generation(
            text=(choice.message.content or "").strip(),
            model=self.name,
            prompt_tokens=r.usage.prompt_tokens if r.usage else 0,
            completion_tokens=r.usage.completion_tokens if r.usage else 0,
            latency_s=latency,
            finish_reason=choice.finish_reason,
        )

    def close(self) -> None:
        self.client.close()
