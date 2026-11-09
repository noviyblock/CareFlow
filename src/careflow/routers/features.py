"""Context encoders for pre-generation routers.

- minilm: sentence embedding of the last seeker turns (all-MiniLM-L6-v2).
- llm_hidden: frozen hidden states of a small LLM (Qwen3.5-0.8B, MLX) over the chat-formatted dialogue history,
  final-norm output, mean-pooled and last-token concatenated. A linear head on top is the text analogue of the
  SpeechLLM router (a classification head on an LLM's hidden states choosing the downstream path); the LLM stays
  frozen, so the router costs one prefill of a 0.8B model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class LLMHiddenEncoder:
    def __init__(self, model_id: str, revision: str, max_tokens: int = 1024) -> None:
        from mlx_lm import load

        self.model, self.tok = load(model_id, revision=revision)
        self.max_tokens = max_tokens

    def _prompt(self, history: Sequence[str], speakers: Sequence[str]) -> list[int]:
        msgs = [{"role": "user" if s == "seeker" else "assistant", "content": t}
                for t, s in zip(history, speakers, strict=True)]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
        return list(ids)[-self.max_tokens:]  # keep the most recent turns

    def encode(self, contexts: Sequence[tuple[Sequence[str], Sequence[str]]]) -> np.ndarray:
        import mlx.core as mx

        inner = self.model.language_model.model if hasattr(self.model, "language_model") else self.model.model
        out = []
        for history, speakers in contexts:
            h = inner(mx.array([self._prompt(history, speakers)]))[0].astype(mx.float32)
            v = mx.concatenate([h.mean(axis=0), h[-1]])
            mx.eval(v)
            out.append(np.array(v))
        x = np.stack(out)
        return x / np.linalg.norm(x, axis=1, keepdims=True)
