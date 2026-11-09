"""Utterance-level emotion labels with a local HF classifier (one forward pass per batch)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class EmotionClassifier:
    def __init__(
        self, model_id: str, revision: str, device: str, max_length: int = 128, multi_label: bool = False
    ) -> None:
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision)
        self.model.to(device).eval()
        self.device = device
        self.max_length = max_length
        self.multi_label = multi_label  # sigmoid per label (e.g. GoEmotions) instead of softmax
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    @torch.inference_mode()
    def predict_proba(self, texts: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = self.tok(
                list(texts[i : i + batch_size]),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**batch).logits.float()
            out.append((logits.sigmoid() if self.multi_label else logits.softmax(-1)).cpu())
        return torch.cat(out) if out else torch.empty(0, len(self.labels))
