"""Blend the local TweetEval ensemble with a Twitter-trained RoBERTa model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def normalize_tweet(text: str) -> str:
    """Use the mention and URL normalization specified by the Cardiff model."""
    return " ".join("@user" if word.startswith("@") else "http" if word.startswith("http") else word for word in text.split())


class TwitterEnsemble:
    def __init__(self, base: Any, transformer_dir: Path, weight: float, card: dict[str, Any] | None = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.base = base
        self.dataset = base.dataset
        self.model_dir = base.model_dir
        self.weight = weight
        self.card = card or base.card
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_dir, local_files_only=True)
        self.transformer = AutoModelForSequenceClassification.from_pretrained(transformer_dir, local_files_only=True)
        self.transformer.eval()

    def transformer_proba(self, texts: list[str], batch_size: int = 24) -> np.ndarray:
        import torch

        blocks = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = [normalize_tweet(text) for text in texts[start : start + batch_size]]
                encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
                logits = self.transformer(**encoded).logits
                blocks.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(blocks, axis=0) if blocks else np.empty((0, 3))

    def base_proba(self, texts: list[str]) -> np.ndarray:
        classes = np.asarray(self.base.pipeline.named_steps["model"].classes_)
        return self.base.pipeline.predict_proba(texts)[:, np.argsort(classes)]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return (1 - self.weight) * self.base_proba(texts) + self.weight * self.transformer_proba(texts)

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        probabilities = self.predict_proba(texts)
        names = self.card["label_names"]
        results = []
        for text, proba in zip(texts, probabilities):
            label = int(np.argmax(proba))
            results.append(
                {
                    "text": text,
                    "label": label,
                    "label_name": names[label],
                    "confidence": float(proba[label]),
                    "probabilities": {name: float(value) for name, value in zip(names, proba)},
                    "threshold": None,
                }
            )
        return results
