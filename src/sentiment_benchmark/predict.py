"""Load an exported model and predict: the single code path used by the CLI,
the Flask app and the tests, so serving can never diverge from evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from . import PROJECT_ROOT
from .evaluate import predictions_from_proba


@dataclass
class LoadedModel:
    dataset: str
    pipeline: Any
    card: dict[str, Any]
    model_dir: Path

    @property
    def label_names(self) -> list[str]:
        return self.card["label_names"]

    @property
    def threshold(self) -> float | None:
        return self.card.get("threshold")

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        proba = self.pipeline.predict_proba(texts)
        classes = np.asarray(self.pipeline.named_steps["model"].classes_)
        order = np.argsort(classes)
        proba = proba[:, order]
        classes = classes[order]
        preds = predictions_from_proba(proba, classes, self.card["task"], self.threshold)
        out = []
        for text, p, y in zip(texts, proba, preds):
            out.append(
                {
                    "text": text,
                    "label": int(y),
                    "label_name": self.label_names[int(y)],
                    "confidence": float(p[int(y)]),
                    "probabilities": {self.label_names[int(c)]: float(v) for c, v in zip(classes, p)},
                    "threshold": self.threshold,
                }
            )
        return out


def available_models(models_dir: Path | None = None) -> list[str]:
    models_dir = Path(models_dir or PROJECT_ROOT / "models")
    return sorted(p.parent.name for p in models_dir.glob("*/best.joblib"))


def load_model(dataset: str, models_dir: Path | None = None) -> LoadedModel:
    models_dir = Path(models_dir or PROJECT_ROOT / "models")
    model_dir = models_dir / dataset
    path = model_dir / "best.joblib"
    if not path.exists():
        raise FileNotFoundError(f"no exported model for {dataset!r} at {path}; run the benchmark first")
    card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
    return LoadedModel(dataset, joblib.load(path), card, model_dir)
