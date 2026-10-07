"""Model registry.

`make_model(name, params, task)` returns an unfitted scikit-learn compatible
estimator. Classical models come from scikit-learn / XGBoost; `cnn` and `lstm`
are PyTorch models wrapped in a scikit-learn estimator (see `neural.py`).
Every estimator exposes `predict_proba`, which the evaluator relies on for
threshold tuning and ROC / PR curves.
"""

from __future__ import annotations

from typing import Any

from .classical import CLASSICAL_MODELS, make_classical

NEURAL_MODELS = ("cnn", "lstm")
ALL_MODELS = tuple(CLASSICAL_MODELS) + NEURAL_MODELS


def make_model(name: str, params: dict[str, Any] | None = None, task: str = "binary", seed: int = 42):
    params = dict(params or {})
    kind = params.pop("type", name)
    if kind in CLASSICAL_MODELS:
        return make_classical(kind, params, task, seed)
    if kind in NEURAL_MODELS:
        from .neural import TorchTextClassifier

        return TorchTextClassifier(arch=kind, seed=seed, **params)
    raise KeyError(f"unknown model {name!r} (type {kind!r}); known: {ALL_MODELS}")


def requires_nonnegative(name: str) -> bool:
    return name == "naive_bayes"


__all__ = ["make_model", "ALL_MODELS", "CLASSICAL_MODELS", "NEURAL_MODELS", "requires_nonnegative"]
