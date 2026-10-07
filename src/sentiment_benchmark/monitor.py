"""Post-deployment monitoring: data drift on incoming text and prediction drift.

At export time `baseline_stats` records, on the training data after cleaning:
token-length distribution, vocabulary (top terms) and the predicted class
distribution. `drift_report` compares a new batch against that baseline with
the Population Stability Index on length bins, the out-of-vocabulary rate and
the predicted-label shift. It is the check a scheduled job (or the `monitor`
CLI) runs before anyone trusts a week of production predictions.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

LENGTH_BINS = [0, 2, 4, 6, 8, 10, 13, 16, 20, 30, 50, 100, 10_000]


def _length_hist(docs: list[str]) -> list[float]:
    lengths = np.array([len(d.split()) for d in docs], dtype=float)
    hist, _ = np.histogram(lengths, bins=LENGTH_BINS)
    return (hist / max(len(lengths), 1)).tolist()


def psi(expected: list[float], actual: list[float], eps: float = 1e-4) -> float:
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def baseline_stats(pipeline, texts: list[str], label_names: list[str], top_terms: int = 5000) -> dict[str, Any]:
    cleaned = pipeline.named_steps["clean"].transform(texts)
    counts = Counter(t for d in cleaned for t in d.split())
    vocab = [w for w, _ in counts.most_common(top_terms)]
    preds = pipeline.predict(texts) if len(texts) <= 20000 else pipeline.predict(texts[:20000])
    pred_dist = np.bincount(np.asarray(preds, dtype=int), minlength=len(label_names)) / max(len(preds), 1)
    return {
        "n": len(texts),
        "length_hist": _length_hist(cleaned),
        "length_bins": LENGTH_BINS,
        "mean_tokens": float(np.mean([len(d.split()) for d in cleaned])) if cleaned else 0.0,
        "vocab_top": vocab,
        "vocab_size": len(counts),
        "empty_after_clean_rate": float(np.mean([d == "" for d in cleaned])) if cleaned else 0.0,
        "pred_dist": pred_dist.tolist(),
        "label_names": label_names,
    }


def drift_report(pipeline, baseline: dict[str, Any], texts: list[str], psi_warn: float = 0.1, psi_alert: float = 0.25, oov_warn: float = 0.3) -> dict[str, Any]:
    cleaned = pipeline.named_steps["clean"].transform(texts)
    vocab = set(baseline["vocab_top"])
    tokens = [t for d in cleaned for t in d.split()]
    oov_rate = float(np.mean([t not in vocab for t in tokens])) if tokens else 0.0
    length_psi = psi(baseline["length_hist"], _length_hist(cleaned))
    preds = np.asarray(pipeline.predict(texts), dtype=int)
    pred_dist = (np.bincount(preds, minlength=len(baseline["label_names"])) / max(len(preds), 1)).tolist()
    pred_psi = psi(baseline["pred_dist"], pred_dist)
    empty_rate = float(np.mean([d == "" for d in cleaned])) if cleaned else 0.0

    def level(value: float, warn: float, alert: float) -> str:
        return "alert" if value >= alert else ("warn" if value >= warn else "ok")

    checks = {
        "length_psi": {"value": length_psi, "status": level(length_psi, psi_warn, psi_alert)},
        "prediction_psi": {"value": pred_psi, "status": level(pred_psi, psi_warn, psi_alert)},
        "oov_rate": {"value": oov_rate, "status": level(oov_rate, oov_warn, oov_warn * 2)},
        "empty_after_clean_rate": {"value": empty_rate, "status": level(empty_rate, 0.05, 0.2)},
    }
    worst = max((c["status"] for c in checks.values()), key=["ok", "warn", "alert"].index)
    return {
        "n": len(texts),
        "status": worst,
        "checks": checks,
        "pred_dist": dict(zip(baseline["label_names"], pred_dist)),
        "baseline_pred_dist": dict(zip(baseline["label_names"], baseline["pred_dist"])),
        "mean_tokens": float(np.mean([len(d.split()) for d in cleaned])) if cleaned else 0.0,
        "baseline_mean_tokens": baseline["mean_tokens"],
    }


def load_baseline(model_dir: Path) -> dict[str, Any]:
    return json.loads((Path(model_dir) / "baseline_stats.json").read_text(encoding="utf-8"))
