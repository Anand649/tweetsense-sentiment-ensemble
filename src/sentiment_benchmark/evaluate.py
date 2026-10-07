"""Metrics, threshold tuning and plots.

Protocol (same for every dataset x feature x model cell):
1. Stratified k-fold on the training split gives out-of-fold probabilities.
   For binary tasks the decision threshold is tuned on those OOF probabilities
   to maximise F1 (the thesis hard-coded 0.3); the test set never sees it.
2. The pipeline is refitted on the whole training split and scored once on the
   held-out test split at the tuned threshold. Both are reported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def tune_threshold(y_true: np.ndarray, p_pos: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Threshold on P(positive) that maximises F1 on the given (OOF) data."""
    grid = np.linspace(0.05, 0.95, 91) if grid is None else grid
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (p_pos >= t).astype(int), zero_division=0)
        if f1 > best_f1 + 1e-12:
            best_t, best_f1 = float(t), float(f1)
    return best_t


def predictions_from_proba(proba: np.ndarray, classes: np.ndarray, task: str, threshold: float | None) -> np.ndarray:
    if task == "binary" and threshold is not None:
        pos_col = int(np.where(classes == 1)[0][0]) if 1 in classes else 1
        return (proba[:, pos_col] >= threshold).astype(int)
    return classes[proba.argmax(axis=1)]


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray, task: str, threshold: float | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = predictions_from_proba(proba, classes, task, threshold)
    avg = "binary" if task == "binary" else "macro"
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
    }
    if task == "binary":
        pos_col = int(np.where(classes == 1)[0][0]) if 1 in classes else 1
        p = proba[:, pos_col]
        out["f1_at_0.5"] = f1_score(y_true, (p >= 0.5).astype(int), zero_division=0)
        if len(np.unique(y_true)) == 2:
            out["roc_auc"] = roc_auc_score(y_true, p)
            out["pr_auc"] = average_precision_score(y_true, p)
    else:
        out["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        if len(np.unique(y_true)) == proba.shape[1]:
            out["roc_auc"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
    return {k: float(v) for k, v in out.items()}


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for i, name in enumerate(label_names):
        mask = np.asarray(y_true) == i
        rows.append(
            {
                "class": name,
                "support": int(mask.sum()),
                "precision": float(precision_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)),
            }
        )
    return rows


def plot_confusion(y_true, y_pred, label_names: list[str], title: str, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_names)), label_names, rotation=30, ha="right")
    ax.set_yticks(range(len(label_names)), label_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_comparison(results, dataset: str, metric: str, path: Path) -> Path:
    """Grouped bar chart: one bar per feature set, grouped by model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame([r for r in results if r["dataset"] == dataset])
    if df.empty:
        return path
    pivot = df.pivot_table(index="model", columns="features", values=metric, aggfunc="max")
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=150)
    pivot.plot(kind="bar", ax=ax, width=0.8, edgecolor="white")
    ax.set_ylabel(metric)
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.set_title(f"{dataset}: test {metric} by model and feature set", fontsize=10)
    ax.legend(title="features", fontsize=7, title_fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Report figures (built from saved probabilities; see benchmark.make_figures)
# --------------------------------------------------------------------------- #


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7})
    return plt


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def best_per_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per model: its best feature set by cross-validated F1."""
    best: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r["model"] not in best or r["cv_f1_mean"] > best[r["model"]]["cv_f1_mean"]:
            best[r["model"]] = r
    return sorted(best.values(), key=lambda r: -r["cv_f1_mean"])


def plot_roc_curves(rows, y_test, label_names, dataset, path: Path) -> Path:
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=150)
    if len(label_names) == 2:
        for r in best_per_model(rows):
            fpr, tpr, _ = roc_curve(y_test, r["_proba_test"][:, 1])
            ax.plot(fpr, tpr, lw=1.4, label=f"{r['model']} / {r['features']} (AUC {auc(fpr, tpr):.3f})")
        ax.set_title(f"{dataset}: ROC curves, best feature set per model")
    else:
        r = best_per_model(rows)[0]
        yb = label_binarize(y_test, classes=list(range(len(label_names))))
        for i, name in enumerate(label_names):
            fpr, tpr, _ = roc_curve(yb[:, i], r["_proba_test"][:, i])
            ax.plot(fpr, tpr, lw=1.4, label=f"{name} (AUC {auc(fpr, tpr):.3f})")
        ax.set_title(f"{dataset}: one-vs-rest ROC, {r['model']} / {r['features']}")
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="grey")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_pr_curves(rows, y_test, label_names, dataset, path: Path) -> Path:
    from sklearn.metrics import average_precision_score, precision_recall_curve
    from sklearn.preprocessing import label_binarize

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=150)
    if len(label_names) == 2:
        for r in best_per_model(rows):
            p, rc, _ = precision_recall_curve(y_test, r["_proba_test"][:, 1])
            ax.plot(rc, p, lw=1.4, label=f"{r['model']} / {r['features']} (AP {average_precision_score(y_test, r['_proba_test'][:, 1]):.3f})")
        ax.axhline(float(np.mean(y_test)), ls="--", lw=0.8, color="grey", label="positive rate")
        ax.set_title(f"{dataset}: precision-recall, best feature set per model")
    else:
        r = best_per_model(rows)[0]
        yb = label_binarize(y_test, classes=list(range(len(label_names))))
        for i, name in enumerate(label_names):
            p, rc, _ = precision_recall_curve(yb[:, i], r["_proba_test"][:, i])
            ax.plot(rc, p, lw=1.4, label=f"{name} (AP {average_precision_score(yb[:, i], r['_proba_test'][:, i]):.3f})")
        ax.set_title(f"{dataset}: one-vs-rest precision-recall, {r['model']} / {r['features']}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    return _save(fig, path)


def plot_calibration(rows, y_test, dataset, path: Path, n_bins: int = 10, top: int = 5) -> Path:
    from sklearn.calibration import calibration_curve

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=150)
    for r in best_per_model(rows)[:top]:
        frac, mean_pred = calibration_curve(y_test, r["_proba_test"][:, 1], n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", ms=3, lw=1.2, label=f"{r['model']} / {r['features']}")
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="grey", label="perfectly calibrated")
    ax.set_xlabel("mean predicted P(positive)")
    ax.set_ylabel("observed positive fraction")
    ax.set_title(f"{dataset}: reliability diagram (quantile bins)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    return _save(fig, path)


def plot_threshold_sweep(row, y_train, dataset, path: Path) -> Path:
    from sklearn.metrics import precision_score, recall_score

    plt = _mpl()
    p = row["_oof"][:, 1]
    grid = np.linspace(0.02, 0.98, 97)
    f1s, precs, recs = [], [], []
    for t in grid:
        pred = (p >= t).astype(int)
        f1s.append(f1_score(y_train, pred, zero_division=0))
        precs.append(precision_score(y_train, pred, zero_division=0))
        recs.append(recall_score(y_train, pred, zero_division=0))
    fig, ax = plt.subplots(figsize=(4.8, 3.4), dpi=150)
    ax.plot(grid, f1s, lw=1.6, label="F1")
    ax.plot(grid, precs, lw=1.1, label="precision")
    ax.plot(grid, recs, lw=1.1, label="recall")
    ax.axvline(row["threshold"], ls="--", lw=0.9, color="black", label=f"chosen {row['threshold']:.2f}")
    ax.axvline(0.5, ls=":", lw=0.8, color="grey", label="default 0.5")
    ax.set_xlabel("decision threshold on P(positive)")
    ax.set_ylabel("score (out-of-fold)")
    ax.set_title(f"{dataset}: threshold sweep, {row['model']} / {row['features']}")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", ncol=2)
    return _save(fig, path)


def plot_cv_variance(rows, dataset, path: Path) -> Path:
    plt = _mpl()
    rs = sorted(rows, key=lambda r: r["cv_f1_mean"])
    labels = [f"{r['model']} / {r['features']}" for r in rs]
    fig, ax = plt.subplots(figsize=(6.0, 0.28 * len(rs) + 1.2), dpi=150)
    y = np.arange(len(rs))
    ax.barh(y, [r["cv_f1_mean"] for r in rs], xerr=[r["cv_f1_std"] for r in rs], color="#4c72b0", alpha=0.85, capsize=2, height=0.6)
    ax.scatter([r["test_f1"] for r in rs], y, color="#dd8452", s=14, zorder=3, label="test F1")
    ax.set_yticks(y, labels)
    ax.set_xlabel("F1 (bars: cross-validated mean and std; dots: held-out test)")
    ax.set_title(f"{dataset}: cross-validation stability and test agreement")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_efficiency(rows, dataset, path: Path) -> Path:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=150)
    models = list(dict.fromkeys(r["model"] for r in rows))
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(models):
        sub = [r for r in rows if r["model"] == m]
        ax.scatter([max(r["fit_seconds"], 0.05) for r in sub], [r["test_f1"] for r in sub], color=cmap(i % 10), s=26, label=m, zorder=3)
        for r in sub:
            ax.annotate(r["features"], (max(r["fit_seconds"], 0.05), r["test_f1"]), fontsize=5.5, xytext=(3, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("training time on the full training split (seconds, log scale)")
    ax.set_ylabel("test F1")
    ax.set_title(f"{dataset}: accuracy versus training cost")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", ncol=2)
    return _save(fig, path)


def plot_heatmap(rows, dataset, path: Path, metric: str = "test_f1") -> Path:
    import pandas as pd

    plt = _mpl()
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    pivot = df.pivot_table(index="model", columns="features", values=metric, aggfunc="max")
    fig, ax = plt.subplots(figsize=(1.1 * pivot.shape[1] + 2.2, 0.42 * pivot.shape[0] + 1.4), dpi=150)
    im = ax.imshow(pivot.values, cmap="YlGnBu", vmin=max(0, np.nanmin(pivot.values) - 0.05), vmax=np.nanmax(pivot.values))
    ax.set_xticks(range(pivot.shape[1]), pivot.columns)
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color="white" if v > np.nanmax(pivot.values) - 0.08 else "black")
    ax.set_title(f"{dataset}: {metric.replace('test_', 'test ')} by model and feature set")
    fig.colorbar(im, ax=ax, fraction=0.03)
    return _save(fig, path)


def plot_per_class(row, y_test, label_names, dataset, path: Path) -> Path:
    plt = _mpl()
    rep = row["per_class"]
    x = np.arange(len(rep))
    fig, ax = plt.subplots(figsize=(4.8, 3.2), dpi=150)
    for key, off in (("precision", -0.25), ("recall", 0.0), ("f1", 0.25)):
        ax.bar(x + off, [c[key] for c in rep], width=0.25, label=key)
    ax.set_xticks(x, [f"{c['class']}\n(n={c['support']})" for c in rep])
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{dataset}: per-class scores, {row['model']} / {row['features']}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", ncol=3)
    return _save(fig, path)
