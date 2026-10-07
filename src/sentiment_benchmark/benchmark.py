"""Benchmark orchestrator: datasets x feature sets x models.

For every dataset the training split is cleaned once and every feature set is
fitted once per cross-validation fold (and once on the full training split);
the cached matrices are shared by all models evaluated on that feature set.
Each model then gets out-of-fold probabilities (binary tasks tune the decision
threshold on them), is refitted on the whole training split and scored once on
the held-out test split. Every cell is an MLflow run; the best pipeline per
dataset (by cross-validated F1, never by test score) is exported with a model
card and the monitoring baseline.

    python -m sentiment_benchmark.cli benchmark --config configs/benchmark.yaml
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from . import __version__
from .config import DatasetConfig, ExperimentConfig
from .datasets import Dataset, load_dataset, write_manifest
from .evaluate import (
    compute_metrics,
    per_class_report,
    plot_calibration,
    plot_comparison,
    plot_confusion,
    plot_cv_variance,
    plot_efficiency,
    plot_heatmap,
    plot_per_class,
    plot_pr_curves,
    plot_roc_curves,
    plot_threshold_sweep,
    predictions_from_proba,
    tune_threshold,
)
from .features import make_feature_extractor
from .models import make_model, requires_nonnegative
from .monitor import baseline_stats
from .preprocess import TextCleaner
from .tracking import Tracker

PRIMARY_METRIC = "f1"


@dataclass
class Cell:
    dataset: str
    features: str
    model: str

    @property
    def name(self) -> str:
        return f"{self.dataset}/{self.model}/{self.features}"


def build_pipeline(cfg: ExperimentConfig, ds_cfg: DatasetConfig, cell: Cell, task: str) -> Pipeline:
    """The canonical clean -> features -> model pipeline for one grid cell."""
    feat_params = dict(cfg.features.get(cell.features, {}))
    feat_params.setdefault("type", cell.features)
    model_params = dict(cfg.models.get(cell.model, {}))
    model_params.setdefault("type", cell.model)
    return Pipeline(
        [
            ("clean", TextCleaner.from_config(ds_cfg.clean)),
            ("features", make_feature_extractor(cell.features, feat_params)),
            ("model", make_model(cell.model, model_params, task=task, seed=cfg.seed)),
        ]
    )


def expand_grid(cfg: ExperimentConfig, only_models: list[str] | None = None, only_features: list[str] | None = None) -> list[tuple[str, str]]:
    cells = []
    for model, feats in cfg.grid.items():
        if only_models and model not in only_models:
            continue
        for f in feats:
            if only_features and f not in only_features:
                continue
            ftype = cfg.features.get(f, {}).get("type", f)
            if requires_nonnegative(cfg.models.get(model, {}).get("type", model)) and ftype not in ("bow", "tfidf"):
                continue  # MultinomialNB needs non-negative features
            cells.append((model, f))
    return cells


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


class FeatureCache:
    """Cleaned text and fitted feature matrices, computed once per dataset.

    Cleaning is the same for every feature set, so it runs once. Each feature
    set is fitted on every CV training fold (transforming its validation fold)
    and once on the full training split (transforming the test split). Models
    consume these cached matrices, which keeps the protocol leakage-free while
    avoiding one feature fit per model.
    """

    def __init__(self, cfg: ExperimentConfig, ds_cfg: DatasetConfig, train: pd.DataFrame, test: pd.DataFrame):
        self.cfg, self.ds_cfg = cfg, ds_cfg
        self.cleaner = TextCleaner.from_config(ds_cfg.clean)
        t0 = time.time()
        self.train_clean = self.cleaner.transform(train["text"].tolist())
        self.test_clean = self.cleaner.transform(test["text"].tolist())
        self.y_train = train["label"].to_numpy()
        self.y_test = test["label"].to_numpy()
        self.clean_seconds = time.time() - t0
        skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
        self.folds = list(skf.split(np.zeros(len(self.y_train)), self.y_train))
        self._fold_cache: dict[str, list[tuple[Any, Any]]] = {}
        self._full_cache: dict[str, tuple[Any, Any, Any]] = {}
        self.seconds: dict[str, float] = {}

    def _extractor(self, name: str):
        params = dict(self.cfg.features.get(name, {}))
        params.setdefault("type", name)
        return make_feature_extractor(name, params)

    def fold_matrices(self, name: str) -> list[tuple[Any, Any]]:
        if name not in self._fold_cache:
            t0 = time.time()
            mats = []
            texts = np.asarray(self.train_clean, dtype=object)
            for tr, va in self.folds:
                ext = self._extractor(name).fit(list(texts[tr]), self.y_train[tr])
                mats.append((ext.transform(list(texts[tr])), ext.transform(list(texts[va]))))
            self._fold_cache[name] = mats
            self.seconds[name] = self.seconds.get(name, 0.0) + time.time() - t0
        return self._fold_cache[name]

    def full(self, name: str) -> tuple[Any, Any, Any]:
        """(fitted extractor, X_train, X_test) on the full training split."""
        if name not in self._full_cache:
            t0 = time.time()
            ext = self._extractor(name).fit(self.train_clean, self.y_train)
            self._full_cache[name] = (ext, ext.transform(self.train_clean), ext.transform(self.test_clean))
            self.seconds[name] = self.seconds.get(name, 0.0) + time.time() - t0
        return self._full_cache[name]


def _aligned_proba(est, X, classes: np.ndarray) -> np.ndarray:
    """predict_proba with columns aligned to the global class order."""
    proba = est.predict_proba(X)
    est_classes = list(est.classes_)
    out = np.zeros((proba.shape[0], len(classes)))
    for j, c in enumerate(classes):
        if c in est_classes:
            out[:, j] = proba[:, est_classes.index(c)]
    return out


def run_cell(cfg: ExperimentConfig, ds: Dataset, ds_cfg: DatasetConfig, cell: Cell, cache: FeatureCache, tracker: Tracker, out_dir: Path) -> dict[str, Any]:
    classes = np.array(sorted(np.unique(cache.y_train)))
    model_params = dict(cfg.models.get(cell.model, {}))
    model_params.setdefault("type", cell.model)
    proto = make_model(cell.model, model_params, task=ds.task, seed=cfg.seed)

    with tracker.run(cell.name, tags={"dataset": cell.dataset, "model": cell.model, "features": cell.features}) as run:
        run.params(
            {
                "dataset": cell.dataset,
                "model": cell.model,
                "features": cell.features,
                "task": ds.task,
                "n_train": len(cache.y_train),
                "n_test": len(cache.y_test),
                "cv_folds": cfg.cv_folds,
                "seed": cfg.seed,
                "feature_params": cfg.features.get(cell.features, {}),
                "model_params": cfg.models.get(cell.model, {}),
                "clean": ds_cfg.clean.__dict__,
                "data_fingerprint": ds.fingerprint(),
            }
        )
        t0 = time.time()
        # 1. out-of-fold probabilities on the training split
        oof = np.zeros((len(cache.y_train), len(classes)))
        for (tr, va), (X_tr, X_va) in zip(cache.folds, cache.fold_matrices(cell.features)):
            est = clone(proto).fit(X_tr, cache.y_train[tr])
            oof[va] = _aligned_proba(est, X_va, classes)
        cv_seconds = time.time() - t0
        threshold = tune_threshold(cache.y_train, oof[:, list(classes).index(1)]) if ds.task == "binary" else None
        cv_metrics = compute_metrics(cache.y_train, oof, classes, ds.task, threshold)
        fold_f1 = [compute_metrics(cache.y_train[va], oof[va], classes, ds.task, threshold)["f1"] for _, va in cache.folds]

        # 2. refit on the full training split, score the held-out test split once
        extractor, X_train, X_test = cache.full(cell.features)
        t1 = time.time()
        final = clone(proto).fit(X_train, cache.y_train)
        fit_seconds = time.time() - t1
        proba_te = _aligned_proba(final, X_test, classes)
        test_metrics = compute_metrics(cache.y_test, proba_te, classes, ds.task, threshold)
        y_pred = predictions_from_proba(proba_te, classes, ds.task, threshold)

        cm_path = plot_confusion(cache.y_test, y_pred, ds.label_names, cell.name, out_dir / "figures" / f"cm_{cell.dataset}_{cell.model}_{cell.features}.png")
        run.artifact(cm_path)
        run.metrics({f"cv_{k}": v for k, v in cv_metrics.items()})
        run.metrics({f"test_{k}": v for k, v in test_metrics.items()})
        run.metrics({"cv_f1_std": float(np.std(fold_f1)), "fit_seconds": fit_seconds, "cv_seconds": cv_seconds, "total_seconds": time.time() - t0})
        if threshold is not None:
            run.metrics({"threshold": threshold})
        if hasattr(final, "describe"):
            run.params({"model_state": final.describe()})

        row = {
            "dataset": cell.dataset,
            "model": cell.model,
            "features": cell.features,
            "task": ds.task,
            "n_train": int(len(cache.y_train)),
            "n_test": int(len(cache.y_test)),
            "threshold": threshold,
            "cv_f1_mean": cv_metrics["f1"],
            "cv_f1_std": float(np.std(fold_f1)),
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "fit_seconds": round(fit_seconds, 1),
            "cv_seconds": round(cv_seconds, 1),
            "per_class": per_class_report(cache.y_test, y_pred, ds.label_names),
        }
        # the deployable artefact: the same clean -> features -> model chain
        row["_pipeline"] = Pipeline([("clean", cache.cleaner), ("features", extractor), ("model", final)])
        row["_oof"], row["_proba_test"] = oof, proba_te
        pred_dir = out_dir / "predictions" / cell.dataset
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pred_dir / f"{cell.model}__{cell.features}.npz", oof=oof.astype(np.float32), proba_test=proba_te.astype(np.float32))
        if not (pred_dir / "labels.npz").exists():
            np.savez_compressed(pred_dir / "labels.npz", y_train=cache.y_train, y_test=cache.y_test, label_names=np.array(ds.label_names))
        return row


def make_figures(rows: list[dict[str, Any]], dataset: str, y_train: np.ndarray, y_test: np.ndarray, label_names: list[str], figures_dir: Path) -> list[Path]:
    """All report figures for one dataset from in-memory (or reloaded) rows."""
    rs = [r for r in rows if r["dataset"] == dataset and "_proba_test" in r]
    if not rs:
        return []
    plain = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rs]
    best = select_best(rs, dataset)
    out = [
        plot_comparison(plain, dataset, "test_f1", figures_dir / f"compare_{dataset}.png"),
        plot_heatmap(rs, dataset, figures_dir / f"heatmap_{dataset}.png"),
        plot_cv_variance(rs, dataset, figures_dir / f"cv_{dataset}.png"),
        plot_efficiency(rs, dataset, figures_dir / f"efficiency_{dataset}.png"),
        plot_roc_curves(rs, y_test, label_names, dataset, figures_dir / f"roc_{dataset}.png"),
        plot_pr_curves(rs, y_test, label_names, dataset, figures_dir / f"pr_{dataset}.png"),
        plot_per_class(best, y_test, label_names, dataset, figures_dir / f"per_class_{dataset}.png"),
        plot_confusion(
            y_test,
            predictions_from_proba(best["_proba_test"], np.arange(len(label_names)), best["task"], best["threshold"]),
            label_names,
            f"{dataset}: {best['model']} / {best['features']}",
            figures_dir / f"confusion_{dataset}.png",
        ),
    ]
    if len(label_names) == 2:
        out.append(plot_calibration(rs, y_test, dataset, figures_dir / f"calibration_{dataset}.png"))
        out.append(plot_threshold_sweep(best, y_train, dataset, figures_dir / f"threshold_{dataset}.png"))
    return out


def load_saved_rows(cfg: ExperimentConfig, dataset: str) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, list[str]]:
    """Rebuild benchmark rows from results.csv and the saved probabilities so
    figures and reports can be regenerated without retraining."""
    table = pd.read_csv(cfg.reports_dir / "results.csv")
    pred_dir = cfg.reports_dir / "predictions" / dataset
    labels = np.load(pred_dir / "labels.npz", allow_pickle=False)
    y_train, y_test, label_names = labels["y_train"], labels["y_test"], [str(x) for x in labels["label_names"]]
    classes = np.arange(len(label_names))
    rows = []
    for _, r in table[table["dataset"] == dataset].iterrows():
        f = pred_dir / f"{r['model']}__{r['features']}.npz"
        if not f.exists():
            continue
        arr = np.load(f)
        row = {k: (None if pd.isna(v) else v) for k, v in r.items()}
        row.setdefault("n_train", int(len(y_train)))
        row.setdefault("n_test", int(len(y_test)))
        row["_oof"], row["_proba_test"] = arr["oof"], arr["proba_test"]
        y_pred = predictions_from_proba(row["_proba_test"], classes, row["task"], row["threshold"])
        row["per_class"] = per_class_report(y_test, y_pred, label_names)
        rows.append(row)
    return rows, y_train, y_test, label_names


def select_best(rows: list[dict[str, Any]], dataset: str) -> dict[str, Any] | None:
    cands = [r for r in rows if r["dataset"] == dataset]
    if not cands:
        return None
    # primary: cross-validated F1 (never the test set), tie-break on test F1
    return max(cands, key=lambda r: (round(r["cv_f1_mean"], 4), r[f"test_{PRIMARY_METRIC}"]))


def export_best(cfg: ExperimentConfig, ds: Dataset, ds_cfg: DatasetConfig, best: dict[str, Any], train: pd.DataFrame, tracker_backend: str) -> Path:
    out = cfg.models_dir / ds.name
    out.mkdir(parents=True, exist_ok=True)
    pipe: Pipeline = best["_pipeline"]
    joblib.dump(pipe, out / "best.joblib")
    card = {
        "dataset": ds.name,
        "task": ds.task,
        "label_names": ds.label_names,
        "model": best["model"],
        "features": best["features"],
        "threshold": best["threshold"],
        "selection": "highest cross-validated F1 on the training split",
        "metrics": {k: v for k, v in best.items() if k.startswith("test_") or k.startswith("cv_")},
        "per_class": best["per_class"],
        "feature_params": cfg.features.get(best["features"], {}),
        "model_params": cfg.models.get(best["model"], {}),
        "clean": ds_cfg.clean.__dict__,
        "training_rows": int(len(train)),
        "data_fingerprint": ds.fingerprint(),
        "config": str(cfg.source_path.relative_to(cfg.reports_dir.parent))
        if cfg.source_path and cfg.source_path.is_relative_to(cfg.reports_dir.parent)
        else str(cfg.source_path),
        "git_commit": _git_commit(),
        "package_version": __version__,
        "python": platform.python_version(),
        "tracking_backend": tracker_backend,
        "created": pd.Timestamp.now("UTC").isoformat(),
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    baseline = baseline_stats(pipe, train["text"].tolist(), ds.label_names)
    (out / "baseline_stats.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    return out


def write_reports(rows: list[dict[str, Any]], cfg: ExperimentConfig, datasets: list[Dataset]) -> tuple[Path, Path]:
    """Write results.csv and results.md. Rows for the datasets of this run
    replace earlier rows for the same datasets; other datasets already in
    results.csv are kept, so runs can be split (for example IMDB alone)."""
    from .datasets import DATASET_META

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_") and k != "per_class"} for r in rows])
    csv_path = cfg.reports_dir / "results.csv"
    run_datasets = set(table["dataset"]) if not table.empty else set()
    if csv_path.exists():
        previous = pd.read_csv(csv_path)
        previous = previous[~previous["dataset"].isin(run_datasets)]
        if not previous.empty:
            table = pd.concat([previous, table], ignore_index=True)
            known = {d.name for d in datasets}
            for name in dict.fromkeys(previous["dataset"]):
                if name not in known:
                    label_names, task = DATASET_META.get(name, (sorted(previous[previous.dataset == name]["task"].unique()), "binary"))
                    datasets = datasets + [Dataset(name, pd.DataFrame({"text": [], "label": []}), list(label_names), task)]
            rows = rows + [dict(r) for _, r in previous.iterrows()]
    table.to_csv(csv_path, index=False)

    lines = [f"# Benchmark results: {cfg.name}", "", f"Generated {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC, {cfg.cv_folds}-fold CV, seed {cfg.seed}.", ""]
    for ds in datasets:
        sub = table[table["dataset"] == ds.name].copy()
        if sub.empty:
            continue
        best = select_best(rows, ds.name)
        metric_cols = ["test_accuracy", "test_precision", "test_recall", "test_f1"] + (
            ["test_roc_auc", "test_pr_auc"] if ds.task == "binary" else ["test_f1_weighted"]
        )
        metric_cols = [c for c in metric_cols if c in sub]
        sub = sub.sort_values("cv_f1_mean", ascending=False)
        lines.append(f"## {ds.name} ({ds.task}, classes: {', '.join(ds.label_names)})")
        lines.append("")
        lines.append(f"Best by cross-validated F1: **{best['model']} + {best['features']}** (test F1 {best['test_f1']:.3f}).")
        lines.append("")
        header = ["model", "features", "cv F1 (std)"] + [c.replace("test_", "") for c in metric_cols] + ["fit s"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for _, r in sub.iterrows():
            vals = (
                [r["model"], r["features"], f"{r['cv_f1_mean']:.3f} ({r['cv_f1_std']:.3f})"]
                + [f"{r[c]:.3f}" for c in metric_cols]
                + [f"{r['fit_seconds']:.0f}"]
            )
            lines.append("| " + " | ".join(str(v) for v in vals) + " |")
        lines.append("")
        for fig in ("compare", "heatmap", "cv", "roc", "pr", "calibration", "threshold", "per_class", "confusion", "efficiency"):
            if (cfg.reports_dir / "figures" / f"{fig}_{ds.name}.png").exists():
                lines.append(f"![{fig}](figures/{fig}_{ds.name}.png)")
        lines.append("")
    md_path = cfg.reports_dir / "results.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def run_benchmark(
    cfg: ExperimentConfig,
    only_datasets: list[str] | None = None,
    only_models: list[str] | None = None,
    only_features: list[str] | None = None,
    export: bool = True,
) -> list[dict[str, Any]]:
    tracker = Tracker(cfg.name, cfg.reports_dir, enabled=cfg.tracking.get("enabled", True), tracking_uri=cfg.tracking.get("uri"))
    print(f"[benchmark] {cfg.name}: tracking via {tracker.backend}, python {platform.python_version()}, {sys.platform}")
    rows: list[dict[str, Any]] = []
    loaded: list[Dataset] = []
    grid = expand_grid(cfg, only_models, only_features)
    for ds_cfg in cfg.datasets:
        if not ds_cfg.enabled or (only_datasets and ds_cfg.name not in only_datasets):
            continue
        ds = load_dataset(ds_cfg.name, cfg.data_dir, ds_cfg)
        loaded.append(ds)
        train, test = ds.split(ds_cfg)
        print(f"[benchmark] {ds.name}: {len(train)} train / {len(test)} test, classes {dict(zip(ds.label_names, np.bincount(train['label'])))}")
        cache = FeatureCache(cfg, ds_cfg, train, test)
        for model, feats in grid:
            cell = Cell(ds.name, feats, model)
            t0 = time.time()
            try:
                row = run_cell(cfg, ds, ds_cfg, cell, cache, tracker, cfg.reports_dir)
            except ImportError as exc:
                print(f"[benchmark] skip {cell.name}: {exc}")
                continue
            rows.append(row)
            print(
                f"[benchmark] {cell.name:45s} cv F1 {row['cv_f1_mean']:.3f}  test F1 {row['test_f1']:.3f}  acc {row['test_accuracy']:.3f}  ({time.time() - t0:.0f}s)"
            )
        print(f"[benchmark] {ds.name}: feature fitting seconds {json.dumps({k: round(v, 1) for k, v in cache.seconds.items()})}")
        figs = make_figures(rows, ds.name, cache.y_train, cache.y_test, ds.label_names, cfg.reports_dir / "figures")
        print(f"[benchmark] {ds.name}: {len(figs)} figures -> {cfg.reports_dir / 'figures'}")
        best = select_best(rows, ds.name)
        if best and export:
            out = export_best(cfg, ds, ds_cfg, best, train, tracker.backend)
            print(f"[benchmark] exported {best['model']} + {best['features']} -> {out}")
            with tracker.run(f"{ds.name}/best", tags={"dataset": ds.name, "stage": "export"}) as run:
                run.params({"model": best["model"], "features": best["features"], "threshold": best["threshold"]})
                run.metrics({k: v for k, v in best.items() if k.startswith("test_") and isinstance(v, float)})
                run.artifact(out / "model_card.json")
                run.model(best["_pipeline"], "best_pipeline")
        if loaded:
            write_manifest(loaded, cfg.data_dir)
            write_reports(rows, cfg, loaded)  # rewritten after every dataset so partial runs leave a report
    if loaded:
        csv_path, md_path = write_reports(rows, cfg, loaded)
        print(f"[benchmark] wrote {csv_path} and {md_path}")
    return rows
