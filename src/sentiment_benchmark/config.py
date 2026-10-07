"""Typed experiment configuration loaded from YAML.

One YAML file fully describes a benchmark run: which datasets, which feature
sets, which models, how to split and cross-validate, where to write outputs.
Keeping the whole grid in configuration (not code) is what makes a run
reproducible and reviewable: the config is logged with every MLflow run.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import PROJECT_ROOT


@dataclass
class CleanConfig:
    lowercase: bool = True
    remove_urls: bool = True
    remove_mentions: bool = True
    keep_hashtags: bool = True
    remove_stopwords: bool = True
    min_word_len: int = 3
    stemmer: str | None = "porter"  # porter | None


@dataclass
class SplitConfig:
    strategy: str = "holdout"  # holdout | official
    test_size: float = 0.2
    seed: int = 42


@dataclass
class DatasetConfig:
    name: str
    enabled: bool = True
    clean: CleanConfig = field(default_factory=CleanConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    max_rows: int | None = None  # subsample for smoke runs
    raw_dir: str | None = None  # override for data/raw/<name>


@dataclass
class ExperimentConfig:
    name: str = "sentiment-benchmark"
    seed: int = 42
    cv_folds: int = 5
    datasets: list[DatasetConfig] = field(default_factory=list)
    features: dict[str, dict[str, Any]] = field(default_factory=dict)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    grid: dict[str, list[str]] = field(default_factory=dict)  # model -> feature names
    reports_dir: Path = PROJECT_ROOT / "reports"
    models_dir: Path = PROJECT_ROOT / "models"
    data_dir: Path = PROJECT_ROOT / "data"
    tracking: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def dataset(self, name: str) -> DatasetConfig:
        for d in self.datasets:
            if d.name == name:
                return d
        raise KeyError(f"dataset {name!r} not in config")


def _dataset_from_dict(d: dict[str, Any]) -> DatasetConfig:
    d = dict(d)
    clean = CleanConfig(**d.pop("clean", {}) or {})
    split = SplitConfig(**d.pop("split", {}) or {})
    return DatasetConfig(clean=clean, split=split, **d)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load a YAML experiment file into an ExperimentConfig.

    `overrides` is a flat dict of top-level keys (e.g. {"cv_folds": 2}) applied
    after loading; useful for CLI flags such as --smoke.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    raw = copy.deepcopy(raw)
    if overrides:
        raw.update(overrides)

    datasets = [_dataset_from_dict(d) for d in raw.get("datasets", [])]
    cfg = ExperimentConfig(
        name=raw.get("name", "sentiment-benchmark"),
        seed=int(raw.get("seed", 42)),
        cv_folds=int(raw.get("cv_folds", 5)),
        datasets=datasets,
        features=raw.get("features", {}) or {},
        models=raw.get("models", {}) or {},
        grid=raw.get("grid", {}) or {},
        reports_dir=_resolve(raw.get("reports_dir"), PROJECT_ROOT / "reports"),
        models_dir=_resolve(raw.get("models_dir"), PROJECT_ROOT / "models"),
        data_dir=_resolve(raw.get("data_dir"), PROJECT_ROOT / "data"),
        tracking=raw.get("tracking", {}) or {},
        source_path=path.resolve(),
        raw=raw,
    )
    _validate(cfg)
    return cfg


def _resolve(value: str | None, default: Path) -> Path:
    if not value:
        return default
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _validate(cfg: ExperimentConfig) -> None:
    if not cfg.datasets:
        raise ValueError("config has no datasets")
    for model, feats in cfg.grid.items():
        if model not in cfg.models:
            raise ValueError(f"grid references unknown model {model!r}")
        for f in feats:
            if f not in cfg.features:
                raise ValueError(f"grid model {model!r} references unknown feature set {f!r}")
    if cfg.cv_folds < 2:
        raise ValueError("cv_folds must be >= 2")
