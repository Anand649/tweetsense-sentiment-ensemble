from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def tiny_binary() -> tuple[list[str], list[int]]:
    pos = ["love this great film", "wonderful acting and story", "best movie ever", "really enjoyed it", "brilliant and fun"]
    neg = ["terrible waste of time", "awful boring plot", "worst movie ever", "hated every minute", "dull and bad"]
    texts = (pos + neg) * 6
    labels = ([1] * 5 + [0] * 5) * 6
    return texts, labels


@pytest.fixture(scope="session")
def smoke_run(root: Path, tmp_path_factory):
    """Run the smoke benchmark once per session into a temp dir on the
    synthetic corpus (data_dir points at an empty folder, so no real data is
    touched); shared by the benchmark, predict, monitor and app tests."""
    from sentiment_benchmark.benchmark import run_benchmark
    from sentiment_benchmark.config import load_config

    out = tmp_path_factory.mktemp("smoke")
    cfg = load_config(
        root / "configs" / "smoke.yaml",
        {"reports_dir": str(out / "reports"), "models_dir": str(out / "models"), "data_dir": str(out / "data")},
    )
    cfg.tracking["enabled"] = False
    rows = run_benchmark(cfg, only_models=["naive_bayes", "logistic_regression", "cnn"])
    return cfg, rows
