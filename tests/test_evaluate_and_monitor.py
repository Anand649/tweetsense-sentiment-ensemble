import numpy as np

from sentiment_benchmark.evaluate import compute_metrics, predictions_from_proba, tune_threshold
from sentiment_benchmark.monitor import psi


def test_threshold_tuning_prefers_low_threshold_for_rare_positives():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.07).astype(int)
    p = np.clip(0.25 * y + rng.normal(0, 0.1, 2000) + 0.1, 0, 1)  # positives score around 0.35
    t = tune_threshold(y, p)
    assert 0.15 <= t <= 0.4
    classes = np.array([0, 1])
    m = compute_metrics(y, np.c_[1 - p, p], classes, "binary", t)
    assert m["f1"] >= m["f1_at_0.5"]
    assert 0 <= m["roc_auc"] <= 1 and "pr_auc" in m


def test_multiclass_metrics_are_macro():
    y = np.array([0, 1, 2, 0, 1, 2])
    proba = np.eye(3)[[0, 1, 2, 0, 1, 1]]
    m = compute_metrics(y, proba, np.array([0, 1, 2]), "multiclass")
    assert m["accuracy"] == 5 / 6 and "f1_weighted" in m and abs(m["f1"] - np.mean([1.0, 0.8, 2 / 3])) < 1e-9


def test_predictions_respect_threshold():
    proba = np.array([[0.8, 0.2], [0.6, 0.4]])
    assert predictions_from_proba(proba, np.array([0, 1]), "binary", 0.3).tolist() == [0, 1]
    assert predictions_from_proba(proba, np.array([0, 1]), "binary", None).tolist() == [0, 0]


def test_psi_zero_for_identical_and_positive_for_shift():
    assert psi([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert psi([0.9, 0.1], [0.5, 0.5]) > 0.25


def test_tracker_sqlite_and_jsonl_replay(tmp_path):
    import json

    from sentiment_benchmark.tracking import Tracker, replay_jsonl

    t = Tracker("tracker-test", tmp_path, enabled=True)
    assert t.backend.startswith("mlflow (sqlite:///")
    with t.run("cell", tags={"dataset": "x"}) as r:
        r.params({"model": "nb", "feature_params": {"k": 1}})
        r.metrics({"test_f1": 0.5})
    rec = json.loads((tmp_path / "runs.jsonl").read_text().strip())
    assert rec["metrics"]["test_f1"] == 0.5 and rec["params"]["feature_params"] == '{"k": 1}'
    off = Tracker("off", tmp_path / "off", enabled=False)
    with off.run("cell2") as r:
        r.metrics({"test_f1": 0.7})
    assert off.backend == "jsonl"
    assert replay_jsonl(tmp_path / "off" / "runs.jsonl", f"sqlite:///{tmp_path / 'replay.db'}") == 1
