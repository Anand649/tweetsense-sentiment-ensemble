"""Runs the smoke grid once (session fixture) and checks every downstream
artefact: results table, model card, exported pipeline, predict API, drift
monitor and the Flask app."""

import json

import pandas as pd

from sentiment_benchmark.monitor import drift_report, load_baseline
from sentiment_benchmark.predict import available_models, load_model


def test_benchmark_writes_reports_and_exports(smoke_run):
    cfg, rows = smoke_run
    assert rows and {r["dataset"] for r in rows} == {"twitter_hate", "vaccination"}
    table = pd.read_csv(cfg.reports_dir / "results.csv")
    assert {"cv_f1_mean", "test_f1", "test_accuracy", "threshold"} <= set(table.columns)
    assert (cfg.reports_dir / "results.md").read_text().startswith("# Benchmark results")
    for fig in ("compare", "heatmap", "cv", "roc", "pr", "per_class", "confusion", "efficiency", "calibration", "threshold"):
        assert (cfg.reports_dir / "figures" / f"{fig}_twitter_hate.png").exists(), fig
    assert not (cfg.reports_dir / "figures" / "calibration_vaccination.png").exists()  # binary-only figure
    assert (cfg.reports_dir / "predictions" / "twitter_hate" / "labels.npz").exists()
    for ds in ("twitter_hate", "vaccination"):
        card = json.loads((cfg.models_dir / ds / "model_card.json").read_text())
        assert card["model"] in {"naive_bayes", "logistic_regression", "cnn"}
        assert card["selection"].startswith("highest cross-validated")
        assert (cfg.models_dir / ds / "best.joblib").exists()
        assert (cfg.models_dir / ds / "baseline_stats.json").exists()
    # binary task tuned a threshold, multiclass did not
    binary = table[table.dataset == "twitter_hate"]
    assert binary["threshold"].notna().all() and table[table.dataset == "vaccination"]["threshold"].isna().all()


def test_predict_api_uses_model_card(smoke_run):
    cfg, _ = smoke_run
    assert set(available_models(cfg.models_dir)) == {"twitter_hate", "vaccination"}
    model = load_model("vaccination", cfg.models_dir)
    out = model.predict(["great vaccine experience", "awful"])
    assert len(out) == 2 and out[0]["label_name"] in model.label_names
    assert abs(sum(out[0]["probabilities"].values()) - 1) < 1e-6


def test_drift_report_flags_shifted_batch(smoke_run):
    cfg, _ = smoke_run
    model = load_model("twitter_hate", cfg.models_dir)
    baseline = load_baseline(model.model_dir)
    from sentiment_benchmark.datasets import synthetic_dataset

    same = drift_report(model.pipeline, baseline, synthetic_dataset("twitter_hate", n=150, seed=11).frame["text"].tolist())
    assert same["checks"]["length_psi"]["status"] == "ok"
    shifted = drift_report(model.pipeline, baseline, ["zzqx " * 80] * 60)
    assert shifted["status"] == "alert" and shifted["checks"]["oov_rate"]["value"] > 0.9


def test_flask_app_routes(smoke_run):
    from app import create_app

    cfg, _ = smoke_run
    client = create_app(models_dir=cfg.models_dir, reports_dir=cfg.reports_dir).test_client()
    assert client.get("/").status_code == 200
    health = client.get("/health").get_json()
    assert health["status"] == "ok" and "twitter_hate" in health["models"]
    r = client.post("/api/predict", json={"dataset": "twitter_hate", "texts": ["hello world"]})
    assert r.status_code == 200 and r.get_json()["results"][0]["label_name"] in ("hate", "not_hate")
    assert client.post("/api/predict", json={}).status_code == 400
    assert client.post("/api/predict", json={"dataset": "missing", "text": "x"}).status_code == 404
    assert client.post("/api/predict", json={"texts": ["x"] * 201}).status_code == 413
    stats = client.get("/api/stats").get_json()
    assert stats["backend"] == "sqlite" and stats["twitter_hate"]["predictions"] >= 1
    assert health["store"] == "sqlite"


def test_report_regenerates_figures_from_saved_probabilities(smoke_run):
    from sentiment_benchmark.benchmark import load_saved_rows, make_figures

    cfg, rows = smoke_run
    reloaded, y_train, y_test, label_names = load_saved_rows(cfg, "twitter_hate")
    assert len(reloaded) == len([r for r in rows if r["dataset"] == "twitter_hate"])
    (cfg.reports_dir / "figures" / "roc_twitter_hate.png").unlink()
    figs = make_figures(reloaded, "twitter_hate", y_train, y_test, label_names, cfg.reports_dir / "figures")
    assert (cfg.reports_dir / "figures" / "roc_twitter_hate.png").exists() and len(figs) == 10


def test_prediction_store_sqlite_and_postgres_url_shape(tmp_path):
    from sentiment_benchmark.storage import PredictionStore, default_url

    store = PredictionStore(f"sqlite:///{tmp_path / 'p.db'}")
    n = store.log_predictions("twitter_hate", "cnn", "sequence", [{"text": "hello", "label": 0, "label_name": "not_hate", "confidence": 0.9}], 3.2)
    assert n == 1 and store.recent_texts("twitter_hate") == ["hello"] and store.recent_texts("vaccination") == []
    store.log_drift("twitter_hate", {"status": "warn", "n": 1, "checks": {}})
    s = store.stats("twitter_hate")
    assert s["predictions"] == 1 and s["by_label"] == {"not_hate": 1} and s["last_drift"]["status"] == "warn"
    assert default_url(tmp_path).startswith("sqlite:///") or default_url(tmp_path).startswith("postgresql")


def test_prediction_store_against_postgres_when_available():
    import os

    import pytest

    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    from sentiment_benchmark.storage import PredictionStore

    store = PredictionStore(url)
    assert store.backend == "postgresql"
    store.log_predictions("vaccination", "lstm", "sequence", [{"text": "pg row", "label": 2, "label_name": "positive", "confidence": 0.8}])
    assert "pg row" in store.recent_texts("vaccination")
