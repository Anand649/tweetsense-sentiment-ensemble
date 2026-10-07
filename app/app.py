"""Flask dashboard and JSON API for the exported models.

    python app/app.py                     # http://127.0.0.1:8765
    curl -X POST localhost:8765/api/predict -H 'content-type: application/json' \
         -d '{"dataset": "twitter_hate", "texts": ["some tweet"]}'

Every prediction is appended to reports/predictions.jsonl so the monitor
command can later run a drift report on real traffic.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sentiment_benchmark import __version__  # noqa: E402
from sentiment_benchmark.predict import available_models, load_model  # noqa: E402
from sentiment_benchmark.storage import PredictionStore  # noqa: E402

MAX_TEXTS = 200
MAX_CHARS = 5000


def create_app(models_dir: str | Path | None = None, reports_dir: str | Path | None = None, database_url: str | None = None) -> Flask:
    models_dir = Path(models_dir or ROOT / "models")
    reports_dir = Path(reports_dir or ROOT / "reports")
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"), static_folder=str(Path(__file__).parent / "static"))
    app.config["MODELS_DIR"] = models_dir
    app.config["REPORTS_DIR"] = reports_dir
    app.config["MODELS"] = {}
    try:
        app.config["STORE"] = PredictionStore(database_url, reports_dir)
    except Exception as exc:  # pragma: no cover
        print(f"[app] prediction store unavailable ({exc}); falling back to reports/predictions.jsonl")
        app.config["STORE"] = None

    def get_model(name: str):
        if name not in app.config["MODELS"]:
            app.config["MODELS"][name] = load_model(name, models_dir)
        return app.config["MODELS"][name]

    def log_predictions(dataset: str, results: list[dict], model=None, latency_ms: float | None = None) -> None:
        store = app.config.get("STORE")
        if store is not None:
            try:
                store.log_predictions(dataset, model.card["model"], model.card["features"], results, latency_ms)
                return
            except Exception as exc:  # pragma: no cover
                print(f"[app] store write failed ({exc}); writing jsonl")
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(reports_dir / "predictions.jsonl", "a", encoding="utf-8") as fh:
            for r in results:
                fh.write(
                    json.dumps({"ts": time.time(), "dataset": dataset, "label": r["label_name"], "confidence": r["confidence"], "text": r["text"][:280]}) + "\n"
                )

    def results_table():
        path = reports_dir / "results.csv"
        if not path.exists():
            return []
        import pandas as pd

        df = pd.read_csv(path).sort_values(["dataset", "cv_f1_mean"], ascending=[True, False])
        cols = [
            c for c in ["dataset", "model", "features", "cv_f1_mean", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"] if c in df
        ]
        return df[cols].round(3).to_dict("records")

    @app.get("/")
    def index():
        names = available_models(models_dir)
        cards = {}
        for n in names:
            try:
                cards[n] = get_model(n).card
            except Exception as exc:  # pragma: no cover
                cards[n] = {"error": str(exc)}
        preferred = "tweet_sentiment" if "tweet_sentiment" in names else (names[0] if names else None)
        return render_template("index.html", models=names, cards=cards, preferred=preferred, version=__version__)

    @app.get("/analysis")
    def analysis():
        names = available_models(models_dir)
        cards = {}
        for name in names:
            try:
                cards[name] = get_model(name).card
            except Exception as exc:  # pragma: no cover
                cards[name] = {"error": str(exc)}
        figures_dir = reports_dir / "figures"
        figures = sorted(p.name for p in figures_dir.glob("*tweet_sentiment*.png")) if figures_dir.exists() else []
        results = [row for row in results_table() if row.get("dataset") == "tweet_sentiment"]
        return render_template("analysis.html", models=names, cards=cards, results=results, figures=figures, version=__version__)

    @app.get("/health")
    def health():
        store = app.config.get("STORE")
        return jsonify({"status": "ok", "version": __version__, "models": available_models(models_dir), "store": store.backend if store else "jsonl"})

    @app.get("/api/models")
    def api_models():
        return jsonify({n: get_model(n).card for n in available_models(models_dir)})

    @app.post("/api/predict")
    def api_predict():
        payload = request.get_json(silent=True) or {}
        dataset = payload.get("dataset") or request.form.get("dataset") or "twitter_hate"
        texts = payload.get("texts")
        if texts is None:
            single = payload.get("text") or request.form.get("text")
            texts = [single] if single else []
        if not isinstance(texts, list) or not texts:
            return jsonify({"error": "provide 'text' or a non-empty 'texts' list"}), 400
        if len(texts) > MAX_TEXTS:
            return jsonify({"error": f"at most {MAX_TEXTS} texts per request"}), 413
        texts = [str(t)[:MAX_CHARS] for t in texts]
        if dataset not in available_models(models_dir):
            return jsonify({"error": f"unknown dataset {dataset!r}", "available": available_models(models_dir)}), 404
        model = get_model(dataset)
        t0 = time.time()
        results = model.predict(texts)
        log_predictions(dataset, results, model, round((time.time() - t0) * 1000, 1))
        return jsonify(
            {
                "dataset": dataset,
                "model": model.card["model"],
                "features": model.card["features"],
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "results": results,
            }
        )

    @app.get("/api/stats")
    def api_stats():
        store = app.config.get("STORE")
        if store is None:
            return jsonify({"backend": "jsonl"})
        return jsonify({d: store.stats(d) for d in available_models(models_dir)} | {"backend": store.backend})

    @app.get("/figures/<name>")
    def figure(name: str):
        from flask import send_from_directory

        return send_from_directory(reports_dir / "figures", name)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=8765, debug=False)
