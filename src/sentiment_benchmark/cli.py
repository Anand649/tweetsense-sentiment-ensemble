"""Command line entry point.

sentiment-benchmark data download --dataset imdb
sentiment-benchmark data samples
sentiment-benchmark benchmark --config configs/benchmark.yaml [--smoke]
sentiment-benchmark predict --dataset twitter_hate "some text" ...
sentiment-benchmark monitor --dataset twitter_hate --batch new_tweets.csv
sentiment-benchmark serve --port 5000
sentiment-benchmark collect --query "#vaccine" --limit 500 --out data/new/vaccine.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import PROJECT_ROOT, __version__

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "benchmark.yaml"
SMOKE_CONFIG = PROJECT_ROOT / "configs" / "smoke.yaml"


def _cmd_data(args: argparse.Namespace) -> int:
    from .config import load_config
    from .datasets import LOADERS, download, load_dataset, write_manifest

    cfg = load_config(args.config)
    if args.data_cmd == "download":
        names = [args.dataset] if args.dataset else list(LOADERS)
        for n in names:
            print(f"[data] {n}: {download(n, cfg.data_dir, force=args.force)}")
        return 0
    if args.data_cmd == "manifest":
        loaded = []
        for ds_cfg in cfg.datasets:
            try:
                loaded.append(load_dataset(ds_cfg.name, cfg.data_dir, ds_cfg))
            except FileNotFoundError as exc:
                print(f"[data] skip {ds_cfg.name}: {exc}")
        if loaded:
            print(f"[data] manifest -> {write_manifest(loaded, cfg.data_dir)}")
        return 0
    if args.data_cmd == "check":
        for ds_cfg in cfg.datasets:
            try:
                ds = load_dataset(ds_cfg.name, cfg.data_dir, ds_cfg)
                counts = ds.frame["label"].value_counts().sort_index().tolist()
                print(f"[data] {ds.name:14s} rows={len(ds.frame):6d} classes={dict(zip(ds.label_names, counts))} fingerprint={ds.fingerprint()}")
            except FileNotFoundError as exc:
                print(f"[data] {ds_cfg.name:14s} MISSING: {exc}")
        return 0
    return 1


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmark import run_benchmark
    from .config import load_config

    path = SMOKE_CONFIG if args.smoke and args.config == str(DEFAULT_CONFIG) else args.config
    overrides = {}
    if args.cv_folds:
        overrides["cv_folds"] = args.cv_folds
    cfg = load_config(path, overrides)
    if args.no_tracking:
        cfg.tracking["enabled"] = False
    rows = run_benchmark(cfg, only_datasets=args.dataset, only_models=args.model, only_features=args.features, export=not args.no_export)
    return 0 if rows else 1


def _cmd_predict(args: argparse.Namespace) -> int:
    from .predict import load_model

    model = load_model(args.dataset, args.models_dir)
    texts = list(args.text)
    if args.file:
        import pandas as pd

        texts += pd.read_csv(args.file)["text"].astype(str).tolist()
    if not texts:
        texts = [line.strip() for line in sys.stdin if line.strip()]
    for r in model.predict(texts):
        if args.json:
            print(json.dumps(r))
        else:
            print(f"{r['label_name']:10s} {r['confidence']:.3f}  {r['text'][:100]}")
    return 0


def _cmd_monitor(args: argparse.Namespace) -> int:
    import pandas as pd

    from .monitor import drift_report, load_baseline
    from .predict import load_base_model

    model = load_base_model(args.dataset, args.models_dir)
    baseline = load_baseline(model.model_dir)
    store = None
    if args.from_db or args.store:
        from .storage import PredictionStore

        store = PredictionStore()
    if args.from_db:
        texts = store.recent_texts(args.dataset, hours=args.hours)
        if not texts:
            print(f"[monitor] no predictions for {args.dataset} in the last {args.hours} h ({store.backend})")
            return 0
    else:
        texts = pd.read_csv(args.batch)["text"].astype(str).tolist()
    report = drift_report(model.pipeline, baseline, texts)
    report["source"] = f"db:{store.backend}" if args.from_db else str(args.batch)
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if store is not None:
        store.log_drift(args.dataset, report)
    return 0 if report["status"] != "alert" else 2


def _cmd_serve(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(PROJECT_ROOT / "app"))
    from app import create_app  # type: ignore

    app = create_app(models_dir=args.models_dir)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Regenerate figures, results.md and the README results block from saved probabilities."""
    import subprocess

    from .benchmark import load_saved_rows, make_figures, write_reports
    from .config import load_config
    from .datasets import DATASET_META, Dataset

    cfg = load_config(args.config)
    import pandas as pd

    table = pd.read_csv(cfg.reports_dir / "results.csv")
    all_rows, datasets = [], []
    for name in list(dict.fromkeys(table["dataset"])):
        rows, y_train, y_test, label_names = load_saved_rows(cfg, name)
        figs = make_figures(rows, name, y_train, y_test, label_names, cfg.reports_dir / "figures")
        print(f"[report] {name}: {len(rows)} cells, {len(figs)} figures")
        all_rows += rows
        meta = DATASET_META.get(name, (label_names, rows[0]["task"] if rows else "binary"))
        datasets.append(Dataset(name, pd.DataFrame({"text": [], "label": []}), list(label_names), meta[1]))
    csv_path, md_path = write_reports(all_rows, cfg, datasets)
    print(f"[report] wrote {md_path}")
    if not args.no_readme:
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "update_readme_results.py")], check=True)
    return 0


def _cmd_tracking(args: argparse.Namespace) -> int:
    from .tracking import replay_jsonl

    n = replay_jsonl(Path(args.jsonl), args.uri, args.experiment)
    print(f"[tracking] replayed {n} runs from {args.jsonl}")
    return 0


def _cmd_collect(args: argparse.Namespace) -> int:
    from .collect import collect

    collect(args.query, args.limit, Path(args.out), lang=args.lang)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sentiment-benchmark", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="download, sample and check datasets")
    d.add_argument("data_cmd", choices=["download", "manifest", "check"])
    d.add_argument("--dataset")
    d.add_argument("--config", default=str(DEFAULT_CONFIG))
    d.add_argument("--force", action="store_true")
    d.set_defaults(func=_cmd_data)

    b = sub.add_parser("benchmark", help="run the datasets x features x models grid")
    b.add_argument("--config", default=str(DEFAULT_CONFIG))
    b.add_argument("--smoke", action="store_true", help="tiny grid on committed samples (CI)")
    b.add_argument("--dataset", action="append")
    b.add_argument("--model", action="append")
    b.add_argument("--features", action="append")
    b.add_argument("--cv-folds", type=int)
    b.add_argument("--no-tracking", action="store_true")
    b.add_argument("--no-export", action="store_true")
    b.set_defaults(func=_cmd_benchmark)

    pr = sub.add_parser("predict", help="classify text with an exported model")
    pr.add_argument("text", nargs="*")
    pr.add_argument("--dataset", default="twitter_hate")
    pr.add_argument("--file", help="CSV with a text column")
    pr.add_argument("--models-dir")
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(func=_cmd_predict)

    m = sub.add_parser("monitor", help="drift report for a new batch of text")
    m.add_argument("--dataset", default="twitter_hate")
    m.add_argument("--batch", help="CSV with a text column")
    m.add_argument("--from-db", action="store_true", help="use recent predictions from DATABASE_URL instead of a CSV")
    m.add_argument("--hours", type=float, default=24.0)
    m.add_argument("--store", action="store_true", help="also record the report in the database")
    m.add_argument("--models-dir")
    m.add_argument("--out")
    m.set_defaults(func=_cmd_monitor)

    s = sub.add_parser("serve", help="start the Flask dashboard and API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=5000)
    s.add_argument("--models-dir")
    s.add_argument("--debug", action="store_true")
    s.set_defaults(func=_cmd_serve)

    rp = sub.add_parser("report", help="regenerate figures and reports from saved probabilities (no retraining)")
    rp.add_argument("--config", default=str(DEFAULT_CONFIG))
    rp.add_argument("--no-readme", action="store_true")
    rp.set_defaults(func=_cmd_report)

    t = sub.add_parser("tracking", help="replay reports/runs.jsonl into an MLflow store")
    t.add_argument("tracking_cmd", choices=["replay"])
    t.add_argument("--jsonl", default=str(PROJECT_ROOT / "reports" / "runs.jsonl"))
    t.add_argument("--uri", help="MLflow tracking URI (default: sqlite:///reports/mlflow.db or MLFLOW_TRACKING_URI)")
    t.add_argument("--experiment")
    t.set_defaults(func=_cmd_tracking)

    c = sub.add_parser("collect", help="collect recent tweets (needs TWITTER_BEARER_TOKEN)")
    c.add_argument("--query", required=True)
    c.add_argument("--limit", type=int, default=500)
    c.add_argument("--out", default="data/new/tweets.csv")
    c.add_argument("--lang", default="en")
    c.set_defaults(func=_cmd_collect)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
