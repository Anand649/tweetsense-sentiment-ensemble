"""Operational store for predictions and drift reports (SQLAlchemy Core).

`DATABASE_URL` selects the backend: PostgreSQL in Docker Compose and on AWS
(RDS), SQLite by default for a laptop. The schema is created on first use.
The API writes every prediction here; the monitor reads recent traffic back
to compute drift, and stores each report so alerts have a history.

    DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/sentiment
    DATABASE_URL=sqlite:///reports/predictions.db          (default)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Column, DateTime, Float, Integer, MetaData, String, Table, Text, create_engine, func, select

from . import PROJECT_ROOT
from .env import env

metadata = MetaData()

predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime(timezone=True), nullable=False, index=True),
    Column("dataset", String(64), nullable=False, index=True),
    Column("model", String(64), nullable=False),
    Column("features", String(64), nullable=False),
    Column("text_hash", String(64), nullable=False),
    Column("text", Text, nullable=False),
    Column("label", Integer, nullable=False),
    Column("label_name", String(64), nullable=False),
    Column("confidence", Float, nullable=False),
    Column("latency_ms", Float),
    Column("source", String(32), default="api"),
)

drift_reports = Table(
    "drift_reports",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime(timezone=True), nullable=False, index=True),
    Column("dataset", String(64), nullable=False, index=True),
    Column("status", String(16), nullable=False),
    Column("n", Integer, nullable=False),
    Column("report", JSON, nullable=False),
)


def default_url(reports_dir: Path | None = None) -> str:
    reports_dir = Path(reports_dir or PROJECT_ROOT / "reports")
    return env("DATABASE_URL") or f"sqlite:///{(reports_dir / 'predictions.db').resolve()}"


class PredictionStore:
    def __init__(self, url: str | None = None, reports_dir: Path | None = None):
        self.url = url or default_url(reports_dir)
        if self.url.startswith("sqlite:///"):
            Path(self.url.replace("sqlite:///", "", 1)).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(self.url, future=True, pool_pre_ping=True)
        metadata.create_all(self.engine)

    @property
    def backend(self) -> str:
        return self.engine.dialect.name

    def log_predictions(
        self, dataset: str, model: str, features: str, results: list[dict[str, Any]], latency_ms: float | None = None, source: str = "api"
    ) -> int:
        now = datetime.now(timezone.utc)
        rows = [
            {
                "ts": now,
                "dataset": dataset,
                "model": model,
                "features": features,
                "text_hash": hashlib.sha256(r["text"].encode("utf-8")).hexdigest(),
                "text": r["text"][:2000],
                "label": int(r["label"]),
                "label_name": r["label_name"],
                "confidence": float(r["confidence"]),
                "latency_ms": latency_ms,
                "source": source,
            }
            for r in results
        ]
        if not rows:
            return 0
        with self.engine.begin() as conn:
            conn.execute(predictions.insert(), rows)
        return len(rows)

    def recent_texts(self, dataset: str, hours: float = 24, limit: int = 5000) -> list[str]:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        stmt = select(predictions.c.text).where(predictions.c.dataset == dataset, predictions.c.ts >= since).order_by(predictions.c.ts.desc()).limit(limit)
        with self.engine.connect() as conn:
            return [r[0] for r in conn.execute(stmt)]

    def log_drift(self, dataset: str, report: dict[str, Any]) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                drift_reports.insert().values(
                    ts=datetime.now(timezone.utc),
                    dataset=dataset,
                    status=report["status"],
                    n=int(report["n"]),
                    report=json.loads(json.dumps(report, default=str)),
                )
            )

    def stats(self, dataset: str | None = None) -> dict[str, Any]:
        where = [predictions.c.dataset == dataset] if dataset else []
        with self.engine.connect() as conn:
            total = conn.execute(select(func.count()).select_from(predictions).where(*where)).scalar_one()
            by_label = conn.execute(select(predictions.c.label_name, func.count()).where(*where).group_by(predictions.c.label_name)).all()
            last_drift = conn.execute(
                select(drift_reports.c.ts, drift_reports.c.status)
                .where(*([drift_reports.c.dataset == dataset] if dataset else []))
                .order_by(drift_reports.c.ts.desc())
                .limit(1)
            ).first()
        return {
            "backend": self.backend,
            "predictions": int(total),
            "by_label": {name: int(c) for name, c in by_label},
            "last_drift": {"ts": str(last_drift[0]), "status": last_drift[1]} if last_drift else None,
        }
