"""Sentiment ML Benchmark.

Reproducible benchmark of classical and neural text classifiers for sentiment
analysis, grown out of a 2022 NUIST BSc thesis (Twitter hate-speech detection
and COVID-19 vaccination sentiment) into an end-to-end, tracked ML pipeline.
"""

from __future__ import annotations

__version__ = "1.0.0"

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

from .env import load_dotenv  # noqa: E402

load_dotenv()
