"""Minimal .env loader (no third-party dependency).

Reads KEY=VALUE lines from the project's `.env` file into `os.environ` without
overriding variables that are already set, so container and CI environments
keep precedence. Everything environment-specific or sensitive (dataset
download locations, database URLs, API tokens, cloud settings) lives there;
`.env.example` documents the keys with placeholder values and is the only
file of the two that is committed.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import PROJECT_ROOT

_LOADED: set[Path] = set()


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    path = Path(path or os.environ.get("SENTIMENT_BENCHMARK_ENV_FILE") or PROJECT_ROOT / ".env")
    if path in _LOADED or not path.exists():
        return {}
    loaded: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    _LOADED.add(path)
    return loaded


def env(key: str, default: str | None = None) -> str | None:
    """Read a setting, loading .env on first use."""
    load_dotenv()
    value = os.environ.get(key)
    return value if value not in (None, "") else default


def require(key: str, hint: str) -> str:
    value = env(key)
    if not value:
        raise RuntimeError(f"{key} is not set. {hint} (copy .env.example to .env)")
    return value
