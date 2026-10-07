"""Collect fresh tweets for a keyword with the X / Twitter API v2.

Mirrors the collection step of the thesis (Tweepy) with one difference that
matters: credentials are read from the environment (see `.env.example`),
never written in code. Output is a CSV with the `text` column the pipeline
expects, ready for `predict --file` or `monitor --batch`.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path


def collect(query: str, limit: int, out: Path, lang: str = "en") -> Path:
    try:
        import tweepy
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install tweepy to collect tweets") from exc

    bearer = os.environ.get("TWITTER_BEARER_TOKEN")
    if not bearer:
        raise RuntimeError("TWITTER_BEARER_TOKEN is not set (copy .env.example to .env)")
    client = tweepy.Client(bearer_token=bearer, wait_on_rate_limit=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "created_at", "text"])
        for tweet in tweepy.Paginator(
            client.search_recent_tweets,
            query=f"{query} lang:{lang} -is:retweet",
            tweet_fields=["created_at"],
            max_results=100,
        ).flatten(limit=limit):
            writer.writerow([tweet.id, tweet.created_at, tweet.text.replace("\n", " ")])
            n += 1
    print(f"[collect] wrote {n} tweets to {out}")
    return out
