"""Download the official TweetEval sentiment text and labels for the demo."""

from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "tweet_sentiment"
FILES = [f"{split}_{kind}.txt" for split in ("train", "val", "test") for kind in ("text", "labels")]


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = DEST / name
        if path.exists() and path.stat().st_size:
            print(f"Present: {path}")
            continue
        print(f"Downloading {name}...")
        try:
            urlretrieve(BASE + name, path)
        except Exception:
            path.unlink(missing_ok=True)
            raise
    print("TweetEval sentiment data is ready.")


if __name__ == "__main__":
    main()
