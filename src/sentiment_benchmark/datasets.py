"""Dataset registry: loading, weak labelling, splitting, manifest and downloads.

Every loader returns a `Dataset` with a DataFrame of exactly two columns,
`text` and `label` (int), plus label names and the task type. Raw files live in
`data/raw/<name>/` and are never committed. Download locations are read from
the environment (`.env`), never written in code.

Datasets
--------
twitter_hate   31,962 labelled tweets, label 1 = racist or sexist (about 7%).
               Thesis experiment 1.
vaccination    11,020 COVID-19 vaccine tweets, weakly labelled negative /
               neutral / positive with TextBlob polarity, as in thesis
               experiment 2.
imdb           Large Movie Review Dataset (Maas et al., 2011), 50,000 reviews,
               official 25k/25k split. Extension beyond the thesis.
synthetic      deterministic generated corpus used by tests and CI so that no
               data has to be committed.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DatasetConfig
from .env import env

DATASET_META: dict[str, tuple[list[str], str]] = {
    "tweet_sentiment": (["negative", "neutral", "positive"], "multiclass"),
    "twitter_hate": (["not_hate", "hate"], "binary"),
    "vaccination": (["negative", "neutral", "positive"], "multiclass"),
    "imdb": (["negative", "positive"], "binary"),
}

# environment keys holding download locations (see .env.example)
ENV_KEYS = {
    "imdb": "IMDB_ARCHIVE_URL",
    "twitter_hate": "KAGGLE_TWITTER_HATE_DATASET",
    "vaccination": "KAGGLE_VACCINATION_DATASET",
}


@dataclass
class Dataset:
    name: str
    frame: pd.DataFrame  # columns: text, label
    label_names: list[str]
    task: str  # binary | multiclass
    official_split: tuple[pd.DataFrame, pd.DataFrame] | None = None
    source_files: list[Path] = field(default_factory=list)
    synthetic: bool = False

    @property
    def n_classes(self) -> int:
        return len(self.label_names)

    def split(self, cfg: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (train, test). Official split when available and requested."""
        if cfg.split.strategy == "official" and self.official_split is not None:
            train, test = self.official_split
        else:
            train, test = train_test_split(
                self.frame,
                test_size=cfg.split.test_size,
                random_state=cfg.split.seed,
                stratify=self.frame["label"],
            )
        if cfg.max_rows:
            train = _stratified_head(train, cfg.max_rows, cfg.split.seed)
            test = _stratified_head(test, max(50, cfg.max_rows // 4), cfg.split.seed)
        return train.reset_index(drop=True), test.reset_index(drop=True)

    def fingerprint(self) -> str:
        """Content hash of the loaded frame, logged with every run."""
        h = hashlib.sha256()
        h.update(pd.util.hash_pandas_object(self.frame, index=False).values.tobytes())
        return h.hexdigest()[:16]


def _stratified_head(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    frac = n / len(df)
    return df.groupby("label", group_keys=False).apply(lambda g: g.sample(frac=frac, random_state=seed) if len(g) * frac >= 1 else g)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def _raw_dir(data_dir: Path, name: str, cfg: DatasetConfig | None) -> Path:
    if cfg and cfg.raw_dir:
        p = Path(cfg.raw_dir)
        return p if p.is_absolute() else data_dir.parent / p
    return data_dir / "raw" / name


def _validate(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "text" not in df or "label" not in df:
        raise ValueError(f"{name}: expected columns text,label")
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""]
    df["label"] = df["label"].astype(int)
    if df["label"].nunique() < 2:
        raise ValueError(f"{name}: need at least two classes")
    return df[["text", "label"]].reset_index(drop=True)


def load_twitter_hate(data_dir: Path, cfg: DatasetConfig | None = None) -> Dataset:
    raw = _raw_dir(data_dir, "twitter_hate", cfg)
    path = _first_existing(raw, ["train.csv", "train_E6oV3lV.csv"], "twitter_hate")
    df = pd.read_csv(path).rename(columns={"tweet": "text"})
    df = _validate(df, "twitter_hate")
    return Dataset("twitter_hate", df, *DATASET_META["twitter_hate"], source_files=[path])


def load_tweet_sentiment(data_dir: Path, cfg: DatasetConfig | None = None) -> Dataset:
    """Load the human-labelled TweetEval sentiment train and test splits."""
    raw = _raw_dir(data_dir, "tweet_sentiment", cfg)

    def read_split(split: str) -> tuple[pd.DataFrame, list[Path]]:
        text_path = raw / f"{split}_text.txt"
        label_path = raw / f"{split}_labels.txt"
        for path in (text_path, label_path):
            if not path.exists():
                raise FileNotFoundError(f"TweetEval file missing: {path}. See data/README.md")
        texts = text_path.read_text(encoding="utf-8").splitlines()
        labels = label_path.read_text(encoding="utf-8").splitlines()
        if len(texts) != len(labels):
            raise ValueError(f"TweetEval {split}: text and label counts differ")
        frame = _validate(pd.DataFrame({"text": texts, "label": labels}), "tweet_sentiment")
        if not frame["label"].isin([0, 1, 2]).all():
            raise ValueError(f"TweetEval {split}: expected labels 0, 1, 2")
        return frame, [text_path, label_path]

    train, train_files = read_split("train")
    test, test_files = read_split("test")
    full = pd.concat([train, test], ignore_index=True)
    return Dataset("tweet_sentiment", full, *DATASET_META["tweet_sentiment"], official_split=(train, test), source_files=train_files + test_files)


def textblob_label(text: str) -> int:
    """Thesis Ex2 weak labelling: TextBlob polarity sign -> 0 neg, 1 neu, 2 pos."""
    from textblob import TextBlob  # local import: optional dependency at import time

    p = TextBlob(text).sentiment.polarity
    return 0 if p < 0 else (1 if p == 0 else 2)


def load_vaccination(data_dir: Path, cfg: DatasetConfig | None = None) -> Dataset:
    raw = _raw_dir(data_dir, "vaccination", cfg)
    path = _first_existing(raw, ["vaccination_tweets.csv"], "vaccination")
    df = pd.read_csv(path)
    if "label" not in df:
        # weak labels are computed on lightly cleaned text, as the thesis did,
        # and cached next to the raw file so they are stable across runs
        cache = raw / "vaccination_labelled.csv"
        if cache.exists():
            df = pd.read_csv(cache)
        else:
            from .preprocess import CleanConfig, clean_corpus

            light = CleanConfig(keep_hashtags=False, remove_stopwords=True, min_word_len=1, stemmer=None)
            df = df[["text"]].dropna().copy()
            df["text_light"] = clean_corpus(df["text"], light)
            df = df.drop_duplicates("text_light")
            df["label"] = [textblob_label(t) for t in df["text_light"]]
            df = df[["text", "label"]]
            df.to_csv(cache, index=False)
    df = _validate(df, "vaccination")
    return Dataset("vaccination", df, *DATASET_META["vaccination"], source_files=[path])


def load_imdb(data_dir: Path, cfg: DatasetConfig | None = None) -> Dataset:
    raw = _raw_dir(data_dir, "imdb", cfg)
    root = raw / "aclImdb"
    cache = raw / "imdb.csv.gz"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        if not root.exists():
            raise FileNotFoundError(f"IMDB not found at {root}. Run: sentiment-benchmark data download --dataset imdb")
        rows = []
        for split in ("train", "test"):
            for label_name, label in (("neg", 0), ("pos", 1)):
                for f in sorted((root / split / label_name).glob("*.txt")):
                    rows.append((f.read_text(encoding="utf-8"), label, split))
        df = pd.DataFrame(rows, columns=["text", "label", "split"])
        df.to_csv(cache, index=False)
    train = _validate(df[df["split"] == "train"], "imdb")
    test = _validate(df[df["split"] == "test"], "imdb")
    full = pd.concat([train, test], ignore_index=True)
    return Dataset("imdb", full, *DATASET_META["imdb"], official_split=(train, test), source_files=[root])


LOADERS: dict[str, Callable[[Path, DatasetConfig | None], Dataset]] = {
    "tweet_sentiment": load_tweet_sentiment,
    "twitter_hate": load_twitter_hate,
    "vaccination": load_vaccination,
    "imdb": load_imdb,
}


# --------------------------------------------------------------------------- #
# Synthetic corpus (tests and CI; nothing real is committed)
# --------------------------------------------------------------------------- #

_LEXICON = {
    "positive": "love great wonderful brilliant happy enjoy best amazing thankful proud excellent fun kind beautiful".split(),
    "negative": "hate terrible awful worst disgusting angry boring waste stupid ugly sad horrible pathetic cruel".split(),
    "hate": "hate disgusting inferior vermin deport racist sexist attack slur bigot threat".split(),
    "neutral": "today update news report schedule meeting number week city office photo link article".split(),
    "filler": "the and with for this that have from about just when they there will your what been more some".split(),
    "tags": "#monday #news #life #mood #update #today #photo #work #friends #weekend".split(),
}


def synthetic_dataset(name: str, n: int = 800, seed: int = 7) -> Dataset:
    """Deterministic pseudo-tweets whose vocabulary correlates with the label."""
    rng = np.random.default_rng(seed)
    label_names, task = DATASET_META.get(name, (["negative", "positive"], "binary"))
    if name == "twitter_hate":
        labels = (rng.random(n) < 0.12).astype(int)
        pools = {0: ["positive", "neutral"], 1: ["hate", "negative"]}
    elif name == "vaccination":
        labels = rng.choice([0, 1, 2], size=n, p=[0.15, 0.5, 0.35])
        pools = {0: ["negative"], 1: ["neutral"], 2: ["positive"]}
    else:
        labels = rng.integers(0, 2, size=n)
        pools = {0: ["negative"], 1: ["positive"]}
    texts = []
    for y in labels:
        signal = [w for p in pools[int(y)] for w in _LEXICON[p]]
        k = int(rng.integers(6, 16))
        n_signal = max(2, k // 2)
        words = list(rng.choice(signal, size=n_signal)) + list(rng.choice(_LEXICON["filler"], size=k - n_signal))
        rng.shuffle(words)
        if rng.random() < 0.5:
            words.append(str(rng.choice(_LEXICON["tags"])))
        if rng.random() < 0.3:
            words.insert(0, "@user")
        texts.append(" ".join(words))
    df = pd.DataFrame({"text": texts, "label": labels.astype(int)})
    return Dataset(name, df, label_names, task, synthetic=True)


def load_dataset(name: str, data_dir: Path, cfg: DatasetConfig | None = None, allow_synthetic: bool | None = None) -> Dataset:
    """Load a dataset; in smoke mode (cfg.max_rows set) fall back to the
    synthetic corpus when the raw files are absent."""
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(LOADERS)}")
    try:
        return LOADERS[name](data_dir, cfg)
    except FileNotFoundError:
        smoke = allow_synthetic if allow_synthetic is not None else bool(cfg and cfg.max_rows)
        if smoke:
            return synthetic_dataset(name, n=max(400, (cfg.max_rows or 0) * 2))
        raise


def _first_existing(folder: Path, names: list[str], dataset: str) -> Path:
    for n in names:
        if (folder / n).exists():
            return folder / n
    raise FileNotFoundError(f"{dataset}: none of {names} found in {folder}. See data/README.md or run: sentiment-benchmark data download --dataset {dataset}")


# --------------------------------------------------------------------------- #
# Manifest and downloads
# --------------------------------------------------------------------------- #


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(datasets: list[Dataset], data_dir: Path) -> Path:
    """Record raw-file hashes and row counts so a run can be tied to exact data."""
    entries = {}
    for ds in datasets:
        entries[ds.name] = {
            "rows": int(len(ds.frame)),
            "classes": {n: int(c) for n, c in zip(ds.label_names, ds.frame["label"].value_counts().sort_index())},
            "fingerprint": ds.fingerprint(),
            "synthetic": ds.synthetic,
            "files": {
                (p.relative_to(data_dir).as_posix() if p.is_relative_to(data_dir) else p.name): (file_sha256(p) if p.is_file() else "dir")
                for p in ds.source_files
            },
        }
    path = data_dir / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return path


def download(name: str, data_dir: Path, force: bool = False) -> Path:
    """Download a dataset into data/raw/<name>/ using the location configured
    in the environment (see .env.example). Prints instructions when the
    location or the Kaggle client is not configured."""
    raw = data_dir / "raw" / name
    raw.mkdir(parents=True, exist_ok=True)
    key = ENV_KEYS.get(name)
    if key is None:
        raise KeyError(f"unknown dataset {name!r}")
    location = env(key)
    if not location:
        print(f"[data] {name}: set {key} in .env (see .env.example and data/README.md), or place the files in {raw} by hand")
        return raw
    if name == "imdb":
        if (raw / "aclImdb").exists() and not force:
            return raw
        archive = raw / "aclImdb_v1.tar.gz"
        if not archive.exists() or force:
            import requests

            print(f"[data] downloading the IMDB archive (about 80 MB) to {archive}")
            with requests.get(location, stream=True, timeout=120) as resp, open(archive, "wb") as out:
                resp.raise_for_status()
                for chunk in resp.iter_content(1 << 20):
                    out.write(chunk)
        with tarfile.open(archive, mode="r:gz") as tar:
            tar.extractall(raw, filter="data")
        return raw
    try:
        import kaggle  # noqa: F401
    except Exception:
        print(f"[data] {name}: Kaggle client not configured (KAGGLE_USERNAME / KAGGLE_KEY). Download the dataset by hand into {raw}.")
        return raw
    import subprocess

    subprocess.run(["kaggle", "datasets", "download", "-d", location, "-p", str(raw), "--unzip"], check=True)
    return raw
