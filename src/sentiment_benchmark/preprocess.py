"""Text cleaning shared by every feature set and model.

Implements the "advanced pre-processing" recipe from the thesis (transform
case, strip handles and URLs, keep letters, filter short words and stop words,
Porter stemming) as a scikit-learn transformer so it lives INSIDE the model
pipeline. That guarantees the exact same cleaning at training, evaluation and
serving time, and that nothing is fitted on test data.

No NLTK corpus downloads are required: the Porter stemmer is pure code and the
stop-word list ships with scikit-learn (the same list the thesis notebooks used
through `CountVectorizer(stop_words="english")`).
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable

from nltk.stem.porter import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .config import CleanConfig

_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_MENTION = re.compile(r"@\w+")
_HTML_TAG = re.compile(r"<[^>]+>")
_KEEP_HASH = re.compile(r"[^a-z#\s]")
_DROP_HASH = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")

STOP_WORDS: frozenset[str] = frozenset(ENGLISH_STOP_WORDS)


def clean_text(text: str, cfg: CleanConfig | None = None, _stemmer: PorterStemmer | None = None) -> str:
    """Clean one document according to `cfg` (defaults to the thesis recipe)."""
    cfg = cfg or CleanConfig()
    text = html.unescape(text or "")
    text = _HTML_TAG.sub(" ", text)
    if cfg.lowercase:
        text = text.lower()
    if cfg.remove_urls:
        text = _URL.sub(" ", text)
    if cfg.remove_mentions:
        text = _MENTION.sub(" ", text)
    text = (_KEEP_HASH if cfg.keep_hashtags else _DROP_HASH).sub(" ", text)
    tokens = text.split()
    if cfg.min_word_len > 1:
        tokens = [t for t in tokens if len(t.lstrip("#")) >= cfg.min_word_len]
    if cfg.remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    if cfg.stemmer == "porter":
        stemmer = _stemmer or PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    elif cfg.stemmer not in (None, "none"):
        raise ValueError(f"unknown stemmer {cfg.stemmer!r}")
    return _SPACES.sub(" ", " ".join(tokens)).strip()


def clean_corpus(texts: Iterable[str], cfg: CleanConfig | None = None) -> list[str]:
    cfg = cfg or CleanConfig()
    stemmer = PorterStemmer() if cfg.stemmer == "porter" else None
    return [clean_text(t, cfg, stemmer) for t in texts]


class TextCleaner(BaseEstimator, TransformerMixin):
    """scikit-learn transformer wrapping `clean_text` (stateless, picklable)."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        keep_hashtags: bool = True,
        remove_stopwords: bool = True,
        min_word_len: int = 3,
        stemmer: str | None = "porter",
    ):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.keep_hashtags = keep_hashtags
        self.remove_stopwords = remove_stopwords
        self.min_word_len = min_word_len
        self.stemmer = stemmer

    @classmethod
    def from_config(cls, cfg: CleanConfig) -> TextCleaner:
        return cls(**cfg.__dict__)

    def _cfg(self) -> CleanConfig:
        return CleanConfig(
            lowercase=self.lowercase,
            remove_urls=self.remove_urls,
            remove_mentions=self.remove_mentions,
            keep_hashtags=self.keep_hashtags,
            remove_stopwords=self.remove_stopwords,
            min_word_len=self.min_word_len,
            stemmer=self.stemmer,
        )

    def fit(self, X, y=None):  # noqa: N803 (sklearn API)
        return self

    def transform(self, X):  # noqa: N803
        return clean_corpus(list(X), self._cfg())
