"""Feature sets: bag-of-words, TF-IDF, Word2Vec, Doc2Vec and raw sequences.

All four thesis feature sets are scikit-learn transformers so they can be
chained after `TextCleaner` and fitted on training folds only. `sequence` is a
pass-through: the neural models tokenize internally so they can also build
their own vocabulary and optional Word2Vec-initialised embedding matrix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer


class MeanWord2Vec(BaseEstimator, TransformerMixin):
    """Skip-gram Word2Vec trained on the training corpus, mean-pooled per doc."""

    def __init__(
        self,
        vector_size: int = 200,
        window: int = 5,
        min_count: int = 2,
        sg: int = 1,
        negative: int = 10,
        epochs: int = 20,
        workers: int = 4,
        seed: int = 34,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.negative = negative
        self.epochs = epochs
        self.workers = workers
        self.seed = seed

    def fit(self, X, y=None):  # noqa: N803
        from gensim.models import Word2Vec

        docs = [str(d).split() for d in X]
        self.model_ = Word2Vec(
            docs,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            negative=self.negative,
            epochs=self.epochs,
            workers=self.workers,
            seed=self.seed,
        )
        return self

    def transform(self, X):  # noqa: N803
        kv = self.model_.wv
        out = np.zeros((len(X), self.vector_size), dtype=np.float32)
        for i, d in enumerate(X):
            toks = [t for t in str(d).split() if t in kv.key_to_index]
            if toks:
                out[i] = kv[toks].mean(axis=0)
        return out

    def embedding_matrix(self, vocab: dict[str, int]) -> np.ndarray:
        """Embedding matrix aligned to an external vocab (index 0 = padding)."""
        kv = self.model_.wv
        rng = np.random.default_rng(self.seed)
        mat = rng.normal(0, 0.1, (len(vocab) + 1, self.vector_size)).astype(np.float32)
        mat[0] = 0
        for w, i in vocab.items():
            if w in kv.key_to_index:
                mat[i] = kv[w]
        return mat


class Doc2VecVectorizer(BaseEstimator, TransformerMixin):
    """PV-DM Doc2Vec; documents are embedded by inference at transform time
    (train and test alike) so train/test representations are comparable."""

    def __init__(
        self,
        vector_size: int = 200,
        window: int = 5,
        min_count: int = 5,
        negative: int = 7,
        epochs: int = 15,
        infer_epochs: int = 20,
        workers: int = 4,
        seed: int = 23,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.negative = negative
        self.epochs = epochs
        self.infer_epochs = infer_epochs
        self.workers = workers
        self.seed = seed

    def fit(self, X, y=None):  # noqa: N803
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument

        tagged = [TaggedDocument(str(d).split(), [i]) for i, d in enumerate(X)]
        self.model_ = Doc2Vec(
            tagged,
            dm=1,
            dm_mean=1,
            vector_size=self.vector_size,
            window=self.window,
            negative=self.negative,
            min_count=self.min_count,
            epochs=self.epochs,
            workers=self.workers,
            seed=self.seed,
        )
        return self

    def transform(self, X):  # noqa: N803
        out = np.zeros((len(X), self.vector_size), dtype=np.float32)
        for i, d in enumerate(X):
            toks = str(d).split()
            if toks:
                out[i] = self.model_.infer_vector(toks, epochs=self.infer_epochs)
        return out


def _identity(x):
    return list(x)


def make_feature_extractor(name: str, params: dict[str, Any] | None = None):
    """Build a feature transformer by registry name."""
    params = dict(params or {})
    kind = params.pop("type", name)
    if kind == "bow":
        params.setdefault("ngram_range", (1, 1))
        params["ngram_range"] = tuple(params["ngram_range"])
        return CountVectorizer(**params)
    if kind == "tfidf":
        params.setdefault("ngram_range", (1, 1))
        params["ngram_range"] = tuple(params["ngram_range"])
        return TfidfVectorizer(**params)
    if kind == "word2vec":
        return MeanWord2Vec(**params)
    if kind == "doc2vec":
        return Doc2VecVectorizer(**params)
    if kind == "sequence":
        return FunctionTransformer(_identity)
    raise KeyError(f"unknown feature set {name!r} (type {kind!r})")


FEATURE_TYPES = ("bow", "tfidf", "word2vec", "doc2vec", "sequence")
