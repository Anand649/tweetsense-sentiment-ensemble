"""CNN and (Bi)LSTM text classifiers in PyTorch, wrapped as a scikit-learn
estimator so they run through the same cross-validation, threshold tuning,
MLflow logging and export path as the classical models.

The thesis listed CNN-LSTM models as future work; this module is that work.
Embeddings are either learned from scratch or initialised from a Word2Vec
model trained on the same training fold (`embedding_init="word2vec"`), which
links the thesis' Word2Vec features to the neural models.
"""

from __future__ import annotations

import io
import math
from collections import Counter
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

try:  # torch is optional at import time so the rest of the package works without it
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

PAD, UNK = 0, 1


def torch_available() -> bool:
    return torch is not None


def pick_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _TextCNN(nn.Module if nn else object):
    def __init__(self, vocab_size, embed_dim, n_classes, kernel_sizes, n_filters, dropout, embedding_matrix=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(embedding_matrix))
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, n_filters, k, padding=k // 2) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)

    def forward(self, x):
        e = self.embedding(x).transpose(1, 2)  # B, E, T
        pooled = [torch.relu(conv(e)).max(dim=2).values for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class _TextLSTM(nn.Module if nn else object):
    def __init__(self, vocab_size, embed_dim, n_classes, hidden, bidirectional, dropout, embedding_matrix=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(embedding_matrix))
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim * 2, n_classes)  # max-pool + mean-pool concat

    def forward(self, x):
        mask = (x != PAD).unsqueeze(-1).float()
        out, _ = self.lstm(self.embedding(x))
        out = out * mask
        max_pool = (out + (mask - 1) * 1e4).max(dim=1).values
        mean_pool = out.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.fc(self.dropout(torch.cat([max_pool, mean_pool], dim=1)))


class TorchTextClassifier(BaseEstimator, ClassifierMixin):
    """scikit-learn estimator around a PyTorch CNN or LSTM.

    fit(X, y) takes cleaned text strings; tokenisation, vocabulary and padding
    happen inside so the estimator is self-contained and picklable.
    """

    def __init__(
        self,
        arch: str = "cnn",
        max_vocab: int = 20000,
        max_len: int = 64,
        embed_dim: int = 128,
        embedding_init: str | None = None,  # None | "word2vec"
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        n_filters: int = 128,
        hidden: int = 128,
        bidirectional: bool = True,
        dropout: float = 0.5,
        epochs: int = 6,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        class_weight: str | None = None,  # None | "balanced"
        patience: int = 2,
        val_fraction: float = 0.1,
        device: str = "auto",
        seed: int = 42,
        verbose: bool = False,
    ):
        self.arch = arch
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.embedding_init = embedding_init
        self.kernel_sizes = kernel_sizes
        self.n_filters = n_filters
        self.hidden = hidden
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.class_weight = class_weight
        self.patience = patience
        self.val_fraction = val_fraction
        self.device = device
        self.seed = seed
        self.verbose = verbose

    # ------------------------------------------------------------------ utils
    def _build_vocab(self, docs: list[list[str]]) -> dict[str, int]:
        counts = Counter(t for d in docs for t in d)
        most = counts.most_common(max(1, self.max_vocab - 2))
        return {w: i + 2 for i, (w, _) in enumerate(most)}

    def _encode(self, texts) -> np.ndarray:
        arr = np.full((len(texts), self.max_len), PAD, dtype=np.int64)
        for i, t in enumerate(texts):
            ids = [self.vocab_.get(w, UNK) for w in str(t).split()][: self.max_len]
            arr[i, : len(ids)] = ids
        return arr

    def _new_model(self, embedding_matrix=None):
        n_classes = len(self.classes_)
        vocab_size = len(self.vocab_) + 2
        embed_dim = embedding_matrix.shape[1] if embedding_matrix is not None else self.embed_dim
        if self.arch == "cnn":
            return _TextCNN(vocab_size, embed_dim, n_classes, tuple(self.kernel_sizes), self.n_filters, self.dropout, embedding_matrix)
        if self.arch == "lstm":
            return _TextLSTM(vocab_size, embed_dim, n_classes, self.hidden, self.bidirectional, self.dropout, embedding_matrix)
        raise KeyError(f"unknown arch {self.arch!r}")

    def _embedding_matrix(self, docs: list[list[str]]):
        if self.embedding_init != "word2vec":
            return None
        from gensim.models import Word2Vec

        w2v = Word2Vec(docs, vector_size=self.embed_dim, window=5, min_count=2, sg=1, negative=10, epochs=10, seed=self.seed, workers=4)
        rng = np.random.default_rng(self.seed)
        mat = rng.normal(0, 0.1, (len(self.vocab_) + 2, self.embed_dim)).astype(np.float32)
        mat[PAD] = 0
        for w, i in self.vocab_.items():
            if w in w2v.wv.key_to_index:
                mat[i] = w2v.wv[w]
        return mat

    # -------------------------------------------------------------------- API
    def fit(self, X, y):  # noqa: N803
        if torch is None:
            raise ImportError("PyTorch is required for neural models: pip install torch")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        texts = [str(t) for t in X]
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        class_index = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_index[v] for v in y], dtype=np.int64)
        docs = [t.split() for t in texts]
        self.vocab_ = self._build_vocab(docs)
        emb = self._embedding_matrix(docs)
        self.device_ = pick_device(self.device)

        ids = self._encode(texts)
        if self.val_fraction and len(texts) >= 50:
            strat = y_idx if np.bincount(y_idx).min() >= 2 else None
            x_tr, x_va, y_tr, y_va = train_test_split(ids, y_idx, test_size=self.val_fraction, random_state=self.seed, stratify=strat)
        else:
            x_tr, y_tr, x_va, y_va = ids, y_idx, None, None

        model = self._new_model(emb).to(self.device_)
        weight = None
        if self.class_weight == "balanced":
            counts = np.bincount(y_tr, minlength=len(self.classes_)).astype(np.float32)
            weight = torch.as_tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32, device=self.device_)
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_state, best_val, bad = None, math.inf, 0
        n = len(x_tr)
        rng = np.random.default_rng(self.seed)
        self.history_ = []
        for epoch in range(self.epochs):
            model.train()
            order = rng.permutation(n)
            total = 0.0
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                xb = torch.as_tensor(x_tr[idx], device=self.device_)
                yb = torch.as_tensor(y_tr[idx], device=self.device_)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                total += loss.item() * len(idx)
            train_loss = total / max(n, 1)
            val_loss = self._eval_loss(model, loss_fn, x_va, y_va) if x_va is not None else train_loss
            self.history_.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
            if self.verbose:
                print(f"[{self.arch}] epoch {epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            if val_loss < best_val - 1e-4:
                best_val, bad = val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model.eval()
        self.n_epochs_ = len(self.history_)
        return self

    def _eval_loss(self, model, loss_fn, x, y) -> float:
        model.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(x), 512):
                xb = torch.as_tensor(x[start : start + 512], device=self.device_)
                yb = torch.as_tensor(y[start : start + 512], device=self.device_)
                total += loss_fn(model(xb), yb).item() * len(xb)
        return total / max(len(x), 1)

    def predict_proba(self, X):  # noqa: N803
        ids = self._encode([str(t) for t in X])
        self.model_.eval()
        outs = []
        with torch.no_grad():
            for start in range(0, len(ids), 512):
                xb = torch.as_tensor(ids[start : start + 512], device=self.device_)
                outs.append(torch.softmax(self.model_(xb), dim=1).cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, len(self.classes_)))

    def predict(self, X):  # noqa: N803
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    # ------------------------------------------------------------ pickling
    def __getstate__(self):
        state = self.__dict__.copy()
        if "model_" in state:
            buf = io.BytesIO()
            torch.save(state["model_"].state_dict(), buf)
            state["model_bytes_"] = buf.getvalue()
            state["model_embed_dim_"] = int(state["model_"].embedding.embedding_dim)
            del state["model_"]
        return state

    def __setstate__(self, state):
        blob = state.pop("model_bytes_", None)
        embed_dim = state.pop("model_embed_dim_", None)
        self.__dict__.update(state)
        if blob is not None:
            self.device_ = "cpu"
            dummy = np.zeros((len(self.vocab_) + 2, embed_dim), dtype=np.float32) if embed_dim != self.embed_dim else None
            model = self._new_model(dummy)
            model.load_state_dict(torch.load(io.BytesIO(blob), map_location="cpu"))
            self.model_ = model.eval()

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params["kernel_sizes"] = tuple(params["kernel_sizes"])
        return params

    def describe(self) -> dict[str, Any]:
        return {
            "arch": self.arch,
            "vocab": len(getattr(self, "vocab_", {})),
            "epochs_run": getattr(self, "n_epochs_", None),
            "device": getattr(self, "device_", None),
        }
