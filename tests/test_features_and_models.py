import io

import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from sentiment_benchmark.features import Doc2VecVectorizer, MeanWord2Vec, make_feature_extractor
from sentiment_benchmark.models import ALL_MODELS, make_model
from sentiment_benchmark.preprocess import TextCleaner


@pytest.mark.parametrize("name", ["bow", "tfidf", "word2vec", "doc2vec", "sequence"])
def test_feature_extractors_fit_transform(name, tiny_binary):
    texts, _ = tiny_binary
    params = {"vector_size": 16, "min_count": 1, "epochs": 2} if name in ("word2vec", "doc2vec") else {}
    ext = make_feature_extractor(name, params)
    out = ext.fit(texts).transform(texts[:4])
    if name == "sequence":
        assert out == texts[:4]
    else:
        assert out.shape[0] == 4
        if name in ("word2vec", "doc2vec"):
            assert out.shape[1] == 16 and np.isfinite(out).all()


def test_word2vec_unknown_words_give_zero_vector():
    w2v = MeanWord2Vec(vector_size=8, min_count=1, epochs=1).fit(["alpha beta", "beta gamma"])
    assert np.all(w2v.transform(["zzz unknown"]) == 0)
    assert w2v.embedding_matrix({"alpha": 1, "zzz": 2}).shape == (3, 8)


def test_doc2vec_infers_for_unseen_docs():
    d2v = Doc2VecVectorizer(vector_size=8, min_count=1, epochs=2).fit(["alpha beta gamma"] * 5)
    assert d2v.transform(["alpha gamma", ""]).shape == (2, 8)


@pytest.mark.parametrize("name", ALL_MODELS)
def test_every_model_trains_predicts_and_pickles(name, tiny_binary):
    texts, labels = tiny_binary
    feature = "sequence" if name in ("cnn", "lstm") else "tfidf"
    params = {"epochs": 2, "embed_dim": 8, "n_filters": 4, "hidden": 4, "max_len": 8, "val_fraction": 0.2} if name in ("cnn", "lstm") else {}
    if name == "xgboost":
        params = {"n_estimators": 10, "max_depth": 2}
    if name == "random_forest":
        params = {"n_estimators": 10}
    if name == "linear_svm":
        params = {"calibration_cv": 2}
    if name == "ensemble":
        params = {"names": ["logistic_regression", "xgboost"], "members": {"xgboost": {"n_estimators": 10, "max_depth": 2}}}
    pipe = Pipeline([("clean", TextCleaner(stemmer=None, min_word_len=1)), ("features", make_feature_extractor(feature)), ("model", make_model(name, params))])
    pipe.fit(texts, labels)
    proba = pipe.predict_proba(["love this wonderful film", "awful boring terrible"])
    assert proba.shape == (2, 2) and np.allclose(proba.sum(axis=1), 1, atol=1e-5)
    assert list(pipe.named_steps["model"].classes_) == [0, 1]
    buf = io.BytesIO()
    joblib.dump(pipe, buf)
    buf.seek(0)
    restored = joblib.load(buf)
    assert np.allclose(restored.predict_proba(["love this wonderful film"]), proba[:1], atol=1e-5)


def test_neural_word2vec_init_and_multiclass(tiny_binary):
    texts, labels = tiny_binary
    labels3 = [(i % 3) for i in range(len(texts))]
    clf = make_model("cnn", {"epochs": 1, "embed_dim": 8, "embedding_init": "word2vec", "max_len": 6, "val_fraction": 0.2, "class_weight": "balanced"})
    clf.fit(texts, labels3)
    assert clf.predict_proba(texts[:3]).shape == (3, 3)
    assert clf.describe()["arch"] == "cnn" and clf.describe()["epochs_run"] == 1


def test_unknown_model_and_feature_raise():
    with pytest.raises(KeyError):
        make_model("transformer")
    with pytest.raises(KeyError):
        make_feature_extractor("glove")
