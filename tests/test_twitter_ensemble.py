"""Check the probability blend used by the web prediction path."""

import numpy as np

from sentiment_benchmark.twitter_ensemble import TwitterEnsemble, normalize_tweet


def test_twitter_normalization():
    assert normalize_tweet("Hi @anand see https://example.com great!") == "Hi @user see http great!"


def test_hybrid_prediction_uses_both_members():
    model = object.__new__(TwitterEnsemble)
    model.weight = 0.75
    model.card = {"label_names": ["negative", "neutral", "positive"]}
    model.base_proba = lambda texts: np.array([[0.8, 0.1, 0.1]])
    model.transformer_proba = lambda texts: np.array([[0.6, 0.3, 0.1]])
    result = model.predict(["Bad writing."])[0]
    assert result["label_name"] == "negative"
    assert np.isclose(result["probabilities"]["negative"], 0.65)
    assert np.isclose(sum(result["probabilities"].values()), 1)
