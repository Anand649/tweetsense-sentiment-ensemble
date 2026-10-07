from sentiment_benchmark.config import CleanConfig
from sentiment_benchmark.preprocess import TextCleaner, clean_corpus, clean_text


def test_thesis_recipe_keeps_hashtags_and_stems():
    raw = "@user thanks for #lyft credit i can't use cause they don't offer wheelchair vans in pdx. #disapointed http://t.co/x"
    out = clean_text(raw)
    assert "@user" not in out and "http" not in out
    assert "#lyft" in out and "#disapoint" in out  # stemmed hashtag kept
    assert "for" not in out and "the" not in out  # stop words removed
    assert out == out.lower()


def test_short_words_and_hashtag_symbol_removed_when_configured():
    cfg = CleanConfig(keep_hashtags=False, min_word_len=4, stemmer=None, remove_stopwords=False)
    assert clean_text("#model i love u take with u all the time", cfg) == "model love take with time"


def test_empty_and_none_safe():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_corpus(["", "   ", "<br/>"]) == ["", "", ""]


def test_cleaner_transformer_is_stateless_and_configurable():
    cleaner = TextCleaner.from_config(CleanConfig(stemmer=None, min_word_len=1, remove_stopwords=False))
    assert cleaner.fit(["x"]).transform(["Hello World!!", "Second ONE"]) == ["hello world", "second one"]
    assert TextCleaner().get_params()["stemmer"] == "porter"


def test_unknown_stemmer_rejected():
    import pytest

    with pytest.raises(ValueError):
        clean_text("hello", CleanConfig(stemmer="snowball"))
