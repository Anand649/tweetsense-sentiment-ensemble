import json
import os

import pandas as pd
import pytest

from sentiment_benchmark.config import load_config
from sentiment_benchmark.datasets import Dataset, download, load_dataset, synthetic_dataset, textblob_label, write_manifest
from sentiment_benchmark.env import load_dotenv


def test_full_config_loads_and_validates(root):
    cfg = load_config(root / "configs" / "benchmark.yaml")
    assert {d.name for d in cfg.datasets} == {"twitter_hate", "vaccination", "imdb"}
    assert cfg.dataset("imdb").enabled is False
    assert cfg.dataset("imdb").split.strategy == "official"
    assert set(cfg.grid) == set(cfg.models)
    assert cfg.cv_folds == 5


def test_config_rejects_unknown_feature(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("datasets: [{name: twitter_hate}]\nfeatures: {bow: {type: bow}}\nmodels: {naive_bayes: {}}\ngrid: {naive_bayes: [tfidf]}\n")
    with pytest.raises(ValueError):
        load_config(bad)


@pytest.mark.parametrize("name", ["twitter_hate", "vaccination", "imdb"])
def test_synthetic_corpus_is_deterministic_and_labelled(name):
    a, b = synthetic_dataset(name, n=300), synthetic_dataset(name, n=300)
    assert a.frame.equals(b.frame) and a.synthetic
    assert list(a.frame.columns) == ["text", "label"]
    assert a.frame["label"].nunique() == a.n_classes
    assert a.frame["text"].str.len().min() > 0


def test_smoke_mode_falls_back_to_synthetic_when_raw_missing(root, tmp_path):
    cfg = load_config(root / "configs" / "smoke.yaml", {"data_dir": str(tmp_path)})
    ds = load_dataset("twitter_hate", tmp_path, cfg.dataset("twitter_hate"))
    assert ds.synthetic and ds.task == "binary"
    train, test = ds.split(cfg.dataset("twitter_hate"))
    assert len(train) <= 300 and set(train["label"]) == {0, 1} and test["label"].nunique() == 2


def test_full_mode_raises_when_raw_missing(root, tmp_path):
    cfg = load_config(root / "configs" / "benchmark.yaml", {"data_dir": str(tmp_path)})
    with pytest.raises(FileNotFoundError):
        load_dataset("twitter_hate", tmp_path, cfg.dataset("twitter_hate"))


def test_textblob_weak_label_signs():
    assert textblob_label("this vaccine is wonderful and safe") == 2
    assert textblob_label("terrible side effects, awful experience") == 0
    assert textblob_label("dose two scheduled tomorrow") == 1


def test_manifest_records_hashes(tmp_path):
    df = pd.DataFrame({"text": [f"doc {i}" for i in range(40)], "label": [i % 2 for i in range(40)]})
    f = tmp_path / "raw" / "x.csv"
    f.parent.mkdir()
    df.to_csv(f, index=False)
    ds = Dataset("twitter_hate", df, ["a", "b"], "binary", source_files=[f])
    manifest = json.loads(write_manifest([ds], tmp_path).read_text())
    assert manifest["twitter_hate"]["rows"] == 40 and manifest["twitter_hate"]["classes"] == {"a": 20, "b": 20}
    assert len(manifest["twitter_hate"]["files"]["raw/x.csv"]) == 64
    assert len(ds.fingerprint()) == 16


def test_download_locations_come_from_environment(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("IMDB_ARCHIVE_URL", raising=False)
    download("imdb", tmp_path)  # no location configured: prints instructions, no network
    assert "IMDB_ARCHIVE_URL" in capsys.readouterr().out
    envfile = tmp_path / ".env"
    envfile.write_text("# comment\nSOME_TEST_KEY='from-dotenv'\nIGNORED LINE\n")
    monkeypatch.delenv("SOME_TEST_KEY", raising=False)
    assert load_dotenv(envfile) == {"SOME_TEST_KEY": "from-dotenv"}
    assert os.environ["SOME_TEST_KEY"] == "from-dotenv"
