# TweetSense

TweetSense is a Twitter sentiment analysis project. It blends a **Twitter-trained RoBERTa model** with a **soft-voting ensemble** trained on the [TweetEval sentiment dataset](https://github.com/cardiffnlp/tweeteval/tree/main/datasets/sentiment) to classify a post as **negative**, **neutral**, or **positive**. A Flask web page lets you enter text and immediately see the prediction, confidence, and probability for each class. A second page shows the model's held-out evaluation.

## How it works

1. `scripts/download_tweeteval.py` downloads the labeled TweetEval train, validation, and test splits.
2. `src/sentiment_benchmark/datasets.py` loads the tweets and keeps the official test split separate.
3. The training pipeline cleans the text and turns it into TF-IDF word and phrase features.
4. Logistic regression, a calibrated linear SVM, and Multinomial Naive Bayes each predict class probabilities. A soft vote averages them for the final prediction.
5. `scripts/upgrade_tweet_model.py` downloads the [Cardiff Twitter RoBERTa sentiment model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), chooses a blend weight on the validation split, and evaluates once on the test split.
6. `app/app.py` loads the blended model. The browser calls `/api/predict` and displays the result. `/analysis` shows test metrics and generated charts.

The dataset and pretrained weights are downloaded locally. Neither the raw tweets nor the trained weights are committed to this repository. The Cardiff model is licensed [CC BY 4.0](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest); credit the model authors when reusing it.

## Run locally

Use Python 3.11 or 3.12. In PowerShell, from the repository root:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e '.[twitter]'
.venv\Scripts\python.exe scripts\download_tweeteval.py
.venv\Scripts\sentiment-benchmark.exe benchmark --config configs\tweet_sentiment.yaml --no-tracking
.venv\Scripts\python.exe scripts\upgrade_tweet_model.py
.venv\Scripts\python.exe app\app.py
```

Open **http://127.0.0.1:8765/** for predictions and **http://127.0.0.1:8765/analysis** for evaluation. On macOS or Linux, use `python3.12 -m venv .venv` and the executables in `.venv/bin/`.

## Results

The checked-in model cards record a local run on TweetEval's **45,615 training tweets**, **2,000 validation tweets**, and **12,284 held-out test tweets**:

| Model | Test accuracy | Test macro F1 |
| --- | ---: | ---: |
| Original TF-IDF soft vote | 58.7% | 55.2% |
| Twitter RoBERTa + soft vote | **72.6%** | **72.8%** |

The blend uses 75% of the Twitter model's probabilities and 25% of the classical soft vote. That weight was chosen on validation data. These are benchmark results, not a guarantee for an individual tweet. Sarcasm, ambiguous wording, and long reviews outside the tweet domain can still be misclassified.

## Project map

| Path | Purpose |
| --- | --- |
| `configs/tweet_sentiment.yaml` | Dataset, feature, and ensemble settings |
| `src/sentiment_benchmark/` | Data loading, preprocessing, training, evaluation, and prediction |
| `app/` | Flask API, prediction page, analysis page, CSS, and JavaScript |
| `scripts/download_tweeteval.py` | Downloads the official TweetEval sentiment files |
| `scripts/upgrade_tweet_model.py` | Downloads Twitter RoBERTa, tunes the blend on validation, and evaluates on test |
| `src/sentiment_benchmark/twitter_ensemble.py` | Runs and blends the two model probabilities |
| `reports/` | Saved evaluation table and charts |
| `models/tweet_sentiment/model_card.json` | Model settings and evaluation metadata |
| `tests/` | Automated checks |

Read the [project guide](docs/PROJECT_GUIDE.md) for setup details, architecture, API examples, evaluation, and troubleshooting. The [quick start](TWEETSENSE_DEMO.md) is a shorter run checklist.

## Source and license

This project adapts [sentiment-ml-benchmark](https://github.com/rhs2/sentiment-ml-benchmark) under its MIT license. The original copyright notice is retained in [LICENSE](LICENSE). TweetEval data has its own terms; review the [dataset repository](https://github.com/cardiffnlp/tweeteval) before redistribution.
