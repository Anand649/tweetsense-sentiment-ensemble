# TweetSense: Twitter sentiment ensemble demo

This local adaptation of [rhs2/sentiment-ml-benchmark](https://github.com/rhs2/sentiment-ml-benchmark) trains a three-class sentiment model on the [TweetEval sentiment dataset](https://github.com/cardiffnlp/tweeteval/tree/main/datasets/sentiment). It serves predictions through a browser page and shows held-out evaluation on `/analysis`.

## What it uses

- Python 3.11 or 3.12, scikit-learn, pandas, Flask, and TF-IDF.
- Soft voting over logistic regression, calibrated linear SVM, and Multinomial Naive Bayes.
- TweetEval's official 45,615-tweet train split and 12,284-tweet test split. Labels are negative, neutral, and positive.
- SQLite for optional prediction logging by default. PostgreSQL is not required.

## Run it

From this repository's root:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe scripts\download_tweeteval.py
.venv\Scripts\sentiment-benchmark.exe benchmark --config configs\tweet_sentiment.yaml --no-tracking
.venv\Scripts\python.exe app\app.py
```

Open `http://127.0.0.1:8765` to enter a tweet and see its predicted sentiment, confidence, and all three class probabilities. Open `http://127.0.0.1:8765/analysis` for the test metrics and charts. The frontend calls `POST /api/predict`.

On macOS or Linux, use `python3.12 -m venv .venv`, then `.venv/bin/python` and `.venv/bin/sentiment-benchmark` in the same commands.

The dataset files live in `data/raw/tweet_sentiment/`; model artifacts appear in `models/tweet_sentiment/`, and evaluation files appear in `reports/`. The raw data and trained model are ignored by Git, so run the download and training commands after a fresh clone.

## Verified local run

On Python 3.12.14 under Windows, the model trained and exported successfully. Held-out test accuracy was 0.587 and macro F1 was 0.552. The local suite finished with 43 passed and 1 skipped. The home page, analysis page, and prediction API returned HTTP 200 using the exported model.

These metrics are model evaluation results, not a guarantee for every individual tweet. The TweetEval sentiment task originates from SemEval Twitter data; see its repository for dataset terms and attribution.
