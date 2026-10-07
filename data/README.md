# Data

## TweetSense demo

Run `python scripts/download_tweeteval.py` from the repository root to place the official [TweetEval sentiment files](https://github.com/cardiffnlp/tweeteval/tree/main/datasets/sentiment) in `data/raw/tweet_sentiment/`. The demo uses the original train and test splits and predicts negative, neutral, or positive sentiment. See `TWEETSENSE_DEMO.md` for the full setup.

No data is committed to this repository: raw corpora, derived labels, samples
and the manifest are all git-ignored. Tests and CI use a deterministic
synthetic corpus generated in code (`datasets.synthetic_dataset`), so the
pipeline can be exercised end to end without any download.

```
data/
  raw/
    twitter_hate/train.csv                 id,label,tweet
    vaccination/vaccination_tweets.csv     text column plus user metadata
    vaccination/vaccination_labelled.csv   generated: text,label after weak labelling
    imdb/aclImdb/                          train/ and test/ with pos/ and neg/
    imdb/imdb.csv.gz                       generated cache
  manifest.json                            generated: rows, class balance, SHA-256 of the raw files
```

## Corpora used in the study

| name | size | task | reference |
|------|------|------|-----------|
| `twitter_hate` | 31,962 labelled tweets, 7.0 percent positive | binary: racist or sexist versus other | Analytics Vidhya practice problem "Twitter Sentiment Analysis" (hate speech detection); thesis experiment 1 |
| `vaccination` | 11,020 tweets, 10,353 after de-duplication | three classes from TextBlob polarity (weak labels) | G. Preda, "Pfizer Vaccine Tweets", Kaggle; thesis experiment 2 |
| `imdb` | 50,000 reviews, balanced, official 25k/25k split | binary | A. L. Maas et al., "Learning Word Vectors for Sentiment Analysis", ACL 2011 |

## Obtaining the files

Download locations are configuration, not code. Put them in `.env`
(see `.env.example`):

| key | meaning |
|-----|---------|
| `IMDB_ARCHIVE_URL` | direct URL of `aclImdb_v1.tar.gz` from the dataset's page |
| `KAGGLE_TWITTER_HATE_DATASET` | Kaggle identifier (`owner/slug`) of a mirror of the hate-speech practice data |
| `KAGGLE_VACCINATION_DATASET` | Kaggle identifier of the vaccine tweets dataset |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | Kaggle API credentials used by the Kaggle client |

Then:

```bash
sentiment-benchmark data download --dataset imdb
sentiment-benchmark data download --dataset twitter_hate
sentiment-benchmark data download --dataset vaccination
sentiment-benchmark data check          # rows, class balance, content fingerprint per dataset
```

Files placed by hand in the paths above work too; the loaders also accept the
original competition file name for the hate-speech set.

## Labels

- `twitter_hate`: 0 = not hate, 1 = racist or sexist (competition labels).
- `vaccination`: no human labels exist. As in the thesis, each de-duplicated
  tweet is labelled by the sign of its TextBlob polarity: negative (0),
  neutral (1), positive (2). Results on this set measure how well a model
  reproduces a lexicon labeller, not human-judged sentiment; the README and
  the model card carry the same caveat.
- `imdb`: 0 = negative (rating 4 or lower), 1 = positive (rating 7 or higher).

## Licensing

Each corpus keeps its original licence and terms of use; this repository
redistributes none of them.
