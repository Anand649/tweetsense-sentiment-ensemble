# Notebooks

`legacy/` holds the three notebooks from the May 2022 thesis, kept as the
historical record with their original outputs:

| notebook | thesis experiment | what it did |
|----------|-------------------|-------------|
| `ex1_twitter_naive_bayes.ipynb` | Ex1 | cleaning, word clouds, hashtag analysis, bag-of-words, Multinomial Naive Bayes |
| `ex1_twitter_lr_svm_rf_xgb.ipynb` | Ex1 | bag-of-words, TF-IDF, Word2Vec, Doc2Vec features; Logistic Regression, SVM, Random Forest, XGBoost; staged XGBoost grid search |
| `ex2_vaccination_lr_svm.ipynb` | Ex2 | TextBlob polarity labels, 1-2 gram counts, Logistic Regression and LinearSVC with grid search |

Two edits were made before publishing: the cell that embedded Twitter API
credentials was replaced by a comment, and the Colab drive mount was removed.
Nothing else was changed, so the notebooks still show the 2022 evaluation
choices (vectorizers fitted on train and test together, a fixed 0.3
threshold) that the maintained pipeline in `src/` corrects. Use the pipeline,
not these notebooks, for any new numbers.
