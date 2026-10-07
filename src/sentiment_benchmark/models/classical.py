"""Classical models from the thesis (Naive Bayes, Logistic Regression, linear
SVM, Random Forest, XGBoost) plus a soft-voting ensemble of the three
strongest linear and boosted learners.

Defaults mirror the thesis notebooks: Random Forest with 400 trees, XGBoost
with the staged grid-search winners (depth 8, min_child_weight 6, eta 0.1,
subsample 0.9, colsample_bytree 0.5, gamma 1.2).
"""

from __future__ import annotations

from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

CLASSICAL_MODELS = ("naive_bayes", "logistic_regression", "linear_svm", "random_forest", "xgboost", "ensemble")


def make_classical(kind: str, params: dict[str, Any], task: str, seed: int):
    params = dict(params)
    if kind == "naive_bayes":
        return MultinomialNB(**{"alpha": 1.0, **params})

    if kind == "logistic_regression":
        p = {"C": 1.0, "max_iter": 2000, "class_weight": None, "random_state": seed}
        p.update(params)
        return LogisticRegression(**p)

    if kind == "linear_svm":
        # LinearSVC has no predict_proba; Platt-style calibration gives scores
        # usable for threshold tuning (the thesis used SVC(probability=True)).
        calib_cv = params.pop("calibration_cv", 3)
        p = {"C": 1.0, "class_weight": None, "random_state": seed, "max_iter": 5000}
        p.update(params)
        return CalibratedClassifierCV(LinearSVC(**p), cv=calib_cv, method="sigmoid")

    if kind == "random_forest":
        p = {"n_estimators": 400, "n_jobs": -1, "random_state": seed, "class_weight": None}
        p.update(params)
        return RandomForestClassifier(**p)

    if kind == "xgboost":
        from xgboost import XGBClassifier

        p = {
            "n_estimators": 300,
            "max_depth": 8,
            "min_child_weight": 6,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.5,
            "gamma": 1.2,
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": seed,
            "eval_metric": "logloss" if task == "binary" else "mlogloss",
        }
        p.update(params)
        if task == "multiclass":
            p.setdefault("objective", "multi:softprob")
        return XGBClassifier(**p)

    if kind == "ensemble":
        # soft vote over logistic regression, calibrated linear SVM and XGBoost;
        # each member takes its own params block, e.g. {"members": {"xgboost": {...}}}
        members = params.pop("members", {}) or {}
        weights = params.pop("weights", None)
        estimators = [
            (name, make_classical(name, members.get(name, {}), task, seed)) for name in params.pop("names", ["logistic_regression", "linear_svm", "xgboost"])
        ]
        return VotingClassifier(estimators=estimators, voting="soft", weights=weights, n_jobs=1)

    raise KeyError(kind)
