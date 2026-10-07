"""Generate the thesis tables from the benchmark outputs.

Reuses the paper's generators (selected, families, features, threshold,
per-class, cost) and adds the full 56-cell results table, the corpus
statistics table and the 2022-versus-2026 comparison, so that no number in
the thesis is typed by hand.

    python thesis/make_tables.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "paper"))
import make_tables as paper_tables  # noqa: E402

THESIS = ROOT / "thesis"
TABLES = THESIS / "tables"
FIGS = THESIS / "figures"
RESULTS = ROOT / "reports" / "results.csv"

DS_LABEL = paper_tables.DS_LABEL
MODEL_LABEL = paper_tables.MODEL_LABEL
FEAT_LABEL = paper_tables.FEAT_LABEL
f = paper_tables.f

# 2022 notebook results on the hate corpus (validation split, fixed threshold 0.3;
# vectorizers fitted on train and test together), transcribed from the legacy notebooks
NOTEBOOK_2022 = {
    "logistic_regression": {"bow": 0.530, "tfidf": 0.545, "word2vec": 0.614, "doc2vec": 0.386},
    "linear_svm": {"bow": 0.510, "tfidf": 0.511, "word2vec": 0.613, "doc2vec": 0.198},
    "random_forest": {"bow": 0.553, "tfidf": 0.562, "word2vec": 0.511, "doc2vec": 0.065},
    "xgboost": {"bow": 0.513, "tfidf": 0.519, "word2vec": 0.658, "doc2vec": 0.365},
}


def table_full(df: pd.DataFrame) -> str:
    lines = [
        r"\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{llrrrrrrrr}",
        r"\caption{Every cell of the benchmark: cross-validated F1 (mean and standard deviation across folds), test metrics at the tuned threshold, and fit time on the full training split. Sorted by corpus and cross-validated F1.}\label{tab:full}\\",
        r"\toprule",
        r"Corpus & Model / features & Thr. & CV F1 & SD & Test F1 & Acc. & ROC-AUC & PR-AUC & Fit s \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Corpus & Model / features & Thr. & CV F1 & SD & Test F1 & Acc. & ROC-AUC & PR-AUC & Fit s \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    short = {
        "naive_bayes": "NB",
        "logistic_regression": "LR",
        "linear_svm": "SVM",
        "random_forest": "RF",
        "xgboost": "XGB",
        "ensemble": "Ens.",
        "cnn": "CNN",
        "lstm": "BiLSTM",
    }
    for ds in paper_tables.DATASETS:
        sub = df[df.dataset == ds].sort_values("cv_f1_mean", ascending=False)
        for k, (_, r) in enumerate(sub.iterrows()):
            thr = f(r["threshold"], 2) if pd.notna(r["threshold"]) else "argmax"
            lines.append(
                f"{DS_LABEL[ds] if k == 0 else ''} & {short[r['model']]} / {FEAT_LABEL[r['features']]} & {thr} & {f(r['cv_f1_mean'])} & {f(r['cv_f1_std'])} & "
                f"{f(r['test_f1'])} & {f(r['test_accuracy'])} & {f(r.get('test_roc_auc'))} & {f(r.get('test_pr_auc'))} & {r['fit_seconds']:.1f} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\end{longtable}"
    return "\n".join(lines)


def table_corpora(df: pd.DataFrame) -> str:
    manifest = ROOT / "data" / "manifest.json"
    counts = json.loads(manifest.read_text()).get("twitter_hate", {}).get("classes") if manifest.exists() else None
    rows = [
        ("Hate (Twitter)", "31,962", "2: not hate 92.99 percent, hate 7.01 percent", "human (competition)", "stratified 80/20"),
        ("Vaccine (Twitter)", "10,353", "3: negative 9.9, neutral 56.6, positive 33.5 percent", "TextBlob polarity sign", "stratified 80/20"),
        ("IMDB", "50,000", "2: balanced", "human (star ratings)", "official 25k/25k"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Corpora used in the study.}",
        r"\label{tab:corpora}",
        r"\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrp{4.6cm}p{2.6cm}l}",
        r"\toprule",
        r"Corpus & Documents & Classes and balance & Labels & Split \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    for ds in paper_tables.DATASETS:
        sub = df[df.dataset == ds]
        if not sub.empty:
            lines.append(f"% {ds}: train {int(sub.iloc[0]['n_train'])}, test {int(sub.iloc[0]['n_test'])}")
    if counts:
        lines.append(f"% manifest classes hate: {counts}")
    return "\n".join(lines)


def table_then_now(df: pd.DataFrame) -> str:
    hate = df[df.dataset == "twitter_hate"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Hate corpus: the 2022 notebook study (validation F1 at a fixed threshold of 0.3, vectorizers fitted on training and test documents together, single split) against the present protocol (test F1 at the out-of-fold tuned threshold, features fitted per fold). The 2022 Naive Bayes figure was an accuracy of 0.941 on a leaky split and is not comparable.}",
        r"\label{tab:thennow}",
        r"\small",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Model & Features & 2022 notebook F1 & 2026 protocol test F1 \\",
        r"\midrule",
    ]
    for m, feats in NOTEBOOK_2022.items():
        for ft, old in feats.items():
            row = hate[(hate.model == m) & (hate.features == ft)]
            new = f(row.iloc[0]["test_f1"]) if not row.empty else "n/a"
            lines.append(f"{MODEL_LABEL[m]} & {FEAT_LABEL[ft]} & {old:.3f} & {new} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(RESULTS)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    selected = paper_tables.table_selected(df).replace("table*", "table")
    selected = selected.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}").replace("\\end{tabular}", "\\end{tabular}}")
    (TABLES / "selected.tex").write_text(selected, encoding="utf-8")
    (TABLES / "families.tex").write_text(paper_tables.table_families(df).replace("table*", "table"), encoding="utf-8")
    (TABLES / "features.tex").write_text(paper_tables.table_features(df), encoding="utf-8")
    (TABLES / "threshold.tex").write_text(paper_tables.table_threshold(df), encoding="utf-8")
    (TABLES / "perclass.tex").write_text(paper_tables.table_perclass(), encoding="utf-8")
    (TABLES / "cost.tex").write_text(paper_tables.table_cost(df), encoding="utf-8")
    (TABLES / "full.tex").write_text(table_full(df), encoding="utf-8")
    (TABLES / "corpora.tex").write_text(table_corpora(df), encoding="utf-8")
    (TABLES / "thennow.tex").write_text(table_then_now(df), encoding="utf-8")
    import textwrap

    card = json.loads((ROOT / "models" / "twitter_hate" / "model_card.json").read_text())
    wrapped = []
    for line in json.dumps(card, indent=2).splitlines():
        indent = len(line) - len(line.lstrip())
        wrapped.extend(textwrap.wrap(line, width=92, subsequent_indent=" " * (indent + 4)) or [""])
    (TABLES / "modelcard.tex").write_text(
        "\\begin{scriptsize}\n\\begin{verbatim}\n" + "\n".join(wrapped) + "\n\\end{verbatim}\n\\end{scriptsize}\n", encoding="utf-8"
    )
    arch = paper_tables.architecture_figure()
    (FIGS / "architecture.png").write_bytes(arch.read_bytes())
    print(f"thesis tables -> {TABLES} ({len(df)} cells); figure -> {FIGS / 'architecture.png'}")


if __name__ == "__main__":
    main()
