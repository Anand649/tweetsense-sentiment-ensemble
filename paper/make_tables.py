"""Generate every table and the architecture figure of the paper from the
benchmark outputs (reports/results.csv, models/*/model_card.json), so the
manuscript never contains a hand-typed number.

    python paper/make_tables.py      # writes paper/tables/*.tex and paper/figures/*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
FIGS = PAPER / "figures"
RESULTS = ROOT / "reports" / "results.csv"

DATASETS = ["twitter_hate", "vaccination", "imdb"]
DS_LABEL = {"twitter_hate": "Hate", "vaccination": "Vaccine", "imdb": "IMDB"}
MODEL_LABEL = {
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic regression",
    "linear_svm": "Linear SVM (calib.)",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "ensemble": "Soft-voting ensemble",
    "cnn": "CNN",
    "lstm": "BiLSTM",
}
FEAT_LABEL = {"bow": "counts", "tfidf": "TF-IDF", "word2vec": "Word2Vec", "doc2vec": "Doc2Vec", "sequence": "sequence"}


def f(v, nd=3):
    return "n/a" if v is None or pd.isna(v) else f"{v:.{nd}f}"


def best_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("cv_f1_mean", ascending=False).groupby("model", sort=False).head(1)


def table_selected(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Model selected per dataset by cross-validated F1, scored once on the held-out test split. F1 is on the positive class for the binary tasks and macro-averaged for the three-class task; ROC-AUC for the three-class task is one-versus-rest, macro-averaged. Thresholds are chosen on out-of-fold predictions (argmax for the three-class task).}",
        r"\label{tab:selected}",
        r"\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrlccccccc}",
        r"\toprule",
        r"Dataset & Task & Train & Test & Model / features & CV F1 & Test F1 & Acc. & Prec. & Rec. & ROC-AUC & Thr. \\",
        r"\midrule",
    ]
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        if sub.empty:
            continue
        b = sub.sort_values(["cv_f1_mean", "test_f1"], ascending=False).iloc[0]
        thr = f(b["threshold"], 2) if pd.notna(b["threshold"]) else "argmax"
        lines.append(
            f"{DS_LABEL[ds]} & {b['task']} & {int(b['n_train'])} & {int(b['n_test'])} & {MODEL_LABEL[b['model']]} / {FEAT_LABEL[b['features']]} & "
            f"{f(b['cv_f1_mean'])} \\textpm\\  {f(b['cv_f1_std'])} & \\textbf{{{f(b['test_f1'])}}} & {f(b['test_accuracy'])} & {f(b['test_precision'])} & {f(b['test_recall'])} & {f(b['test_roc_auc'])} & {thr} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def table_families(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Every model family at its best feature set per dataset (feature set chosen by cross-validated F1). Cells give test F1 with the cross-validated mean \textpm{} standard deviation across folds underneath. The best test F1 per column is in bold.}",
        r"\label{tab:families}",
        r"\begin{tabular}{l" + "l" * len(DATASETS) + "}",
        r"\toprule",
        "Model & " + " & ".join(DS_LABEL[d] for d in DATASETS) + r" \\",
        r"\midrule",
    ]
    best_test = {d: df[df.dataset == d]["test_f1"].max() for d in DATASETS}
    for m in MODEL_LABEL:
        cells = []
        for d in DATASETS:
            sub = df[(df.dataset == d) & (df.model == m)]
            if sub.empty:
                cells.append("n/a")
                continue
            r = sub.sort_values("cv_f1_mean", ascending=False).iloc[0]
            val = f(r["test_f1"])
            if abs(r["test_f1"] - best_test[d]) < 1e-9:
                val = r"\textbf{" + val + "}"
            cells.append(f"{val} ({FEAT_LABEL[r['features']]}) \\\\ \\scriptsize CV {f(r['cv_f1_mean'])} \\textpm\\  {f(r['cv_f1_std'])}")
        row = " & ".join(r"\begin{tabular}[t]{@{}l@{}}" + c + r"\end{tabular}" if "CV" in c else c for c in cells)
        lines.append(f"{MODEL_LABEL[m]} & {row} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def table_features(df: pd.DataFrame) -> str:
    feats = ["bow", "tfidf", "word2vec", "doc2vec", "sequence"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Feature representations: best model per feature set and dataset (test F1, model in parentheses). Doc2Vec was not run on IMDB.}",
        r"\label{tab:features}",
        r"\begin{tabular}{l" + "l" * len(DATASETS) + "}",
        r"\toprule",
        "Features & " + " & ".join(DS_LABEL[d] for d in DATASETS) + r" \\",
        r"\midrule",
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
    for ft in feats:
        cells = []
        for d in DATASETS:
            sub = df[(df.dataset == d) & (df.features == ft)]
            if sub.empty:
                cells.append("n/a")
            else:
                r = sub.sort_values("cv_f1_mean", ascending=False).iloc[0]
                cells.append(f"{f(r['test_f1'])} ({short[r['model']]})")
        lines.append(f"{FEAT_LABEL[ft]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_threshold(df: pd.DataFrame) -> str:
    sub = df[df.dataset == "twitter_hate"].sort_values("cv_f1_mean", ascending=False)
    sub = best_rows(sub)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Effect of the decision threshold on the hate-speech task (best feature set per model). F1 at the threshold tuned on out-of-fold predictions versus F1 at 0.5 on the same test split; the last column is the gap between the cross-validated F1 and the test F1 at the tuned threshold (threshold transfer).}",
        r"\label{tab:threshold}",
        r"\footnotesize\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Model & Feat. & Thr. & F1 tuned & F1 at 0.5 & Gain & CV minus test \\",
        r"\midrule",
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
    for _, r in sub.iterrows():
        lines.append(
            f"{short[r['model']]} & {FEAT_LABEL[r['features']]} & {f(r['threshold'], 2)} & {f(r['test_f1'])} & {f(r['test_f1_at_0.5'])} & "
            f"{r['test_f1'] - r['test_f1_at_0.5']:+.3f} & {r['cv_f1_mean'] - r['test_f1']:+.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_perclass() -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-class test scores of the selected model on each dataset.}",
        r"\label{tab:perclass}",
        r"\begin{tabular}{llrccc}",
        r"\toprule",
        r"Dataset & Class & Support & Prec. & Rec. & F1 \\",
        r"\midrule",
    ]
    for ds in DATASETS:
        card = ROOT / "models" / ds / "model_card.json"
        if not card.exists():
            continue
        c = json.loads(card.read_text())
        for i, p in enumerate(c["per_class"]):
            name = DS_LABEL[ds] if i == 0 else ""
            lines.append(f"{name} & {p['class'].replace('_', ' ')} & {p['support']} & {f(p['precision'])} & {f(p['recall'])} & {f(p['f1'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_cost(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Training cost of the best cell per model family: seconds to fit on the full training split (laptop CPU, 8 cores), hate-speech and IMDB tasks.}",
        r"\label{tab:cost}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & \multicolumn{2}{c}{Hate} & \multicolumn{2}{c}{IMDB} \\",
        r" & F1 & s & F1 & s \\",
        r"\midrule",
    ]
    for m in MODEL_LABEL:
        cells = []
        for d in ("twitter_hate", "imdb"):
            sub = df[(df.dataset == d) & (df.model == m)]
            if sub.empty:
                cells += ["n/a", "n/a"]
            else:
                r = sub.sort_values("cv_f1_mean", ascending=False).iloc[0]
                cells += [f(r["test_f1"]), f"{max(r['fit_seconds'], 0.1):.1f}"]
        lines.append(f"{MODEL_LABEL[m]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def architecture_figure() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.2, 3.1), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef2f8", ec="#4c72b0", fs=7.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#333"))

    # top row: experiment pipeline
    box(1, 27, 15, 12, "Raw corpora\n(env-configured\nlocations)")
    box(19, 27, 15, 12, "Loaders, weak\nlabels, splits,\nmanifest (SHA-256)")
    box(37, 27, 15, 12, "TextCleaner\n(thesis recipe,\nconfigurable)")
    box(55, 27, 15, 12, "Feature cache\nfit per CV fold,\nshared by models")
    box(73, 27, 12, 12, "8 model\nfamilies")
    box(88, 27, 11, 12, "OOF proba,\nthreshold,\ntest score")
    for x in (16, 34, 52, 70, 85):
        arrow(x, 33, x + 3, 33)
    # middle: tracking and export
    box(37, 12, 24, 10, "MLflow run per cell\n(params, metrics, artifacts)", fc="#f4f1e8", ec="#8c6d1f")
    box(65, 12, 34, 10, "Export: pipeline + model card +\nmonitoring baseline", fc="#f4f1e8", ec="#8c6d1f")
    arrow(93, 27, 82, 22)
    arrow(93, 27, 49, 22)
    # bottom: serving and monitoring
    box(1, 1, 28, 8, "Flask API and dashboard\n(predict, health, stats)", fc="#eaf4ec", ec="#2e7d4f")
    box(33, 1, 30, 8, "PostgreSQL store\n(predictions, drift reports)", fc="#eaf4ec", ec="#2e7d4f")
    box(67, 1, 32, 8, "Drift monitor: PSI on length and\npredictions, OOV rate (scheduled)", fc="#eaf4ec", ec="#2e7d4f")
    arrow(82, 12, 15, 9)
    arrow(29, 5, 33, 5)
    arrow(63, 5, 67, 5)
    arrow(82, 12, 83, 9)
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "architecture.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    df = pd.read_csv(RESULTS)
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "selected.tex").write_text(table_selected(df), encoding="utf-8")
    (TABLES / "families.tex").write_text(table_families(df), encoding="utf-8")
    (TABLES / "features.tex").write_text(table_features(df), encoding="utf-8")
    (TABLES / "threshold.tex").write_text(table_threshold(df), encoding="utf-8")
    (TABLES / "perclass.tex").write_text(table_perclass(), encoding="utf-8")
    (TABLES / "cost.tex").write_text(table_cost(df), encoding="utf-8")
    architecture_figure()
    print(f"tables -> {TABLES}, figure -> {FIGS / 'architecture.png'} ({len(df)} cells)")


if __name__ == "__main__":
    main()
