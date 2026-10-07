"""Refresh the Results section of README.md from reports/results.csv.

    python scripts/update_readme_results.py

The headline tables and the figure gallery are always the output of the last
benchmark run, never hand-edited numbers.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULTS = ROOT / "reports" / "results.csv"
FIGURES = ROOT / "reports" / "figures"

TASK_NOTE = {"binary": "F1 on the positive class", "multiclass": "macro F1"}
KEY_FIGURES = ["roc", "pr", "calibration", "threshold", "cv", "efficiency"]
FIGURE_CAPTIONS = [
    ("roc", "ROC curves (best feature set per model)"),
    ("pr", "Precision-recall curves"),
    ("calibration", "Reliability diagram"),
    ("threshold", "Threshold sweep on out-of-fold predictions"),
    ("cv", "Cross-validation stability versus test agreement"),
    ("heatmap", "Test F1 by model and feature set"),
    ("efficiency", "Accuracy versus training cost"),
    ("per_class", "Per-class precision, recall and F1 of the selected model"),
    ("confusion", "Confusion matrix of the selected model"),
]


CAPTION = dict(FIGURE_CAPTIONS)


def fmt(v, nd=3):
    return "n/a" if v is None or pd.isna(v) else f"{v:.{nd}f}"


def build_section(df: pd.DataFrame) -> str:
    datasets = list(dict.fromkeys(df["dataset"]))
    models = list(dict.fromkeys(df["model"]))
    lines = []

    lines.append("### Table 1. Selected model per dataset")
    lines.append("")
    lines.append("Selection uses cross-validated F1 on the training split only; the test split is scored once, after selection.")
    lines.append("")
    lines.append("| dataset | task | n train / test | model + features | test F1 | accuracy | precision | recall | ROC-AUC | PR-AUC | threshold |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for ds in datasets:
        sub = df[df.dataset == ds]
        best = sub.sort_values(["cv_f1_mean", "test_f1"], ascending=False).iloc[0]
        n = f"{int(best['n_train'])} / {int(best['n_test'])}" if "n_train" in sub else ""
        thr = fmt(best.get("threshold"), 2) if pd.notna(best.get("threshold")) else "argmax"
        lines.append(
            f"| `{ds}` | {best['task']} | {n} | {best['model']} + {best['features']} | **{fmt(best['test_f1'])}** | "
            f"{fmt(best['test_accuracy'])} | {fmt(best['test_precision'])} | {fmt(best['test_recall'])} | "
            f"{fmt(best.get('test_roc_auc'))} | {fmt(best.get('test_pr_auc'))} | {thr} |"
        )
    lines.append("")
    lines.append(
        f"F1 is {TASK_NOTE['binary']} for binary tasks and {TASK_NOTE['multiclass']} for the three-class task. ROC-AUC for the three-class task is one-versus-rest, macro-averaged."
    )
    lines.append("")

    lines.append("### Table 2. Every model family at its best feature set")
    lines.append("")
    lines.append("Cell format: test F1 (feature set), with the cross-validated F1 and its standard deviation across folds in the second line.")
    lines.append("")
    lines.append("| model | " + " | ".join(f"`{d}`" for d in datasets) + " |")
    lines.append("|---|" + "---|" * len(datasets))
    for m in models:
        cells = []
        for d in datasets:
            sub = df[(df.model == m) & (df.dataset == d)]
            if sub.empty:
                cells.append("n/a")
            else:
                r = sub.sort_values("cv_f1_mean", ascending=False).iloc[0]
                cells.append(f"{fmt(r['test_f1'])} ({r['features']})<br><sub>cv {fmt(r['cv_f1_mean'])} ± {fmt(r['cv_f1_std'])}</sub>")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("### Table 3. Feature sets (best model per feature set, test F1)")
    lines.append("")
    feats = list(dict.fromkeys(df["features"]))
    lines.append("| feature set | " + " | ".join(f"`{d}`" for d in datasets) + " |")
    lines.append("|---|" + "---|" * len(datasets))
    for f in feats:
        cells = []
        for d in datasets:
            sub = df[(df.features == f) & (df.dataset == d)]
            if sub.empty:
                cells.append("n/a")
            else:
                r = sub.sort_values("cv_f1_mean", ascending=False).iloc[0]
                cells.append(f"{fmt(r['test_f1'])} ({r['model']})")
        lines.append(f"| {f} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("### Figures")
    lines.append("")
    lines.append("Key figures per dataset (two per row); the complete gallery follows, collapsed.")
    lines.append("")
    for ds in datasets:
        lines.append(f"**{ds}**")
        lines.append("")
        keys = [k for k in KEY_FIGURES if (FIGURES / f"{k}_{ds}.png").exists()]
        lines.append("<table><tbody>")
        for i in range(0, len(keys), 2):
            pair = keys[i : i + 2]
            lines.append(
                "<tr>"
                + "".join(
                    f'<td width="50%" align="center"><img src="reports/figures/{k}_{ds}.png" alt="{CAPTION[k]}" width="100%"><br><sub>{CAPTION[k]}</sub></td>'
                    for k in pair
                )
                + "</tr>"
            )
        lines.append("</tbody></table>")
        lines.append("")
    lines.append("<details><summary><b>Complete figure gallery (all datasets, all figure types)</b></summary>")
    lines.append("")
    for ds in datasets:
        lines.append(f"#### {ds}")
        lines.append("")
        for key, caption in FIGURE_CAPTIONS:
            p = FIGURES / f"{key}_{ds}.png"
            if p.exists():
                lines.append(f"*{caption}*")
                lines.append("")
                lines.append(f"![{caption}](reports/figures/{key}_{ds}.png)")
                lines.append("")
    lines.append("</details>")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(RESULTS)
    section = build_section(df)
    text = README.read_text(encoding="utf-8")
    new = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", lambda _: f"<!-- RESULTS:START -->\n{section}\n<!-- RESULTS:END -->", text, flags=re.S)
    README.write_text(new, encoding="utf-8")
    print(f"updated {README} from {RESULTS} ({len(df)} cells)")


if __name__ == "__main__":
    main()
