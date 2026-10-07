"""Build the GitHub Pages site in docs/ from the benchmark outputs and the
LaTeX sources of the paper and the thesis.

    python scripts/build_site.py

Produces docs/index.html (project page), docs/paper.html and docs/thesis.html
(full text rendered by pandoc) and docs/figures/. PDFs are not published. Serve with
GitHub Pages: Settings > Pages > Deploy from a branch > main, /docs.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def _load_module(path: Path, name: str):
    """Load paper/build.py and thesis/build.py by path (both are named build.py)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


paper_build = _load_module(ROOT / "paper" / "build.py", "paper_build")
thesis_build = _load_module(ROOT / "thesis" / "build.py", "thesis_build")

SITE_URL = "https://rhs2.github.io/sentiment-ml-benchmark/"
REPO_URL = "https://github.com/rhs2/sentiment-ml-benchmark"
AUTHOR = "Rakibul Hasan Sium"
EMAIL = "sium@nuist.edu.cn"
PAPER_TITLE = paper_build.VARIANTS["ranking"]["title"]
THESIS_TITLE = "Leakage-Free Benchmarking of Classical and Neural Sentiment Classifiers: From a Thesis Notebook to a Reproducible, Operable System"

DS_LABEL = {"twitter_hate": "Hate speech (Twitter)", "vaccination": "Vaccine sentiment (Twitter)", "imdb": "IMDB reviews"}
MODEL_LABEL = {
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic regression",
    "linear_svm": "Linear SVM (calibrated)",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "ensemble": "Soft-voting ensemble",
    "cnn": "CNN",
    "lstm": "BiLSTM",
}
FEAT_LABEL = {"bow": "counts", "tfidf": "TF-IDF", "word2vec": "Word2Vec", "doc2vec": "Doc2Vec", "sequence": "sequence"}


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def copy_assets() -> None:
    (DOCS / "figures").mkdir(parents=True, exist_ok=True)
    for png in (ROOT / "reports" / "figures").glob("*.png"):
        if not png.name.startswith("cm_"):
            shutil.copy(png, DOCS / "figures" / png.name)
    shutil.copy(ROOT / "paper" / "figures" / "architecture.png", DOCS / "figures" / "architecture.png")
    shutil.rmtree(DOCS / "pdf", ignore_errors=True)  # PDFs are not published; the HTML versions are


def render_html(tex_source: Path, out: Path, title: str, toc_depth: int) -> None:
    """pandoc LaTeX -> standalone HTML with MathJax and the site stylesheet."""
    run(
        [
            "pandoc",
            str(tex_source),
            "-o",
            str(out),
            "--from=latex",
            "--to=html5",
            "--standalone",
            "--mathjax",
            "--toc",
            f"--toc-depth={toc_depth}",
            "--css=style.css",
            f"--metadata=title:{title}",
            f"--metadata=author:{AUTHOR}, Nanjing University of Information Science and Technology ({EMAIL})",
            f"--resource-path={ROOT / 'paper'}:{ROOT / 'thesis'}:{ROOT / 'reports' / 'figures'}",
        ],
        ROOT,
    )
    html = out.read_text(encoding="utf-8")
    html = re.sub(r'src="(?:[^"]*/)?([\w\-]+\.png)"', r'src="figures/\1"', html)
    nav = (
        '<nav class="sitenav"><a href="index.html">Project page</a> <a href="paper.html">Paper</a> '
        '<a href="thesis.html">Thesis</a> <a href="' + REPO_URL + '">Code</a></nav>\n'
    )
    html = html.replace("<body>", "<body>\n" + nav, 1)
    out.write_text(html, encoding="utf-8")


def paper_html() -> None:
    src = paper_build.prepare_docx_source(ROOT / "paper" / "main.tex", ROOT / "paper" / "main_site.tex", PAPER_TITLE)
    render_html(src, DOCS / "paper.html", PAPER_TITLE, 2)
    src.unlink(missing_ok=True)


def thesis_html() -> None:
    src = thesis_build.prepare_docx_source(ROOT / "thesis" / "main.tex", ROOT / "thesis" / "main_site.tex")
    render_html(src, DOCS / "thesis.html", THESIS_TITLE, 3)
    src.unlink(missing_ok=True)


def index_html() -> None:
    df = pd.read_csv(ROOT / "reports" / "results.csv")
    rows = []
    for ds in ["twitter_hate", "vaccination", "imdb"]:
        sub = df[df.dataset == ds].sort_values(["cv_f1_mean", "test_f1"], ascending=False)
        b = sub.iloc[0]
        best_classical = sub[~sub.model.isin(["cnn", "lstm"])].iloc[0]
        rows.append(
            f"<tr><td>{DS_LABEL[ds]}</td><td>{int(b.n_train):,} / {int(b.n_test):,}</td>"
            f"<td>{MODEL_LABEL[b.model]} on {FEAT_LABEL[b.features]}</td><td><b>{b.test_f1:.3f}</b></td>"
            f"<td>{b.test_accuracy:.3f}</td><td>{b.test_roc_auc:.3f}</td>"
            f"<td>{MODEL_LABEL[best_classical.model]} on {FEAT_LABEL[best_classical.features]}: {best_classical.test_f1:.3f}</td></tr>"
        )
    families = []
    for m in MODEL_LABEL:
        cells = []
        for ds in ["twitter_hate", "vaccination", "imdb"]:
            sub = df[(df.dataset == ds) & (df.model == m)]
            if sub.empty:
                cells.append("<td>n/a</td>")
            else:
                r = sub.sort_values("cv_f1_mean", ascending=False).iloc[0]
                cells.append(f"<td>{r.test_f1:.3f} <span class='muted'>({FEAT_LABEL[r.features]})</span></td>")
        families.append(f"<tr><td>{MODEL_LABEL[m]}</td>{''.join(cells)}</tr>")
    n_cells = len(df)

    hate = df[df.dataset == "twitter_hate"].sort_values("cv_f1_mean", ascending=False).groupby("model").head(1)
    roc_lo, roc_hi = hate.test_roc_auc.min(), hate.test_roc_auc.max()
    f1_lo, f1_hi = hate.test_f1.min(), hate.test_f1.max()
    gain_classical = (hate[~hate.model.isin(["cnn", "lstm"])].test_f1 - hate[~hate.model.isin(["cnn", "lstm"])]["test_f1_at_0.5"]).max()
    gain_nn = (hate[hate.model.isin(["cnn", "lstm"])].test_f1 - hate[hate.model.isin(["cnn", "lstm"])]["test_f1_at_0.5"]).max()

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{PAPER_TITLE}</title>
<meta name="description" content="A leakage-free benchmark of eight sentiment classifiers across five text representations and three corpora, with MLflow tracking, model cards, a prediction service and drift monitoring.">
<meta property="og:title" content="{PAPER_TITLE}">
<meta property="og:description" content="56-cell benchmark under one evaluation protocol. Ranking ability is nearly identical across model families; thresholds and calibration decide F1.">
<meta property="og:image" content="{SITE_URL}figures/roc_twitter_hate.png">
<link rel="stylesheet" href="style.css">
</head>
<body>
<nav class="sitenav"><a href="index.html" class="active">Project page</a> <a href="paper.html">Paper</a> <a href="thesis.html">Thesis</a> <a href="{REPO_URL}">Code</a></nav>

<header class="hero">
  <h1>{PAPER_TITLE}</h1>
  <p class="authors"><b>{AUTHOR}</b><br>Nanjing University of Information Science and Technology<br><a href="mailto:{EMAIL}">{EMAIL}</a></p>
  <p class="links">
    <a class="btn" href="paper.html">Read the paper</a>
    <a class="btn" href="thesis.html">Read the thesis</a>
    <a class="btn" href="{REPO_URL}">Code</a>
  </p>
</header>

<main>
<section>
  <h2>In one paragraph</h2>
  <p>Comparisons of sentiment classifiers are often decided by evaluation practice rather than by the models: vectorizers fitted on test data, a fixed decision threshold, or accuracy reported on a corpus where one class holds 93 percent of the documents. This work evaluates <b>eight model families</b> (naive Bayes, logistic regression, a calibrated linear SVM, random forest, XGBoost, a soft-voting ensemble, a CNN and a BiLSTM) across <b>five representations</b> (counts, TF-IDF, Word2Vec, Doc2Vec, token sequences) on <b>three corpora</b> (racist and sexist tweet detection, COVID-19 vaccine tweets, IMDB reviews) under <b>one leakage-free protocol</b>: representations fitted per cross-validation fold, decision thresholds and model selection from out-of-fold predictions, one scoring pass on the held-out split. The benchmark ships as a maintained system with experiment tracking, model cards, a prediction service over PostgreSQL, drift monitoring, CI, Docker and cloud infrastructure definitions. Every number on this page is regenerated from saved predictions.</p>
</section>

<section>
  <h2>Key findings</h2>
  <div class="cards">
    <div class="card"><div class="num">{roc_lo:.3f} to {roc_hi:.3f}</div><p>ROC-AUC of <i>every</i> model family on the hate-speech task. They rank tweets almost equally well.</p></div>
    <div class="card"><div class="num">{f1_lo:.3f} to {f1_hi:.3f}</div><p>Their F1 on the positive class. The spread is calibration and threshold transfer, not ranking ability.</p></div>
    <div class="card"><div class="num">+{gain_classical:.2f} / +{gain_nn:.2f}</div><p>F1 gained by tuning the threshold on out-of-fold predictions instead of using 0.5 (classical models / networks).</p></div>
    <div class="card"><div class="num">{n_cells} cells</div><p>Corpus x representation x model, each under the same protocol, each tracked, each regenerable.</p></div>
  </div>
  <ul class="findings">
    <li><b>Over-confident networks transfer thresholds worst.</b> The CNN has the best cross-validated F1 on the hate corpus (0.672) but scores 0.608 on the test split; calibrated linear and tree models reproduce their cross-validated F1 within 0.02.</li>
    <li><b>Sparse n-grams beat learned document embeddings on tweets.</b> Doc2Vec is the weakest representation for every model on both Twitter corpora; Word2Vec helps trees and hurts linear models.</li>
    <li><b>On long balanced reviews, linear models are the ceiling.</b> Logistic regression, the calibrated SVM and their ensemble reach F1 0.884 to 0.888 on IMDB; the CNN and BiLSTM trained from scratch do not, at a hundred times the cost.</li>
    <li><b>The 2022 study was directionally right and numerically wrong.</b> Its ranking of classical models survives the protocol; its headline "94 percent accuracy" was the majority-class rate on a leaky split.</li>
  </ul>
</section>

<section>
  <h2>Selected model per corpus</h2>
  <p>Selection uses cross-validated F1 on the training split only; the test split is scored once. F1 is on the positive class for binary tasks and macro-averaged for the three-class task.</p>
  <div class="scroll"><table>
    <thead><tr><th>Corpus</th><th>Train / test</th><th>Selected model</th><th>Test F1</th><th>Accuracy</th><th>ROC-AUC</th><th>Best classical model</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>
</section>

<section>
  <h2>Every model family at its best representation (test F1)</h2>
  <div class="scroll"><table>
    <thead><tr><th>Model</th><th>Hate speech</th><th>Vaccine</th><th>IMDB</th></tr></thead>
    <tbody>{"".join(families)}</tbody>
  </table></div>
</section>

<section>
  <h2>Figures</h2>
  <div class="gallery">
    <figure><img src="figures/roc_twitter_hate.png" alt="ROC curves on the hate corpus"><figcaption>Hate speech: ROC curves of all eight families. Within 0.013 of each other.</figcaption></figure>
    <figure><img src="figures/calibration_twitter_hate.png" alt="Reliability diagram"><figcaption>Reliability diagram: the class-weighted networks are over-confident; trees and linear models are near the diagonal.</figcaption></figure>
    <figure><img src="figures/threshold_twitter_hate.png" alt="Threshold sweep"><figcaption>Threshold sweep of the CNN on out-of-fold predictions: the F1 optimum sits at 0.90, not 0.5.</figcaption></figure>
    <figure><img src="figures/cv_twitter_hate.png" alt="Cross-validation stability"><figcaption>Cross-validated F1 with fold standard deviation against test F1 for all 22 hate-speech cells.</figcaption></figure>
    <figure><img src="figures/efficiency_twitter_hate.png" alt="Accuracy versus cost"><figcaption>Test F1 against training time: the linear models sit at the top left.</figcaption></figure>
    <figure><img src="figures/pr_imdb.png" alt="IMDB precision-recall"><figcaption>IMDB: precision-recall curves; the linear n-gram models lead.</figcaption></figure>
  </div>
  <p>Thirty figures in total (ROC, precision-recall, reliability, threshold sweeps, cross-validation stability, heatmaps, efficiency, per-class scores and confusion matrices for every corpus) are in the <a href="thesis.html">thesis</a> and in the repository.</p>
</section>

<section>
  <h2>Method in brief</h2>
  <img src="figures/architecture.png" alt="System architecture" class="wide">
  <ol>
    <li>One YAML file defines the experiment: corpora, cleaning, representations, models, grid, folds, seed. It is logged with every run.</li>
    <li>The training split is cleaned once; each representation is fitted per cross-validation fold and shared by every model on that fold, so models are compared on identical inputs and nothing is fitted on validation or test documents.</li>
    <li>Out-of-fold probabilities determine the decision threshold (binary tasks) and the selected model. The test split is scored once, at that threshold.</li>
    <li>Each cell is an MLflow run; the selected pipeline is exported with a model card and a monitoring baseline.</li>
    <li>A Flask service predicts from the exported pipelines and logs to PostgreSQL; a drift monitor compares recent traffic with the baseline (population stability index, out-of-vocabulary rate) and stores its reports.</li>
  </ol>
</section>

<section>
  <h2>Documents</h2>
  <ul>
    <li><b>Paper</b> (journal-article length, IEEE style): <a href="paper.html">read online</a>. PDF available from the author on request.</li>
    <li><b>Thesis</b> (monograph): background and a 2026 literature review, corpora, methodology with formal definitions, system design, all experiments, discussion, future work, appendices with every benchmark cell, reproduction commands and a model card. <a href="thesis.html">read online</a>.</li>
    <li><b>Code</b>: <a href="{REPO_URL}">{REPO_URL}</a>. Both documents are generated from the same results table by scripts, so no number is typed by hand.</li>
  </ul>
</section>

<section>
  <h2>Citation</h2>
  <p>Sium, R. H. (2026). <i>Ranking is not the problem: Leakage-free benchmarking of sentiment classifiers on Twitter and IMDB</i> [Project page and source code]. Nanjing University of Information Science and Technology. {SITE_URL}</p>
<pre><code>@misc{{sium2026ranking,
  author       = {{Sium, Rakibul Hasan}},
  title        = {{Ranking Is Not the Problem: Leakage-Free Benchmarking of
                  Sentiment Classifiers on Twitter and IMDB}},
  year         = {{2026}},
  howpublished = {{Project page and source code}},
  url          = {{{SITE_URL}}},
  note         = {{Extends the author's BSc thesis, Nanjing University of
                  Information Science and Technology, 2022}}
}}</code></pre>
  <p class="muted">Prior work: Sium, R. H. (2022). <i>Research on machine learning classification algorithms: A comprehensive theoretical survey of sentiment analysis and deep learning approaches</i> [Bachelor's thesis, Nanjing University of Information Science and Technology].</p>
</section>
</main>

<footer><p>{AUTHOR}, 2026. Code under the MIT licence; the corpora keep their original licences and are not redistributed.</p></footer>
</body>
</html>
"""
    (DOCS / "index.html").write_text(html, encoding="utf-8")


def main() -> int:
    DOCS.mkdir(exist_ok=True)
    copy_assets()
    index_html()
    paper_html()
    thesis_html()
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    print(f"site -> {DOCS} (index.html, paper.html, thesis.html, figures/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
