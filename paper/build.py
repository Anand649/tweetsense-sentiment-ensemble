"""Build the paper in one or more title variants: tables and figure from the
benchmark outputs, PDF with tectonic, DOCX with pandoc.

    python paper/build.py                 # builds every variant in VARIANTS
    python paper/build.py --variant ranking
    python paper/build.py --list

Outputs are named <Author>_<Year>_<ShortSlug>.pdf / .docx.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
AUTHOR_LINE = "Rakibul Hasan Sium, Nanjing University of Information Science and Technology, Nanjing, China. sium@nuist.edu.cn"

VARIANTS = {
    "ranking": {
        "title": "Ranking Is Not the Problem: Leakage-Free Benchmarking of Sentiment Classifiers on Twitter and IMDB",
        "short": "Ranking Is Not the Problem",
        "file": "Sium_2026_Ranking_Is_Not_the_Problem",
    },
    "classical-vs-neural": {
        "title": "Classical Versus Neural Sentiment Classification on Twitter and IMDB: A Leakage-Free Benchmark",
        "short": "Classical Versus Neural Sentiment Classification",
        "file": "Sium_2026_Classical_vs_Neural_Sentiment_Benchmark",
    },
}


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def tex_escape(text: str) -> str:
    return text.replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")


def prepare_docx_source(src: Path, dst: Path, title: str) -> Path:
    """pandoc drops IEEEtran-specific macros and the thebibliography
    environment, so for the Word build: inline the tables, number captions,
    resolve \\ref, number the \\bibitem entries, replace \\cite keys by [n],
    emit the references as a numbered list, and expand the IEEE macros."""
    tex = src.read_text(encoding="utf-8")
    tex = re.sub(r"\\IfFileExists\{title\.tex\}\{\\input\{title\}\}\{.*?\}\n", "", tex, count=1, flags=re.S)
    tex = tex.replace("\\title{\\PaperTitle}", f"\\title{{{tex_escape(title)}}}")
    tex = re.sub(r"\\input\{tables/(\w+)\}", lambda mm: (src.parent / "tables" / f"{mm.group(1)}.tex").read_text(encoding="utf-8"), tex)
    tex = tex.replace("\\begin{table*}", "\\begin{table}").replace("\\end{table*}", "\\end{table}")
    tex = tex.replace("\\textpm{}", "\u00b1").replace("\\textpm\\ ", "\u00b1 ")
    counts = {"table": 0, "figure": 0}
    labels: dict[str, str] = {}
    out, pos = [], 0
    for env in re.finditer(r"\\begin\{(table|figure)\}", tex):
        kind = env.group(1)
        end = tex.find(f"\\end{{{kind}}}", env.end())
        cap = tex.find("\\caption{", env.end())
        if cap == -1 or cap > end:
            continue
        counts[kind] += 1
        lab = re.search(r"\\label\{([^}]+)\}", tex[env.end() : end])
        if lab:
            labels[lab.group(1)] = str(counts[kind])
        prefix = f"Table {counts['table']}. " if kind == "table" else f"Fig. {counts['figure']}. "
        out.append(tex[pos : cap + len("\\caption{")] + prefix)
        pos = cap + len("\\caption{")
    tex = "".join(out) + tex[pos:]
    for n, sec in enumerate(re.finditer(r"\\section\{[^}]*\}\s*(?:\\label\{([^}]+)\})?", tex), start=1):
        if sec.group(1):
            labels[sec.group(1)] = str(n)
    tex = re.sub(r"(~?)\\ref\{([^}]+)\}", lambda mm: (" " if mm.group(1) else "") + labels.get(mm.group(2), "?"), tex)
    m = re.search(r"\\begin\{thebibliography\}\{\d+\}(.*?)\\end\{thebibliography\}", tex, flags=re.S)
    items = re.findall(r"\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem\{|$)", m.group(1), flags=re.S) if m else []
    order = {key: i + 1 for i, (key, _) in enumerate(items)}

    def cite(match):
        keys = [k.strip() for k in match.group(2).split(",")]
        return (" " if match.group(1) else "") + "[" + ", ".join(str(order.get(k, "?")) for k in keys) + "]"

    tex = re.sub(r"(~?)\\cite\{([^}]+)\}", cite, tex)
    refs = "\\section*{References}\n\\begin{enumerate}\n" + "\n".join(f"\\item {body.strip()}" for _, body in items) + "\n\\end{enumerate}\n"
    tex = re.sub(r"\\begin\{thebibliography\}\{\d+\}.*?\\end\{thebibliography\}", lambda _: refs, tex, flags=re.S)
    tex = re.sub(r"\\IEEEPARstart\{(\w)\}\{(\w+)\}", r"\1\2", tex)
    tex = re.sub(r"\\begin\{IEEEkeywords\}\s*(.*?)\s*\\end\{IEEEkeywords\}", r"\\noindent\\textbf{Index Terms:} \1", tex, flags=re.S)
    tex = re.sub(r"\\markboth\{[^}]*\}\{[^}]*\}", "", tex)
    dst.write_text(tex, encoding="utf-8")
    return dst


def build_variant(key: str, spec: dict[str, str]) -> list[Path]:
    title_tex = PAPER / "title.tex"
    title_tex.write_text(
        f"\\newcommand{{\\PaperTitle}}{{{tex_escape(spec['title'])}}}\n\\newcommand{{\\PaperShortTitle}}{{{tex_escape(spec['short'])}}}\n",
        encoding="utf-8",
    )
    outputs: list[Path] = []
    if shutil.which("tectonic"):
        run(["tectonic", "--keep-logs", "main.tex"], PAPER)
    else:
        for _ in range(2):
            run(["pdflatex", "-interaction=nonstopmode", "main.tex"], PAPER)
    pdf = PAPER / f"{spec['file']}.pdf"
    shutil.move(PAPER / "main.pdf", pdf)
    outputs.append(pdf)
    print(f"PDF  [{key}] -> {pdf.name} ({pdf.stat().st_size // 1024} KB)")
    if shutil.which("pandoc"):
        docx_src = prepare_docx_source(PAPER / "main.tex", PAPER / "main_docx.tex", spec["title"])
        docx = PAPER / f"{spec['file']}.docx"
        run(
            [
                "pandoc",
                docx_src.name,
                "-o",
                docx.name,
                "--from=latex",
                "--resource-path=.:figures:../reports/figures",
                f"--metadata=title:{spec['title']}",
                f"--metadata=author:{AUTHOR_LINE}",
            ],
            PAPER,
        )
        docx_src.unlink(missing_ok=True)
        outputs.append(docx)
        print(f"DOCX [{key}] -> {docx.name} ({docx.stat().st_size // 1024} KB)")
    else:
        print("pandoc not found; DOCX skipped")
    for junk in ("main.log", "main.aux", "title.tex"):
        (PAPER / junk).unlink(missing_ok=True)
    return outputs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", action="append", choices=sorted(VARIANTS), help="build only this variant (repeatable)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k, v in VARIANTS.items():
            print(f"{k:22s} {v['file']}.pdf / .docx\n{'':22s} {v['title']}")
        return 0
    run([sys.executable, str(PAPER / "make_tables.py")], ROOT)
    for key in args.variant or list(VARIANTS):
        build_variant(key, VARIANTS[key])
    return 0


if __name__ == "__main__":
    sys.exit(main())
