"""Build the thesis: tables from the benchmark outputs, PDF with tectonic,
DOCX with pandoc.

    python thesis/build.py

Output: thesis/Sium_2026_Thesis_Sentiment_Classification_Benchmark.pdf / .docx
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

THESIS = Path(__file__).resolve().parent
ROOT = THESIS.parent
OUT = "Sium_2026_Thesis_Sentiment_Classification_Benchmark"
TITLE = "Leakage-Free Benchmarking of Classical and Neural Sentiment Classifiers: From a Thesis Notebook to a Reproducible, Operable System"
AUTHOR_LINE = "Rakibul Hasan Sium, Nanjing University of Information Science and Technology, Nanjing, China. sium@nuist.edu.cn"


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def inline_inputs(tex: str, base: Path) -> str:
    def repl(m):
        p = base / (m.group(1) + ".tex")
        return inline_inputs(p.read_text(encoding="utf-8"), base) if p.exists() else m.group(0)

    return re.sub(r"\\input\{([^}]+)\}", repl, tex)


def prepare_docx_source(src: Path, dst: Path) -> Path:
    """Flatten \\input, drop the LaTeX title page and front-matter commands,
    number captions, resolve \\ref (chapters, sections, tables, figures),
    convert \\cite to [n] and the bibliography to a numbered list."""
    tex = inline_inputs(src.read_text(encoding="utf-8"), src.parent)
    tex = re.sub(r"\\IfFileExists\{title\.tex\}.*?\n", "", tex, count=1, flags=re.S)
    tex = re.sub(r"\\begin\{titlepage\}.*?\\end\{titlepage\}", "", tex, flags=re.S)
    for cmd in (
        r"\\pagenumbering\{\w+\}",
        r"\\tableofcontents",
        r"\\listoffigures",
        r"\\listoftables",
        r"\\clearpage",
        r"\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}",
    ):
        tex = re.sub(cmd, "", tex)
    tex = tex.replace("\\appendix", "\\APPENDIXMARK")
    tex = tex.replace("\\begin{table*}", "\\begin{table}").replace("\\end{table*}", "\\end{table}")
    tex = tex.replace("\\resizebox{\\textwidth}{!}{\\begin{tabular}", "\\begin{tabular}").replace("\\end{tabular}}", "\\end{tabular}")
    tex = tex.replace("\\textpm{}", "\u00b1").replace("\\textpm\\ ", "\u00b1 ")
    # longtable -> table with tabular (pandoc does not read longtable headers)
    tex = re.sub(r"\\begin\{longtable\}\{([^}]*)\}", r"\\begin{table}\\begin{tabular}{\1}", tex)
    tex = tex.replace("\\end{longtable}", "\\end{tabular}\\end{table}")
    tex = re.sub(r"\\endfirsthead.*?\\endfoot", "", tex, flags=re.S)
    tex = re.sub(r"\\caption\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\\label\{([^}]+)\}\\\\", r"\\caption{\1}\\label{\2}", tex)

    # numbering: chapters and sections in order, tables and figures in order
    labels: dict[str, str] = {}
    chap = 0
    sec = 0

    appendix = False

    def number_heading(m):
        nonlocal chap, sec
        kind, title, lab = m.group(1), m.group(2), m.group(3)
        if kind == "chapter":
            chap += 1
            sec = 0
            num = chr(ord("A") + chap - 1) if appendix else str(chap)
        else:
            sec += 1
            num = f"{chr(ord('A') + chap - 1) if appendix else chap}.{sec}"
        if lab:
            labels[lab] = num
        return f"\\{kind}{{{title}}}"

    heading = r"\\(chapter|section)\{([^}]*)\}\s*(?:\\label\{([^}]+)\})?"
    if "\\APPENDIXMARK" in tex:
        main_part, app_part = tex.split("\\APPENDIXMARK", 1)
        main_part = re.sub(heading, number_heading, main_part)
        appendix, chap = True, 0
        app_part = re.sub(heading, number_heading, app_part)
        tex = main_part + app_part
    else:
        tex = re.sub(heading, number_heading, tex)
    counts = {"table": 0, "figure": 0}
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
        prefix = f"Table {counts['table']}. " if kind == "table" else f"Figure {counts['figure']}. "
        out.append(tex[pos : cap + len("\\caption{")] + prefix)
        pos = cap + len("\\caption{")
    tex = "".join(out) + tex[pos:]
    tex = re.sub(r"(~?)\\ref\{([^}]+)\}", lambda mm: (" " if mm.group(1) else "") + labels.get(mm.group(2), "?"), tex)

    m = re.search(r"\\begin\{thebibliography\}\{\d+\}(.*?)\\end\{thebibliography\}", tex, flags=re.S)
    items = re.findall(r"\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem\{|$)", m.group(1), flags=re.S) if m else []
    order = {key: i + 1 for i, (key, _) in enumerate(items)}

    def cite(match):
        keys = [k.strip() for k in match.group(2).split(",")]
        return (" " if match.group(1) else "") + "[" + ", ".join(str(order.get(k, "?")) for k in keys) + "]"

    tex = re.sub(r"(~?)\\cite\{([^}]+)\}", cite, tex)
    refs = "\\chapter*{References}\n\\begin{enumerate}\n" + "\n".join(f"\\item {body.strip()}" for _, body in items) + "\n\\end{enumerate}\n"
    tex = re.sub(r"\\begin\{thebibliography\}\{\d+\}.*?\\end\{thebibliography\}", lambda _: refs, tex, flags=re.S)
    tex = tex.replace("\\chapter*{Abstract}", "\\chapter*{Abstract}").replace("\\chapter*{Acknowledgements}", "\\chapter*{Acknowledgements}")
    dst.write_text(tex, encoding="utf-8")
    return dst


def main() -> int:
    run([sys.executable, str(THESIS / "make_tables.py")], ROOT)
    if shutil.which("tectonic"):
        run(["tectonic", "--keep-logs", "main.tex"], THESIS)
    else:
        for _ in range(3):
            run(["pdflatex", "-interaction=nonstopmode", "main.tex"], THESIS)
    pdf = THESIS / f"{OUT}.pdf"
    shutil.move(THESIS / "main.pdf", pdf)
    print(f"PDF  -> {pdf.name} ({pdf.stat().st_size // 1024} KB)")
    if shutil.which("pandoc"):
        src = prepare_docx_source(THESIS / "main.tex", THESIS / "main_docx.tex")
        docx = THESIS / f"{OUT}.docx"
        run(
            [
                "pandoc",
                src.name,
                "-o",
                docx.name,
                "--from=latex",
                "--resource-path=.:figures:../reports/figures",
                "--toc",
                f"--metadata=title:{TITLE}",
                f"--metadata=author:{AUTHOR_LINE}",
            ],
            THESIS,
        )
        src.unlink(missing_ok=True)
        print(f"DOCX -> {docx.name} ({docx.stat().st_size // 1024} KB)")
    for junk in ("main.log", "main.aux", "main.toc", "main.lof", "main.lot", "main.out", "title.tex"):
        (THESIS / junk).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
