"""Build a manuscript PDF from a Markdown file and the built figures.

    python manuscript/build_pdf.py [--md manuscript/MAIN_nature.md] [--figs manuscript/figures/out] [--out manuscript/out]
                                   [--name NAME] [--compact] [--inline-figures]

A small converter for the Markdown subset the manuscript uses (#/##/### headings, paragraphs, numbered lists, **bold**,
*italic*, `code`, ^superscript^ citations, unicode superscripts) writes LaTeX, and tectonic compiles it.
Default layout: figures with their legends on their own pages after the text (submission style). With --inline-figures
each figure is placed after the paragraph that first cites it (reading style), and the legends section is dropped.
Extended Data legends are set as text, with the images above them where a built file exists. Nothing here alters the
Markdown source.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, ".."))
FIG_FILES = {str(i): [f"fig{i}.pdf"] for i in range(1, 7)}
ED_FILES = {
    "11": [os.path.join(EXP, "results", "biology", "individuality", "fig_bff_slope.pdf")],
    "12": [os.path.join(EXP, "manuscript", "figures", "out", "ed12.pdf")],
    "13": [os.path.join(EXP, "manuscript", "figures", "out", "ed13.pdf")],
    "14": [os.path.join(EXP, "manuscript", "figures", "out", "ed14.pdf")],
}

# v4 layout (MAIN_v4.md): Fig. 4 is the variation figure, Fig. 5 gains the dial panel, Fig. 6 is redrawn; eight Extended Data
# figures and two tables, numbered by first citation (detectors, census, atlas, confinement, closure stages, scan, BFF inflow, lethal tar).
FIG_FILES_V4 = {"1": ["fig1.pdf"], "2": ["fig2.pdf"], "3": ["fig3.pdf"], "4": ["fig4v4.pdf"], "5": ["fig5v4.pdf"], "6": ["fig6v4.pdf"]}
ED_FILES_V4 = {
    "1": [os.path.join(EXP, "manuscript", "figures", "out", "ed12.pdf")],
    "2": [os.path.join(EXP, "manuscript", "figures", "out", "ed_census.pdf")],
    "3": [os.path.join(EXP, "manuscript", "figures", "out", "fig4.pdf")],
    "4": [os.path.join(EXP, "manuscript", "figures", "out", "ed_confine.pdf")],
    "5": [os.path.join(EXP, "manuscript", "figures", "out", "ed_closure.pdf")],
    "6": [os.path.join(EXP, "manuscript", "figures", "out", "ed14.pdf")],
    "7": [os.path.join(EXP, "manuscript", "figures", "out", "ed_bffinflow.pdf")],
    "8": [os.path.join(EXP, "manuscript", "figures", "out", "ed13.pdf")],
}

SYM = {
    "≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$", "→": r"$\rightarrow$", "←": r"$\leftarrow$", "∞": r"$\infty$",
    "≈": r"$\approx$", "⇔": r"$\Leftrightarrow$", "⇒": r"$\Rightarrow$", "Σ": r"$\Sigma$", "−": r"$-$", "≠": r"$\neq$",
    "±": r"$\pm$", "·": r"\textperiodcentered{}", "…": r"\ldots{}", "°": r"\textdegree{}", "µ": r"$\mu$", "μ": r"$\mu$",
    "σ": r"$\sigma$", "ε": r"$\varepsilon$", "δ": r"$\delta$", "λ": r"$\lambda$", "α": r"$\alpha$", "β": r"$\beta$",
    "γ": r"$\gamma$", "θ": r"$\theta$", "π": r"$\pi$", "ρ": r"$\rho$", "τ": r"$\tau$", "φ": r"$\varphi$", "ω": r"$\omega$",
    "Δ": r"$\Delta$", "∈": r"$\in$", "∑": r"$\sum$", "√": r"$\surd$", "≡": r"$\equiv$", "∝": r"$\propto$",
    "½": r"\textonehalf{}", "∫": r"$\int$", "⌈": r"$\lceil$", "⌉": r"$\rceil$", "⌊": r"$\lfloor$", "⌋": r"$\rfloor$",
    "∎": r"$\blacksquare$", "ℓ": r"$\ell$", "□": r"$\square$", "⊆": r"$\subseteq$", "∅": r"$\emptyset$",
    "∂": r"$\partial$", "₂": r"$_2$", "₁": r"$_1$", "ₙ": r"$_n$", "∇": r"$\nabla$", "≫": r"$\gg$", "≪": r"$\ll$", "∪": r"$\cup$", "∩": r"$\cap$", "⊂": r"$\subset$",
}
SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "−", "⁺": "+"}


def esc(s: str) -> str:
    """Escape LaTeX specials in plain text (symbols are mapped later, by symbols())."""
    s = s.replace("\\", "\u0000")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}")):
        s = s.replace(a, b)
    return s.replace("\u0000", r"\textbackslash{}")


def symbols(s: str) -> str:
    return "".join(SYM.get(ch, ch) for ch in s)


def inline(s: str) -> str:
    """Code spans, superscripts, bold, italic → LaTeX. Code spans are escaped verbatim; text is escaped then marked up."""
    out = []
    for p in re.split(r"(`[^`]*`)", s):
        if p.startswith("`") and p.endswith("`") and len(p) >= 2:
            code = symbols(esc(p[1:-1]))
            for brk in ("/", r"\_", "-", "."):
                code = code.replace(brk, brk + r"\allowbreak{}")
            out.append(r"\texttt{" + code + "}")
            continue
        t = esc(p)
        t = re.sub("[" + "".join(SUP) + "]+", lambda m: r"\textsuperscript{" + "".join(SUP[c] for c in m.group(0)) + "}", t)
        t = re.sub(r"\^([^^\s$\\]{1,12})\^", r"\\textsuperscript{\1}", t)
        t = re.sub(r"\^(-|\u2212)?([A-Za-z0-9]+)", r"\\textsuperscript{\1\2}", t)
        t = t.replace("^", r"\textasciicircum{}")
        t = re.sub(r"\*\*([^*\n]+?)\*\*", r"\\textbf{\1}", t)
        t = re.sub(r"(?<![*\w])\*([^*\n]+?)\*(?![*\w])", r"\\emph{\1}", t)
        out.append(symbols(t))
    return "".join(out)


def figure_block(files: list[str], legend_tex: str, out_dir: str, floating: bool) -> list[str]:
    """Images (copied next to the .tex) above a legend; a float placed near the citation, or a fixed block with a page
    break after it."""
    body = [r"\begin{figure}[!htbp]\centering" if floating else r"\noindent\begin{minipage}{\textwidth}\centering"]
    present = [f for f in files if os.path.exists(f)]
    for f in present:
        dst = os.path.join(out_dir, os.path.basename(f))
        if os.path.abspath(f) != os.path.abspath(dst):
            shutil.copy(f, dst)
        h = (r"0.34\textheight" if len(present) > 1 else r"0.62\textheight") if floating else (r"0.36\textheight" if len(present) > 1 else r"0.70\textheight")
        body.append(r"\includegraphics[width=\textwidth,height=%s,keepaspectratio]{%s}\par\vspace{2mm}" % (h, os.path.basename(f)))
    if not present:
        body.append(r"\fbox{\parbox{0.9\textwidth}{\centering\vspace{20mm}\small figure file not built\vspace{20mm}}}\par\vspace{2mm}")
    if floating:
        body += [r"{\small " + legend_tex + r"\par}", r"\end{figure}", ""]
    else:
        body += [r"\end{minipage}\par\vspace{3mm}", r"{\small " + legend_tex + r"\par}", r"\clearpage", ""]
    return body


COMPACT = [False]
INLINE = [False]
NOTE = [r"Review copy assembled DATE; author list and affiliations to be added."]


def convert(md_text: str, figs_dir: str, out_dir: str) -> str:
    lines = md_text.splitlines()
    legends: dict[str, str] = {}
    for l in lines:
        m = re.match(r"\*\*Fig\. (\d+) \| ", l)
        if m:
            legends[m.group(1)] = l.strip()
    body: list[str] = []
    title = "Manuscript"
    title_set = False
    section = ""
    para: list[str] = []
    in_list = False
    placed: set[str] = set()

    def main_fig_block(n: str, floating: bool) -> list[str]:
        files = [os.path.join(figs_dir, f) for f in FIG_FILES.get(n, [])]
        return figure_block(files, inline(legends[n]), out_dir, floating)

    def flush_para():
        nonlocal para
        if not para:
            return
        text = " ".join(x.strip() for x in para)
        para = []
        if text.startswith("**Extended Data"):
            text = re.sub(r"\s*Source: .*$", "", text)       # provenance stays in the Markdown, not in the review copy
        if text.startswith("**Extended Data Table"):
            pending_table_legend.append(inline(text))       # kept with its table (see flush_table)
            return
        m = re.match(r"\*\*Fig\. (\d+) \| ", text)
        if m and section.startswith("Figure legends"):
            if INLINE[0] and m.group(1) in placed:
                return
            body.extend(main_fig_block(m.group(1), floating=False))
            placed.add(m.group(1))
            return
        m = re.match(r"\*\*Extended Data Fig\. (\d+) \| ", text)
        if m and m.group(1) in ED_FILES and any(os.path.exists(f) for f in ED_FILES[m.group(1)]):
            body.extend(figure_block(ED_FILES[m.group(1)], inline(text), out_dir, floating=True))
            return
        if text.startswith("*") and text.endswith("*") and text.count("*") == 2:
            body.append(r"{\small\color{gray}" + inline(text) + "}")
            body.append("")
            return
        body.append(inline(text))
        body.append("")
        if INLINE[0] and not section.startswith(("Figure legends", "Extended Data", "References")):
            for n in re.findall(r"(?<!Extended Data )Fig\.\s*(\d)", text):
                if n in legends and n not in placed:
                    body.extend(main_fig_block(n, floating=True))
                    placed.add(n)

    table_rows: list[str] = []
    pending_table_legend: list[str] = []

    def flush_table():
        """A Markdown pipe table as a small booktabs tabular (alignment from the separator row)."""
        rows = [r for r in table_rows]
        table_rows.clear()
        if not rows:
            return
        cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
        sep = next((i for i, r in enumerate(cells) if all(re.fullmatch(r":?-{2,}:?", c) for c in r if c)), None)
        aligns = ["l"] * len(cells[0])
        if sep is not None:
            aligns = ["r" if c.endswith(":") and not c.startswith(":") else ("c" if c.startswith(":") and c.endswith(":") else "l") for c in cells[sep]]
        head = cells[:sep] if sep is not None else []
        data = cells[sep + 1:] if sep is not None else cells
        for k in range(len(aligns)):                       # a long text column wraps
            if max(len(r[k]) for r in data if k < len(r)) > 70:
                aligns[k] = r">{\raggedright\arraybackslash}p{0.55\linewidth}"
        out = []
        if pending_table_legend:
            out.append(r"\par\noindent\begin{minipage}{\linewidth}")
            out.append(pending_table_legend.pop(0))
            out.append(r"\par\vspace{2pt}")
        out += [r"{\footnotesize\setlength{\tabcolsep}{4pt}\begin{tabular}{" + "".join(aligns) + "}", r"\toprule"]
        for h in head:
            out.append(" & ".join(r"\textbf{" + inline(c) + "}" for c in h) + r" \\")
        if head:
            out.append(r"\midrule")
        for r in data:
            out.append(" & ".join(inline(c) for c in r) + r" \\")
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}\par}")
        if out[0].startswith(r"\par\noindent\begin{minipage}"):
            out.append(r"\end{minipage}\par\vspace{4mm}")
        body.extend(out)
        body.append("")

    def close_list():
        nonlocal in_list
        if in_list:
            body.append(r"\end{enumerate}")
            body.append("")
            in_list = False

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("# ") and not title_set:
            flush_para()
            title = inline(line[2:].strip())
            title_set = True
            continue
        if line.startswith("## "):
            flush_para(); close_list()
            leaving_figs = section.startswith("Figure legends")
            section = line[3:].strip()
            if section.startswith("Figure legends") and INLINE[0] and set(legends) <= placed:
                continue
            if section.startswith("Figure legends") or leaving_figs:
                body.append(r"\clearpage")
            body.append(r"\section*{" + inline(section) + "}")
            body.append("")
            continue
        if line.startswith("### "):
            flush_para(); close_list()
            body.append(r"\subsection*{" + inline(line[4:].strip()) + "}")
            body.append("")
            continue
        if line.strip() == "---":
            flush_para(); close_list()
            continue
        if line.lstrip().startswith("|"):
            flush_para(); close_list()
            table_rows.append(line.strip())
            continue
        if table_rows:
            flush_table()
        m = re.match(r"^(\d+)\. (.*)", line)
        if m:
            flush_para()
            if not in_list:
                body.append(r"\begin{enumerate}[leftmargin=*,itemsep=1pt,parsep=0pt]")
                in_list = True
            body.append(r"\item " + inline(m.group(2)))
            continue
        if line.startswith("- "):
            flush_para()
            body.append(r"\begin{itemize}[leftmargin=*,itemsep=1pt]\item " + inline(line[2:]) + r"\end{itemize}")
            continue
        if not line.strip():
            flush_para(); close_list()
            continue
        if in_list:
            body[-1] += " " + inline(line.strip())
            continue
        para.append(line)
    flush_para(); close_list()
    if table_rows:
        flush_table()

    today = dt.date.today().isoformat()
    margin = "14mm" if COMPACT[0] else "20mm"
    preamble = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=""" + margin + r"""]{geometry}
\usepackage{fontspec}
\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\setsansfont{texgyreheros}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\setmonofont{texgyrecursor}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,Scale=0.9]
\usepackage{graphicx}
\usepackage{amssymb}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{array}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt plus 1pt}
\linespread{1.08}
\pagestyle{plain}
\begin{document}
""" + (r"\fontsize{9}{11.2}\selectfont\setlength{\parskip}{3pt}" if COMPACT[0] else "") + r"""
{\sffamily\bfseries\LARGE """ + title + r"""\par}
\vspace{3mm}
{\small\color{gray}""" + NOTE[0].replace("DATE", today) + r"""\par}
\vspace{6mm}
"""
    return preamble + "\n".join(body) + "\n\\end{document}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=os.path.join(HERE, "MAIN_nature.md"))
    ap.add_argument("--figs", default=os.path.join(HERE, "figures", "out"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--name", default=None, help="output PDF basename (default: the Markdown file's basename)")
    ap.add_argument("--compact", action="store_true", help="9 pt type and 14 mm margins (one-pagers)")
    ap.add_argument("--inline-figures", action="store_true", help="place each figure after the paragraph that first cites it")
    ap.add_argument("--layout", default=None, choices=["v3", "v4"], help="figure-file mapping (default: v4 for MAIN_v4*.md, else v3)")
    a = ap.parse_args()
    COMPACT[0] = a.compact
    INLINE[0] = a.inline_figures
    layout = a.layout or ("v4" if os.path.basename(a.md).startswith("MAIN_v4") else "v3")
    if layout == "v4":
        FIG_FILES.clear(); FIG_FILES.update(FIG_FILES_V4)
        ED_FILES.clear(); ED_FILES.update(ED_FILES_V4)
    if os.path.basename(a.md).startswith("MAIN_v4") and not a.inline_figures:
        pass  # keep the review-copy note
    elif os.path.basename(a.md) != "MAIN_nature.md" and not os.path.basename(a.md).startswith("MAIN_v4"):
        NOTE[0] = "Assembled DATE from " + r"\texttt{" + os.path.basename(a.md).replace("_", r"\_") + "}."
    if a.inline_figures and (os.path.basename(a.md) == "MAIN_nature.md" or os.path.basename(a.md).startswith("MAIN_v4")):
        NOTE[0] = NOTE[0].replace("Review copy assembled", "Reading copy (figures placed in the text) assembled")
    os.makedirs(a.out, exist_ok=True)
    name = a.name or os.path.splitext(os.path.basename(a.md))[0]
    tex = convert(open(a.md, encoding="utf-8").read(), a.figs, a.out)
    tex_path = os.path.join(a.out, name + ".tex")
    open(tex_path, "w", encoding="utf-8").write(tex)
    cmd = ["tectonic", "--keep-logs", "--outdir", a.out, tex_path]
    print("running:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout[-3000:])
    sys.stderr.write(r.stderr[-6000:])
    if r.returncode != 0:
        sys.exit(r.returncode)
    print("wrote", os.path.join(a.out, name + ".pdf"))


if __name__ == "__main__":
    main()
