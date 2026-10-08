"""Build the manuscript PDF from MAIN_nature.md and the built figures.

    python manuscript/build_pdf.py [--md manuscript/MAIN_nature.md] [--figs manuscript/figures/out] [--out manuscript/out]

A small converter for the Markdown subset the manuscript uses (#/##/### headings, paragraphs, numbered lists,
**bold**, *italic*, `code`, ^superscript^ citations) writes LaTeX, and tectonic compiles it. Figure legends in the
"Figure legends" section pull in the corresponding figure files above the legend text; Extended Data legends are set
as text. Nothing here alters the manuscript source.
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
FIG_FILES = {"1": ["fig1.pdf"], "2": ["fig2_ab.pdf", "fig2_cde.pdf"], "3": ["fig3.pdf"], "4": ["fig4.pdf"], "5": ["fig5.pdf"]}

SYM = {
    "≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$", "→": r"$\rightarrow$", "←": r"$\leftarrow$", "∞": r"$\infty$",
    "≈": r"$\approx$", "⇔": r"$\Leftrightarrow$", "⇒": r"$\Rightarrow$", "Σ": r"$\Sigma$", "−": r"$-$", "≠": r"$\neq$",
    "±": r"$\pm$", "·": r"\textperiodcentered{}", "…": r"\ldots{}", "°": r"\textdegree{}", "µ": r"$\mu$", "μ": r"$\mu$",
    "σ": r"$\sigma$", "ε": r"$\varepsilon$", "δ": r"$\delta$", "λ": r"$\lambda$", "α": r"$\alpha$", "β": r"$\beta$",
    "γ": r"$\gamma$", "θ": r"$\theta$", "π": r"$\pi$", "ρ": r"$\rho$", "τ": r"$\tau$", "φ": r"$\varphi$", "ω": r"$\omega$",
    "Δ": r"$\Delta$", "∈": r"$\in$", "∑": r"$\sum$", "√": r"$\surd$", "≡": r"$\equiv$", "∝": r"$\propto$", "½": r"\textonehalf{}", "∫": r"$\int$", "⌈": r"$\lceil$", "⌉": r"$\rceil$", "⌊": r"$\lfloor$",
    "⌋": r"$\rfloor$", "∂": r"$\partial$", "∇": r"$\nabla$", "≫": r"$\gg$", "≪": r"$\ll$", "∪": r"$\cup$", "∩": r"$\cap$", "⊂": r"$\subset$",
}


SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "−", "⁺": "+"}


def esc(s: str) -> str:
    """Escape LaTeX specials in plain text (symbols are mapped later, by symbols())."""
    s = s.replace("\\", "\u0000")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}")):
        s = s.replace(a, b)
    return s.replace("\u0000", r"\textbackslash{}")


def symbols(s: str) -> str:
    """Replace symbols the text fonts may lack by math-mode equivalents."""
    return "".join(SYM.get(ch, ch) for ch in s)


def inline(s: str) -> str:
    """Code spans, superscripts, bold, italic → LaTeX. Code spans are escaped verbatim; text is escaped then marked up."""
    out = []
    parts = re.split(r"(`[^`]*`)", s)
    for p in parts:
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


COMPACT = [False]
NOTE = [r"Draft assembled DATE from \texttt{manuscript/MAIN\_nature.md} and \texttt{manuscript/figures/out}; author list and affiliations to be added."]


def convert(md_text: str, figs_dir: str, out_dir: str) -> str:
    lines = md_text.splitlines()
    body = []
    title = "Manuscript"
    section = ""
    para: list[str] = []
    in_list = False

    def flush_para():
        nonlocal para
        if not para:
            return
        text = " ".join(x.strip() for x in para)
        para = []
        m = re.match(r"\*\*Fig\. (\d+) \| ", text)
        if section.startswith("Figure legends") and m:
            files = [f for f in FIG_FILES.get(m.group(1), []) if os.path.exists(os.path.join(figs_dir, f))]
            # fixed placement (no float): images, then the legend, then a page break
            body.append(r"\noindent\begin{minipage}{\textwidth}\centering")
            for f in files:
                shutil.copy(os.path.join(figs_dir, f), os.path.join(out_dir, f))
                h = r"0.36\textheight" if len(files) > 1 else r"0.70\textheight"
                body.append(r"\includegraphics[width=\textwidth,height=%s,keepaspectratio]{%s}\par\vspace{2mm}" % (h, f))
            if not files:
                body.append(r"\fbox{\parbox{0.9\textwidth}{\centering\vspace{20mm}\small figure file not built\vspace{20mm}}}\par\vspace{2mm}")
            body.append(r"\end{minipage}\par\vspace{3mm}")
            body.append(r"{\small " + inline(text) + r"\par}")
            body.append(r"\clearpage")
            body.append("")
            return
        if text.startswith("*") and text.endswith("*") and text.count("*") == 2:
            body.append(r"{\small\color{gray}" + inline(text) + "}")
            body.append("")
            return
        body.append(inline(text))
        body.append("")

    def close_list():
        nonlocal in_list
        if in_list:
            body.append(r"\end{enumerate}")
            body.append("")
            in_list = False

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("# ") and not title_set[0]:
            flush_para()
            title = inline(line[2:].strip())
            title_set[0] = True
            continue
        if line.startswith("## "):
            flush_para(); close_list()
            leaving_figs = section.startswith("Figure legends")
            section = line[3:].strip()
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
            # continuation line of a list item
            body[-1] += " " + inline(line.strip())
            continue
        para.append(line)
    flush_para(); close_list()

    today = dt.date.today().isoformat()
    margin = "14mm" if COMPACT[0] else "20mm"
    preamble = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=""" + margin + r"""]{geometry}
\usepackage{fontspec}
\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\setsansfont{texgyreheros}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\setmonofont{texgyrecursor}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,Scale=0.9]
\usepackage{graphicx}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{textcomp}
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


title_set = [False]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=os.path.join(HERE, "MAIN_nature.md"))
    ap.add_argument("--figs", default=os.path.join(HERE, "figures", "out"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--name", default=None, help="output PDF basename (default: the Markdown file's basename)")
    ap.add_argument("--compact", action="store_true", help="9 pt type and 14 mm margins (one-pagers)")
    a = ap.parse_args()
    COMPACT[0] = a.compact
    if os.path.basename(a.md) != "MAIN_nature.md":
        NOTE[0] = "Assembled DATE from " + r"\texttt{" + os.path.basename(a.md).replace("_", r"\_") + "}."
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
