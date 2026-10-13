"""Build a manuscript PDF from a Markdown file and the built figures.

    python manuscript/build_pdf.py --v8

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
FIG_FILES_V4 = {"1": ["fig1.pdf"], "2": ["fig2.pdf"], "3": ["fig3.pdf"], "4": ["fig4v5.pdf"], "5": ["fig5v4.pdf"], "6": ["fig6v5.pdf"]}
ED_FILES_V4 = {
    # revision 3: ten Extended Data items, numbered by first citation; three former items moved to the Supplementary
    "1": [os.path.join(EXP, "manuscript", "figures", "out", "ed12.pdf")],
    "2": [os.path.join(EXP, "manuscript", "figures", "out", "fig4.pdf")],
    "3": [os.path.join(EXP, "manuscript", "figures", "out", "ed_confine.pdf")],
    "4": [os.path.join(EXP, "manuscript", "figures", "out", "ed_closure.pdf")],
    "5": [os.path.join(EXP, "manuscript", "figures", "out", "ed_invasions2.pdf")],
    "6": [os.path.join(EXP, "manuscript", "figures", "out", "ed_switch.pdf")],
    "7": [os.path.join(EXP, "manuscript", "figures", "out", "ed_assembly.pdf")],
    "8": [os.path.join(EXP, "manuscript", "figures", "out", "ed_dial.pdf")],
    "9": [os.path.join(EXP, "manuscript", "figures", "out", "ed13.pdf")],
}
SUP_FILES = {
    "1": [os.path.join(EXP, "manuscript", "figures", "out", "ed14.pdf")],
    "2": [os.path.join(EXP, "manuscript", "figures", "out", "ed_bffinflow.pdf")],
    "3": [os.path.join(EXP, "manuscript", "figures", "out", "ed_census.pdf")],
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
    "∂": r"$\partial$", "′": r"$'$", "₂": r"$_2$", "₁": r"$_1$", "ₙ": r"$_n$", "∇": r"$\nabla$", "≫": r"$\gg$", "≪": r"$\ll$", "∪": r"$\cup$", "∩": r"$\cap$", "⊂": r"$\subset$", "𝒪": r"$\mathcal{O}$",
}
SUP = {"ᵏ": "k", "ᵐ": "m", "ⁿ": "n", "ˣ": "x", "ᵈ": "d", "⁽": "(", "⁾": ")", "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "−", "⁺": "+"}


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
        t = re.sub(r"https?://[A-Za-z0-9./:\-]+[A-Za-z0-9/]", lambda m: r"\url{" + m.group(0) + "}", t)   # breakable URLs
        t = re.sub(r"(?<=[A-Za-z′Σ])\\_([A-Za-z0-9]{1,8})(?![A-Za-z0-9])", r"\\textsubscript{\1}", t)   # s_j, t_rep → subscripts
        t = re.sub("[" + "".join(SUP) + "]+", lambda m: r"\textsuperscript{" + "".join(SUP[c] for c in m.group(0)) + "}", t)
        t = re.sub(r"\^([^^\s$\\]{1,12})\^", r"\\textsuperscript{\1}", t)
        t = re.sub(r"\^(-|\u2212)?([A-Za-z0-9]+)", r"\\textsuperscript{\1\2}", t)
        t = t.replace("^", r"\textasciicircum{}")
        t = re.sub(r'(^|[\s(\[{\u2013\u2014/])"', r"\1``", t)       # straight quotes: opening after space or bracket
        t = t.replace('"', "''")
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
        body += [LEGEND_OPEN + legend_tex + r"\par}", r"\end{figure}", ""]
    else:
        body += [r"\end{minipage}\par\vspace{3mm}", LEGEND_OPEN + legend_tex + r"\par}", r"\clearpage", ""]
    return body


# legends are set flush left and justified, not centred like the images above them
LEGEND_OPEN = r"{\small\leftskip=0pt\rightskip=0pt\parfillskip=0pt plus 1fil\relax "
COMPACT = [False]
INLINE = [False]
NOTE = [r"Review copy assembled DATE; author list and affiliations to be added."]


def convert(md_text: str, figs_dir: str, out_dir: str) -> str:
    lines = md_text.splitlines()
    legends: dict[str, str] = {}
    for l in lines:
        m = re.match(r"\*\*Fig\. (\d+) \| ", l)
        if m:
            legends[m.group(1)] = re.sub(r"\s*Sources?: .*$", "", l.strip())   # provenance stays in the Markdown
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
        if text.startswith(("**Extended Data", "**Supplementary")) or re.match(r"\*\*Fig\. \d+ \| ", text):
            text = re.sub(r"\s*Sources?: .*$", "", text)     # provenance stays in the Markdown, not in the review copy
        if text.startswith(("**Extended Data Table", "**Supplementary Table")):
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
        m = re.match(r"\*\*Supplementary Fig\. (\d+) \| ", text)
        if m and m.group(1) in SUP_FILES and any(os.path.exists(f) for f in SUP_FILES[m.group(1)]):
            body.extend(figure_block(SUP_FILES[m.group(1)], inline(text), out_dir, floating=True))
            return
        if text.startswith("*") and text.endswith("*") and text.count("*") == 2:
            body.append(r"{\small\color{gray}" + inline(text) + "}")
            body.append("")
            return
        body.append(inline(text))
        body.append("")
        if INLINE[0] and not section.startswith(("Figure legends", "Extended Data", "References", "Supplementary")):
            for n in re.findall(r"(?<!Extended Data )(?<!Supplementary )Fig\.\s*(\d)", text):
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
        long_table = len(data) > 30                         # a long table breaks across pages instead of overflowing one
        env = "longtable" if long_table else "tabular"
        if pending_table_legend:
            out.append(r"\par\noindent" + (r"\begin{minipage}{\linewidth}" if not long_table else ""))
            out.append(pending_table_legend.pop(0))
            out.append(r"\par\vspace{2pt}")
        out += [r"{\footnotesize\setlength{\tabcolsep}{4pt}\begin{" + env + "}{" + "".join(aligns) + "}", r"\toprule"]
        for h in head:
            out.append(" & ".join(r"\textbf{" + inline(c) + "}" for c in h) + r" \\")
        if head:
            out.append(r"\midrule")
        for r in data:
            out.append(" & ".join(inline(c) for c in r) + r" \\")
        out.append(r"\bottomrule")
        out.append(r"\end{" + env + r"}\par}")
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
            if section.startswith(("Figure legends", "References", "Supplementary")) or leaving_figs:
                body.append(r"\clearpage")                   # flushes pending floats, so no list or heading is split by them
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
\usepackage{flafter}
\usepackage{longtable}
\usepackage{xurl}
\urlstyle{same}
\emergencystretch=1.5em
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
{\raggedright\sffamily\bfseries\LARGE """ + title + r"""\par}
\vspace{3mm}
{\small\color{gray}""" + NOTE[0].replace("DATE", today) + r"""\par}
\vspace{6mm}
"""
    return preamble + "\n".join(body) + "\n\\end{document}\n"


# v8 uses a separate, strict converter; archived v3/v4 conversion stays available.
from pathlib import Path
import collections
import hashlib
import json
import tempfile
import unicodedata


SOURCE_PATH = re.compile(
    r"(?<![\w:/])(?:results/|algocell_exp/|manuscript/|/Users/)[^\s;,`\]]*"
    r"|(?<![\w/])[-\w]+\.(?:md|py|csv|json|wgsl|gz|pdf)(?!\w)"
)


def bracket_audit(text: str) -> list[dict]:
    """Classify every balanced bracket, including nested mathematical/code brackets.

    A local file/directory token makes a bracket provenance unless it is a
    Markdown link label, code/math, or an explicit author placeholder. Qualifiers
    (registered, theorem, etc.) do not change a path bracket's classification.
    Unknown non-path brackets are retained and reported for manual inspection.
    """
    protected = [(m.start(), m.end(), 'code') for m in re.finditer(r'`[^`]*`', text)]
    protected += [(m.start(), m.end(), 'math') for m in re.finditer(r'\\\[.*?\\\]', text, re.S)]
    stack, pairs = [], []
    for i, c in enumerate(text):
        if c == '[':
            stack.append(i)
        elif c == ']' and stack:
            pairs.append((stack.pop(), i + 1))
    if stack:
        raise ValueError('Unbalanced opening brackets in manuscript')
    records = []
    for start, end in sorted(pairs):
        content = text[start + 1:end - 1]
        region = next((kind for a, b, kind in protected if a <= start < b), None)
        if region:
            kind = region
        elif text[end:end + 1] == '(':
            kind = 'link'
        elif content.startswith('AUTHOR TO '):
            kind = 'placeholder'
        elif re.fullmatch(r'\d+(?:\s*[,–-]\s*\d+)*', content):
            kind = 'citation'
        elif SOURCE_PATH.search(content):
            kind = 'source-path'
        elif re.fullmatch(r'[\w\s+−*/=(),.]+', content) and re.search(r'[+−=*/]', content):
            kind = 'math'
        else:
            kind = 'ambiguous-retained'
        records.append(dict(line=text.count('\n', 0, start) + 1, start=start, end=end,
                            bracket=text[start:end], classification=kind))
    return records


def clean_v8(text: str, audit: list[dict]) -> str:
    for record in reversed(audit):
        if record['classification'] == 'source-path':
            text = text[:record['start']] + text[record['end']:]
    text = re.sub(r'^\*Review copy\..*?\*\s*\n', '', text, count=1, flags=re.M)
    text = re.sub(r' +([.,;:)])', r'\1', text)
    return re.sub(r' {2,}', ' ', text)


def v8_plain(text: str) -> str:
    text = esc(text)
    text = re.sub('[' + ''.join(SUP) + ']+',
                  lambda m: r'\textsuperscript{' + symbols(''.join(SUP[c] for c in m.group())) + '}', text)
    text = re.sub(r'\^\(([^()]*)\)', r'\\textsuperscript{\1}', text)
    text = re.sub(r'\^([−-]?[\w]+|∞)', r'\\textsuperscript{\1}', text)
    text = text.replace('^', r'\textasciicircum{}')
    text = re.sub(r'\\_([A-Za-z0-9]+)', r'\\textsubscript{\1}', text)
    return symbols(text).replace('₀', r'$_0$')


def v8_code(text: str) -> str:
    # Break paths/identifiers inside narrow table columns without altering them.
    parts = []
    run = 0
    for char in text:
        parts.append(symbols(esc(char)).replace('^', r'\textasciicircum{}'))
        run += 1
        if char in '/_.-+,=' or (run >= 10 and not char.isspace()):
            parts.append(r'\allowbreak{}')
            run = 0
        if char.isspace():
            run = 0
    return r'\texttt{' + ''.join(parts) + '}'


def v8_inline(text: str, *, provenance: bool = True) -> str:
    token = re.compile(r'`[^`]*`|\[[^\[\]]*\]\(https?://[^\s]*\)|\*\*[^*]+\*\*|(?<!\*)\*[^*]+\*(?!\*)|\[[^\[\]]*\]')
    pieces, pos = [], 0
    for m in token.finditer(text):
        pieces.append(v8_plain(text[pos:m.start()]))
        value = m.group()
        if value.startswith('`'):
            rendered = v8_code(value[1:-1])
        elif value.startswith('**'):
            rendered = r'\textbf{' + v8_inline(value[2:-2], provenance=provenance) + '}'
        elif value.startswith('*'):
            rendered = r'\emph{' + v8_inline(value[1:-1], provenance=provenance) + '}'
        elif '](' in value:
            label, url = value[1:-1].split('](', 1)
            rendered = r'\url{' + url + '}' if label == url else v8_plain(label) + r' (\url{' + url + '})'
        elif provenance and bracket_audit(value)[0]['classification'] == 'source-path':
            rendered = r'{\footnotesize\color{gray}' + v8_inline(value, provenance=False) + '}'
        else:
            # Paths in source brackets need discretionary breaks too.
            content, fragments, cursor = value[1:-1], [], 0
            for part in re.finditer(r'`[^`]*`|[^\s;,]+', content):
                fragments.append(v8_plain(content[cursor:part.start()]))
                word = part.group()
                if word.startswith('`') and word.endswith('`'):
                    fragments.append(v8_code(word[1:-1]))
                else:
                    fragments.append(v8_code(word) if SOURCE_PATH.fullmatch(word) or '_' in word else v8_plain(word))
                cursor = part.end()
            fragments.append(v8_plain(content[cursor:]))
            rendered = '[' + ''.join(fragments) + ']'
        pieces.append(rendered)
        pos = m.end()
    pieces.append(v8_plain(text[pos:]))
    return ''.join(pieces)


def v8_math(text: str) -> str:
    for char, tex in SYM.items():
        text = text.replace(char, tex.strip('$'))
    return text.replace('₀', '_0')


def v8_table(block: str) -> list[str]:
    rows = [[c.strip() for c in line.strip().strip('|').split('|')] for line in block.splitlines()]
    if not all(len(row) == len(rows[0]) for row in rows):
        raise ValueError('Ragged Markdown table')
    n = len(rows[0])
    if n == 4:
        weights = [.17, .26, .29, .28]
    elif rows[0][0] == 'Table':
        weights = [.08, .33, .59]
    elif rows[0][0] == 'Extended Data item':
        weights = [.25, .25, .50]
    else:
        weights = [1 / n] * n
    colspec = ''.join(r'>{\raggedright\arraybackslash}p{\dimexpr ' + str(w)
                      + r'\textwidth-' + str(2 * 4) + 'pt\\relax}' for w in weights)
    head = ' & '.join(r'\textbf{' + v8_inline(c) + '}' for c in rows[0]) + r' \\'
    result = [r'\par\begingroup\footnotesize\setlength{\tabcolsep}{4pt}',
              r'\renewcommand{\arraystretch}{1.18}',
              r'\begin{longtable}{' + colspec + '}', r'\toprule', head, r'\midrule\endfirsthead',
              r'\toprule', head, r'\midrule\endhead', r'\bottomrule\endfoot', r'\bottomrule\endlastfoot']
    result.extend(' & '.join(v8_inline(c) for c in row) + r' \\' for row in rows[2:])
    result.extend([r'\end{longtable}\endgroup', ''])
    return result


def v8_figure(path: Path, legend: str, out_dir: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    dst = out_dir / path.name
    if path.resolve() != dst.resolve():
        shutil.copyfile(path, dst)
    # Fixed content blocks enforce citation order. Captions can continue onto
    # another page. Tall Extended Data pages are capped at 86% of the text height
    # in this reading PDF; the submitted figure files keep their full size.
    return [r'\par\addvspace{10pt}',
            r'\noindent\includegraphics[width=\textwidth,height=0.86\textheight,keepaspectratio]{' + path.name + r'}\par\nobreak',
            r'\vspace{4pt}{\small ' + v8_inline(legend) + r'\par}', r'\par\addvspace{10pt}', '']


def convert_v8(text: str, figs_dir: Path, out_dir: Path, *, author: str,
               supplementary: bool = False, ed_text: str = '') -> tuple[str, list[str]]:
    blocks = re.split(r'\n\s*\n', text.strip())
    title = blocks.pop(0)[2:].strip()
    author_tex = v8_inline(author)
    affiliation = ''
    if not supplementary:
        if blocks.pop(0).strip() != author:
            raise ValueError('Unexpected author block')
        affiliation = v8_inline(blocks.pop(0).strip())
    legends = {}
    for i, block in enumerate(blocks):
        m = re.fullmatch(r'\*\*Figure (\d+)\..*\*\*', block.strip())
        if m:
            legends[m[1]] = block.strip() + '\n' + blocks[i + 1].strip()
    ed_legends = {}
    if supplementary:
        for m in re.finditer(r'^## (Extended Data (\d+)\.[^\n]+)\n(.*?)(?=^## |\Z)', ed_text, re.M | re.S):
            caption = re.search(r'\*\*Legend(?: draft)? \(\d+ words\)\.\*\*\s*(.*?)(?=\n\n)', m[3], re.S)
            if not caption:
                raise ValueError('Missing ED legend: ' + m[2])
            ed_legends[m[2]] = '**' + m[1] + '** ' + caption[1].strip()
        if set(ed_legends) != {str(n) for n in range(1, 10)}:
            raise ValueError('Expected nine Extended Data legends')
    body, placed, section = [], [], ''
    references_open = False
    in_legends = False
    skip_caption = False
    for block in blocks:
        block = block.strip()
        heading = re.fullmatch(r'(#{2,6}) (.+)', block)
        if heading:
            if references_open:
                body.append(r'\end{enumerate}')
                references_open = False
            section = heading[2]
            in_legends = section == 'Figure legends'
            if section in ('Methods', 'References', 'Supplementary Tables and generated-source index', 'Extended Data display map'):
                body.append(r'\clearpage')
            command = 'section' if len(heading[1]) == 2 else 'subsection'
            body.append('\\' + command + '*{' + v8_inline(section) + '}')
            continue
        if in_legends and re.fullmatch(r'\*\*Figure \d+\..*\*\*', block):
            skip_caption = True
            continue
        if skip_caption:
            skip_caption = False
            continue
        if block.startswith('|'):
            body.extend(v8_table(block))
            continue
        if block.startswith(r'\[') and block.endswith(r'\]'):
            body.append(v8_math(block))
            continue
        numbered = re.match(r'^(\d+)\. (.*)', block, re.S)
        if numbered:
            if not references_open:
                body.append(r'\begin{enumerate}[label=\arabic*.,widest=15,leftmargin=*,itemsep=4pt,parsep=0pt]')
                references_open = True
            body.append(r'\item ' + v8_inline(numbered[2].replace('\n', ' ')))
            continue
        if block.startswith('- '):
            body.append(r'\begin{itemize}[leftmargin=*]')
            body.extend(r'\item ' + v8_inline(line[2:]) for line in block.splitlines())
            body.append(r'\end{itemize}')
            continue
        body.extend([v8_inline(block.replace('\n', ' ')), ''])
        if not supplementary and not in_legends:
            for number in re.findall(r'(?<!Extended Data )Fig\.\s*(\d+)', block):
                if number in legends and number not in placed:
                    body.extend(v8_figure(figs_dir / f'fig{number}.pdf', legends[number], out_dir))
                    placed.append(number)
    if references_open:
        body.append(r'\end{enumerate}')
    if supplementary:
        body.append(r'\clearpage')
        for n, legend in ed_legends.items():
            body.extend(v8_figure(figs_dir / f'ed8_{n}.pdf', legend, out_dir))
            body.append(r'\clearpage')
            placed.append(n)
    elif set(placed) != set(legends) or len(placed) != 5:
        raise ValueError('Expected exactly five figures at their first citations')
    preamble = r'''\documentclass[11pt,a4paper]{article}
\usepackage[left=29mm,right=29mm,top=18mm,bottom=18mm]{geometry}
\usepackage{fontspec}
\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\setmonofont{texgyrecursor}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,Scale=0.9]
\usepackage{graphicx,longtable,xurl,amssymb,enumitem,xcolor,textcomp,booktabs,array}
\urlstyle{same}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt plus 1pt}
\setlength{\emergencystretch}{4em}
\tolerance=2500
\raggedright
\linespread{1.15}
\clubpenalty=10000
\widowpenalty=10000
\displaywidowpenalty=10000
\pagestyle{plain}
\begin{document}
{\raggedright\bfseries\LARGE ''' + v8_inline(title) + r'''\par}
\vspace{8pt}
''' + author_tex + r'''\par
''' + (r'{\small ' + affiliation + r'\par}' if affiliation else '') + r'''
\vspace{12pt}
'''
    return preamble + '\n'.join(body) + '\n\\end{document}\n', placed


def cached_v8_bundle(destination: Path) -> Path:
    """Expose only existing Tectonic cache files as a local directory bundle.

    A local bundle avoids Tectonic's macOS HTTP-client initialization crash,
    even with --only-cached. No resources are fetched and the cache is read-only.
    """
    root = Path.home() / 'Library/Caches/Tectonic/bundles'
    hash_file = root / 'hashes/https,58,,47,,47,relay.fullyjustified.net,47,default_bundle_v33.tar'
    if not hash_file.is_file():
        raise FileNotFoundError('No cached Tectonic v33 bundle; network downloads are disabled')
    digest = hash_file.read_text().strip()
    if not re.fullmatch(r'[0-9a-f]{64}', digest):
        raise ValueError('Invalid cached Tectonic bundle digest')
    cache = root / 'data' / digest
    destination.mkdir()
    for resource in cache.iterdir():
        if resource.is_file():
            (destination / resource.name).symlink_to(resource)
    (destination / 'SHA256SUM').write_text(digest + '\n')
    return destination


def build_v8(out_dir: Path, engine: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    main = (Path(HERE) / 'MAIN_v8.md').read_text(encoding='utf-8')
    si = (Path(HERE) / 'SI_v8.md').read_text(encoding='utf-8')
    figs = Path(HERE) / 'figures/out/v8'
    ed = (figs / 'ED_V8.md').read_text(encoding='utf-8')
    author = re.split(r'\n\s*\n', main)[1].strip()
    audit = bracket_audit(main)
    audit_path = out_dir / 'MAIN_v8_bracket_audit.json'
    audit_path.write_text(json.dumps(dict(rule=bracket_audit.__doc__, brackets=audit,
        counts=dict(collections.Counter(r['classification'] for r in audit)),
        ambiguous=[r for r in audit if r['classification'] == 'ambiguous-retained']), indent=2, ensure_ascii=False) + '\n')
    print('Bracket audit:', collections.Counter(r['classification'] for r in audit))
    for r in audit:
        if r['classification'] == 'ambiguous-retained':
            print('AMBIGUOUS, retained:', r['line'], r['bracket'])
    bundle_temp = tempfile.TemporaryDirectory(prefix='algocell-tectonic-v8-')
    bundle = cached_v8_bundle(Path(bundle_temp.name) / 'bundle')
    jobs = [('MAIN_v8_review', main, False), ('MAIN_v8_clean', clean_v8(main, audit), False), ('SI_v8', si, True)]
    for name, source, supplementary in jobs:
        tex, placed = convert_v8(source, figs, out_dir, author=author,
                                 supplementary=supplementary, ed_text=ed)
        tex_path = out_dir / (name + '.tex')
        tex_path.write_text(tex, encoding='utf-8')
        cmd = [engine, '--only-cached', '--bundle', str(bundle), '--keep-logs', '--outdir', str(out_dir), str(tex_path)]
        print('running:', ' '.join(cmd), flush=True)
        run = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(run.stdout)
        sys.stderr.write(run.stderr)
        if run.returncode:
            raise SystemExit(run.returncode)
        log = (out_dir / (name + '.log')).read_text(errors='replace')
        issues = re.findall(r'(?:Overfull \\[hv]box|Missing character:)[^\n]*', log)
        if issues:
            raise RuntimeError(name + ': layout/glyph errors: ' + '\n'.join(issues))
        print('wrote', out_dir / (name + '.pdf'), 'figures:', ', '.join(placed))
    bundle_temp.cleanup()
    verify_v8(out_dir, main, si, ed, figs)


def verify_v8(out_dir: Path, main: str, si: str, ed: str, figs: Path) -> None:
    """Extract all text, check headings/embedding, and render review previews."""
    import pymupdf as fitz  # already installed in the experiments venv

    def normalize(value: str) -> str:
        value = re.sub(r'\*\*?|`', '', value)
        return ' '.join(unicodedata.normalize('NFKC', value).split())

    report = {}
    for name, source, count in [('MAIN_v8_review', main, 5), ('MAIN_v8_clean', main, 5), ('SI_v8', si, 9)]:
        pdf = out_dir / (name + '.pdf')
        doc = fitz.open(pdf)
        text_path = out_dir / (name + '.txt')
        if shutil.which('pdftotext'):
            subprocess.run(['pdftotext', '-layout', str(pdf), str(text_path)], check=True)
            extracted = text_path.read_text(encoding='utf-8')
        else:
            extracted = '\n'.join(page.get_text() for page in doc)
            text_path.write_text(extracted, encoding='utf-8')
        norm = normalize(extracted)
        if name == 'MAIN_v8_review':
            # Footers can interrupt a multi-page provenance bracket.
            without_footers = re.sub(r'(?m)^\s*\d+\s*$', '', extracted)
            compact = lambda value: re.sub(r'\s+', '', unicodedata.normalize('NFKC', value))
            missing_sources = [r for r in bracket_audit(main) if r['classification'] == 'source-path'
                               and compact(r['bracket']) not in compact(without_footers)]
            if missing_sources:
                raise RuntimeError('Review PDF lost source brackets: ' + repr(missing_sources))
        headings = re.findall(r'^#{1,6} (.+)$', source, re.M)
        if name == 'SI_v8':
            headings += re.findall(r'^## (.+)$', ed, re.M)
        missing = [heading for heading in headings if normalize(heading) not in norm]
        if missing:
            raise RuntimeError(name + ': missing headings: ' + repr(missing))
        embedded = [(p + 1, x) for p in range(len(doc)) for x in doc.get_page_xobjects(p) if x[2] == 0]
        if len(embedded) != count:
            raise RuntimeError(name + ': expected ' + str(count) + ' PDF Forms, found ' + str(len(embedded)))
        expected_files = [figs / (f'ed8_{n}.pdf' if name == 'SI_v8' else f'fig{n}.pdf') for n in range(1, count + 1)]
        for (_, form), expected in zip(embedded, expected_files):
            source_pdf = fitz.open(expected)
            bounds = fitz.Rect(form[3])
            if abs(bounds.width - source_pdf[0].rect.width) > .1 or abs(bounds.height - source_pdf[0].rect.height) > .1:
                raise RuntimeError(name + ': embedded figure size/order mismatch: ' + expected.name)
            source_pdf.close()
            if hashlib.sha256(expected.read_bytes()).digest() != hashlib.sha256((out_dir / expected.name).read_bytes()).digest():
                raise RuntimeError('Copied figure differs from source: ' + expected.name)
        leaks = [marker for marker in ('**', '`') if marker in extracted]
        if name == 'MAIN_v8_clean':
            leaks += [marker for marker in ('[results/', '[registered:', '[exploratory:', 'Review copy.') if marker in extracted]
            preserved = [r for r in bracket_audit(main) if r['classification'] in ('citation', 'placeholder', 'math')]
            missing_preserved = [r['bracket'] for r in preserved if normalize(r['bracket']) not in norm]
            if missing_preserved:
                raise RuntimeError('Clean PDF lost retained brackets: ' + repr(missing_preserved))
        if leaks:
            raise RuntimeError(name + ': leaked markup/provenance: ' + repr(leaks))
        # Check complete legend text as well as its heading and image.
        compact_text = re.sub(r'\s+', '', normalize(re.sub(r'(?m)^\s*\d+\s*$', '', extracted)))
        if name == 'SI_v8':
            captions = []
            for match in re.finditer(r'^## (Extended Data \d+\.[^\n]+)\n(.*?)(?=^## |\Z)', ed, re.M | re.S):
                draft = re.search(r'\*\*Legend(?: draft)? \(\d+ words\)\.\*\*\s*(.*?)(?=\n\n)', match[2], re.S)
                captions.append(match[1] + ' ' + draft[1])
        else:
            captions = [match[0] for match in re.finditer(r'^\*\*Figure \d+\..*?\*\*\n\n[^\n]+', main, re.M)]
        if len(captions) != count or any(re.sub(r'\s+', '', normalize(c)) not in compact_text for c in captions):
            raise RuntimeError(name + ': incomplete figure legends')
        # TeX cannot diagnose text already inside an embedded PDF figure.
        margin = 29 * 72 / 25.4
        for page in doc:
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines', []):
                    for span in line['spans']:
                        bounds = fitz.Rect(span['bbox'])
                        if bounds.x0 < margin - 1 or bounds.x1 > page.rect.width - margin + 1:
                            raise RuntimeError(name + ': text outside horizontal margins: ' + span['text'])
        figure_pages = [p for p, _ in embedded]
        previews = []
        for label, page_number in [('page1', 1), ('figure', figure_pages[0])]:
            path = out_dir / f'preview_{name}_{label}.png'
            doc[page_number - 1].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(path)
            previews.append(str(path))
        report[name] = dict(pages=len(doc), headings_checked=len(headings), missing_headings=missing,
                            embedded_figures=[dict(file=p.name, page=item[0]) for p, item in zip(expected_files, embedded)],
                            raw_markup_leaks=leaks, previews=previews,
                            source_brackets_retained=sum(r['classification'] == 'source-path' for r in bracket_audit(main))
                            if name == 'MAIN_v8_review' else None,
                            complete_legends_checked=len(captions),
                            horizontal_margin_violations=0, overfull_boxes=0, missing_glyphs=0)
        print(name + ':', len(doc), 'pages;', len(headings), 'headings;', count, 'embedded displays; verified')
        doc.close()
    (out_dir / 'v8_verification.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=os.path.join(HERE, "MAIN_nature.md"))
    ap.add_argument("--figs", default=os.path.join(HERE, "figures", "out"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--name", default=None, help="output PDF basename (default: the Markdown file's basename)")
    ap.add_argument("--compact", action="store_true", help="9 pt type and 14 mm margins (one-pagers)")
    ap.add_argument("--inline-figures", action="store_true", help="place each figure after the paragraph that first cites it")
    ap.add_argument("--layout", default=None, choices=["v3", "v4"], help="figure-file mapping (default v4; v3 for the archived manuscript/archive/MAIN_nature_v3.md)")
    ap.add_argument("--v8", action="store_true", help="build and verify v8 MAIN review/clean and SI using only cached TeX resources")
    ap.add_argument("--tectonic", default="/opt/homebrew/bin/tectonic", help="TeX engine for the v8 build")
    a = ap.parse_args()
    if a.v8:
        build_v8(Path(a.out).resolve(), a.tectonic)
        return
    COMPACT[0] = a.compact
    INLINE[0] = a.inline_figures
    layout = a.layout or ("v3" if "_v3" in os.path.basename(a.md) else "v4")
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
