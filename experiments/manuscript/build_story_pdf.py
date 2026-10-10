"""Build a reading-style PDF of the plain-language account (STORY_FOR_NON_EXPERTS.md).

    .venv/bin/python manuscript/build_story_pdf.py      # manuscript/out/STORY_FOR_NON_EXPERTS.pdf

Reuses the escaping and inline markup of build_pdf.py, and adds what this document needs: bullet lists with continuation
lines, nested bullets and continuation paragraphs, numbered lists that keep their numbers, block quotes, a glossary and a
reference list with hanging indents, curly quotes and a table of contents. The Markdown source is
not altered.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from build_pdf import inline  # noqa: E402


def smart(s: str) -> str:
    """Curly quotes outside code spans."""
    out = []
    for p in re.split(r"(`[^`]*`)", s):
        if p.startswith("`"):
            out.append(p)
            continue
        p = re.sub(r'(^|[\s(\[—–-])"', "\\1\u201c", p)
        p = p.replace('"', "\u201d")
        p = re.sub(r"(^|[\s(\[—–-])'", "\\1\u2018", p)
        p = p.replace("'", "\u2019")
        out.append(p)
    return "".join(out)


def tx(s: str) -> str:
    return inline(smart(s))


def heading_plain(s: str) -> str:
    """Text for the table of contents: markup removed, escaped."""
    return tx(re.sub(r"[*`]", "", s))


def blocks(lines):
    """Split into blocks separated by blank lines."""
    cur = []
    for l in lines:
        if l.strip():
            cur.append(l.rstrip())
        elif cur:
            yield cur
            cur = []
    if cur:
        yield cur


def list_items(block, numbered=False):
    """[(level, number, text)] from a bullet or numbered block with indented continuation lines."""
    items = []
    pat = re.compile(r"^(\s*)(\d+)\. (.*)") if numbered else re.compile(r"^(\s*)- (.*)")
    for l in block:
        m = pat.match(l)
        if m:
            ind = len(m.group(1))
            if numbered:
                items.append([ind // 2, int(m.group(2)), m.group(3)])
            else:
                items.append([ind // 2, None, m.group(2)])
        else:
            items[-1][2] += " " + l.strip()
    return items


PAR = "\x01PAR\x01"


def render_bullets(items, plain=False):
    opt = r"[label={},leftmargin=1.4em,itemindent=-1.4em,itemsep=2pt,parsep=0pt,topsep=3pt]" if plain else \
        r"[leftmargin=1.3em,itemsep=3pt,parsep=0pt,topsep=3pt,label=\textbullet]"
    nested = r"[leftmargin=1.3em,itemsep=2pt,parsep=0pt,topsep=2pt,label=\textendash]"
    out = [r"\begin{itemize}" + opt]
    level = 0
    for lv, _, text in items:
        while lv > level:
            out.append(r"\begin{itemize}" + nested)
            level += 1
        while lv < level:
            out.append(r"\end{itemize}")
            level -= 1
        out.append(r"\item " + tx(text).replace(PAR, r"\par\smallskip "))
    while level > 0:
        out.append(r"\end{itemize}")
        level -= 1
    out.append(r"\end{itemize}")
    return out


def convert(md: str) -> str:
    lines = md.splitlines()
    title = ""
    body: list[str] = []
    section = ""
    pending: list = []        # items of the bullet list being read (it may continue across blank lines)
    subtitle_done = False

    def flush():
        nonlocal pending
        if pending:
            body.extend(render_bullets(pending, plain=section in ("Glossary", "Further reading")))
            pending = []

    for b in blocks(lines):
        first = b[0]
        if first.startswith("  ") and pending:
            # an indented paragraph continues the last item; any bullets after it continue the list
            k = next((i for i, l in enumerate(b) if l.startswith("- ")), len(b))
            pending[-1][2] += PAR + " ".join(l.strip() for l in b[:k])
            if k < len(b):
                pending += list_items(b[k:])
            continue
        if first.startswith("- "):
            pending += list_items(b)
            continue
        flush()
        if first.startswith("# ") and not title:
            title = tx(first[2:].strip())
            continue
        if first.strip() == "---":
            continue
        if first.startswith("## "):
            section = first[3:].strip()
            plain = heading_plain(section)
            if section == "Glossary" or section.startswith("Prologue"):
                body.append(r"\clearpage")
            elif section.startswith("Part"):
                body.append(r"\bigskip")
            body.append(r"\section*{" + tx(section) + "}")
            if section != "Further reading":
                body.append(r"\addcontentsline{toc}{section}{" + plain + "}")
            continue
        if first.startswith("### "):
            sub = first[4:].strip()
            body.append(r"\subsection*{" + tx(sub) + "}")
            if section.startswith("Part") or section.startswith("Prologue"):
                body.append(r"\addcontentsline{toc}{subsection}{" + heading_plain(sub) + "}")
            continue
        if first.startswith("> "):
            text = " ".join(l[2:].strip() for l in b)
            body += [r"\begin{keyquote}", tx(text), r"\end{keyquote}"]
            continue
        if re.match(r"^\d+\. ", first):
            items = list_items(b, numbered=True)
            body.append(r"\begin{enumerate}[start=%d,leftmargin=1.6em,itemsep=3pt,parsep=0pt,topsep=3pt]" % items[0][1])
            body += [r"\item " + tx(t) for _, _, t in items]
            body.append(r"\end{enumerate}")
            continue
        text = " ".join(l.strip() for l in b)
        if not subtitle_done and not section:
            body.append(r"{\itshape\color{subtle}" + tx(text.strip("*")) + r"\par}")
            body.append(r"\vspace{5mm}{\sffamily\bfseries\large Contents\par}\vspace{1mm}")
            body.append(r"{\small\setlength{\parskip}{0pt}\tableofcontents}")
            subtitle_done = True
            continue
        body.append(tx(text))
        body.append("")
    flush()
    pre = r"""\documentclass[11pt,letterpaper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{fontspec}
\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\setsansfont{texgyreheros}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\setmonofont{Inconsolatazi4}[Extension=.otf,UprightFont=*-Regular,BoldFont=*-Bold,Scale=0.95]
\usepackage{amssymb}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{titlesec}
\usepackage[hidelinks]{hyperref}
\definecolor{accent}{HTML}{2F4F6F}
\definecolor{subtle}{HTML}{4A4A4A}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt plus 1pt}
\linespread{1.12}
\titleformat{\section}{\sffamily\bfseries\Large\color{accent}}{}{0pt}{}
\titlespacing*{\section}{0pt}{4pt}{10pt}
\titleformat{\subsection}{\sffamily\bfseries\large}{}{0pt}{}
\titlespacing*{\subsection}{0pt}{14pt}{4pt}
\renewcommand{\contentsname}{}
\makeatletter\renewcommand\tableofcontents{\@starttoc{toc}}\makeatother
\newenvironment{keyquote}{\par\medskip\begin{list}{}{\leftmargin=1.6em\rightmargin=1.6em}\item[]\large\color{accent}}{\end{list}\medskip}
\pagestyle{plain}
\begin{document}
{\raggedright\sffamily\bfseries\Huge\color{accent} """ + title + r"""\par}
\vspace{4mm}
"""
    return pre + "\n".join(body) + "\n\\end{document}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=os.path.join(EXP, "STORY_FOR_NON_EXPERTS.md"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(a.md))[0]
    tex_path = os.path.join(a.out, name + ".tex")
    open(tex_path, "w", encoding="utf-8").write(convert(open(a.md, encoding="utf-8").read()))
    r = subprocess.run(["tectonic", "--keep-logs", "--outdir", a.out, tex_path], capture_output=True, text=True)
    sys.stdout.write(r.stdout[-2000:])
    sys.stderr.write(r.stderr[-4000:])
    if r.returncode:
        sys.exit(r.returncode)
    print("wrote", os.path.join(a.out, name + ".pdf"))


if __name__ == "__main__":
    main()
