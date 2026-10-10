"""Composite figures for the Nature manuscript. Every number plotted comes from a generated table under results/ or a
recorded run file; the conceptual panels come from concept.py (code-drawn stand-ins until the designer's set arrives).

    python manuscript/figures/make_figures.py [--out manuscript/figures/out] [--only fig2,fig3]

Main figures (180 mm double column, depth <= 170 mm, Nature profile from figstyle):
  fig1  The first replicator and its closure     a design panel (refs/designer_fig1_round3.png if present) | b emergence by L (KM)
  fig2  One world, watched                        a seven lattice frames (seed 2002) | b its time course | c a byte-resolution window
  fig3  The first replicator is open, the successor closed
                                                 a copies b self-damage c information inflow (first -> final per world, slope charts)
                                                 d heritable fraction vs step | e convergence counts | f control flow of the closers (concept)
  fig4  What the instruction set must provide    a atlas forest | b size axis + unit fitness | c dead-zone switch | d L = 9 reversal
  fig5  One instruction decides how life begins in BFF   a heritable fraction vs epoch | b first vs final openness | c the all-P wave | d classification (concept)
  fig6  Closure requires a cycle                 the theorem as a diagram (concept, single panel)
Extended Data drawn here: ed12 (assembly measure against the culture test), ed13 (lethal tar in the first machine).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, EXP)
import figstyle as fs  # noqa: E402
sys.path.insert(0, HERE)
import concept as cp  # noqa: E402
import figcheck  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

R = os.path.join(EXP, "results")
L_COL = dict(fs.LEN_COLOR)     # one length palette for every figure (figstyle.LEN_COLOR)
INK, GREY, RULE, TEAL, RED = cp.INK, "#6B7280", "#B4BAC1", cp.TEAL, cp.RED
LOOP_MS = 7          # marker area for the slope charts
H_LABEL = r"$H(Y \mid X = x)$"   # mathtext: the bar renders as a relation bar, not as an I or l
H_AXIS = "information from the partner,\n" + H_LABEL + " (bits)"


def save(fig, path_no_ext):
    """Run the layout checks, print the report, then save. A figure with problems is still written so it can be inspected."""
    figcheck.print_report(figcheck.check(fig), os.path.basename(path_no_ext))
    fs.save(fig, path_no_ext)


def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def label(ax, letter, dx=0.052, dy=0.006):
    """Panel letter at a fixed offset from the panel's box, so letters align across rows and panel types."""
    ax.apply_aspect()
    pos = ax.get_position()
    ax.figure.text(pos.x0 - dx, pos.y1 + dy, letter, fontsize=8, fontweight="bold", va="bottom", ha="left", gid="panel-label")


def placeholder(ax, text):
    ax.set_axis_off()
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=6, color="#777777", transform=ax.transAxes, wrap=True)
    for sp in ax.spines.values():
        sp.set_visible(False)


def km_curve(times, horizon):
    """Kaplan–Meier fraction of worlds WITHOUT the event vs step (events at `times`; NaN or < 0 = censored at horizon)."""
    t = np.array([x if (x is not None and x == x and x >= 0) else np.inf for x in times], float)
    xs = [0.0]
    ys = [1.0]
    for v in sorted(set(t[np.isfinite(t)])):
        d = int((t == v).sum())
        at_risk = int((t >= v).sum())
        s = ys[-1] * (1 - d / at_risk)
        xs += [v, v]
        ys += [ys[-1], s]
    xs.append(horizon)
    ys.append(ys[-1])
    return np.array(xs), np.array(ys)


def loop_flags(d, which):
    return (d[f"{which}_has_cf"].astype(bool) | d[f"{which}_has_block"].astype(bool)).values


# ----------------------------------------------------------------------------------------------------------------- fig 1
def km_panel(ax, ncol=3):
    """b: Kaplan–Meier emergence by tape length (Stage E, none@nominal, 128 steps, k = 4)."""
    A = pd.read_csv(os.path.join(R, "stageE", "assays.csv"))
    A = A[(A["label"] == "none@nominal") & (A["steps"] == 128) & (A["k"] == 4)]
    if "replicate" in A:
        A = A[A["replicate"].isna()]
    Ls = (8, 9, 16, 36, 64, 100)
    for L in Ls:
        g = A[A["tape_len"] == L]
        if g.empty:
            continue
        x, y = km_curve(g["t_rep"].tolist(), 300000)
        ax.step(np.maximum(x, 40), 1 - y, where="post", color=L_COL.get(L, "#444444"), lw=0.9, label=f"L = {L}")
    ax.set_xscale("log")
    ax.set_xlim(40, 3.5e5)
    fs.log10_ticks(ax)
    ax.set_ylim(-0.03, 1.03)
    fs.tidy(ax, "step", "fraction of worlds with a heritable replicator")
    ax.set_gid("allow-clip")
    # the key reads by row (8, 9, 16 / 36, 64, 100): matplotlib fills columns, so hand it the entries column by column
    h, l = ax.get_legend_handles_labels()
    nrow = int(np.ceil(len(h) / ncol))
    order = [r * ncol + c for c in range(ncol) for r in range(nrow) if r * ncol + c < len(h)]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, fontsize=5,
              frameon=False, columnspacing=1.0, handlelength=1.4)


COMPOSE_TEX = r"""\documentclass{article}
\usepackage[paperwidth=180mm,paperheight=%(h)smm,margin=0mm,top=2mm,headheight=0pt,headsep=0pt]{geometry}
\usepackage{fontspec}
\usepackage{xcolor}
\usepackage{tikz}
\IfFontExistsTF{Helvetica}{\setsansfont{Helvetica}}{\setsansfont{texgyreheros}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold]}
\newfontfamily\monofont{IBMPlexMono-Regular.otf}
\newcommand{\mono}[1]{{\monofont #1}}
\definecolor{ink}{HTML}{1C2733}\definecolor{mute}{HTML}{6B7280}\definecolor{verm}{HTML}{B24431}
\pagestyle{empty}\setlength{\parindent}{0pt}
\begin{document}
\sffamily
\noindent\begin{minipage}[t]{%(wa)smm}\vspace{0pt}{\bfseries\fontsize{8}{9}\selectfont\strut a}\par\vspace{0.4mm}%%
\begin{tikzpicture}[x=1mm,y=1mm]
\node[anchor=south west,inner sep=0] at (0,0) {\includegraphics[trim=%(trim)s,clip,width=%(wa)smm]{%(a)s}};
%(overlay)s
\end{tikzpicture}\end{minipage}\hfill
\begin{minipage}[t]{%(wb)smm}\vspace{0pt}{\bfseries\fontsize{8}{9}\selectfont\strut b}\par\vspace{0.4mm}\includegraphics{%(b)s}\end{minipage}
\end{document}
"""

# The designer's panel (Chromium PDF: one 600-dpi raster of the drawing, labels as vector text on top) is rewritten:
#  - every horizontal label (the serif and Inter text, the poster sentences, and the monospace labels that need aligning)
#    is removed and set again by FIG1_LABELS in Helvetica, with code and bytes in IBM Plex Mono (the cube labels' face);
#  - the cube labels (rotated with the perspective) are kept, enlarged about their centre where they would print below 5 pt;
#  - the raster is recoloured (_fig1_recolour): every byte of organism A teal, and the bytes A writes into B teal too, so
#    that the vermilion tint means one thing only, the return instruction c0 and its return of control (as in Fig. 3f);
#  - a tick with "A | B" marks where A ends and B begins in each strip that holds both.
FIG1_MIN_PT = 5.0
FIG1_PX_PER_PT = 4252 / 510.0           # the raster spans the 510 pt page width at 600 dpi
# (text, x, baseline y from the top, both in designer points; size in final pt; colour; TikZ anchor; bold)
FIG1_LABELS = [
    (r"\mono{01 c5} × 8", 80.8, 60.0, 5.5, "ink", "base", False),
    ("first four bytes", 80.8, 69.5, 5.5, "mute", "base", False),
    (r"\mono{LD BC,nn}", 76.1, 116.0, 5.0, "ink", "base", False),          # centred under its bracket
    (r"\mono{PUSH BC}", 113.8, 116.0, 5.0, "ink", "base", False),           # under its tick, inside the circle
    ("operand = output", 80.8, 128.7, 5.5, "ink", "base", False),
    ("writes itself", 201.5, 59.4, 6.0, "ink", "base west", False),
    ("runs into B", 263.8, 145.3, 6.0, "ink", "base west", False),
    ("comes back", 219.8, 195.4, 6.0, "ink", "base west", False),
    ("stack starts", 413.9, 67.8, 5.5, "ink", "base west", False),
    ("next push", 307.2, 82.8, 5.5, "ink", "base west", False),
    (r"\mono{c5 01}", 422.2, 119.0, 5.5, "ink", "base west", False),
    ("first write", 422.2, 127.5, 5.5, "mute", "base west", False),
    ("Organism A · 16 bytes", 141.6, 172.0, 5.5, "ink", "base west", False),
    ("Partner B · 16 occupied bytes", 240.9, 172.0, 5.5, "ink", "base west", False),
    ("Partner B", 364.7, 205.6, 5.5, "ink", "base west", False),
    (r"\mono{\textcolor{verm}{c0} = RET NZ}", 277.8, 268.8, 5.5, "ink", "base", False),   # under the c0 cube
    ("Organism A", 221.8, 291.0, 5.5, "ink", "base west", False),
    ("writes into occupied B", 304.9, 291.0, 5.5, "ink", "base west", False),
]
FIG1_BADGES = []                        # (digit, x, y of the red disc's centre), filled from the drawing by _fig1_badges
FIG1_AB_TICKS = [(226.3, 114.0), (321.5, 230.1)]   # top-back corner of the first B cube after A's last cube, per strip
# raster regions, in pixels (x0, y0, x1, y1)
_FIG1_GREY_A = [(322, 668, 515, 866), (680, 652, 884, 850), (335, 1222, 668, 1512), (830, 1130, 1142, 1398), (1490, 990, 1782, 1242)]
_FIG1_WRITTEN = [(2840, 660, 3340, 915), (3040, 1750, 3310, 1965)]        # bytes written into B, and the next-push arrow
_FIG1_ARCENDS = [(2850, 430, 3200, 700), (2900, 1900, 3260, 2120)]        # the write arcs' warm-tinted ends
_FIG1_ARCSKIP = [(3120, 655, 3200, 700), (2850, 1880, 2965, 2030), (2965, 1800, 3150, 1978)]   # cube faces inside those
_OKM1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566], [0.0883024619, 0.2817188376, 0.6299787005]])
_OKM2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099], [0.0259040371, 0.7827717662, -0.8086757660]])


def _oklab(rgb):
    c = rgb / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return np.cbrt(lin @ _OKM1.T) @ _OKM2.T


def _srgb(lab):
    lin = np.clip(((lab @ np.linalg.inv(_OKM2).T) ** 3) @ np.linalg.inv(_OKM1).T, 0, 1)
    return np.clip(np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055) * 255.0, 0, 255)


def _fig1_recolour(img):
    """The designer's raster with A's grey `01` cubes made teal (grey -> teal fitted on the front and top faces in OKLab),
    and the bytes written into B, the next-push arrow and the write arcs' warm ends turned to the arcs' teal."""
    img = img.astype(float).copy()
    h_teal = np.radians(-160.0)

    def blend(box, new, w):
        x0, y0, x1, y1 = box
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - w[..., None]) + new * w[..., None]

    def teal(L, C):
        return _srgb(np.stack([L, C * np.cos(h_teal), C * np.sin(h_teal)], -1))

    for box in _FIG1_GREY_A:
        x0, y0, x1, y1 = box
        p = img[y0:y1, x0:x1]
        L = _oklab(p)[..., 0]
        mn = p.min(axis=2)
        thr = 8.5 - 127.0 * (L - 0.90)               # G - R separates grey (3-12) from teal (14-28) at the face's lightness
        w = np.clip((thr + 2 - (p[..., 1] - p[..., 0])) / 4, 0, 1) * np.clip((mn - 120) / 30, 0, 1) * np.clip((242 - mn) / 6, 0, 1)
        C = np.clip(0.030 + (0.016 - 0.030) * (L - 0.869) / (0.926 - 0.869), 0.0, 0.06)
        blend(box, teal(1.28 * L - 0.287, C), w)
    for box in _FIG1_WRITTEN:
        x0, y0, x1, y1 = box
        p = img[y0:y1, x0:x1]
        lab = _oklab(p)
        w = np.clip((p[..., 0] - p[..., 1] - 6) / 10, 0, 1)
        blend(box, teal(lab[..., 0] + 0.008, 0.65 * np.hypot(lab[..., 1], lab[..., 2])), w)
    for box in _FIG1_ARCENDS:
        x0, y0, x1, y1 = box
        p = img[y0:y1, x0:x1]
        lab = _oklab(p)
        C = np.hypot(lab[..., 1], lab[..., 2])
        h = np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))
        cov = np.clip((252 - p.min(axis=2)) / 25, 0, 1)
        w = (((h > -100) & (h < 140)) | (C < 0.022)) * cov
        for a0, b0, a1, b1 in _FIG1_ARCSKIP:
            xs0, xs1, ys0, ys1 = max(a0, x0) - x0, min(a1, x1) - x0, max(b0, y0) - y0, min(b1, y1) - y0
            if xs1 > xs0 and ys1 > ys0:
                w[ys0:ys1, xs0:xs1] = 0
        blend(box, teal(lab[..., 0] + 0.01, np.maximum(0.65 * C, 0.034 * cov)), w)
    return np.round(img).astype(np.uint8)


def _pdf_obj(data, num):
    m = re.search(rb"(?:^|\n)%d 0 obj\b" % num, data)
    return m.start() + (1 if data[m.start():m.start() + 1] == b"\n" else 0)


def _pdf_stream(data, num):
    """(start of object, end of object, decompressed stream) for a FlateDecode stream object with a direct /Length."""
    import zlib
    o = _pdf_obj(data, num)
    head = re.compile(rb"%d 0 obj\s*<<(.*?)>>\s*stream\r?\n" % num, re.S).match(data, o)
    n = int(re.search(rb"/Length (\d+)", head.group(1)).group(1))
    s0 = head.end()
    end = data.index(b"endobj", s0 + n) + len(b"endobj")
    return o, end, zlib.decompress(data[s0:s0 + n])


def _tounicode(data, font_num):
    """CID -> text for a Type0 font, from its ToUnicode CMap (bfchar and bfrange)."""
    o = _pdf_obj(data, font_num)
    tu = int(re.search(rb"/ToUnicode (\d+) 0 R", data[o:data.index(b"endobj", o)]).group(1))
    cmap = _pdf_stream(data, tu)[2].decode("latin1")
    out = {}
    for blk in re.findall(r"beginbfchar(.*?)endbfchar", cmap, re.S):
        for a, b in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", blk):
            out[int(a, 16)] = bytes.fromhex(b).decode("utf-16-be")
    for blk in re.findall(r"beginbfrange(.*?)endbfrange", cmap, re.S):
        for a, b, c in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", blk):
            base = bytes.fromhex(c).decode("utf-16-be")
            for k in range(int(a, 16), int(b, 16) + 1):
                out[k] = base[:-1] + chr(ord(base[-1]) + k - int(a, 16))
    return out


def _designer_text_edit(src, dst, drop_horizontal=True, min_size=None, image_fn=None):
    """Rewrite the designer's single-page PDF: drop every text object that runs horizontally (all labels except the cube
    labels, which are rotated with the perspective), scale up (font size and glyph advances, about the centre of the
    baseline) every kept text object whose effective size on the page is below `min_size`, and pass the page's raster
    through `image_fn`. Returns [(text, size_before, size_after, x, y_from_top)] for every text object kept."""
    import zlib
    data = open(src, "rb").read()
    page = data[_pdf_obj(data, 2):]
    page = page[:page.index(b"endobj")]
    contents = int(re.search(rb"/Contents (\d+) 0 R", page).group(1))
    height = float(re.search(rb"/MediaBox \[0 0 [\d.]+ ([\d.]+)\]", page).group(1))
    fonts = {name.decode(): _tounicode(data, int(num)) for name, num in re.findall(rb"/(F\d+) (\d+) 0 R", re.search(rb"/Font <<(.*?)>>", page, re.S).group(1))}
    o0, o1, stream = _pdf_stream(data, contents)
    txt = stream.decode("latin1")
    tok = re.compile(r"<[0-9A-Fa-f]*>|/[^\s/\[\]<>()]+|[-+]?(?:\d+\.?\d*|\.\d+)|[A-Za-z'\"*]+")
    ctm, stack, ops = np.eye(3), [], []
    edits, report, block, used_fonts = [], [], None, set()
    for m in tok.finditer(txt):
        t = m.group(0)
        if not (t[0].isalpha() or t[0] in "'\"*"):
            ops.append(m)
            continue
        vals = [o.group(0) for o in ops]
        if t == "q":
            stack.append(ctm.copy())
        elif t == "Q":
            ctm = stack.pop()
        elif t == "cm":
            a, b, c, d_, e, f = map(float, vals[-6:])
            ctm = np.array([[a, b, 0], [c, d_, 0], [e, f, 1]]) @ ctm
        elif t == "BT":
            block = {"start": m.start(), "tf": None, "tm": np.eye(3), "td": [], "text": ""}
        elif block is not None and t == "Tf":
            block["font"], block["size"], block["tf"] = vals[-2][1:], float(vals[-1]), ops[-1]
        elif block is not None and t == "Tm":
            a, b, c, d_, e, f = map(float, vals[-6:])
            block["tm"] = np.array([[a, b, 0], [c, d_, 0], [e, f, 1]])
            block["tm_ops"] = ops[-6:]
        elif block is not None and t == "Td":
            block["td"] += ops[-2:]
        elif block is not None and t in ("Tj", "TJ"):
            cmap = fonts[block["font"]]
            for h in re.findall(r"<([0-9A-Fa-f]*)>", " ".join(vals)):
                block["text"] += "".join(cmap.get(int(h[i:i + 4], 16), "?") for i in range(0, len(h), 4))
        elif block is not None and t == "ET":
            M = block["tm"] @ ctm
            size = block["size"] * np.sqrt(abs(np.linalg.det(M[:2, :2])))
            x, y = M[2, 0], height - M[2, 1]
            if drop_horizontal and abs(np.degrees(np.arctan2(M[0, 1], M[0, 0]))) < 0.5:
                edits.append((block["start"], m.end(), ""))
            else:
                used_fonts.add(block["font"])
                k = max(1.0, min_size / size) if min_size else 1.0
                if k > 1.0:
                    for o in [block["tf"]] + block["td"]:
                        edits.append((o.start(), o.end(), f"{float(o.group(0)) * k:.5f}"))
                    # keep the label centred where it was: shift its origin back by half of the added width
                    adv = [float(o.group(0)) for o in block["td"][0::2]]
                    w = sum(adv) + (np.mean(adv) if adv else block["size"] * 0.6)
                    a, b = float(block["tm_ops"][0].group(0)), float(block["tm_ops"][1].group(0))
                    for o, v in ((block["tm_ops"][4], -(k - 1) * w / 2 * a), (block["tm_ops"][5], -(k - 1) * w / 2 * b)):
                        edits.append((o.start(), o.end(), f"{float(o.group(0)) + v:.5f}"))
                report.append((block["text"], size, size * k, x, y))
            block = None
        ops = []
    for s0, s1, rep in sorted(edits, reverse=True):
        txt = txt[:s0] + rep + txt[s1:]
    body = zlib.compress(txt.encode("latin1"), 9)
    new_objs = {contents: b"%d 0 obj\n<</Filter /FlateDecode\n/Length %d>> stream\n" % (contents, len(body)) + body + b"\nendstream\nendobj"}
    if image_fn is not None:
        xo = int(re.search(rb"/XObject <</X\d+ (\d+) 0 R>>", page).group(1))
        io0, io1, raw = _pdf_stream(data, xo)
        head = re.compile(rb"%d 0 obj\s*<<(.*?)>>\s*stream\r?\n" % xo, re.S).match(data, io0).group(1)
        W, H = int(re.search(rb"/Width (\d+)", head).group(1)), int(re.search(rb"/Height (\d+)", head).group(1))
        img = image_fn(np.frombuffer(raw, np.uint8).reshape(H, W, 3))
        body = zlib.compress(np.ascontiguousarray(img, np.uint8).tobytes(), 9)
        head = re.sub(rb"/Length \d+", b"/Length %d" % len(body), head)
        new_objs[xo] = b"%d 0 obj\n<<" % xo + head + b">> stream\n" + body + b"\nendstream\nendobj"
    out = data
    for num in sorted(new_objs, key=lambda n: _pdf_obj(data, n), reverse=True):
        a0 = _pdf_obj(out, num)
        # the old object ends at the first "endobj" after its stream's end
        s_old = re.compile(rb"%d 0 obj\s*<<(.*?)>>\s*stream\r?\n" % num, re.S).match(out, a0)
        n_old = int(re.search(rb"/Length (\d+)", s_old.group(1)).group(1))
        a1 = out.index(b"endobj", s_old.end() + n_old) + len(b"endobj")
        out = out[:a0] + new_objs[num] + out[a1:]
    # drop the fonts no kept label uses (the serif and Inter faces) from the page's resources, so they are not embedded
    p0 = _pdf_obj(out, 2)
    p1 = out.index(b"endobj", p0)
    fdict = re.search(rb"/Font <<(.*?)>>", out[p0:p1], re.S)
    keep = b"\n".join(b"/%s %s 0 R" % (n, r) for n, r in re.findall(rb"/(F\d+) (\d+) 0 R", fdict.group(1)) if n.decode() in used_fonts)
    out = out[:p0 + fdict.start(1)] + keep + out[p0 + fdict.end(1):]
    # rebuild the classic xref table for the shifted offsets
    sx = out.rindex(b"\nxref") + 1
    nobj = int(re.search(rb"/Size (\d+)", out[sx:]).group(1))
    offs = {k: _pdf_obj(out[:sx], k) for k in range(1, nobj)}
    xref = b"xref\n0 %d\n0000000000 65535 f \n" % nobj + b"".join(b"%010d 00000 n \n" % offs[k] for k in range(1, nobj))
    trailer = out[out.index(b"trailer", sx):out.index(b"startxref", sx)]
    open(dst, "wb").write(out[:sx] + xref + trailer + b"startxref\n%d\n%%%%EOF\n" % sx)
    return report


def _render_gray(pdf, dpi):
    import subprocess
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", pdf, os.path.join(tmp, "p")], check=True)
        return np.asarray(Image.open(os.path.join(tmp, "p.png")).convert("RGB")).astype(int)


def _ink_bbox(pdf, pad_pt=3.0):
    """Bounding box (x0, y0_top, x1, y1_top) in PDF points of everything drawn on page 1 that is not white."""
    im = _render_gray(pdf, 288).min(axis=2)
    ys, xs = np.where(im < 250)
    k = 72.0 / 288
    return xs.min() * k - pad_pt, ys.min() * k - pad_pt, (xs.max() + 1) * k + pad_pt, (ys.max() + 1) * k + pad_pt


def _fig1_badges(pdf):
    """Centres (designer points) of the three red step discs, which are vector paths: the digits are set again on them."""
    im = _render_gray(pdf, 288)
    red = (im[..., 0] > 180) & (im[..., 0] - im[..., 1] > 70) & (im[..., 0] - im[..., 2] > 70)
    todo = set(zip(*np.where(red)))
    k = 72.0 / 288
    found = []
    while todo:                                   # 4-connected components of the red mask
        seed = todo.pop()
        comp, stack = [seed], [seed]
        while stack:
            y, x = stack.pop()
            for nb in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if nb in todo:
                    todo.remove(nb)
                    comp.append(nb)
                    stack.append(nb)
        ys, xs = np.array(comp).T
        wx, wy = int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1
        if 300 < len(xs) < 4000 and abs(wx - wy) < 6 and len(xs) > 0.7 * wx * wy:      # a small filled disc (pi/4 of its box)
            found.append(((xs.min() + xs.max() + 1) / 2 * k, (ys.min() + ys.max() + 1) / 2 * k))
    return found


def _pdf_size_pt(pdf):
    data = open(pdf, "rb").read()
    x0, y0, x1, y1 = map(float, re.search(rb"/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]", data).groups())
    return x1 - x0, y1 - y0


def _fig1_overlay(x0, y0, kmm, ha):
    """TikZ commands for the labels, the step digits and the A|B ticks; kmm = mm per designer point."""
    X = lambda x: (x - x0) * kmm
    Y = lambda y: ha - (y - y0) * kmm
    cmds = []
    for text, x, y, size, col, anchor, bold in FIG1_LABELS:
        font = r"\fontsize{%.2f}{%.2f}\selectfont%s" % (size, size * 1.2, r"\bfseries" if bold else "")
        cmds.append(r"\node[anchor=%s,inner sep=0,text=%s,font=%s] at (%.3f,%.3f) {%s};" % (anchor, col, font, X(x), Y(y), text))
    for digit, x, y in FIG1_BADGES:
        cmds.append(r"\node[anchor=center,inner sep=0,text=white,font=\fontsize{5.5}{6.6}\selectfont\bfseries] at (%.3f,%.3f) {%s};" % (X(x), Y(y), digit))
    for x, y in FIG1_AB_TICKS:
        cmds.append(r"\draw[ink,line width=0.5pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (X(x), Y(y - 1.0), X(x), Y(y - 9.0)))
        font = r"\fontsize{5}{6}\selectfont"
        cmds.append(r"\node[anchor=base east,inner sep=0,text=ink,font=%s] at (%.3f,%.3f) {A};" % (font, X(x) - 0.45, Y(y - 6.6)))
        cmds.append(r"\node[anchor=base west,inner sep=0,text=ink,font=%s] at (%.3f,%.3f) {B};" % (font, X(x) + 0.45, Y(y - 6.6)))
    return "\n".join(cmds)


def _fig1_label_box(x0, y0, x1, y1, scale):
    """Grow the crop (designer points) to hold every overlay label: width estimated at 0.6 em per character."""
    for text, x, y, size, col, anchor, bold in FIG1_LABELS:
        plain = re.sub(r"\\[a-z]+\{|\{|\}", "", re.sub(r"\\textcolor\{[a-z]+\}", "", text))
        sz = size / scale
        w = 0.6 * sz * len(plain)
        lx = x - (w / 2 if anchor == "base" else (w if anchor.endswith("east") else 0))
        x0, x1 = min(x0, lx - 2), max(x1, lx + w + 2)
        y0, y1 = min(y0, y - 0.8 * sz - 2), max(y1, y + 0.3 * sz + 2)
    return x0, y0, x1, y1


def fig1(out):
    """a: the designer's conceptual panel (vector PDF, round 4), rewritten by _designer_text_edit and composed with
    tectonic (labels set by FIG1_LABELS); b: Kaplan–Meier emergence by L."""
    import shutil
    import subprocess
    design_pdf = os.path.join(HERE, "refs", "designer_fig1_round4.pdf")
    design_png = os.path.join(HERE, "refs", "designer_fig1_round3.png")
    if os.path.exists(design_pdf) and shutil.which("tectonic") and shutil.which("pdftoppm"):
        PT = 72.0 / 25.4
        fig = plt.figure(figsize=(55 * fs.MM, 70 * fs.MM))
        ax = fig.add_axes([0.17, 0.1, 0.8, 0.76])
        km_panel(ax, ncol=3)
        figcheck.print_report(figcheck.check(fig), "fig1_km")
        fs.save(fig, os.path.join(out, "fig1_km"), formats=("pdf",))
        wb_pt, hb_pt = _pdf_size_pt(os.path.join(out, "fig1_km.pdf"))     # b is placed at its natural size (scale 1)
        wa = 180.0 - wb_pt / PT - 1.5                                     # a takes the rest of the 180 mm width
        a_pdf = os.path.join(out, "fig1a_design.pdf")
        page_w, page_h = _pdf_size_pt(design_pdf)
        # the step discs are found once, on the drawing with its text removed and without the recolouring
        _designer_text_edit(design_pdf, a_pdf, min_size=None)
        discs = sorted(_fig1_badges(a_pdf), key=lambda p: p[1])          # steps 1, 2, 3 run down the page
        assert len(discs) == 3, f"expected three step discs, found {discs}"
        FIG1_BADGES[:] = [(str(i + 1), x, y) for i, (x, y) in enumerate(discs)]
        min_design = FIG1_MIN_PT                                          # iterate: the crop and the scale depend on each other
        for _ in range(4):
            rep = _designer_text_edit(design_pdf, a_pdf, min_size=min_design, image_fn=_fig1_recolour)
            x0, y0, x1, y1 = _ink_bbox(a_pdf)
            scale = wa * PT / (x1 - x0)
            x0, y0, x1, y1 = _fig1_label_box(x0, y0, x1, y1, scale)
            x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(page_w, x1), min(page_h, y1)
            scale = wa * PT / (x1 - x0)
            smallest = min(r[2] for r in rep) * scale
            if smallest >= FIG1_MIN_PT - 1e-6:
                break
            min_design = FIG1_MIN_PT / scale * 1.005
        kmm = wa / (x1 - x0)
        ha = (y1 - y0) * kmm
        print(f"  fig1a: {len(rep)} cube labels kept, {sum(r[2] > r[1] * 1.0001 for r in rep)} enlarged; {len(FIG1_LABELS)} labels set "
              f"in Helvetica/Plex Mono; panel scale {scale:.3f}; smallest cube label {smallest:.2f} pt at 180 mm; badges {FIG1_BADGES}")
        trim = f"{x0:.2f} {page_h - y1:.2f} {page_w - x1:.2f} {y0:.2f}"                   # left bottom right top, bp
        tex_path = os.path.join(out, "fig1_compose.tex")
        h = max(ha, hb_pt / PT) + 15.0
        for _ in range(2):                    # compose on a deep page, then again on a page cut 1 mm below the lowest ink
            open(tex_path, "w").write(COMPOSE_TEX % {"h": f"{h:.2f}", "wa": f"{wa:.2f}", "wb": f"{wb_pt / PT:.2f}", "trim": trim,
                                                     "a": a_pdf, "b": os.path.join(out, "fig1_km.pdf"),
                                                     "overlay": _fig1_overlay(x0, y0, kmm, ha)})
            subprocess.run(["tectonic", "--outdir", out, tex_path], check=True, capture_output=True)
            h = _ink_bbox(os.path.join(out, "fig1_compose.pdf"), pad_pt=0.0)[3] / PT + 1.0
        os.replace(os.path.join(out, "fig1_compose.pdf"), os.path.join(out, "fig1.pdf"))
        for stale in ("fig1.svg", "fig1_compose.tex"):
            if os.path.exists(os.path.join(out, stale)):
                os.remove(os.path.join(out, stale))
        subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", os.path.join(out, "fig1.pdf"), os.path.join(out, "fig1")], check=False)
        print("  fig1 composed from the designer's vector PDF (a, rewritten) and the KM panel (b)")
        return
    fig = plt.figure(figsize=(fs.DOUBLE, 72 * fs.MM))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[112, 58], wspace=0.22, left=0.012, right=0.99, top=0.9, bottom=0.14)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    if os.path.exists(design_png):
        axa.imshow(plt.imread(design_png))
        axa.set_axis_off()
    else:
        cp.fig1a(axa)
    axa.set_anchor("NW")
    label(axa, "a", dx=0.01)
    try:
        km_panel(axb, ncol=3)
        label(axb, "b")
    except Exception as e:  # noqa: BLE001
        placeholder(axb, f"b  (data missing: {e})")
    save(fig, os.path.join(out, "fig1"))


# ----------------------------------------------------------------------------------------------------------------- fig 2
VIDEO_STEM = os.path.join(EXP, "runs", "video", "video_L16_st128_k4_s2002")
FRAME_STEPS = [5, 320, 600, 1200, 10000, 41000, 100000]
FRAME_CAPS = ["random programs", "tar: zeros spread", "first replicators", "the wave", "the open phase", "closure spreads", "closed"]
WINDOW_STEP, WIN_W, WIN_H = 600, 24, 16


def _frame_rgb(soup, cc, k=4, min_count=20):
    """Lattice map, one pixel per tape: the k most common classes (>= min_count tapes, not zero-rich) in the mean colour of
    their bytes (the same rule as the supplementary video); every other tape grey, lighter the more of its bytes are zero,
    so the tar flood and the zero pockets are visible as bleaching."""
    uniq, inv, counts = np.unique(soup, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    zf = (uniq == 0).mean(axis=1)
    base = np.array([168.0, 173.0, 181.0])
    col = (base[None, :] + (255.0 - base)[None, :] * zf[:, None]).astype(np.uint8)
    order = np.argsort(-counts, kind="stable")
    top = []
    for i in order[:k]:
        if counts[i] >= min_count and zf[i] < 0.5:
            col[i] = cc.colour(uniq[i].tobytes())
            top.append(i)
    return col[inv].reshape(125, 160, 3), [(uniq[i], int(counts[i])) for i in top]


def _best_window(soup, top_class):
    """Window (row, col) of WIN_W x WIN_H tapes whose share of the top class is closest to 0.45 (a patch edge)."""
    member = (soup == top_class[None, :]).all(axis=1).reshape(125, 160).astype(float)
    best, score = (0, 0), 9.0
    for r in range(0, 125 - WIN_H + 1, 2):
        for c in range(0, 160 - WIN_W + 1, 2):
            s = abs(member[r:r + WIN_H, c:c + WIN_W].mean() - 0.45)
            if s < score:
                best, score = (r, c), s
    return best


def fig2(out):
    import soup_stills as ss
    rows = [json.loads(l) for l in open(VIDEO_STEM + ".jsonl") if '"kind": "sample"' in l]
    rows.sort(key=lambda r: r["step"])
    S = pd.DataFrame([{"step": r["step"], "zero": r["zero_frac"], "q": r.get("q_share", np.nan), "unique": r.get("unique", np.nan)} for r in rows])
    S = S[S["step"] > 0]
    snaps = ss.snapshot_files(VIDEO_STEM)
    files = {st: min(snaps, key=lambda t: abs((t[1] if t[1] is not None else -1) - st)) for st in FRAME_STEPS}
    cc = ss.ClassColours(k=8)
    frames = {}
    for st in FRAME_STEPS:
        name, sp, f = files[st]
        soup = ss.load(f, 16)
        frames[st] = (sp, soup, *_frame_rgb(soup, cc))

    fig = plt.figure(figsize=(fs.DOUBLE, 84 * fs.MM))
    gs_top = GridSpec(1, 7, figure=fig, wspace=0.06, left=0.02, right=0.99, top=0.9, bottom=0.6)
    gs_bot = GridSpec(1, 2, figure=fig, width_ratios=[2.35, 1.0], wspace=0.12, left=0.075, right=0.99, top=0.44, bottom=0.11)
    for i, st in enumerate(FRAME_STEPS):
        ax = fig.add_subplot(gs_top[0, i])
        sp, soup, rgb, top = frames[st]
        ax.imshow(np.repeat(np.repeat(rgb, 4, axis=0), 4, axis=1), interpolation="nearest")
        ax.set_axis_off()
        ax.set_title(FRAME_CAPS[i], fontsize=5.5, color=INK, pad=2)
        ax.text(0.5, -0.06, f"step {sp:,}", transform=ax.transAxes, ha="center", va="top", fontsize=5, color=GREY, gid="allow-outside")
        ax.text(0.03, 0.97, str(i + 1), transform=ax.transAxes, ha="left", va="top", fontsize=5, color=INK, gid="allow-outside",
                bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="none"))
        if st == WINDOW_STEP:
            r0, c0 = _best_window(soup, top[0][0])
            ax.add_patch(Rectangle((c0 * 4 - 0.5, r0 * 4 - 0.5), WIN_W * 4, WIN_H * 4, fill=False, ec=RED, lw=0.7))
        if i == 0:
            label(ax, "a", dx=0.012)
    # b: the world's time course with the frames marked
    axb = fig.add_subplot(gs_bot[0, 0])
    axb.plot(S["step"], S["zero"], color=GREY, lw=0.9, label="zero bytes (fraction of all bytes)")
    axb.plot(S["step"], S["q"], color=TEAL, lw=0.9, label="occupancy of the dominant tape")
    axb.plot(S["step"], S["unique"] / 20000.0, color=INK, lw=0.8, ls="--", label="distinct tapes (fraction of 20,000)")
    for i, st in enumerate(FRAME_STEPS):
        sp = frames[st][0]
        axb.axvline(sp, color=RULE, lw=0.5, ls=":", zorder=0)
        axb.text(sp, 1.03, str(i + 1), ha="center", va="bottom", fontsize=5, color=INK, gid="allow-outside")
    axb.set_xscale("log")
    axb.set_xlim(4, 1.3e5)
    fs.log10_ticks(axb)
    axb.set_ylim(0, 1.0)
    axb.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1.00"])     # one number of decimals, as in Fig. 4a
    fs.tidy(axb, "step", "fraction")
    axb.set_gid("allow-clip")
    axb.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=5, frameon=False, columnspacing=1.2, handlelength=1.6)
    label(axb, "b")
    # c: a byte-resolution window of frame 3
    axc = fig.add_subplot(gs_bot[0, 1])
    sp, soup, rgb, top = frames[WINDOW_STEP]
    r0, c0 = _best_window(soup, top[0][0])
    idx = np.array([[(r0 + r) * 160 + (c0 + c) for c in range(WIN_W)] for r in range(WIN_H)]).ravel()
    # one 4 x 4 block per tape with a one-pixel white gutter, so the tapes read as tiles
    win = np.full((WIN_H * 5 + 1, WIN_W * 5 + 1, 3), 255, np.uint8)
    for r in range(WIN_H):
        for c in range(WIN_W):
            win[1 + r * 5:5 + r * 5, 1 + c * 5:5 + c * 5] = ss.LUT[soup[idx[r * WIN_W + c]]].reshape(4, 4, 3)
    axc.imshow(np.repeat(np.repeat(win, 5, axis=0), 5, axis=1), interpolation="nearest")
    for sp_ in axc.spines.values():
        sp_.set_edgecolor(RED)
        sp_.set_linewidth(0.7)
    axc.set_xticks([])
    axc.set_yticks([])
    word = " ".join(f"{b:02x}" for b in top[0][0][:2])
    axc.set_title(f"{WIN_W} × {WIN_H} tapes of frame 3 at byte resolution", fontsize=5.5, color=INK, pad=2)
    axc.text(0.5, -0.05, f"one block per tape, one pixel per byte; zero bytes white;\nthe replicator is the repeated word {word}", transform=axc.transAxes,
             ha="center", va="top", fontsize=5, color=GREY, gid="allow-outside")
    label(axc, "c", dx=0.03)
    save(fig, os.path.join(out, "fig2"))
    # the numbers the legend quotes
    z = S.set_index("step")["zero"]
    print("  fig2 numbers: zero peak %.3f at step %d; tq_10 %s; zero at 100k %.3f; q at 100k %.3f; distinct at 100k %d" % (
        z.max(), z.idxmax(), json.load(open(VIDEO_STEM + ".summary.json")).get("tq_10"), z.iloc[-1], S["q"].iloc[-1], S["unique"].iloc[-1]))
    print("  fig2 window: rows %d–%d, cols %d–%d of frame at step %d; top classes %s" % (r0, r0 + WIN_H, c0, c0 + WIN_W, sp,
          [(" ".join(f"{b:02x}" for b in t[:4]), n) for t, n in top]))


# ----------------------------------------------------------------------------------------------------------------- fig 3
COUNT_TOL_PT, COUNT_SPREAD_PT = 1.5, 3.0   # slope charts: a cluster chains within half a marker and spans at most one marker
JIT_HALF = 0.14                            # slope charts: points spread sideways uniformly by +-0.14 of the column spacing


def slope_chart(ax, groups, first, final, loop_first, loop_final, ylab, ylim, yticks, seed=0, header_y=None, final_marker="o",
                count_tol=None, count_min=2, count_lines=(), count_pad=0.0, count_spread=np.inf, counts=True, ytick_labels=None,
                count_side="above"):
    """Paired first -> final per world, grouped (one group per tape length): grey lines join the same world; filled
    vermilion = loop instruction present, open grey = absent (the convention of every first/final chart in the paper).
    Points spread sideways within their column (uniform, +-0.14 of the column spacing); heights are exact.
    final_marker "s" draws the final dominants as squares. Coincident points are counted: points of a column whose values
    chain within count_tol overlap; every such cluster of at least count_min points and at most count_spread from its lowest
    to its highest value is labelled with its size in grey, beside the column on the side away from the joining lines
    (left of first, right of final), at the cluster's mean value; a number within count_pad of one of count_lines
    (threshold lines) is set just above (count_side "above") or below that line instead. By default (count_tol None) a
    cluster chains within half a marker (COUNT_TOL_PT), spans at most one marker (COUNT_SPREAD_PT), and is labelled only
    if it holds more points than its column can show side by side: more than (1 + spread / d)(1 + band / d), with d the
    marker diameter and band the sideways spread, all in points. counts=False draws no numbers."""
    rng = np.random.default_rng(seed)
    ax.set_ylim(*ylim)
    lab_h = 5.6 * (ylim[1] - ylim[0]) / (ax.get_position().height * ax.figure.get_figheight() * 72.0)   # a 5 pt number's height
    auto = counts and count_tol is None
    if auto:                                    # half a marker and one marker, in data units of this panel
        pos = ax.get_position()
        per_pt = (ylim[1] - ylim[0]) / (pos.height * ax.figure.get_figheight() * 72.0)
        count_tol, count_spread = COUNT_TOL_PT * per_pt, min(count_spread, COUNT_SPREAD_PT * per_pt)
        xpt = pos.width * ax.figure.get_figwidth() * 72.0 / ((len(groups) - 1) * 3.0 + 2.4)     # points per x unit
        d_pt = 2.0 * np.sqrt(LOOP_MS / np.pi)                                                   # marker diameter
        band_pt = 2 * JIT_HALF * xpt
    left_count = right_count = False
    for gi, (name, idx) in enumerate(groups):
        x0 = gi * 3.0
        j = rng.uniform(-JIT_HALF, JIT_HALF, len(idx))
        for k, w in enumerate(idx):
            ax.plot([x0 + j[k], x0 + 1 + j[k]], [first[w], final[w]], color=RULE, lw=0.5, alpha=0.9, zorder=1)
        for dx, vals, loop, mk in ((0, first, loop_first, "o"), (1, final, loop_final, final_marker)):
            y = np.array([vals[w] for w in idx])
            lp = np.array([loop[w] for w in idx], bool)
            xs = x0 + dx + j
            ms = LOOP_MS * (0.8 if mk == "s" else 1.0)            # a square of the circle's area reads as larger
            ax.scatter(xs[~lp], y[~lp], s=ms, marker=mk, facecolors="white", edgecolors=GREY, lw=0.6, zorder=3)
            ax.scatter(xs[lp], y[lp], s=ms, marker=mk, color=RED, lw=0, zorder=3)
            if counts and count_tol is not None and len(y):
                ys = np.sort(y)
                last_c = -np.inf                                 # numbers beside one column never overlap: each sits above the last
                for cl in np.split(ys, np.where(np.diff(ys) > count_tol)[0] + 1):
                    if len(cl) < count_min or cl[-1] - cl[0] > count_spread:     # only clusters that are truly coincident
                        continue
                    if auto and len(cl) <= (1 + (cl[-1] - cl[0]) / per_pt / d_pt) * (1 + band_pt / d_pt):
                        continue                                 # the sideways spread already shows every point
                    yc, va = float(np.mean(cl)), "center"
                    for line in count_lines:                     # a threshold line through the number: set it beside the line
                        if abs(yc - line) < count_pad:
                            yc, va = (line + 0.1 * count_pad, "bottom") if count_side == "above" else (line - 0.1 * count_pad, "top")
                    if va == "center":
                        yc = max(yc, last_c + lab_h)
                        last_c = yc
                    ax.text(x0 + dx + (-0.32 if dx == 0 else 0.32), yc, f"{len(cl)}", ha="right" if dx == 0 else "left", va=va,
                            fontsize=5, color=GREY)
                    left_count |= (gi == 0 and dx == 0)
                    right_count |= (gi == len(groups) - 1 and dx == 1)
        ax.text(x0 + 0.5, header_y if header_y is not None else ylim[1], name, ha="center", va="bottom", fontsize=6, color=INK)
    ax.set_xticks([gi * 3.0 + dx for gi in range(len(groups)) for dx in (0, 1)], ["first", "final"] * len(groups), fontsize=5)
    ax.set_xlim(-1.05 if left_count else -0.7, (len(groups) - 1) * 3.0 + (2.05 if right_count else 1.7))   # room for the end counts
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks, ytick_labels)
    fs.tidy(ax, None, ylab)


def fig3(out):
    g = pd.read_csv(os.path.join(R, "stageG", "stageG", "stage_g_runs.csv"))
    c4 = pd.read_csv(os.path.join(R, "stageG", "c4", "functional.csv"))
    Ls = [16, 20, 50, 64]
    H = 144.0      # mm: rows a-c and d-e keep their size, f is about 30% shallower than before (52.8 -> 37 mm)
    fig = plt.figure(figsize=(fs.DOUBLE, H * fs.MM))
    gs1 = GridSpec(1, 3, figure=fig, wspace=0.42, left=0.075, right=0.99, top=1 - 8.0 / H, bottom=1 - 41.6 / H)
    gs2 = GridSpec(1, 2, figure=fig, width_ratios=[1.45, 1.0], wspace=0.3, left=0.075, right=0.99, top=1 - 59.2 / H, bottom=1 - 92.8 / H)
    gs3 = GridSpec(1, 1, figure=fig, left=0.03, right=0.99, top=1 - 106.4 / H, bottom=0.005)
    axa, axb, axc = fig.add_subplot(gs1[0, 0]), fig.add_subplot(gs1[0, 1]), fig.add_subplot(gs1[0, 2])
    axd, axe = fig.add_subplot(gs2[0, 0]), fig.add_subplot(gs2[0, 1])
    axf = fig.add_subplot(gs3[0, 0])
    # a, b: partner test, first replicator -> final dominant per world
    groups, first_c, final_c, first_d, final_d, lf, ll = [], {}, {}, {}, {}, {}, {}
    for L in Ls:
        d = g[g["L"] == L].reset_index(drop=True)
        idx = [f"{L}:{i}" for i in range(len(d))]
        hz = int(d["horizon"].iloc[0])
        groups.append((f"L = {L}", idx))
        for i, key in enumerate(idx):
            first_c[key], final_c[key] = d.loc[i, "first_copied"], d.loc[i, "final_copied"]
            first_d[key], final_d[key] = d.loc[i, "first_damaged"], d.loc[i, "final_damaged"]
            lf[key], ll[key] = loop_flags(d, "first")[i], loop_flags(d, "final")[i]
    slope_chart(axa, groups, first_c, final_c, lf, ll, "fraction of partners\ncopied (≥ 75%)", (-0.04, 1.12), [0, 0.25, 0.5, 0.75, 1.0], seed=0, header_y=1.04)
    slope_chart(axb, groups, first_d, final_d, lf, ll, "fraction of encounters with\nself-damage (≥ 25% lost)", (-0.04, 1.12), [0, 0.25, 0.5, 0.75, 1.0], seed=1, header_y=1.04)
    h = [plt.Line2D([], [], marker="o", ls="none", color=RED, ms=3, label="loop instruction"),
         plt.Line2D([], [], marker="o", ls="none", mfc="white", mec=GREY, ms=3, label="no loop instruction")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.5, frameon=False, columnspacing=2.0)
    label(axa, "a")
    label(axb, "b")
    # c: information inflow H(o | x) in bits (analysis B)
    try:
        P = pd.read_csv(os.path.join(R, "biology", "individuality", "per_replicator.csv"))
        P = P[P["machine"] == "z80"].copy()
        P["loop"] = P["has_loop"].map(lambda v: str(v) == "True")
        groups_i, fH, lH, lfH, llH = [], {}, {}, {}, {}
        for L in Ls:
            d = P[P["group"].astype(int) == L]
            first = d[d["which"] == "first"].set_index("world")
            final = d[d["which"] == "final"].set_index("world")
            worlds = first.index.intersection(final.index)
            idx = [f"{L}:{w}" for w in worlds]
            groups_i.append((f"L = {L}", idx))
            for w, key in zip(worlds, idx):
                fH[key], lH[key] = first.loc[w, "H_bits"], final.loc[w, "H_bits"]
                lfH[key], llH[key] = first.loc[w, "loop"], final.loc[w, "loop"]
        slope_chart(axc, groups_i, fH, lH, lfH, llH, H_AXIS, (-0.3, 9.0), [0, 2, 4, 6, 8], seed=2, header_y=8.35,
                    count_lines=(8.0,), count_pad=0.35, count_side="below")
        axc.axhline(8.0, color=RULE, lw=0.5, ls=":", zorder=0)
        axc.text(axc.get_xlim()[0] + 0.15, 7.6, "8-bit ceiling", ha="left", va="top", fontsize=5, color=GREY)
        label(axc, "c")
    except Exception as e:  # noqa: BLE001
        placeholder(axc, f"c (data missing: {e})")
    # d: heritable fraction of random cells vs step per L (median + IQR)
    for L in Ls:
        c = c4[c4["tape_len"] == L].groupby("step")["frac_heritable"]
        med, lo, hi = c.median(), c.quantile(0.25), c.quantile(0.75)
        axd.plot(med.index, med.values, color=L_COL[L], lw=0.9, label=f"L = {L}", zorder=3)
        # interquartile band: a light fill of the length's colour, no outline (outlines of four bands crossed unreadably)
        axd.fill_between(med.index, lo.values, hi.values, color=L_COL[L], alpha=0.10, lw=0, zorder=1)
    axd.set_xscale("log")
    axd.set_xlim(40, 1.2e6)
    fs.log10_ticks(axd)
    axd.set_ylim(-0.02, 1.02)
    fs.tidy(axd, "step", "heritable fraction of random cells")
    axd.set_gid("allow-clip")
    axd.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    label(axd, "d")
    # e: convergence and closure counts per L as a count table (filled proportion behind each count)
    rows = []
    for L in Ls:
        d = g[g["L"] == L]
        modal = d["final_tape"].mode().iloc[0]
        hz = int(d["horizon"].iloc[0])
        rows.append({"L": L, "hz": hz, "loop": int(loop_flags(d, "final").sum()), "closed": int((d["final_copied"] >= 0.95).sum()),
                     "identical": int((d["final_tape"] == modal).sum()), "n": len(d)})
    T = pd.DataFrame(rows)
    cols = [("loop", "loop\ninstruction"), ("closed", "copies ≥ 95%\nof partners"), ("identical", "byte-identical\nto the modal tape")]
    axe.set_xlim(0, 3.9)
    axe.set_ylim(-0.1, len(Ls) + 1.1)
    axe.set_axis_off()
    for j, (_, head) in enumerate(cols):
        axe.text(1.3 + j * 0.9, len(Ls) + 0.05, head, ha="center", va="bottom", fontsize=5, color=INK, linespacing=1.1)
    axe.text(0.42, len(Ls) + 0.05, "worlds of 20\nby horizon", ha="center", va="bottom", fontsize=5, color=GREY, linespacing=1.1)

    for i, r in T.iterrows():
        y = len(Ls) - 1 - i
        axe.text(0.42, y + 0.5, f"L = {r['L']}\n{fs.steps(r['hz'])} steps", ha="center", va="center", fontsize=5, color=INK, linespacing=1.1)
        for j, (key, _) in enumerate(cols):
            x = 0.85 + j * 0.9
            axe.add_patch(Rectangle((x + 0.03, y + 0.08), 0.84, 0.84, facecolor="#F1F2F4", edgecolor="none"))   # plain cell, no encoding
            axe.text(x + 0.45, y + 0.5, f"{r[key]}", ha="center", va="center", fontsize=6, color=INK)
    label(axe, "e", dx=0.03)
    # f: control flow of the first replicator and four closed successors (conceptual)
    cp.fig2d(axf)
    axf.set_anchor("NW")
    axa.apply_aspect()
    axf.apply_aspect()
    fig.text(axa.get_position().x0 - 0.052, axf.get_position().y1 + 0.006, "f", fontsize=8, fontweight="bold", va="bottom", ha="left", gid="panel-label")
    save(fig, os.path.join(out, "fig3"))
    print("  fig3 table:", T.to_dict("records"))


# ----------------------------------------------------------------------------------------------------------------- fig 4
def fig4(out):
    fig = plt.figure(figsize=(fs.DOUBLE, 100 * fs.MM))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.25, 1.0], hspace=0.55, wspace=0.45, left=0.2, right=0.98, top=0.96, bottom=0.1)
    axa, axb, axc, axd = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    # a: atlas forest (Stage C, 128 steps, 1/16). One colour; filled = all ten worlds alive; open = fewer; arrow = no emergence.
    try:
        C = pd.read_csv(os.path.join(R, "stageC", "stage_c", "c3_ablations_st128_k4.csv"))
        C = C[C["label"] != "none"].copy()
        C["ratio"] = C["km_ratio_vs_none"].replace([np.inf, -np.inf], np.nan)
        C = C.sort_values("ratio", na_position="last")
        y = np.arange(len(C))
        for yi, (_, r) in zip(y, C.iterrows()):
            alive, n = int(r["t_rep_n"]), int(r["n"])
            if np.isfinite(r["ratio"]):
                if alive == n:
                    axa.plot(r["ratio"], yi, "o", color=fs.MARK, ms=3.2)
                else:
                    axa.plot(r["ratio"], yi, "o", mfc="white", mec=fs.MARK, mew=0.7, ms=3.2)
            else:
                axa.annotate("", xy=(3000, yi), xytext=(600, yi), arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))
            dagger = "†" if r["steps_run_min"] >= 1_000_000 else ""          # these arms were counted to 1,000,000 steps
            axa.text(1.03, yi, f"{alive}/{n}{dagger}", transform=axa.get_yaxis_transform(), fontsize=5, va="center", color=INK if alive == n else GREY, gid="allow-outside")
        axa.axvline(1, color=INK, lw=0.5, ls=":")
        axa.set_xscale("log")
        axa.set_xlim(0.3, 3000)
        fs.log10_ticks(axa)                              # full-size superscripts: no exponent below 5 pt
        axa.set_yticks(y, C["label"].tolist())
        axa.set_ylim(-0.7, len(C) + 0.3)
        fs.tidy(axa, "emergence delay against the unablated soup\n(ratio of Kaplan–Meier medians; dotted line, no change)")
        axa.text(1.03, len(C) + 0.05, "alive", transform=axa.get_yaxis_transform(), fontsize=5, va="center", color=GREY, gid="allow-outside")
        h = [plt.Line2D([], [], marker="o", ls="none", color=fs.MARK, ms=3, label="all 10 worlds alive"),
             plt.Line2D([], [], marker="o", ls="none", mfc="white", mec=fs.MARK, ms=3, label="fewer alive"),
             plt.Line2D([], [], marker=r"$\rightarrow$", ls="none", color=GREY, ms=5, label="no median: fewer than half the worlds alive")]
        axa.legend(handles=h, fontsize=5, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, frameon=False, columnspacing=1.0, handletextpad=0.3)
    except Exception as e:  # noqa: BLE001
        placeholder(axa, f"a (data missing: {e})")
    # b: size axis: fraction alive by L (none @nominal) with Wilson CI, plus the pusher's isolated heritability
    try:
        S = pd.read_csv(os.path.join(R, "stageE", "stage_e", "size_arms.csv"))
        S = S[(S["ablation"] == "none") & (S["arm"] == "nominal")].sort_values("tape_len")
        axb.errorbar(S["tape_len"], S["t_rep_frac"], yerr=[S["t_rep_frac"] - S["t_rep_lo"], S["t_rep_hi"] - S["t_rep_frac"]], fmt="o", color=fs.MARK, ms=3, lw=0.6, capsize=1.5, label="fraction of worlds alive by 300,000 steps")
        U = pd.read_csv(os.path.join(R, "stageE", "stage_e", "unit_fitness_vs_L.csv"))
        U = U[(U["unit"].str.startswith("pusher")) & (U["steps"] == 128)].sort_values("L")
        axb.plot(U["L"], U["gen2"], color=INK, lw=0.9, ls="--", label="heredity (gen2) of the pusher tiling in isolation")   # ink: teal is the transmitter class
        axb.axhline(0.3, color=RULE, lw=0.5, ls=":")
        axb.set_xscale("log")
        axb.set_xticks([3, 5, 8, 12, 16, 25, 36, 50, 64, 100], ["3", "5", "8", "12", "16", "25", "36", "50", "64", "100"])
        axb.minorticks_off()
        axb.set_ylim(-0.03, 1.03)
        fs.tidy(axb, "tape length L (bytes)", "fraction of worlds, or gen2")
        axb.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)
    except Exception as e:  # noqa: BLE001
        placeholder(axb, f"b (data missing: {e})")
    # c: dead-zone switch (Stage F4 + Stage E reference)
    try:
        F = pd.read_csv(os.path.join(R, "stageF", "stage_f", "rings.csv"))
        F = F[(F["ablation"] == "none") & (F["L"].isin([8, 10, 12]))].sort_values(["L", "P"])
        # every point at its true P, except where two lengths share a ring (P = 28: L = 10 and 12), drawn 0.45 byte either side
        shared = F.groupby("P")["L"].nunique()
        for L, mk, side in ((8, "s", -1), (10, "^", -1), (12, "o", 1)):
            col = L_COL[L]                                     # the paper's length palette (figstyle.LEN_COLOR)
            d = F[F["L"] == L]
            frac = d["t_rep_n"] / d["n"]
            dx = np.where(d["P"].map(shared).values > 1, 0.45 * side, 0.0)
            axc.errorbar(d["P"] + dx, frac, yerr=[frac - d["t_rep_lo"], d["t_rep_hi"] - frac], fmt=mk, color=col, ms=3.2, lw=0.6, capsize=1.5, label=f"L = {L}")
            nat = d[d["P"] == 2 * d["L"]]
            axc.scatter(nat["P"] + dx[(d["P"] == 2 * d["L"]).values], nat["t_rep_n"] / nat["n"], s=40, facecolors="none", edgecolors=INK, lw=0.6, zorder=5)
        assert int((shared > 1).sum()) == 1 and shared.get(28, 0) == 2 and set(F[F["P"] == 28]["L"]) == {10, 12}, "ED Fig. 2c: shared rings changed"
        axc.scatter([], [], s=40, facecolors="none", edgecolors=INK, lw=0.6, label="native ring, P = 2L")
        axc.set_ylim(-0.03, 1.03)
        fs.tidy(axc, "pair memory ring P (bytes)", "fraction of worlds alive\nby 300,000 steps")
        axc.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, columnspacing=1.0)
    except Exception as e:  # noqa: BLE001
        placeholder(axc, f"c (data missing: {e})")
    # d: L = 9 reversal (Stage D)
    try:
        D = pd.read_csv(os.path.join(R, "stageD", "stage_d", "cells_D.csv"))
        if "k" in D:
            D = D[D["k"] == 4]
        D = D.drop_duplicates(subset=["label", "steps"])
        arms = [a for a in ["none", "stack-writes", "stack-write-only", "stack-read-only", "push", "call-rst-write"] if a in set(D["label"])]
        y = np.arange(len(arms))
        for st, mk, off, col in ((128, "o", -0.15, fs.MARK), (512, "s", 0.15, INK)):      # ink, not teal (the transmitter class)
            d = D[D["steps"] == st].set_index("label").reindex(arms)
            axd.errorbar(d["t_rep_frac"], y + off, xerr=[d["t_rep_frac"] - d["t_rep_lo"], d["t_rep_hi"] - d["t_rep_frac"]], fmt=mk, color=col, ms=3, lw=0.6, capsize=1.5, label=f"{st} instructions per encounter")
        axd.set_yticks(y, arms)
        axd.set_xlim(-0.03, 1.03)
        fs.tidy(axd, "fraction of 20 worlds alive at L = 9")
        axd.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)
    except Exception as e:  # noqa: BLE001
        placeholder(axd, f"d (data missing: {e})")
    # panel letters: one baseline per row, at the top of the row's keys; a and c at the left edge of the row labels, b and d
    # at a fixed offset from their axes
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    x_left = min(inv.transform((t.get_window_extent(rend).x0, 0))[0] for ax_ in (axa, axc) for t in ax_.get_yticklabels() if t.get_text())
    for row, (ax_l, ax_r) in ((0, (axa, axb)), (1, (axc, axd))):
        top = max(inv.transform((0, lg.get_window_extent(rend).y1))[1] for lg in (ax_l.get_legend(), ax_r.get_legend()) if lg is not None)
        for letter, x in (("ac"[row], x_left), ("bd"[row], ax_r.get_position().x0 - 0.052)):
            fig.text(x, top, letter, fontsize=8, fontweight="bold", va="top", ha="left", gid="panel-label")
    save(fig, os.path.join(out, "fig4"))


# ----------------------------------------------------------------------------------------------------------------- fig 5
def _allp_share(s):
    """Share of the all-`P` class (hex 50) among the ten largest classes of a sample; 0 when it is not among them."""
    for t in s["top"]:
        tp = t["tape"]
        if tp and set(tp[i:i + 2] for i in range(0, len(tp), 2)) == {"50"}:
            return t["share"]
    return 0.0


def fig5(out):
    from matplotlib import ticker as mticker
    bdir = os.path.join(EXP, "runs", "bff_modal", "bff")
    runs = pd.read_csv(os.path.join(R, "bff", "runs.csv"))
    runs["variant"] = runs["variant"].replace({"stdlit": "lit"})
    variants = [v for v in ["std", "wrap", "lit", "wraplit", "wraplitnh"] if v in set(runs["variant"])]
    xticks, xlabels = [0, 64, 256, 1024, 4096, 16384], ["0", "64", "256", "1,024", "4,096", "16,384"]

    def epoch_axis(ax):
        ax.set_xscale("symlog", linthresh=64)
        ax.set_xticks(xticks, xlabels)
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlim(0, 17500)
        ax.set_ylim(-0.02, 1.02)

    fig = plt.figure(figsize=(fs.DOUBLE, 120 * fs.MM))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.4, 1.0, 1.0], height_ratios=[1.0, 1.0], hspace=0.5, wspace=0.45, left=0.07, right=0.99, top=0.96, bottom=0.02)
    gs2 = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.0], hspace=0.5, left=0.065, right=0.99, top=0.96, bottom=0.02)
    axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    axd = fig.add_subplot(gs2[1, 0])
    for v in variants:
        for run in runs[runs["variant"] == v]["run"]:
            p = os.path.join(bdir, run, "samples.jsonl")
            if not os.path.exists(p):
                continue
            S = pd.DataFrame([{"epoch": s["epoch"], "h": s["frac_heritable"]} for s in map(json.loads, open(p))])
            axa.plot(S["epoch"], S["h"].rolling(4, min_periods=1).mean(), color=fs.BFF_VARIANT[v], lw=0.6, alpha=0.75)
        axa.plot([], [], color=fs.BFF_VARIANT[v], lw=0.9, alpha=0.9, label=f"{fs.BFF_VARIANT_LABEL[v]} (n = {int((runs['variant'] == v).sum())})")
    epoch_axis(axa)
    fs.tidy(axa, "epoch", "heritable fraction of random tapes")
    label(axa, "a")
    handles, labels = axa.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.49), ncol=5, frameon=False, fontsize=5, handlelength=1.8, columnspacing=1.6)
    rng = np.random.default_rng(0)
    ROW, SUB, JIT = 2.6, 0.45, 0.22
    yticks, ylabels = [], []
    for i, v in enumerate(variants):
        d = runs[(runs["variant"] == v) & runs["t_top"].notna()]
        y0 = -ROW * i
        for which, dy, mk in (("first", SUB, "o"), ("final", -SUB, "s")):
            y = y0 + dy + rng.uniform(-JIT, JIT, len(d))
            loop = d[f"{which}_loop"].astype(bool).values
            vals = d[f"{which}_entered"].values
            axb.scatter(vals[~loop], y[~loop], s=6, marker=mk, facecolors="none", edgecolors=fs.BFF_VARIANT[v], lw=0.6)
            axb.scatter(vals[loop], y[loop], s=6, marker=mk, color=fs.BFF_VARIANT[v], lw=0)
            yticks.append(y0 + dy)
            ylabels.append(which)
        axb.text(0.0, y0 + SUB + JIT + 0.3, fs.BFF_VARIANT_LABEL[v], ha="left", va="bottom", fontsize=5, fontweight="bold", color="black")
    axb.set_yticks(yticks, ylabels, fontsize=5)
    axb.set_ylim(-ROW * (len(variants) - 1) - SUB - JIT - 0.35, SUB + JIT + 0.3 + 0.75)
    axb.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    axb.set_xlim(-0.04, 1.04)
    axb.tick_params(axis="y", length=2)
    fs.tidy(axb, "encounters whose pointer\nenters the partner")
    label(axb, "b")
    for v in ("wraplit", "wraplitnh"):
        if v not in variants:
            continue
        for run in runs[runs["variant"] == v]["run"]:
            p = os.path.join(bdir, run, "samples.jsonl")
            if not os.path.exists(p):
                continue
            S = pd.DataFrame([{"epoch": s["epoch"], "share": _allp_share(s)} for s in map(json.loads, open(p))])
            axc.plot(S["epoch"], S["share"], color=fs.BFF_VARIANT[v], lw=0.6, alpha=0.6)
    epoch_axis(axc)
    fs.tidy(axc, "epoch", "share of the all-P tape in the soup")
    label(axc, "c")
    cp.fig4d(axd)
    axd.set_anchor("NW")
    label(axd, "d")
    save(fig, os.path.join(out, "fig5"))


# ----------------------------------------------------------------------------------------------------------------- fig 6
def fig6(out):
    """Theorem 2 as a diagram, a single panel (no letter): 120 mm wide."""
    fig = plt.figure(figsize=(120 * fs.MM, 52 * fs.MM))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    cp.fig5a(ax)
    ax.set_anchor("NW")
    save(fig, os.path.join(out, "fig6"))


def fig6v4(out):
    """v4 Fig. 6 | What is proved: a Theorem 2 (budget pigeonhole, the Z80 case); b Proposition 3 with Proposition 4's count."""
    fig = plt.figure(figsize=(fs.DOUBLE, 58 * fs.MM))
    axa = fig.add_axes([0.02, 0.02, 0.55, 0.9])
    axb = fig.add_axes([0.6, 0.02, 0.39, 0.9])
    cp.fig6a(axa)
    cp.fig6b(axb)
    for ax, letter in ((axa, "a"), (axb, "b")):
        ax.set_anchor("NW")
        label(ax, letter, dx=0.02, dy=0.0)
    save(fig, os.path.join(out, "fig6v4"))


# ------------------------------------------------------------------------------------------------------ Extended Data 12
def ed12(out):
    A = os.path.join(R, "biology", "assembly")
    fig = plt.figure(figsize=(fs.DOUBLE, 62 * fs.MM))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.25, 1.0, 1.0], height_ratios=[1.0, 0.8], hspace=0.12, wspace=0.42, left=0.075, right=0.99, top=0.9, bottom=0.14)
    axa1, axa2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    axb, axc = fig.add_subplot(gs[:, 1]), fig.add_subplot(gs[:, 2])
    world = "none@closure_L16_st128_k4_s2001"
    S = pd.read_csv(os.path.join(A, "per_sample.csv"))
    S = S[S["world"] == world].sort_values("step")
    S = S[S["step"] > 0]
    W = pd.read_csv(os.path.join(A, "per_world.csv"))
    w = W[W["world"] == world].iloc[0]
    axa1.plot(S["step"], S["A_top10"], color=TEAL, lw=0.8)
    axa1.axhline(w["threshold"], color=RULE, lw=0.5, ls=":")
    axa1.text(2.6e4, w["threshold"] * 1.35, "10 × baseline", fontsize=5, color=GREY, va="bottom", ha="right")
    axa1.set_yscale("log")
    axa1.set_ylim(1, 1e4)
    axa1.set_yticks([1, 10, 100, 1000, 10000], ["1", "10", "100", "1,000", "10,000"])
    fs.tidy(axa1, None, "assembly measure,\nten commonest classes")
    axa1.set_xticklabels([])
    axa2.plot(S["step"], S["hoe"], color=INK, lw=0.8)
    axa2.set_ylim(0, 4.2)
    axa2.set_yticks([0, 1, 2, 3, 4])
    fs.tidy(axa2, "step", "high-order entropy\n(bits per byte)")
    for ax in (axa1, axa2):
        ax.set_xscale("log")
        ax.set_xlim(1, 3.5e5)
        ax.axvline(w["t_rep"], color=RED, lw=0.6, ls="--")
        ax.set_gid("allow-clip")
    axa1.tick_params(axis="x", labelbottom=False)
    axa1.xaxis.set_major_locator(plt.matplotlib.ticker.LogLocator(base=10))
    fs.log10_ticks(axa2)
    axa1.text(w["t_rep"] * 1.15, 5e3, "first heritable\nreplicator", fontsize=5, color=RED, va="top")
    axa1.plot([w["step_first_cross"]], [w["A_first_cross"]], "o", mfc="white", mec=TEAL, ms=4, mew=0.8)
    axa1.text(w["step_first_cross"] / 1.25, w["A_first_cross"] * 1.9, "first tenfold rise", fontsize=5, color=TEAL, va="bottom", ha="right")
    label(axa1, "a")
    # b: first tenfold rise vs the first heritable replicator, all Stage G worlds
    mk = {16: "o", 20: "s", 50: "^", 64: "D"}
    top_edge = 2.0e4
    for L in (16, 20, 50, 64):
        d = W[W["L"] == L]
        y = d["step_first_cross"].astype(float).values
        ok = np.isfinite(y)
        axb.scatter(d["t_rep"].values[ok], y[ok], s=9, marker=mk[L], facecolors="white", edgecolors=INK, lw=0.6, label=f"L = {L} (n = {len(d)})")
        if (~ok).any():
            axb.scatter(d["t_rep"].values[~ok], np.full((~ok).sum(), top_edge), s=12, marker="x", color=RED, lw=0.7)
    axb.plot([60, 1e4], [60, 1e4], color=RULE, lw=0.5, ls=":")
    axb.set_xscale("log")
    axb.set_yscale("log")
    axb.set_xlim(60, 1e4)
    axb.set_ylim(1, 3e4)
    fs.log10_ticks(axb)
    fs.log10_ticks(axb, "y")
    axb.text(70, 1.4, "rises before\nthe replicator", fontsize=5, color=GREY, va="bottom")
    axb.text(70, 1.6e4, "no rise (x)", fontsize=5, color=RED, va="center")
    fs.tidy(axb, "first heritable replicator (step)", "first tenfold rise of the measure (step)")
    axb.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, columnspacing=0.8, handletextpad=0.2)
    label(axb, "b")
    # c: AUC of the two detectors against the heredity event by tape length (Stage E, snapshot steps)
    U = pd.read_csv(os.path.join(A, "auc.csv"))
    recs = []
    for _, r in U.iterrows():
        m = re.match(r"E \| L=(\d+) \| all arms \| snapshot steps", str(r["stratum"]))
        if m and np.isfinite(r["AUC"]):
            recs.append({"L": int(m.group(1)), "det": r["detector"], "auc": r["AUC"]})
    T = pd.DataFrame(recs)
    for det, col, mk_, lab in (("A_top10", TEAL, "o", "assembly measure"), ("hoe", INK, "s", "high-order entropy")):
        d = T[T["det"] == det].sort_values("L")
        axc.plot(d["L"], d["auc"], ls="none", color=col, marker=mk_, ms=3, label=lab, mfc="white" if det == "hoe" else col)
    # unablated worlds only (results/detectors/NUMBERS_DETECTORS.md, "by ablation and L", none@nominal)
    rows = []
    for line in open(os.path.join(R, "detectors", "NUMBERS_DETECTORS.md")):
        if line.startswith("| none@nominal"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            try:
                rows.append({"L": int(c[1]), "auc": float(c[4])})
            except ValueError:
                pass
    U0 = pd.DataFrame(rows).dropna().sort_values("L")
    axc.plot(U0["L"], U0["auc"], ls="none", color="#7B3294", marker="D", ms=2.8, mfc="#7B3294", label="high-order entropy, unablated worlds")
    axc.axhline(0.5, color=RULE, lw=0.5, ls=":")
    axc.set_xscale("log")
    axc.set_xticks([4, 8, 16, 32, 64, 100], ["4", "8", "16", "32", "64", "100"])
    axc.minorticks_off()
    axc.set_ylim(-0.03, 1.03)
    fs.tidy(axc, "tape length L (bytes)", "AUC against the heredity event")
    axc.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)
    label(axc, "c")
    save(fig, os.path.join(out, "ed12"))
    print("  ed12 numbers:", {k: w[k] for k in ("t_rep", "step_first_cross", "A_first_cross", "threshold")}, "AUC rows", len(T))


# ------------------------------------------------------------------------------------------------------ Extended Data 13
def ed13(out):
    leth = pd.read_csv(os.path.join(R, "stageI", "c4", "functional.csv"))
    ben = pd.read_csv(os.path.join(R, "stageG", "c4", "functional.csv"))
    ben = ben[(ben["tape_len"] == 16) & (ben["label"].str.startswith("none@closure"))]
    gl = pd.read_csv(os.path.join(R, "stageI", "stageI", "stage_g_runs.csv"))
    gb = pd.read_csv(os.path.join(R, "stageG", "stageG", "stage_g_runs.csv"))
    gb = gb[gb["L"] == 16]
    fig = plt.figure(figsize=(fs.DOUBLE, 55 * fs.MM))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 0.9], wspace=0.42, left=0.07, right=0.99, top=0.86, bottom=0.17)
    axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    for ax, col, ylab in ((axa, "frac_heritable", "heritable fraction of random cells"), (axb, "zero_frac", "zero bytes (fraction of all bytes)")):
        for seed, d in ben.groupby("seed"):
            d = d.sort_values("step")
            ax.plot(d["step"], d[col], color=RULE, lw=0.5, alpha=0.9)
        for seed, d in leth.groupby("seed"):
            d = d.sort_values("step")
            ax.plot(d["step"], d[col], color=LETHAL_COL, lw=0.6, alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlim(40, 3.5e5)
        fs.log10_ticks(ax)
        ax.set_ylim(-0.02, 1.02 if col == "frac_heritable" else 0.5)
        fs.tidy(ax, "step", ylab)
        ax.set_gid("allow-clip")
    h = [plt.Line2D([], [], color=LETHAL_COL, lw=0.9, label="zero byte halts the pair (lethal tar, 10 worlds)"),
         plt.Line2D([], [], color=RULE, lw=0.9, label="zero byte is a no-op (benign tar, 20 worlds)")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.42, 1.0), ncol=2, fontsize=5.5, frameon=False, columnspacing=2.0)
    label(axa, "a")
    label(axb, "b")
    # c: the first heritable replicator's step, lethal vs benign
    rng = np.random.default_rng(0)
    for y, d, col, name in ((1, gl, LETHAL_COL, "lethal"), (0, gb, GREY, "benign")):
        t = d["t_rep"].astype(float).values
        t = np.sort(t[np.isfinite(t) & (t > 0)])
        axc.scatter(t, y + rng.uniform(-0.12, 0.12, len(t)), s=8, facecolors="white" if name == "benign" else col, edgecolors=col, lw=0.6, zorder=3)
        med = float(t[int(np.ceil(len(t) / 2)) - 1])   # Kaplan-Meier median (no censoring)
        axc.plot([med, med], [y - 0.3, y + 0.3], color=INK, lw=0.8, zorder=4)
        axc.text(med, y + 0.36, f"median {med:,.0f}", ha="center", va="bottom", fontsize=5, color=INK)
        print(f"  ed13 {name}: n = {len(t)}, median t_rep = {med:,.0f}")
    axc.set_xscale("log")
    axc.set_xlim(60, 3.5e5)
    fs.log10_ticks(axc)
    axc.set_ylim(-0.6, 1.9)
    axc.set_yticks([0, 1], ["benign", "lethal"])
    fs.tidy(axc, "first heritable clone, t_rep (step)")
    label(axc, "c")
    save(fig, os.path.join(out, "ed13"))


# ------------------------------------------------------------------------------------------------------ Extended Data 14
def ed14(out):
    """Mutational scan (results/mutscan/mutscan_tapes.csv): transmissible sites, capacity and robustness, first -> final per world."""
    T = pd.read_csv(os.path.join(R, "mutscan", "mutscan_tapes.csv"))
    Ls = [16, 20, 50, 64]
    fig = plt.figure(figsize=(fs.DOUBLE, 58 * fs.MM))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3, left=0.075, right=0.99, top=0.84, bottom=0.14)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    specs = [("capacity_bits", "single-mutant variation score", (-15, 470), [0, 100, 200, 300, 400], 425),
             ("robustness_h", "fraction of single mutants\nthat remain heritable", (0.4, 1.07), [0.4, 0.6, 0.8, 1.0], 1.025)]
    for ax, (col, ylab, ylim, yt, hy), sd in zip(axes, specs, (1, 2)):
        groups, first, final, lf, ll = [], {}, {}, {}, {}
        for L in Ls:
            f = T[(T.L == L) & (T.which == "first")].set_index("seed")
            n = T[(T.L == L) & (T.which == "final")].set_index("seed")
            idx = [f"{L}:{sd_}" for sd_ in f.index.intersection(n.index)]
            groups.append((f"L = {L}", idx))
            for sd_, key in zip(f.index.intersection(n.index), idx):
                first[key], final[key] = f.loc[sd_, col], n.loc[sd_, col]
                lf[key], ll[key] = bool(f.loc[sd_, "has_loop"]), bool(n.loc[sd_, "has_loop"])
        slope_chart(ax, groups, first, final, lf, ll, ylab, ylim, yt, seed=sd, header_y=hy)
    h = [plt.Line2D([], [], marker="o", ls="none", color=RED, ms=3, label="loop instruction"),
         plt.Line2D([], [], marker="o", ls="none", mfc="white", mec=GREY, ms=3, label="no loop instruction")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.5, frameon=False, columnspacing=2.0)
    for ax, letter in zip(axes, "ab"):
        label(ax, letter)
    save(fig, os.path.join(out, "ed14"))


# ------------------------------------------------------------------------------------------------ v4 Extended Data, new
def ed_census(out):
    """Every two-byte word (results/census2): the 254 heritable words by the culture test and by the partner test."""
    H = pd.read_csv(os.path.join(R, "census2", "heritable.csv"))
    fx = set(pd.read_csv(os.path.join(R, "census2", "fixed_points.csv"))["word"]) - {"00 00"}
    cf = H["control_flow"].astype(str) != "-"
    fig = plt.figure(figsize=(fs.DOUBLE, 62 * fs.MM))
    gs = GridSpec(1, 2, figure=fig, wspace=0.32, left=0.075, right=0.98, top=0.82, bottom=0.16)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(5)
    for ax, (xc, yc, xl, yl) in ((axa, ("score", "gen2", "culture-test score (32 partners)", "heredity of the copies (gen2)")),
                                (axb, ("copied", "damaged", "fraction of 256 partners copied", "fraction of 256 encounters with self-damage"))):
        jx, jy = np.abs(rng.normal(0, 0.006, len(H))), np.abs(rng.normal(0, 0.006, len(H)))   # one-sided, up: no point drawn below its value
        if ax is axb:
            jx = np.where(H[xc].values == 0, 0.0, jx)   # the words that copy no partner stay exactly at 0: jitter only upwards
        ax.scatter((H[xc] + jx)[cf].clip(0, 1), (H[yc] + jy)[cf].clip(0, 1), s=5, color=RED, lw=0, alpha=0.7, label=f"with a call, jump or return ({int(cf.sum())})", zorder=2)
        nf = (~cf) & (~H["word"].isin(fx))
        ax.scatter(H[xc][nf], H[yc][nf], s=10, facecolors="white", edgecolors=INK, lw=0.7, label=f"other straight-line words ({int(nf.sum())})", zorder=3)
        sel = H["word"].isin(fx)
        ax.scatter(H[xc][sel], H[yc][sel], s=14, color=fs.MARK, lw=0, label="the five self-writers: 01 c5, 11 d5, 21 e5, 2a e5, e5 2a", zorder=4)   # slate: pushers, as in Fig. 4
        ax.set_xlim(-0.05, 1.02)
        ax.set_ylim(-0.05, 1.05)
        fs.tidy(ax, xl, yl)
    axa.axhline(0.3, color=RULE, lw=0.5, ls=":")
    axa.text(1.0, 0.31, "heredity threshold", fontsize=5, color=GREY, ha="right", va="bottom")
    axb.axvline(0.95, color=RULE, lw=0.5, ls=":")
    axb.text(0.94, 1.0, "copy threshold 0.95", fontsize=5, color=GREY, ha="right", va="top")
    h, l = axa.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.5, frameon=False, columnspacing=1.6)
    label(axa, "a"); label(axb, "b")
    save(fig, os.path.join(out, "ed_census"))


def ed_confine(out):
    """Pointer confinement against information inflow, and the executed fraction of the tape (results/exectrace)."""
    X = pd.read_csv(os.path.join(R, "exectrace", "per_replicator.csv"))
    fig = plt.figure(figsize=(fs.DOUBLE, 66 * fs.MM))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.8, 1.2], wspace=0.3, left=0.07, right=0.99, top=0.84, bottom=0.14)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(7)
    fi, fl = X[X.which == "first"], X[X.which == "final"]
    lp = fl.has_loop.astype(bool)
    fl_ = fi.has_loop.astype(bool)
    # every replicator fetches a partner byte in no encounter or in all of them (entered_frac is exactly 0 or 1), so the
    # x axis is two categories, each split into first replicators and final dominants; points spread only sideways
    # within their column, and every value is drawn at its exact height
    assert set(np.round(X.entered_frac, 9)) <= {0.0, 1.0}, "entered_frac is no longer binary: redraw ED Fig. 3a"
    XPOS = {(0, "first"): 0.0, (0, "final"): 1.0, (1, "first"): 2.6, (1, "final"): 3.6}
    for d, which, kw, lab in ((fi[~fl_], "first", dict(marker="o", facecolors="white", edgecolors=GREY, lw=0.5, s=8), f"first replicators, no loop instruction ({int((~fl_).sum())})"),
                              (fi[fl_], "first", dict(marker="o", color=RED, lw=0, s=9), f"first replicators with a loop, lethal tar ({int(fl_.sum())})"),
                              (fl[~lp], "final", dict(marker="s", facecolors="white", edgecolors=GREY, lw=0.5, s=8), f"final dominants, no loop instruction ({int((~lp).sum())})"),
                              (fl[lp], "final", dict(marker="s", color=RED, lw=0, s=9), f"final dominants with a loop instruction ({int(lp.sum())})")):
        x = np.array([XPOS[(int(round(e)), which)] for e in d.entered_frac]) + rng.uniform(-0.32, 0.32, len(d))
        axa.scatter(x, d.H_bits, label=lab, zorder=3, **kw)
    for (cat, which), x0 in XPOS.items():                 # counts of coincident points at 0 bits and at the 8-bit ceiling
        d = (fi if which == "first" else fl)
        here = np.round(d.entered_frac) == cat
        n0 = int((here & (d.H_bits.abs() < 1e-9)).sum())
        n8 = int((here & (d.H_bits >= 7.9)).sum())         # 7.9-8 bits: within a marker's height of the ceiling
        if n0 > 1:
            axa.text(x0, 0.45, f"{n0}", ha="center", va="bottom", fontsize=5, color=GREY)
        if n8 > 1:
            axa.text(x0, 8.42, f"{n8}", ha="center", va="bottom", fontsize=5, color=GREY)
    axa.axhline(8.0, color=RULE, lw=0.5, ls=":")
    axa.set_xlim(-0.55, 4.15)
    axa.set_ylim(-0.4, 9.2)
    axa.set_xticks(list(XPOS.values()), ["first", "final", "first", "final"])
    axa.tick_params(axis="x", length=0)
    for x0, t in ((0.5, "0 (never)"), (3.1, "1 (every encounter)")):
        axa.annotate(t, xy=(x0, 0), xycoords=("data", "axes fraction"), xytext=(0, -10.5), textcoords="offset points", ha="center", va="top", fontsize=6, color=INK, gid="allow-outside")
    fs.tidy(axa, None, H_AXIS)
    axa.set_xlabel("fraction of encounters in which the\npointer fetches a partner byte", labelpad=13)
    axa.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)
    X["exec_frac"] = X.exec_union / X.L
    groups, first, final, lf, ll = [], {}, {}, {}, {}
    for st, L, name in (("G", 16, "L = 16"), ("G", 20, "L = 20"), ("K", 32, "L = 32"), ("G", 50, "L = 50"), ("G", 64, "L = 64"), ("I", 16, "L = 16,\nlethal tar")):
        f = X[(X.stage == st) & (X.L == L) & (X.which == "first")].set_index("seed")
        n = X[(X.stage == st) & (X.L == L) & (X.which == "final")].set_index("seed")
        idx = [f"{st}{L}:{sd}" for sd in f.index.intersection(n.index)]
        groups.append((name, idx))
        for sd, k in zip(f.index.intersection(n.index), idx):
            first[k], final[k] = f.loc[sd, "exec_frac"], n.loc[sd, "exec_frac"]
            lf[k], ll[k] = bool(f.loc[sd, "has_loop"]), bool(n.loc[sd, "has_loop"])
    slope_chart(axb, groups, first, final, lf, ll, "fraction of the tape executed\n(union over 256 encounters)", (-0.03, 1.2), [0, 0.25, 0.5, 0.75, 1.0], seed=3, header_y=1.06,
                final_marker="s")
    label(axa, "a"); label(axb, "b")
    save(fig, os.path.join(out, "ed_confine"))


def ed_closure(out):
    """Closure across the stages added in revision: aligned L = 32 (K), ten million steps (L), the 8080 subset (M)."""
    sets = [("stageL", 16, "Z80, L = 16", 1e7), ("stageL", 20, "Z80, L = 20", 1e7), ("stageK", 32, "Z80, L = 32", 1e6),
            ("stageM", 16, "8080, L = 16", 3e5), ("stageM", 32, "8080, L = 32", 1e6)]
    fig = plt.figure(figsize=(fs.DOUBLE, 64 * fs.MM))
    gs = GridSpec(1, 2, figure=fig, wspace=0.28, left=0.07, right=0.99, top=0.8, bottom=0.12)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    for ax, col, ylab, sd in ((axes[0], "copied", "fraction of 256 partners\ncopied (≥ 75%)", 1),
                              (axes[1], "damaged", "fraction of 256 encounters with\nself-damage (≥ 25% lost)", 2)):
        groups, first, final, lf, ll = [], {}, {}, {}, {}
        for st, L, name, hz in sets:
            d = pd.read_csv(os.path.join(R, st, st, "stage_g_runs.csv"))
            d = d[d.L == L].reset_index(drop=True)
            assert (d["horizon"] == hz).all(), f"ED Fig. 4: horizon of {st} L = {L} is not {hz:,.0f}"
            idx = [f"{st}{L}:{sd_}" for sd_ in d.seed]
            groups.append(("", idx))
            lf_, ll_ = loop_flags(d, "first"), loop_flags(d, "final")
            for i, k in enumerate(idx):
                first[k], final[k] = d.loc[i, f"first_{col}"], d.loc[i, f"final_{col}"]
                lf[k], ll[k] = bool(lf_[i]), bool(ll_[i])
        slope_chart(ax, groups, first, final, lf, ll, ylab, (-0.04, 1.2), [0, 0.25, 0.5, 0.75, 1.0], seed=sd, header_y=1.05)
        lo, hi = ax.get_xlim()
        ax.set_xlim(min(lo, -1.0), max(hi, (len(sets) - 1) * 3.0 + 2.0))   # room for the outer headers' horizons
        for gi, (_, L, name, hz) in enumerate(sets):    # the machine and length (6 pt) over the horizon, written out (5 pt)
            ax.text(gi * 3.0 + 0.5, 1.05, f"{fs.steps(hz)} steps", ha="center", va="bottom", fontsize=5, color=INK)
            ax.annotate(name, xy=(gi * 3.0 + 0.5, 1.05), xytext=(0, 6.2), textcoords="offset points", ha="center", va="bottom", fontsize=6, color=INK)
    h = [plt.Line2D([], [], marker="o", ls="none", color=RED, ms=3, label="loop instruction"),
         plt.Line2D([], [], marker="o", ls="none", mfc="white", mec=GREY, ms=3, label="no loop instruction")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.5, frameon=False, columnspacing=2.0)
    label(axes[0], "a"); label(axes[1], "b")
    save(fig, os.path.join(out, "ed_closure"))


def ed_bffinflow(out):
    """Information inflow of first replicators and final dominants in BFF, by variant (results/biology/individuality)."""
    P = pd.read_csv(os.path.join(R, "biology", "individuality", "per_replicator.csv"))
    P = P[P.machine == "bff"]
    names = {"std": "BFF as published", "wrap": "wrapping pointer", "lit": "literal push", "wraplit": "wrap + literal push", "wraplitnh": "wrap + literal push,\nharmless brackets"}
    fig = plt.figure(figsize=(fs.DOUBLE, 62 * fs.MM))
    ax = fig.add_axes([0.07, 0.13, 0.92, 0.72])
    groups, first, final, lf, ll = [], {}, {}, {}, {}
    VARS = ("std", "wrap", "lit", "wraplit", "wraplitnh")
    for v in VARS:
        d = P[P.group.astype(str) == v]
        f, l = d[d.which == "first"].set_index("world"), d[d.which == "final"].set_index("world")
        idx = [f"{v}:{w}" for w in f.index.intersection(l.index)]
        groups.append(("", idx))                       # headers drawn below: name with its Fig. 5 swatch, then the count
        for w, k in zip(f.index.intersection(l.index), idx):
            first[k], final[k] = f.loc[w, "H_bits"], l.loc[w, "H_bits"]
            lf[k], ll[k] = bool(f.loc[w, "has_loop"]), bool(l.loc[w, "has_loop"])
    slope_chart(ax, groups, first, final, lf, ll, H_AXIS, (-0.45, 10.9), [0, 2, 4, 6, 8], seed=4, header_y=8.75,
                count_tol=0.12, count_min=2, count_lines=(8.0, 1.0, 0.5), count_pad=0.22, count_spread=0.15)
    # group headers: the variant's name (as in Fig. 5a) after a short swatch of its Fig. 5 colour, and the number of soups
    heads = []
    for gi, (v, (_, idx)) in enumerate(zip(VARS, groups)):
        ax.text(gi * 3.0 + 0.5, 8.75, f"({len(idx)} soups with a replicator)", ha="center", va="bottom", fontsize=5.5, color=INK)
        heads.append(ax.annotate(names[v], xy=(gi * 3.0 + 0.5, 8.75), xytext=(0, 7.6), textcoords="offset points", ha="center", va="bottom",
                                 fontsize=6, color=INK, linespacing=1.1))
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pt = fig.dpi / 72.0
    for v, t in zip(VARS, heads):
        bb = t.get_window_extent(rend)
        y = bb.y1 - 0.42 * 6 * pt                       # the middle of the first line's lower-case letters
        (x0, y0), (x1, _) = ax.transData.inverted().transform([(bb.x0 - 9.0 * pt, y), (bb.x0 - 1.8 * pt, y)])
        ax.plot([x0, x1], [y0, y0], color=fs.BFF_VARIANT[v], lw=1.8, solid_capstyle="butt", clip_on=False)
    for y, t in ((8.0, "8-bit ceiling"), (1.0, "1 bit"), (0.5, "0.5 bit")):       # every threshold label above its line
        ax.axhline(y, color=RULE, lw=0.5, ls=":", zorder=0)
        ax.text(ax.get_xlim()[1] - 0.05, y + 0.05, t, fontsize=5, color=GREY, va="bottom", ha="right")
    ax.set_gid("allow-clip")
    h = [plt.Line2D([], [], marker="o", ls="none", color=RED, ms=3, label="loop instruction"),
         plt.Line2D([], [], marker="o", ls="none", mfc="white", mec=GREY, ms=3, label="no loop instruction")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.47, 1.0), ncol=2, fontsize=5.5, frameon=False, columnspacing=2.0)
    save(fig, os.path.join(out, "ed_bffinflow"))


# ------------------------------------------------------------------------------------------- the variation figure (v4 Fig. 4)
def figvar(out):
    """The cost of closure and the birth of the genotype: a executed/transmissible position maps of exemplar genomes;
    b the matched-pair invasion (jump-word share vs step); c capacity of the dominant tape over evolutionary time."""
    sys.path.insert(0, EXP)
    from algocell_exp import exectrace as X
    fig = plt.figure(figsize=(fs.DOUBLE, 118 * fs.MM))
    gs_a = GridSpec(1, 1, figure=fig, left=0.2, right=0.84, top=0.9, bottom=0.6)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0], wspace=0.3, left=0.07, right=0.99, top=0.45, bottom=0.1)
    axa = fig.add_subplot(gs_a[0, 0])
    axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    # a: position maps
    gG = pd.read_csv(os.path.join(R, "stageG", "stageG", "stage_g_runs.csv"))
    gI = pd.read_csv(os.path.join(R, "stageI", "stageI", "stage_g_runs.csv"))
    sG = pd.read_csv(os.path.join(R, "mutscan", "mutscan_sites.csv"))
    sI = pd.read_csv(os.path.join(R, "mutscan_I", "mutscan_sites.csv"))
    ex_rows = [("the first replicator (L = 50)", gG, 50, 2001, "first", sG, False),
               ("its successor: the pusher with a jump (L = 50)", gG, 50, 2001, "final", sG, False),
               ("return closer (L = 16)", gG, 16, 2001, "final", sG, False),
               ("block copy with a skipped segment (L = 16)", gG, 16, 2002, "final", sG, False),
               ("born closed under lethal tar (L = 16)", gI, 16, 4009, "final", sI, True)]
    y = 0
    for title, g, L, seed, which, sites, zh in ex_rows:
        w = g[(g.L == L) & (g.seed == seed)].iloc[0]
        tape = np.array([int(b, 16) for b in str(w[f"{which}_tape"]).split()], np.uint8)
        rng = np.random.default_rng([20261010, L, seed])
        Rp = rng.integers(0, 256, size=(64, L), dtype=np.uint8)
        res, masks = X.execute_pairs_traced(np.concatenate([np.repeat(tape[None, :], 64, 0), Rp], 1), L, 128, zero_halts=zh)
        union = X.exec_positions(masks, 2 * L)[:, :L].any(axis=0)
        st = sites[(sites.L == L) & (sites.seed == seed) & (sites.which == which)].sort_values("pos")
        tr = st["transmissible"].astype(bool).values if len(st) == L else np.zeros(L, bool)
        scale = 64.0 / L
        for i in range(L):
            axa.add_patch(Rectangle((i * scale, y), scale, 0.8, facecolor=EXEC_C if union[i] else "#FFFFFF", edgecolor=RULE, lw=0.3))
            if tr[i]:
                axa.plot(i * scale + scale / 2, y + 0.4, "o", color=RED, ms=2.6, mec="white", mew=0.3, zorder=5)
        axa.text(-1.0, y + 0.4, title, ha="right", va="center", fontsize=5.5, color=INK)
        axa.text(65.0, y + 0.4, f"executed {int(union.sum())}/{L} · transmissible {int(tr.sum())}" + (f" ({int((tr & ~union).sum())} unexecuted)" if tr.any() else ""), ha="left", va="center", fontsize=5, color=GREY)
        y += 1.2
    axa.set_xlim(-0.5, 64.5)
    axa.set_ylim(-0.2, y)
    axa.invert_yaxis()
    axa.set_axis_off()
    h = [Rectangle((0, 0), 1, 1, facecolor=EXEC_C, edgecolor=RULE, lw=0.3, label="byte executed (fetched as instruction stream)"),
         Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=RULE, lw=0.3, label="byte never executed"),
         plt.Line2D([], [], marker="o", ls="none", color=RED, ms=3, label="transmissible site: a mutation here is inherited")]
    axa.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=5.5, frameon=False, columnspacing=1.5)
    label(axa, "a", dx=0.185)
    # b: matched-pair invasion
    try:
        d = pd.read_csv(os.path.join(R, "invasion_pair", "invasion_pair.csv"))
        for (res_, inv), col, ls, lab in ((("pusher", "closed"), RED, "-", "closed form seeded at 1% into a pusher world"),
                                          (("closed", "pusher"), GREY, "-", "pusher seeded at 1% into a closed world"),
                                          (("pusher", "none"), INK, ":", "pusher world, no seeding (closure arises by mutation)"),
                                          (("closed", "none"), RULE, ":", "closed world, no seeding")):
            g_ = d[(d.resident == res_) & (d.invader == inv)]
            first = True
            for sd, e in g_.groupby("seed"):
                e = e[e.step > 0].sort_values("step")
                axb.plot(e.step, e.jump_share, color=col, ls=ls, lw=0.8, alpha=0.85, label=lab if first else None)
                first = False
        axb.set_xscale("log")
        axb.set_xlim(8, 2.2e4)
        axb.set_ylim(-0.02, 1.0)
        fs.tidy(axb, "step", "cells carrying the jump word")
        axb.set_gid("allow-clip")
        axb.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)
        label(axb, "b")
    except Exception as e:  # noqa: BLE001
        placeholder(axb, f"b (data missing: {e})")
    # c: capacity of the dominant tape over time
    try:
        C = pd.read_csv(os.path.join(R, "capacity_time", "capacity_over_time.csv"))
        C = C[C.tape != "all-zero"].dropna(subset=["capacity_bits"])
        for L, col in ((16, L_COL[16]), (20, L_COL[20]), (50, L_COL[50]), (64, L_COL[64])):
            d = C[(C.stage == "G") & (C.L == L)]
            if d.empty:
                continue
            med = d.groupby("step").capacity_bits.median()
            lo, hi = d.groupby("step").capacity_bits.quantile(0.25), d.groupby("step").capacity_bits.quantile(0.75)
            axc.plot(med.index, med.values, color=col, lw=0.9, label=f"L = {L}")
            axc.fill_between(med.index, lo.values, hi.values, color=col, alpha=0.15, lw=0)
        axc.set_xscale("log")
        axc.set_xlim(400, 1.2e6)
        fs.tidy(axc, "step", "capacity for inherited variation\nof the dominant tape (bits)")
        axc.set_gid("allow-clip")
        axc.legend(fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
        label(axc, "c")
    except Exception as e:  # noqa: BLE001
        placeholder(axc, f"c (data missing: {e})")
    save(fig, os.path.join(out, "figvar"))


# ------------------------------------------------------------------------------------------------ v4 Fig. 4 and Fig. 5
LETHAL_COL = "#7B3294"      # lethal tar (Z80 Stage I), distinct from vermilion (loop) and the L = 64 orange


def _position_maps(axa):
    """Executed and transmissible positions of five exemplar genomes (fig4v4 a)."""
    sys.path.insert(0, EXP)
    from algocell_exp import exectrace as X
    gG = pd.read_csv(os.path.join(R, "stageG", "stageG", "stage_g_runs.csv"))
    gI = pd.read_csv(os.path.join(R, "stageI", "stageI", "stage_g_runs.csv"))
    sG = pd.read_csv(os.path.join(R, "mutscan", "mutscan_sites.csv"))
    sI = pd.read_csv(os.path.join(R, "mutscan_I", "mutscan_sites.csv"))
    ex_rows = [("first replicator, L = 50", gG, 50, 2001, "first", sG, False),
               ("its closed successor, L = 50", gG, 50, 2001, "final", sG, False),
               ("return closer, L = 16", gG, 16, 2001, "final", sG, False),
               ("block copier, L = 16", gG, 16, 2002, "final", sG, False),
               ("lethal-tar closer, L = 16", gI, 16, 4009, "final", sI, True)]
    y = 0
    for title, g, L, seed, which, sites, zh in ex_rows:
        w = g[(g.L == L) & (g.seed == seed)].iloc[0]
        tape = np.array([int(b, 16) for b in str(w[f"{which}_tape"]).split()], np.uint8)
        rng = np.random.default_rng([20261010, L, seed])
        Rp = rng.integers(0, 256, size=(64, L), dtype=np.uint8)
        res, masks = X.execute_pairs_traced(np.concatenate([np.repeat(tape[None, :], 64, 0), Rp], 1), L, 128, zero_halts=zh)
        union = X.exec_positions(masks, 2 * L)[:, :L].any(axis=0)
        st = sites[(sites.L == L) & (sites.seed == seed) & (sites.which == which)].sort_values("pos")
        tr = st["transmissible"].astype(bool).values if len(st) == L else np.zeros(L, bool)
        scale = 64.0 / L
        for k in range(L):
            axa.add_patch(Rectangle((k * scale, y), scale, 0.8, facecolor=EXEC_C if union[k] else "#FFFFFF", edgecolor=RULE, lw=0.3))
            if tr[k]:
                axa.plot(k * scale + scale / 2, y + 0.4, "o", color=RED, ms=2.6, mec="white", mew=0.3, zorder=5)
        axa.text(-1.0, y + 0.4, title, ha="right", va="center", fontsize=5.5, color=INK)
        n_tr, n_un = int(tr.sum()), int((tr & ~union).sum())
        note = f"{int(union.sum())}/{L} executed · {n_tr} site{'' if n_tr == 1 else 's'}" + (f" ({n_un} unexecuted)" if n_un else "")
        axa.text(65.0, y + 0.4, note, ha="left", va="center", fontsize=5.5, color=GREY)
        y += 1.2
    axa.set_xlim(-0.5, 64.5)
    axa.set_ylim(-0.2, y)
    axa.invert_yaxis()
    axa.set_axis_off()
    h = [Rectangle((0, 0), 1, 1, facecolor=EXEC_C, edgecolor=RULE, lw=0.3, label="byte executed (fetched as instruction stream)"),
         Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=RULE, lw=0.3, label="byte never executed"),
         plt.Line2D([], [], marker="o", ls="none", color=RED, ms=3, label="transmissible site")]
    axa.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=5.5, frameon=False, columnspacing=1.5)


def _sites_slope(ax):
    """Transmissible sites, first replicator -> final dominant, per Stage G world."""
    T = pd.read_csv(os.path.join(R, "mutscan", "mutscan_tapes.csv"))
    groups, first, final, lf, ll = [], {}, {}, {}, {}
    for L in (16, 20, 50, 64):
        f = T[(T.L == L) & (T.which == "first")].set_index("seed")
        n = T[(T.L == L) & (T.which == "final")].set_index("seed")
        idx = [f"{L}:{sd_}" for sd_ in f.index.intersection(n.index)]
        groups.append((f"L = {L}", idx))
        for sd_, key in zip(f.index.intersection(n.index), idx):
            first[key], final[key] = f.loc[sd_, "n_sites"], n.loc[sd_, "n_sites"]
            lf[key], ll[key] = bool(f.loc[sd_, "has_loop"]), bool(n.loc[sd_, "has_loop"])
    slope_chart(ax, groups, first, final, lf, ll, "transmissible sites (positions)", (-1.5, 42.0), [0, 10, 20, 30], seed=0, header_y=38.5)
    h = [plt.Line2D([], [], marker="o", ls="none", color=RED, ms=3, label="loop instruction"),
         plt.Line2D([], [], marker="o", ls="none", mfc="white", mec=GREY, ms=3, label="no loop instruction")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=5.5, frameon=False)


def _invasion_panel(ax):
    d = pd.read_csv(os.path.join(R, "invasion_pair", "invasion_pair.csv"))
    for (res_, inv), col, ls, lw, lab in ((("pusher", "closed"), RED, "-", 0.8, "closed form seeded at 1% into an ancestor world"),
                                          (("closed", "pusher"), GREY, "-", 0.8, "ancestor seeded at 1% into a closed world"),
                                          (("pusher", "none"), INK, ":", 0.9, "ancestor world, no seeding"),
                                          (("closed", "none"), "#2B8CBE", (0, (3, 2)), 0.9, "closed world, no seeding")):
        g_ = d[(d.resident == res_) & (d.invader == inv)]
        first = True
        for sd, e in g_.groupby("seed"):
            e = e[e.step > 0].sort_values("step")
            ax.plot(e.step, e.jump_share, color=col, ls=ls, lw=lw, alpha=0.9, label=lab if first else None, zorder=3 if inv == "none" else 2)
            first = False
    ax.set_xscale("log")
    ax.set_xlim(8, 2.2e4)
    ax.set_ylim(-0.02, 1.0)
    fs.tidy(ax, "step", "fraction of cells carrying 20 f0")
    ax.set_gid("allow-clip")
    ax.legend(fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)


def _capacity_panel(ax):
    """Capacity for inherited variation of the most common tape against step: Stage G (L = 50, 64), K (L = 32), I (lethal
    tar) and L (L = 16, 20 to ten million steps). Medians only at steps where every world of the series has a heritable
    most common tape on record (snapshots taken in a few worlds only are dropped); thin lines, the individual Stage L worlds."""
    frames = [pd.read_csv(p) for p in (os.path.join(R, "capacity_time", "capacity_over_time.csv"), os.path.join(R, "capacity_time_L", "capacity_over_time.csv")) if os.path.exists(p)]
    C = pd.concat(frames, ignore_index=True).drop_duplicates(["stage", "L", "seed", "step"])
    C = C[(C.tape != "all-zero") & C.ctrl_herit.fillna(False).astype(bool)].dropna(subset=["capacity_bits"])
    series = [("L", 16, L_COL[16], "-", "L = 16"), ("L", 20, L_COL[20], "-", "L = 20"), ("K", 32, "#009E73", "-", "L = 32"),
              ("G", 50, L_COL[50], "-", "L = 50"), ("G", 64, L_COL[64], "-", "L = 64"), ("I", 16, LETHAL_COL, "--", "L = 16, lethal tar")]
    for stage, L, col, ls, lab in series:
        d = C[(C.stage == stage) & (C.L == L)]
        if d.empty:
            continue
        n_worlds = d.seed.nunique()
        full = d.groupby("step").seed.nunique()
        d = d[d.step.isin(full[full >= max(1, int(np.ceil(0.8 * n_worlds)))].index)]
        if stage == "L":
            for _, w in d.groupby("seed"):
                w = w.sort_values("step")
                ax.plot(w.step, w.capacity_bits, color=col, lw=0.35, alpha=0.35, zorder=1)
        g = d.groupby("step")
        med = g.capacity_bits.median()
        closed = g.entered_frac.apply(lambda v: float((v < 0.5).mean())) >= 0.5
        ax.plot(med.index, med.values, color=col, ls=ls, lw=0.9, label=lab, zorder=3)
        ax.scatter([med.index[0]], [med.values[0]], s=9, facecolors="white", edgecolors=col, lw=0.7, zorder=4)
        if closed.any():
            k = int(np.argmax(closed.values))
            ax.scatter([med.index[k]], [med.values[k]], s=11, color=col, zorder=5, lw=0)
    ax.set_xscale("log")
    ax.set_xlim(300, 1.3e7)
    fs.tidy(ax, "step", "capacity for inherited variation\nof the most common tape (bits)")
    ax.set_gid("allow-clip")
    ax.legend(fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, columnspacing=1.2)


def _row_letters(fig, axes_letters, y):
    for ax, letter in axes_letters:
        ax.apply_aspect()
        fig.text(ax.get_position().x0 - 0.052, y, letter, fontsize=8, fontweight="bold", va="bottom", ha="left", gid="panel-label")


def fig4v4(out):
    """v4 Fig. 4 | Closure wins the competition and inherits nothing."""
    fig = plt.figure(figsize=(fs.DOUBLE, 132 * fs.MM))
    gs_a = GridSpec(1, 1, figure=fig, left=0.245, right=0.765, top=0.93, bottom=0.715)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.9, 1.0, 1.1], wspace=0.5, left=0.065, right=0.99, top=0.505, bottom=0.075)
    axa = fig.add_subplot(gs_a[0, 0])
    axb, axc, axd = (fig.add_subplot(gs[0, k]) for k in range(3))
    for ax, fn, letter in ((axa, _position_maps, "a"), (axb, _sites_slope, "b"), (axc, _invasion_panel, "c"), (axd, _capacity_panel, "d")):
        try:
            fn(ax)
        except Exception as e:  # noqa: BLE001
            placeholder(ax, f"{letter} (data missing: {e})")
    fig.text(0.01, 0.975, "a", fontsize=8, fontweight="bold", va="top", ha="left", gid="panel-label")
    _row_letters(fig, ((axb, "b"), (axc, "c"), (axd, "d")), 0.635)
    save(fig, os.path.join(out, "fig4v4"))


# ------------------------------------------------------------------------------------------------ v5 Fig. 4 (round 2)
OPEN_C, REGEN_C, TRANS_C, INTER_C, J50_C = fs.MARK, RED, TEAL, "#E69F00", fs.LEN_COLOR[50]   # the L = 50 jump genome: the L = 50 blue
# Fig. 4's neutrals differ by panel, so that no grey means two things (slate is the open class of d and the pushers of b):
LOST_C, ERASED_C = "#A39D95", "#E3E0DB"   # a: lineage lost (warm mid grey), alive with the allele erased (warm light grey)
EXEC_C = "#C6C6C6"   # c: byte executed (pure neutral, light, under the dark-teal site markers)
SITE_C, SITE_MS = "#0B4F4F", 3.6                  # Fig. 4c sites: dark teal fill (carried) against white fill, dark rim (first copy)
SITE_COLS = (70.0, 79.0, 88.0, 97.5)             # Fig. 4c count columns, in strip units right of the 64-unit strips
# one name per group in a and b: regenerators and transmitters as classed in d (at most two, five or more sites)
SR_ROWS = [("open first replicators", OPEN_C, [("pusher16", "pusher, L = 16"), ("pusher50", "pusher, L = 50"), ("pusher64", "pusher, L = 64"), ("pusher32_8080", "pusher, 8080 subset, L = 32")]),
           ("evolved regenerators", REGEN_C, [("ret16", "return closer, L = 16"), ("ldir20", "block-copy tiling, L = 20"), ("ldir32", "block-copy tiling, L = 32")]),
           ("L = 50 jump genome", J50_C, [("jr50", "pusher with three jumps, L = 50")]),
           ("evolved transmitters", TRANS_C, [("lethal16_s4009", "lethal-tar closer, L = 16"), ("genome16_s6006", "transient genome, L = 16"), ("genome20_s6003", "transient genome, L = 20 (world 3)"), ("genome20_s6004", "transient genome, L = 20 (world 4)")]),
           ("constructed transmitter", TRANS_C, [("closer32_8080_p1", "8080 closer + payload 1, L = 32"), ("closer32_8080_p2", "8080 closer + payload 2, L = 32")])]


def _sr_summary():
    return pd.read_csv(os.path.join(R, "serial_retention", "sr_summary.csv")).set_index("key")


def _sr_partition(ax):
    """Fate of every single-byte allele after eight serial transfers into fresh random partners (all lineages counted)."""
    S = _sr_summary()
    y, yt, yl = 0.0, [], []
    for gi, (gname, gcol, rows) in enumerate(SR_ROWS):
        if gi:
            y += 0.6                     # each header sits with its own group: 0.6 of a row more above it than below it
        ax.text(-0.02, y + 0.02, gname, ha="right", va="bottom", fontsize=5.5, color=gcol, fontweight="bold", transform=ax.get_yaxis_transform(), gid="allow-outside")
        y += 0.5
        for key, lab in rows:
            r = S.loc[key]
            x0 = 0.0
            for v, c in ((r.lost_g8, LOST_C), (r.erased_g8, ERASED_C), (r.retained_g8, TRANS_C)):
                ax.barh(y, v, left=x0, height=0.72, color=c, lw=0)
                x0 += v
            ax.text(1.02, y, f"{r.ctrl_alive_g8:.2f}", ha="left", va="center", fontsize=5.5, color=GREY, transform=ax.get_yaxis_transform(), gid="allow-outside")
            yt.append(y)
            yl.append(lab.strip())
            y += 1.0
        y += 0.35
    ax.set_yticks(yt, yl, fontsize=5.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y - 0.6, -0.6)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1.00"])
    fs.tidy(ax, "share of mutant lineages after eight transfers", None)
    ax.spines["left"].set_visible(False)
    ax.text(1.02, -0.6, "unmutated\nlineages alive", ha="left", va="bottom", fontsize=5.0, color=GREY, transform=ax.get_yaxis_transform(), gid="allow-outside")
    h = [Rectangle((0, 0), 1, 1, color=LOST_C, label="lineage lost"), Rectangle((0, 0), 1, 1, color=ERASED_C, label="alive, allele erased"),
         Rectangle((0, 0), 1, 1, color=TRANS_C, label="alive, allele carried")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, fontsize=5.5, frameon=False, columnspacing=1.2, handlelength=1.0)


def _sr_curves(ax):
    """Share of all mutant lineages carrying the allele against the number of transfers."""
    S = _sr_summary()
    gx = np.array([1, 2, 4, 8])
    # near zero the pushers, the L = 50 jump genome and the regenerators coincide: each class is dodged sideways by a
    # fixed fraction of an octave and the regenerators are drawn on top
    dodge = {OPEN_C: -0.07, J50_C: 0.0, REGEN_C: 0.07, TRANS_C: 0.0}
    zord = {TRANS_C: 2, OPEN_C: 3, J50_C: 4, REGEN_C: 5}
    for gname, gcol, rows in SR_ROWS:
        for key, lab in rows:
            if key not in S.index:
                continue
            r = S.loc[key]
            col = J50_C if key == "jr50" else gcol
            ax.plot(gx * 2 ** dodge[col], [r[f"retained_g{g}"] for g in gx], color=col, lw=0.8, marker="o", ms=2.2, mew=0, alpha=0.9, zorder=zord[col])
    ax.set_xscale("log", base=2)
    ax.set_xticks(gx, [str(g) for g in gx])
    ax.minorticks_off()
    ax.set_xlim(0.85, 9.5)
    ax.set_ylim(-0.02, 0.8)
    fs.tidy(ax, "serial transfers", "share of mutant lineages\ncarrying the allele")
    for txt, yy, col in (("transmitters, evolved and constructed", 0.72, TRANS_C), ("pushers (L = 16, 50, 64; 8080, L = 32)", 0.33, OPEN_C),
                         ("pusher with three jumps, L = 50", 0.26, J50_C), ("evolved regenerators", 0.19, REGEN_C)):
        ax.text(9.3, yy, txt, ha="right", va="center", fontsize=5.0, color=col)


def _position_maps_sr(ax):
    """Executed bytes and serially transmissible sites of six genomes (S): filled dot, the allele is still carried after
    eight transfers by at least half of its values; open dot, only into first-generation copies."""
    S = _sr_summary()
    st = pd.read_csv(os.path.join(R, "serial_retention", "sr_sites.csv"))
    rows = [("pusher50", "pusher, L = 50"), ("ret16", "return closer, L = 16"), ("ldir32", "block-copy tiling, L = 32"),
            ("lethal16_s4009", "lethal-tar closer, L = 16"), ("genome20_s6004", "transient genome, L = 20 (world 4)"), ("closer32_8080_p1", "constructed 8080 closer, L = 32")]
    y = 0
    for key, title in rows:
        r = S.loc[key]
        L = int(r.L)
        ex = np.array([c == "1" for c in str(r.executed).zfill(L)])
        s1 = st[(st.key == key) & (st.gen == 1)].set_index("pos").transmissible.reindex(range(L)).fillna(False).astype(bool).values
        s8 = st[(st.key == key) & (st.gen == 8)].set_index("pos").transmissible.reindex(range(L)).fillna(False).astype(bool).values
        scale = 64.0 / L
        for k in range(L):
            ax.add_patch(Rectangle((k * scale, y), scale, 0.8, facecolor=EXEC_C if ex[k] else "#FFFFFF", edgecolor=RULE, lw=0.3))
            if s8[k]:
                ax.plot(k * scale + scale / 2, y + 0.4, "o", ms=SITE_MS, mfc=SITE_C, mec=SITE_C, mew=0.5, zorder=5)
            elif s1[k]:
                ax.plot(k * scale + scale / 2, y + 0.4, "o", ms=SITE_MS, mfc="white", mec=SITE_C, mew=0.9, zorder=5)
        ax.text(-1.0, y + 0.4, title, ha="right", va="center", fontsize=5.5, color=INK)
        n1, n8, nu = int(s1.sum()), int(s8.sum()), int((s8 & ~ex).sum())
        for xc, v in zip(SITE_COLS, (f"{int(ex.sum())}/{L}", n1, n8, nu)):
            ax.text(xc, y + 0.4, f"{v}", ha="center", va="center", fontsize=5.5, color=INK, gid="allow-outside")
        y += 1.2
    for xc, head in zip(SITE_COLS, ("bytes\nexecuted", "sites after\n1 transfer", "sites after\n8 transfers", "of these, never\nexecuted")):
        ax.text(xc, -0.3, head, ha="center", va="bottom", fontsize=5.0, color=GREY, linespacing=1.1, gid="allow-outside")
    ax.set_xlim(-0.5, 64.5)
    ax.set_ylim(-0.2, y)
    ax.invert_yaxis()
    ax.set_axis_off()
    h = [Rectangle((0, 0), 1, 1, facecolor=EXEC_C, edgecolor=RULE, lw=0.3, label="byte executed"),
         Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=RULE, lw=0.3, label="byte never executed"),
         plt.Line2D([], [], marker="o", ls="none", ms=SITE_MS, mfc="white", mec=SITE_C, mew=0.9, label="site, first copy only"),
         plt.Line2D([], [], marker="o", ls="none", ms=SITE_MS, mfc=SITE_C, mec=SITE_C, mew=0.5, label="site, carried through eight transfers")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.5, frameon=False, columnspacing=1.4)



CLASS_PARTS = (("not heritable", "#ECEDEF"), ("open", "#6F7C8B"), ("closed, ≤ 2 sites", REGEN_C), ("closed, 3–4 sites", INTER_C), ("closed, ≥ 5 sites", TRANS_C))


def _class_bars_h(ax, rows, xlabel):
    """Horizontal stacked bars of cell classes. rows: (label, frame of per-world or per-soup class shares) or None for a gap."""
    y, yt, yl = 0.0, [], []
    for row in rows:
        if row is None:
            y += 0.5
            continue
        lab, d = row
        h = d.frac_heritable
        vals = (1 - h, h * d.frac_open_of_heritable.fillna(0), h * d.frac_regenerator_of_heritable.fillna(0),
                h * d.frac_intermediate_of_heritable.fillna(0), h * d.frac_transmitter_of_heritable.fillna(0))
        x0 = 0.0
        for v, (_, c) in zip(vals, CLASS_PARTS):
            m = float(v.mean())
            ax.barh(y, m, left=x0, height=0.72, color=c, lw=0)
            x0 += m
        ax.text(1.02, y, f"{len(d)}", ha="left", va="center", fontsize=5.0, color=GREY, transform=ax.get_yaxis_transform(), gid="allow-outside")
        yt.append(y)
        yl.append(lab)
        y += 1.0
    ax.set_yticks(yt, yl, fontsize=5.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y - 0.4, -0.6)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1.0"])
    fs.tidy(ax, xlabel, None)
    ax.spines["left"].set_visible(False)
    h = [Rectangle((0, 0), 1, 1, color=c, label=n) for n, c in CLASS_PARTS]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=5.5, frameon=False, columnspacing=1.0, handlelength=1.0)

_S3, _S6, _S7 = fs.steps(3e5), fs.steps(1e6), fs.steps(1e7)       # horizons written out, as in the text
POP_ROWS = [("G", 16, f"Z80, L = 16, {_S3} steps"), ("G", 20, f"L = 20, {_S6}"), ("K", 32, f"L = 32, {_S6}"), ("G", 50, f"L = 50, {_S3}"), ("G", 64, f"L = 64, {_S6}"), None,
            ("L", 16, f"L = 16, {_S7}"), ("L", 20, f"L = 20, {_S7}"), None, ("I", 16, f"L = 16, lethal tar, {_S3}"), None, ("M", 16, f"8080 subset, L = 16, {_S3}"), ("M", 32, f"L = 32, {_S6}")]


def _pop_classes(ax):
    """Composition of 64 random cells at the final snapshot, mean over worlds (Q2)."""
    S = pd.read_csv(os.path.join(R, "population", "classes_snapshots.csv"))
    F = S[S.snapshot == "final"]
    rows = []
    for item in POP_ROWS:
        if item is None:
            rows.append(None)
            continue
        st, L, lab = item
        d = F[(F.stage == st) & (F.L == L)]
        if len(d):
            rows.append((lab, d))
    _class_bars_h(ax, rows, "share of random cells at the last snapshot")


def _pop_time(ax):
    """Share of heritable cells that are closed transmitters, against step: means over worlds (Q2). Each line is named in
    the panel by its length, stage and number of worlds; L = 32 (Stage K) and L = 50 (Stage G) are left out (printed)."""
    S = pd.read_csv(os.path.join(R, "population", "classes_snapshots.csv"))
    S = S[S.step.notna() & (S.snapshot != "emergence") & (S.frac_heritable >= 0.2)]

    def series(st, L):
        d = S[(S.stage == st) & (S.L == L)]
        nw = d.seed.nunique()
        full = d.groupby("step").seed.nunique()
        d = d[d.step.isin(full[full >= max(1, int(np.ceil(0.8 * nw)))].index)]
        return d.groupby("step").frac_transmitter_of_heritable.mean(), nw
    for st, L, col, ls, lab in (("L", 16, L_COL[16], "-", "L = 16, Stage L"), ("L", 20, L_COL[20], "-", "L = 20, Stage L"),
                                ("G", 64, L_COL[64], "-", "L = 64, Stage G"), ("I", 16, LETHAL_COL, "--", "L = 16, lethal tar, Stage I")):
        m, nw = series(st, L)
        if len(m):
            ax.plot(m.index, m.values, color=col, ls=ls, lw=0.9, label=f"{lab}, {nw} worlds")
    for st, L in (("K", 32), ("G", 50)):
        m, nw = series(st, L)
        print(f"  fig4e: not drawn, Stage {st} L = {L} ({nw} worlds): mean share at most {m.max():.3f} over {len(m)} snapshots")
    ax.set_xscale("log")
    first = min(float(np.min(l.get_xdata())) for l in ax.lines if len(l.get_xdata()))
    assert first >= 1e3, f"Fig. 4e: a snapshot at {first} steps falls left of the trimmed axis"
    ax.set_xlim(1e3, 1.3e7)                 # the first snapshot drawn is at 2,000 steps: no empty decade
    fs.log10_ticks(ax)
    ax.set_ylim(-0.02, 1.0)
    fs.tidy(ax, "step", "share of heritable cells that are\nclosed with ≥ 5 sites (mean over worlds)")
    ax.set_gid("allow-clip")
    ax.legend(fontsize=5.0, loc="center", bbox_to_anchor=(0.5, 0.53), ncol=1, frameon=False, handlelength=2.2)


def fig4v5(out):
    """v5 Fig. 4 | Closure regenerates or transmits."""
    fig = plt.figure(figsize=(fs.DOUBLE, 168 * fs.MM))
    gs_top = GridSpec(1, 2, figure=fig, width_ratios=[1.35, 1.0], wspace=0.55, left=0.215, right=0.985, top=0.935, bottom=0.62)
    gs_mid = GridSpec(1, 1, figure=fig, left=0.215, right=0.69, top=0.535, bottom=0.36)
    gs_bot = GridSpec(1, 2, figure=fig, width_ratios=[1.35, 1.0], wspace=0.55, left=0.215, right=0.985, top=0.27, bottom=0.06)
    axa, axb = fig.add_subplot(gs_top[0, 0]), fig.add_subplot(gs_top[0, 1])
    axc = fig.add_subplot(gs_mid[0, 0])
    axd, axe = fig.add_subplot(gs_bot[0, 0]), fig.add_subplot(gs_bot[0, 1])
    for ax, fn, letter in ((axa, _sr_partition, "a"), (axb, _sr_curves, "b"), (axc, _position_maps_sr, "c"), (axd, _pop_classes, "d"), (axe, _pop_time, "e")):
        try:
            fn(ax)
        except Exception as e:  # noqa: BLE001
            placeholder(ax, f"{letter} (data missing: {e})")
    for ax, letter, x, y in ((axa, "a", 0.01, 0.975), (axb, "b", None, 0.975), (axc, "c", 0.01, 0.585), (axd, "d", 0.01, 0.33), (axe, "e", None, 0.33)):
        if x is None:
            ax.apply_aspect()
            x = ax.get_position().x0 - 0.075
        fig.text(x, y, letter, fontsize=8, fontweight="bold", va="top", ha="left", gid="panel-label")
    save(fig, os.path.join(out, "fig4v5"))
    w, h = _pdf_size_pt(os.path.join(out, "fig4v5.pdf"))
    print(f"  fig4v5 page {w * 25.4 / 72:.1f} x {h * 25.4 / 72:.1f} mm")


def _marker_vs_confined(ax):
    """V: share of cells carrying the jump word against the confined share of 512 traced cells, closed form seeded at 1%
    into an ancestor world at L = 50 (three soups)."""
    M = pd.read_csv(os.path.join(R, "marker_check", "marker_check.csv"))
    for sd, d in M.groupby("seed"):
        d = d[d.step > 0].sort_values("step")
        ax.plot(d.step, d.marker_share_soup, color=J50_C, lw=0.8, marker="o", ms=2.0, mew=0, label="cells carrying the jump word 20 f0" if sd == 1 else None)
        ax.plot(d.step, d.confined_share, color=REGEN_C, lw=0.8, marker="o", ms=2.0, mew=0, label="cells confined in 16 of 16 encounters" if sd == 1 else None)
    ax.set_xscale("log")
    ax.set_xlim(40, 1200)
    fs.log10_ticks(ax)
    ax.set_ylim(-0.02, 1.0)
    fs.tidy(ax, "step", "share of cells")
    ax.legend(fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)


def _a1_curves(ax):
    """A1: the constructed 8080 closer seeded at 1% into a world of the 8080 pusher (L = 32): share of cells carrying the
    closer's loop body (post hoc) and share within Hamming 8 of the pusher, five soups; unseeded pusher worlds dotted."""
    D = pd.read_csv(os.path.join(R, "invasion_closer", "A1.csv"))
    first = True
    for sd, d in D[(D.resident == "pusher") & (D.invader == "closer")].groupby("seed"):
        d = d[d.step > 0].sort_values("step")
        ax.plot(d.step, d.loop_share_posthoc, color=TRANS_C, lw=0.8, label="closer's loop body, closer seeded" if first else None)
        ax.plot(d.step, d.pusher_class_share, color=OPEN_C, lw=0.8, label="pusher class, closer seeded" if first else None)
        first = False
    first = True
    for sd, d in D[(D.resident == "pusher") & (D.invader == "none")].groupby("seed"):
        d = d[d.step > 0].sort_values("step")
        ax.plot(d.step, d.pusher_class_share, color=OPEN_C, lw=0.8, ls=":", label="pusher class, no seeding" if first else None)
        first = False
    ax.set_xscale("log")
    ax.set_xlim(8, 2.2e4)
    fs.log10_ticks(ax)
    ax.set_ylim(-0.02, 1.02)
    fs.tidy(ax, "step", "share of cells")
    ax.legend(fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)


INV_ROWS = [("A1", "A1_pusher_none", "8080 subset: pusher world"), ("A1", "A1_pusher_closer", "constructed closer seeded into it"), ("A1", "A1_closer_none", "constructed-closer world"),
            ("A1", "A1_closer_pusher", "pusher seeded into it"), None, ("A2", "A2_ldir_none", "Z80: block-copy world"), ("A2", "A2_ldir_closer", "constructed closer seeded into it"),
            ("A2", "A2_closer_none", "constructed-closer world"), ("A2", "A2_closer_ldir", "block copier seeded into it")]


def _inv_classes(ax):
    """Composition of 64 random cells at the last step of every invasion soup (A1, 20,000 steps; A2, 50,000), mean over
    soups; classes as in Fig. 4d."""
    rows = []
    for item in INV_ROWS:
        if item is None:
            rows.append(None)
            continue
        tag, pre, lab = item
        C = pd.read_csv(os.path.join(R, "invasion_closer", f"{tag}_classes.csv"))
        rows.append((lab, C[C.file.str.startswith(pre + "_s")]))
    _class_bars_h(ax, rows, "share of random cells at the last step")


def ed_invasions(out):
    """Extended Data: three invasion tests of round 2 (V, A1, A2)."""
    fig = plt.figure(figsize=(fs.DOUBLE, 112 * fs.MM))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35, left=0.08, right=0.985, top=0.86, bottom=0.6)
    gs2 = GridSpec(1, 1, figure=fig, left=0.33, right=0.80, top=0.40, bottom=0.08)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs2[0, 0])
    for ax, fn, letter in ((axa, _marker_vs_confined, "a"), (axb, _a1_curves, "b"), (axc, _inv_classes, "c")):
        try:
            fn(ax)
        except Exception as e:  # noqa: BLE001
            placeholder(ax, f"{letter} (data missing: {e})")
    for ax, letter, y in ((axa, "a", 0.985), (axb, "b", 0.985), (axc, "c", 0.53)):
        ax.apply_aspect()
        fig.text(0.01 if letter == "c" else max(0.005, ax.get_position().x0 - 0.07), y, letter, fontsize=8, fontweight="bold", va="top", ha="left", gid="panel-label")
    save(fig, os.path.join(out, "ed_invasions"))


BFF_NAMES = {"std": "BFF as published", "wrap": "wrapping pointer", "lit": "literal push", "wraplit": "wrap + literal push",
             "wraplitnh": "wrap + literal push,\nharmless brackets"}
P_COLS = {0.0: fs.BFF_VARIANT["wraplitnh"], 0.01: "#BCC3CA", 0.03: "#969FAA", 0.1: "#6F7A87", 0.3: "#3E4753", 1.0: fs.BFF_VARIANT["wraplit"]}
EPOCH_TICKS = ([64, 256, 1024, 4096, 16384], ["64", "256", "1,024", "4,096", "16,384"])


def _epoch_axis(ax, ylab):
    from matplotlib import ticker as mticker
    ax.set_xscale("log")
    ax.set_xticks(*EPOCH_TICKS)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(56, 19000)
    ax.set_ylim(-0.02, 1.02)
    fs.tidy(ax, "epoch", ylab)


def _samples(path, fn):
    S = [json.loads(l) for l in open(path)]
    return pd.DataFrame([{"epoch": s["epoch"], "v": fn(s)} for s in S if s["epoch"] > 0])


def _dial_lethality(ax):
    """Share of the all-P tape against epoch by the probability p that an unmatched bracket halts (wrap + literal push)."""
    bdir = os.path.join(EXP, "runs", "bff_modal")
    runs = pd.read_csv(os.path.join(R, "bff", "runs.csv"))
    series = [(0.0, os.path.join(bdir, "bff", r, "samples.jsonl")) for r in runs[runs.variant == "wraplitnh"].run]
    series += [(1.0, os.path.join(bdir, "bff", r, "samples.jsonl")) for r in runs[runs.variant == "wraplit"].run]
    for pv in (0.01, 0.03, 0.1, 0.3):
        series += [(pv, q) for q in sorted(glob.glob(os.path.join(bdir, f"bff_dial_hp{pv}", "**", "samples.jsonl"), recursive=True))]
    n_by_p = {}
    for pv, q in series:
        if not os.path.exists(q):
            continue
        S = _samples(q, _allp_share)
        ax.plot(S.epoch, S.v, color=P_COLS[pv], lw=0.6, zorder=4 if pv == 0.0 else 2)    # p = 0 on top of the greys near 1
        n_by_p[pv] = n_by_p.get(pv, 0) + 1
    for pv in sorted(n_by_p):
        lab = {0.0: "p = 0 (harmless brackets)", 1.0: "p = 1 (published bracket rule)"}.get(pv, f"p = {pv:g}")
        ax.plot([], [], color=P_COLS[pv], lw=0.9, label=f"{lab}, n = {n_by_p[pv]}")
    _epoch_axis(ax, "share of the all-P tape in the soup")
    ax.text(0.99, 0.55, "every soup: wrap + literal push", transform=ax.transAxes, ha="right", va="center", fontsize=5.5, color=GREY)
    ax.legend(fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, columnspacing=1.2)


def _dial_write_ratio(ax):
    """Information inflow of the first replicator by the literal's write ratio r (literal push, no wrapping pointer)."""
    D2 = pd.read_csv(os.path.join(R, "bff_dials", "d2_bandwidth.csv"))
    D2 = D2[D2["transition"].astype(bool)]
    for r_, d in D2.groupby("lit_rep"):
        d = d.sort_values("H_bits")
        x = r_ + (np.linspace(-0.3, 0.3, len(d)) if len(d) > 1 else np.zeros(1))   # evenly spread sideways: no point hides another
        loop = d["first_loop"].astype(bool).values
        ax.scatter(x[~loop], d["H_bits"].values[~loop], s=9, facecolors="white", edgecolors=GREY, lw=0.7)
        ax.scatter(x[loop], d["H_bits"].values[loop], s=9, color=RED, lw=0)
        ax.text(r_, 8.45, f"n = {len(d)}", ha="center", va="bottom", fontsize=5.5, color=GREY)
    ax.axhline(8.0, color=RULE, lw=0.5, ls=":")
    ax.set_xticks([1, 2, 3], ["1", "2", "3"])
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(-0.4, 9.3)
    ax.set_yticks([0, 2, 4, 6, 8])
    fs.tidy(ax, "copies of its word the literal writes per execution, r", H_AXIS)


def fig5v4(out):
    """v4 Fig. 5 | Two properties of the substrate decide the beginning."""
    runs = pd.read_csv(os.path.join(R, "bff", "runs.csv"))
    runs["variant"] = runs["variant"].replace({"stdlit": "lit"})
    bdir = os.path.join(EXP, "runs", "bff_modal", "bff")
    variants = [v for v in ["std", "wrap", "lit", "wraplit", "wraplitnh"] if v in set(runs["variant"])]
    fig = plt.figure(figsize=(fs.DOUBLE, 172 * fs.MM))
    gsA = GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1.0], wspace=0.32, left=0.07, right=0.99, top=0.965, bottom=0.77)
    gsD = GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1.0], wspace=0.32, left=0.07, right=0.99, top=0.565, bottom=0.37)
    gsE = GridSpec(1, 1, figure=fig, left=0.065, right=0.99, top=0.29, bottom=0.01)
    axa, axb = fig.add_subplot(gsA[0, 0]), fig.add_subplot(gsA[0, 1])
    axc, axd = fig.add_subplot(gsD[0, 0]), fig.add_subplot(gsD[0, 1])
    axe = fig.add_subplot(gsE[0, 0])
    zorder = {"std": 6, "wrap": 2, "lit": 3, "wraplit": 4, "wraplitnh": 5}     # as published on top: its 9 transitions stay visible
    for v in variants:
        for run in runs[runs["variant"] == v]["run"]:
            q = os.path.join(bdir, run, "samples.jsonl")
            if not os.path.exists(q):
                continue
            S = _samples(q, lambda s: s["frac_heritable"])
            axa.plot(S.epoch, S.v.rolling(4, min_periods=1).mean(), color=fs.BFF_VARIANT[v], lw=0.6, alpha=0.75 if v != "std" else 0.95, zorder=zorder[v])
        axa.plot([], [], color=fs.BFF_VARIANT[v], lw=0.9, label=f"{BFF_NAMES[v].replace(chr(10), ' ')} (n = {int((runs['variant'] == v).sum())})")
    _epoch_axis(axa, "heritable fraction of random tapes")
    h_, l_ = axa.get_legend_handles_labels()
    fig.legend(h_, l_, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, 0.715), ncol=3, frameon=False, columnspacing=1.6)
    rng = np.random.default_rng(0)
    ROW, SUB, JIT = 2.9, 0.45, 0.2      # jitter is vertical only, within each first/final row: the x value is drawn exactly
    yticks, ylabels = [], []
    for k, v in enumerate(variants):
        d = runs[(runs["variant"] == v) & runs["t_top"].notna()]
        y0 = -ROW * k
        for which, dy, mk in (("first", SUB, "o"), ("final", -SUB, "s")):
            y = y0 + dy + rng.uniform(-JIT, JIT, len(d))
            x = d[f"{which}_entered"].values
            loop = d[f"{which}_loop"].astype(bool).values
            axb.scatter(x[~loop], y[~loop], s=5, marker=mk, facecolors="white", edgecolors=fs.BFF_VARIANT[v], lw=0.5)
            axb.scatter(x[loop], y[loop], s=5, marker=mk, color=fs.BFF_VARIANT[v], lw=0)
            vals, cnt = np.unique(np.round(x, 9), return_counts=True)     # coincident points (same x): their number, outside
            for xv, n_ in zip(vals, cnt):
                if n_ < 2:
                    continue
                assert xv in (0.0, 1.0), f"Fig. 5b: a cluster of {n_} at {xv} needs a placed count"
                axb.text(-0.025 if xv == 0 else 1.025, y0 + dy, f"{n_}", ha="right" if xv == 0 else "left", va="center", fontsize=5, color=GREY)
            yticks.append(y0 + dy)
            ylabels.append(which)
        axb.text(0.0, y0 + SUB + JIT + 0.3, f"{BFF_NAMES[v].replace(chr(10), ' ')} ({len(d)} of {int((runs['variant'] == v).sum())} soups)", ha="left", va="bottom", fontsize=5.5, color=INK)
    axb.set_yticks(yticks, ylabels, fontsize=5.5)
    axb.set_ylim(-ROW * (len(variants) - 1) - SUB - JIT - 0.35, SUB + JIT + 0.3 + 0.8)
    axb.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    axb.set_xlim(-0.08, 1.08)
    axb.tick_params(axis="y", length=2)
    fs.tidy(axb, "fraction of encounters in which\nthe pointer enters the partner")
    for fn, ax in ((_dial_lethality, axc), (_dial_write_ratio, axd)):
        try:
            fn(ax)
        except Exception as e:  # noqa: BLE001
            placeholder(ax, f"(data missing: {e})")
    lit = runs[runs["variant"] == "lit"]        # the map's "literal push" entry (typed in concept.fig4d): open, then dies out
    assert len(lit) == 12 and (lit["first_class"] == "open").sum() == 12 and (lit["final_heritable"] < 0.1).sum() == 12, "Fig. 5e: literal push entry"
    cp.fig4d(axe)
    axe.set_anchor("NW")
    for ax_, letter, y in ((None, "a", 0.985), (axb, "b", 0.985), (None, "c", 0.66), (axd, "d", 0.66), (None, "e", 0.315)):
        if ax_ is None:
            x = 0.01
        else:
            ax_.apply_aspect()
            x = ax_.get_position().x0 - 0.052
        fig.text(x, y, letter, fontsize=8, fontweight="bold", va="top", ha="left", gid="panel-label")
    save(fig, os.path.join(out, "fig5v4"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fs.setup()
    for name, fn in (("fig1", fig1), ("fig2", fig2), ("fig3", fig3), ("fig4", fig4), ("fig5", fig5), ("fig6", fig6), ("ed12", ed12), ("ed13", ed13), ("ed14", ed14), ("figvar", figvar), ("fig4v4", fig4v4), ("fig4v5", fig4v5), ("ed_invasions", ed_invasions), ("fig5v4", fig5v4), ("fig6v4", fig6v4), ("ed_census", ed_census), ("ed_confine", ed_confine), ("ed_closure", ed_closure), ("ed_bffinflow", ed_bffinflow)):
        if a.only and name not in a.only.split(","):
            continue
        try:
            fn(a.out)
            print("built", name)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("FAILED", name, repr(e))


if __name__ == "__main__":
    main()
