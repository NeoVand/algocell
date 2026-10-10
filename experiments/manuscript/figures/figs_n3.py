"""Extended Data figure (N3, exploratory): how the first self-confined copier (the founder) of each world was made.

    .venv/bin/python manuscript/figures/figs_n3.py      # manuscript/figures/out/ed_assembly.{pdf,svg,png}

Every number drawn is read or computed at build time from:
  results/lod/founding.csv    the founding encounter of each first founder (lod_founding.py): executor, side, how often the
                              executor writes the founder into random partners, whether its pointer enters them, bytes of
                              the founder in neither parent, the earliest producer on the chain and how it was made
  results/lod/founders2.csv   first founders (`first`), chain from the newest open copier older than the founder
  runs/lod_v6_modal/lod_v6/L16_benign_s<seed>/line_records.npz   the exact line of descent (panels a and b)
Tape classes: lod_traj.classify_tapes (64 random partners, seed 3); instruction starts: lod_origin.roles (another 64
random partners, seed 5); both run 128 steps.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_figures as mf  # noqa: E402
from make_figures import fs, plt, save, INK, GREY, RED  # noqa: E402

EXP = mf.EXP
sys.path.insert(0, EXP)
from lod_origin import roles  # noqa: E402
from lod_traj import classify_tapes  # noqa: E402

LOD = os.path.join(EXP, "results", "lod")
RUNS = os.path.join(EXP, "runs", "lod_v6_modal", "lod_v6")
L = 16
MINUS = "−"

# byte values (panels a, b). PUSH is slate, the open-copier class colour, on purpose (a pusher is mostly PUSH).
SLATE = fs.MARK
SKY = "#56B4E9"              # LD rr,nn
ORANGE = "#E69F00"           # EX (SP),HL
PURPLE = "#CC79A7"           # RET, RET cc
OTHER = "#CDD2D8"            # any other byte
OTHER_EDGE = "#9AA3AD"
BYTE_KEY = (("PUSH", SLATE), ("LD rr,nn", SKY), ("EX (SP),HL", ORANGE), ("RET, RET cc", PURPLE), ("other", OTHER))
# vermilion (RED) only for the return-closer motif and confined copiers; every point in c-f is one first founder.
EVENT = {1: "copy", 2: "partial overwrite", 3: "rewrite", 4: "point mutation", 5: "copy of the partner"}
MADE = {"rewrite (neither copy nor ≥ 75% own)": "rewrite", "point mutation": "point mutation"}
MOTIF = re.compile(r"^(..) e3 21 e3 21 (e0|c0) \1 (e0|c0)")       # as lod_parts_first.py
W_MM, H_MM = 180.0, 150.0
CW, RH = 3.1, 3.05                                                    # mm per byte, mm per row (panels a, b)


def _hx(a) -> str:
    return " ".join(f"{int(x):02x}" for x in a)


def byte_colour(b: int) -> str:
    if b in (0xE5, 0xC5, 0xD5, 0xF5):
        return SLATE
    if b in (0x21, 0x01, 0x11, 0x31):
        return SKY
    if b == 0xE3:
        return ORANGE
    if b == 0xC9 or (b & 0xC7) == 0xC0:                 # RET, RET cc
        return PURPLE
    return OTHER


def axmm(fig, x, y, w, h):
    """Axes at (x, y) mm from the figure's top-left corner, w x h mm."""
    return fig.add_axes([x / W_MM, 1 - (y + h) / H_MM, w / W_MM, h / H_MM])


def letter(fig, x, y, s):
    fig.text(x / W_MM, 1 - y / H_MM, s, fontsize=8, fontweight="bold", va="top", ha="left", gid="panel-label")


def founder_marker(ax, x, y, open_writer, s=11, **kw):
    """One first founder: filled = written by an open non-copier in that encounter; hollow = written by a producer."""
    if open_writer:
        ax.scatter(x, y, s=s, marker="s", color=RED, lw=0, zorder=3, **kw)
    else:
        ax.scatter(x, y, s=s * 0.8, marker="s", facecolor="white", edgecolor=RED, lw=0.7, zorder=3, **kw)


# ------------------------------------------------------------------------------------------------------------- data
def load():
    F2 = pd.read_csv(os.path.join(LOD, "founders2.csv"))
    F1 = F2[F2["first"].astype(bool)]
    FD = pd.read_csv(os.path.join(LOD, "founding.csv"))
    M = F1.merge(FD, on="seed", suffixes=("", "_fd"))
    assert len(M) == len(F1) == len(FD) and (M.F_tape == M.F_tape_fd).all()
    M["produces"] = np.maximum(M.executor_writes_F, M.executor_self_F)          # FOUNDING.md: producer if >= 0.5
    M["open_exec"] = M.executor_enters >= 0.5
    M["producer"] = M.produces >= 0.5
    M["open_writer"] = ~M.producer & M.open_exec
    M["route"] = np.select([M.open_writer, M.producer & ~M.open_exec, M.producer & M.open_exec],
                           ["open writer", "confined producer", "open producer"], "other")
    assert (M.producer == M.producer_on_chain.astype(bool)).all() and (M.route != "other").all()
    return M.reset_index(drop=True)


def choose_worlds(M):
    """Per route shown (open writer; confined producer), the world whose chain is closest to the median first-founder chain."""
    mr, ms = M.chain_records.median(), M.steps_elapsed.median()
    d = (M.chain_records - mr).abs() / mr + np.log(M.steps_elapsed / ms).abs()
    out = []
    for rt in ("open writer", "confined producer"):
        g = M.assign(d=d)[M.route == rt].sort_values("d")
        print(f"[ed_assembly] {rt}: distance to the median chain ({mr:.0f} records, {ms:.0f} steps): " +
              ", ".join(f"{r.seed} {r.d:.2f}" for r in g.head(3).itertuples()))
        out.append(int(g.seed.iloc[0]))
    return out


def line_of(row):
    """Records of the first founder's chain (founders2.csv row): from the newest open copier older than F to F on the first
    line holding F (lod_founding.py's convention), with the founding encounter's partner and the bytes in neither parent."""
    z = dict(np.load(os.path.join(RUNS, f"L16_benign_s{row.seed}", "line_records.npz")))     # in memory: npz access re-reads
    tapes = [_hx(t) for t in z["tape"]]
    cand = {i for i in np.where(z["step"] == row.F_step)[0] if tapes[i] == row.F_tape}
    off = np.concatenate([[0], np.cumsum(z["line_lengths"])])
    for li in range(len(z["line_lengths"])):
        ids = z["line_ids"][off[li]:off[li + 1]]                     # newest first
        hit = [q for q in range(len(ids)) if ids[q] in cand]
        if hit:
            fq = hit[0]
            break
    on_line = [tapes[i] for i in ids[fq:]]
    C = classify_tapes(sorted(set(on_line)))
    lo = next(q for q in range(fq + 1, len(ids)) if C[tapes[ids[q]]]["class"] == "open copier")
    chain = [int(ids[q]) for q in range(lo, fq - 1, -1)]              # chain start ... F
    F = chain[-1]
    # the founding encounter (as lod_founding.py): own = F's cell before, oth = the other tape
    p1 = int(z["p1"][F])
    if int(z["kind"][F]) == 3:
        own, oth = (z["tape"][p1], z["p2_tape"][F]) if z["cell"][p1] == z["cell"][F] else (z["p2_tape"][F], z["tape"][p1])
    else:
        own, oth = z["other"][F], z["tape"][p1]
    exe, par = (own, oth) if row.side == "executor's own half" else (oth, own)
    assert _hx(exe) == row.executor and _hx(exe) == tapes[chain[-2]], f"world {row.seed}: executor is not the record before F"
    neither = (z["tape"][F] != own) & (z["tape"][F] != oth)
    assert int(neither.sum()) == row.bytes_in_neither_parent
    ts = [tapes[i] for i in chain] + [_hx(par)]
    C = classify_tapes(sorted(set(ts)))
    R = roles(sorted(set(ts)))
    rec = pd.DataFrame({"step": [int(z["step"][i]) for i in chain], "tape": ts[:-1], "event": [EVENT[int(z["kind"][i])] for i in chain]})
    rec["cls"] = [C[t]["class"] for t in rec.tape]
    rec["start"] = [R[t]["start"] >= 0.5 for t in rec.tape]
    assert len(rec) - 1 == row.chain_records and rec.step.iloc[-1] - rec.step.iloc[0] == row.steps_elapsed
    assert rec.step.iloc[0] == row.last_open_step and rec.tape.iloc[0] == row.last_open_tape
    assert rec.cls.iloc[0] == "open copier" and rec.cls.iloc[-1] == "confined copier" and not (rec.cls.iloc[1:-1] == "open copier").any()
    partner = {"tape": _hx(par), "cls": C[_hx(par)]["class"], "start": R[_hx(par)]["start"] >= 0.5}
    return rec, partner, neither


# ----------------------------------------------------------------------------------------------------------- panels
def _row(ax, y, tape, start, cls, dashed=False, box=False):
    for p, b in enumerate(int(x, 16) for x in tape.split()):
        c = byte_colour(b)
        if start[p]:
            ax.add_patch(plt.Rectangle((p + 0.06, y + 0.08), 0.88, 0.84, facecolor=c, edgecolor="none", lw=0))
        else:
            ax.add_patch(plt.Rectangle((p + 0.1, y + 0.12), 0.8, 0.76, facecolor="white", edgecolor=OTHER_EDGE if c == OTHER else c, lw=0.7))
        ax.text(p + 0.5, y + 0.53, f"{b:02x}", fontsize=5.0, ha="center", va="center", color="white" if (start[p] and c == SLATE) else INK)
    mk = {"open copier": ("o", SLATE, SLATE), "confined copier": ("s", RED, RED), "non-copier": ("o", "white", GREY)}[cls]
    ax.plot(16.75, y + 0.5, marker=mk[0], ms=3.0, mfc=mk[1], mec=mk[2], mew=0.6, ls="none", clip_on=False)
    if dashed or box:
        ax.add_patch(plt.Rectangle((-0.02, y + 0.0), 16.04, 1.0, facecolor="none", edgecolor=INK if box else GREY, lw=0.6,
                                   ls=(0, (2, 1.5)) if dashed else "-"))


def panel_line(ax, row, title, xr, yr, key):
    rec, partner, neither = line_of(row)
    f_step = int(rec.step.iloc[-1])
    n = len(rec)
    y = 0
    prod_q = n - 2                                                          # the founding executor is the record before F
    for q, e in enumerate(rec.itertuples()):
        if q == n - 1:                                                      # the partner of the founding encounter, off the line
            _row(ax, y, partner["tape"], partner["start"], partner["cls"], dashed=True)
            ax.text(-0.35, y + 0.5, "partner", fontsize=6, ha="right", va="center", color=INK)
            ax.text(20.1, y + 0.5, "off the line", fontsize=5.5, ha="left", va="center", color=GREY)
            y += 1
        is_f = q == n - 1
        _row(ax, y, e.tape, e.start, e.cls, box=is_f)
        d = e.step - f_step
        ax.text(19.55, y + 0.5, "0" if d == 0 else f"{MINUS}{-d:,}", fontsize=5.5, va="center", ha="right",
                color=INK if is_f else GREY, fontweight="bold" if is_f else "normal")
        if q > 0:
            same = q > 0 and rec.step.iloc[q - 1] == e.step
            ax.text(20.1, y + 0.5, e.event + (", same step" if same else ""), fontsize=5.5, va="center", ha="left",
                    color=INK if is_f else GREY, fontweight="bold" if is_f else "normal")
        lab = {0: "chain start", n - 1: "founder"}.get(q)
        if q == prod_q:
            lab = "producer (executor)" if row.producer else "executor"
        if lab:
            ax.text(-0.35, y + 0.5, lab, fontsize=6, ha="right", va="center", color=INK, fontweight="bold" if is_f else "normal")
        y += 1
    yf = y - 1
    # bytes of the founder in neither parent; the return-closer motif
    for p in np.where(neither)[0]:
        ax.plot(p + 0.5, yf + 1.32, marker="o", ms=1.8, color=INK, mew=0, ls="none", clip_on=False)
    ft = rec.tape.iloc[-1].split()
    for s in range(L):
        if MOTIF.search(" ".join(ft[s:] + ft[:s])):
            x0, x1, yb = s + 0.1, s + 7.9, yf + 1.75
            ax.plot([x0, x0, x1, x1], [yb - 0.18, yb, yb, yb - 0.18], color=RED, lw=0.8, clip_on=False, solid_capstyle="butt")
            ax.text(x1 + 0.3, yb, "return-closer motif", fontsize=5.5, color=RED, ha="left", va="center")
            break
    # header
    ax.text(-4.6, -2.2, title, fontsize=6, color=INK, ha="left", va="center")
    ax.text(0.0, -0.75, f"byte position 0 … {L - 1} (hex)", fontsize=5.5, color=GREY, va="center")
    ax.text(19.55, -0.75, "step", fontsize=5.5, color=GREY, va="center", ha="right")
    ax.text(20.1, -0.75, "made by", fontsize=5.5, color=GREY, va="center", ha="left")
    if key:
        hy = -7.6
        pos = ((0.0, 0), (4.0, 0), (9.1, 0), (0.0, 1), (5.4, 1))
        for (lab, col), (x, k) in zip(BYTE_KEY, pos):
            yy = hy + k * 1.1
            ax.add_patch(plt.Rectangle((x + 0.06, yy + 0.08), 0.88, 0.84, facecolor=col, edgecolor="none", clip_on=False))
            ax.text(x + 1.25, yy + 0.5, lab, fontsize=5.5, va="center", color=INK)
        yy = hy + 2.2
        ax.add_patch(plt.Rectangle((0.06, yy + 0.08), 0.88, 0.84, facecolor=SKY, edgecolor="none", clip_on=False))
        ax.add_patch(plt.Rectangle((1.1, yy + 0.12), 0.8, 0.76, facecolor="white", edgecolor=SKY, lw=0.7, clip_on=False))
        ax.text(2.25, yy + 0.5, "instruction start in ≥ / < half of 64 runs", fontsize=5.5, va="center", color=INK)
        yy = hy + 3.3
        ax.plot(0.5, yy + 0.5, marker="o", ms=1.8, color=INK, mew=0, ls="none", clip_on=False)
        ax.text(1.25, yy + 0.5, "founder byte in neither parent", fontsize=5.5, va="center", color=INK)
        ax.add_patch(plt.Rectangle((10.6 + 0.06, yy + 0.08), 0.88, 0.84, facecolor="none", edgecolor=GREY, lw=0.6, ls=(0, (2, 1.5)), clip_on=False))
        ax.text(10.6 + 1.25, yy + 0.5, "off the line", fontsize=5.5, va="center", color=INK)
        for k, (lab, mk, fc, ec) in enumerate((("open copier", "o", SLATE, SLATE), ("non-copier", "o", "white", GREY), ("confined copier", "s", RED, RED))):
            yy = hy + k * 1.1 + 0.5
            ax.plot(16.75, yy, marker=mk, ms=3.0, mfc=fc, mec=ec, mew=0.6, ls="none", clip_on=False)
            ax.text(17.45, yy, lab, fontsize=5.5, va="center", color=INK)
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    ax.set_axis_off()
    print(f"[ed_assembly] line of world {row.seed} ({row.route}): {len(rec)} records, chain of {int(row.chain_records)} records over "
          f"{int(row.steps_elapsed)} steps ({int(rec.step.iloc[0]):,} to {f_step:,}); events {rec.event.iloc[1:].value_counts().to_dict()}; "
          f"between: {rec.cls.iloc[1:-1].value_counts().to_dict()}; executor at {int(rec.step.iloc[-2]) - f_step} ({rec.event.iloc[-2]}); "
          f"partner {partner['cls']}; founder on the {row.side}; bytes in neither parent {int(neither.sum())}")
    return rec


def panel_event(ax, M, shown):
    """Dot plot: how often the founding executor makes the founder with random partners (binned to 0.05), by whether its
    pointer enters the partner; one square per first founder."""
    step = 0.05
    xb = np.round(M.produces.values / step) * step
    base = {True: 7.0, False: 0.0}
    seen = {}
    order = sorted(range(len(M)), key=lambda i: (M.seed.values[i] not in shown, M.seed.values[i]))
    pos = {}
    for i in order:
        key = (bool(M.open_exec.values[i]), xb[i])
        k = seen.get(key, 0)
        seen[key] = k + 1
        pos[i] = (xb[i], base[key[0]] + k)
        founder_marker(ax, [xb[i]], [base[key[0]] + k], M.open_writer.values[i], s=10)
    for sd in shown:
        i = int(np.where(M.seed.values == sd)[0][0])
        ax.scatter(*pos[i], s=36, marker="o", facecolor="none", edgecolor=INK, lw=0.6, zorder=4)
    n = M.route.value_counts()
    ow = M[M.open_writer].produces
    top_open = max(v for (o, _), v in seen.items() if o)
    ax.text(0.24, base[True] + 3.2, f"open non-copiers that wrote it\nin that encounter ({n['open writer']})",
            fontsize=5.5, color=INK, ha="left", va="center")
    ax.text(0.94, base[True], f"open producer ({n.get('open producer', 0)})", fontsize=5.5, color=INK, ha="right", va="center")
    ax.text(0.94, base[False] + 2.0, f"confined producers ({n.get('confined producer', 0)})", fontsize=5.5, color=INK, ha="right", va="center")
    nc = int((M.executor_class == "non-copier").sum())
    ax.text(0.45, (base[True] + base[False]) / 2 + 0.6, f"every founding executor is a non-copier ({nc} of {len(M)})", fontsize=5.5, color=GREY, ha="center", va="center")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.8, base[True] + top_open + 0.2)
    h = [plt.Line2D([], [], marker="s", ls="none", ms=3.2, mfc=RED, mec=RED, mew=0, label="written by an open non-copier in that encounter"),
         plt.Line2D([], [], marker="s", ls="none", ms=2.9, mfc="white", mec=RED, mew=0.7, label="written by a producer"),
         plt.Line2D([], [], marker="o", ls="none", ms=5.0, mfc="none", mec=INK, mew=0.6, label="the worlds in a and b")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(-0.02, 1.02), ncol=1, fontsize=5.5, frameon=False, handletextpad=0.3,
              borderaxespad=0.0, labelspacing=0.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yticks([base[False], base[True]], ["pointer\nconfined", "pointer enters\nthe partner"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fs.tidy(ax, "share of 64 random partners with which\nthe founding executor makes the founder", None)
    print(f"[ed_assembly] panel c: executors non-copier {nc} of {len(M)}; routes {n.to_dict()}; open writers make F with "
          f"{ow.min():.3f}-{ow.max():.3f} of random partners; producers {M[M.producer].produces.min():.2f}-{M[M.producer].produces.max():.2f}; "
          f"pointer enters in >= half: {int(M.open_exec.sum())}; side {M.side.value_counts().to_dict()}")


def panel_routes(ax, M):
    P = M[M.producer]
    made = P.producer_made_by.map(MADE)
    rows = [("open non-copier, same encounter", int(M.open_writer.sum()), True),
            ("confined, made by rewrite", int(((P.route == "confined producer") & (made == "rewrite")).sum()), False),
            ("confined, made by point mutation", int(((P.route == "confined producer") & (made == "point mutation")).sum()), False),
            ("open, made by point mutation", int(((P.route == "open producer") & (made == "point mutation")).sum()), False),
            ("point mutation of a copier", int((M.F_kind == "mut").sum()), True)]          # founders made by any point mutation: an upper bound
    assert sum(r[1] for r in rows) == len(M) and len(P) == rows[1][1] + rows[2][1] + rows[3][1]
    ys = [0, 2, 3, 4, 5.2]
    for y, (lab, v, filled) in zip(ys, rows):
        if filled:
            ax.barh(y, v, height=0.62, color=RED, lw=0)
        else:
            ax.barh(y, v, height=0.56, facecolor="white", edgecolor=RED, lw=0.7)
        ax.text(v + 0.3, y, f"{v}", fontsize=5.5, va="center", ha="left", color=INK)
    lo, hi = P.producer_steps_before_F.min(), P.producer_steps_before_F.max()
    ax.text(-0.02, 1.0, f"producer that arose {lo:.0f}–{hi:.0f} steps before:", fontsize=5.5, color=INK, ha="right", va="center",
            transform=ax.get_yaxis_transform(), gid="allow-outside")
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(5.75, -0.55)
    ax.set_xlim(0, max(r[1] for r in rows) + 2.5)
    ax.set_xticks([0, 5, 10])
    fs.tidy(ax, "first founders written by", None)
    print(f"[ed_assembly] panel d: " + "; ".join(f"{r[0]} {r[1]}" for r in rows) + f"; producer arose {P.producer_steps_before_F.min():.0f}-"
          f"{P.producer_steps_before_F.max():.0f} steps before the founder; completing events {M.F_kind.value_counts().to_dict()}")


def _swarm(x, dx, dy):
    order = np.argsort(x)
    ys = np.zeros(len(x))
    placed = []
    for i in order:
        for k in range(40):
            cand = (k + 1) // 2 * dy * (1 if k % 2 else -1)
            if all(abs(x[i] - x[j]) >= dx or abs(cand - ys[j]) >= dy * 0.999 for j in placed):
                ys[i] = cand
                break
        placed.append(i)
    return ys


def panel_steps(ax, M, shown):
    v = M.steps_elapsed.values.astype(float)
    y = _swarm(np.log10(v), 0.085, 0.5)
    for i in range(len(v)):
        founder_marker(ax, [v[i]], [y[i]], M.open_writer.values[i])
    for sd in shown:
        i = int(np.where(M.seed.values == sd)[0][0])
        ax.scatter([v[i]], [y[i]], s=36, marker="o", facecolor="none", edgecolor=INK, lw=0.6, zorder=4)
    m = float(np.median(v))
    ax.axvline(m, color=GREY, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
    ax.text(m * 0.9, 1.45, f"median {m:.0f}", fontsize=5.5, color=GREY, ha="right", va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.7, 600)
    ax.set_xticks([1, 10, 100], ["1", "10", "100"])
    ax.set_ylim(-1.5, 1.8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    fs.tidy(ax, "steps from the newest open\ncopier to the founder", None)
    print(f"[ed_assembly] panel e: steps median {m:.0f} (range {v.min():.0f}-{v.max():.0f}); records median {M.chain_records.median():.0f}; "
          f"only non-copiers between {int((M.all_between_noncopiers == True).sum())} of {len(M)}")  # noqa: E712


def panel_new(ax, M, shown):
    v = M.bytes_in_neither_parent.values.astype(int)
    seen = {}
    order = sorted(range(len(v)), key=lambda i: (not M.open_writer.values[i], M.seed.values[i] not in shown))
    pos = {}
    for i in order:
        k = seen.get(int(v[i]), 0)
        seen[int(v[i])] = k + 1
        pos[i] = (v[i], k + 1)
        founder_marker(ax, [v[i]], [k + 1], M.open_writer.values[i])
    for sd in shown:
        i = int(np.where(M.seed.values == sd)[0][0])
        ax.scatter(*pos[i], s=36, marker="o", facecolor="none", edgecolor=INK, lw=0.6, zorder=4)
    m = float(np.median(v))
    top = max(seen.values())
    ax.axvline(m, color=GREY, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
    ax.text(m - 0.4, top + 1.3, f"median {m:.0f}", fontsize=5.5, color=GREY, ha="right", va="center")
    ax.set_xlim(-0.5, L + 0.5)
    ax.set_xticks([0, 4, 8, 12, 16])
    ax.set_ylim(0.3, top + 1.9)
    ax.set_yticks(range(0, top + 1, 2) if top > 3 else range(0, top + 1))
    fs.tidy(ax, f"founder bytes (of {L}) in\nneither parent", "first founders")
    b = M.bytes_in_neither_best_shift
    print(f"[ed_assembly] panel f: bytes in neither parent median {m:.0f} (range {v.min()}-{v.max()}); at best shifts median "
          f"{b.median():.0f} ({b.min()}-{b.max()}); counts {dict(sorted(seen.items()))}")


# ----------------------------------------------------------------------------------------------------------- figure
def _title(r):
    if r.route == "open writer":
        side = "its own half" if r.side == "executor's own half" else "its partner"
        return f"World {r.seed}: an open non-copier writes the founder into {side} in one encounter"
    return (f"World {r.seed}: a {r.route}, made by {MADE[r.producer_made_by]}, writes the founder "
            f"{int(r.producer_steps_before_F)} steps later")


def ed_assembly(out):
    M = load()
    shown = choose_worlds(M)
    rows = [M[M.seed == s].iloc[0] for s in shown]
    fig = plt.figure(figsize=(W_MM * fs.MM, H_MM * fs.MM))
    xr = (-6.4, 27.3)
    x0, w = 1.0, (xr[1] - xr[0]) * CW
    # a, b: one line of descent per route (the key heads a)
    na = int(rows[0].chain_records) + 2
    nb = int(rows[1].chain_records) + 2
    ya, yra = 4.0, (na + 2.4, -8.0)
    ha = (yra[0] - yra[1]) * RH
    yb, yrb = ya + ha + 3.0, (nb + 2.4, -2.9)
    hb = (yrb[0] - yrb[1]) * RH
    axa = axmm(fig, x0, ya, w, ha)
    axb = axmm(fig, x0, yb, w, hb)
    panel_line(axa, rows[0], _title(rows[0]), xr, yra, key=True)
    panel_line(axb, rows[1], _title(rows[1]), xr, yrb, key=False)
    # c-f: the founding encounter of every first founder
    axc = axmm(fig, 126.0, 15.0, 52.0, 30.0)
    axd = axmm(fig, 152.0, 62.0, 22.0, 28.0)
    axe = axmm(fig, 112.0, 110.0, 29.0, 18.0)
    axf = axmm(fig, 151.0, 110.0, 26.0, 18.0)
    panel_event(axc, M, shown)
    panel_routes(axd, M)
    panel_steps(axe, M, shown)
    panel_new(axf, M, shown)
    letter(fig, 0.5, 1.0, "a")
    letter(fig, 0.5, yb - 1.5, "b")
    letter(fig, 107.0, 1.0, "c")
    letter(fig, 107.0, 56.0, "d")
    letter(fig, 107.0, 102.0, "e")
    letter(fig, 143.0, 102.0, "f")
    save(fig, os.path.join(out, "ed_assembly"))


if __name__ == "__main__":
    fs.setup()
    ed_assembly(os.path.join(HERE, "out"))
