"""Extended Data figure (N3, exploratory): the first self-confined copiers (founders) arise among non-copiers.

    .venv/bin/python manuscript/figures/figs_n3.py      # manuscript/figures/out/ed_assembly.{pdf,svg,png}

Every number drawn is read or computed at build time from:
  results/lod/founding.csv    the founding encounter of each first founder (lod_founding.py): the executor, whether it
                              copies the bytes it executes (executor_exec_copier), whether its pointer entered the partner in
                              that encounter, the share of 512 soup partners with which it makes the founder, the event kinds
                              on the chain before the founder (chain_events) and the founding event (F_kind)
  results/lod/founders2.csv   first founders (`first`); chain from the newest open copier older than the founder
  runs/lod_v6_modal/lod_v6/L16_benign_s<seed>/line_records.npz   the exact line of descent (panels a and b)
Tape classes (whole tape): lod_traj.classify_tapes (64 random partners, seed 3); instruction starts: lod_origin.roles
(another 64 random partners, seed 5); both run 128 steps.
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
from make_figures import fs, plt, save, INK, GREY, RED, RULE  # noqa: E402

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
# vermilion (RED) only for the return-closer motif and confined copiers; every square in c-e is one first founder.
CHAIN_C = "#8A95A1"          # events on the chains (panel d)
EVENT = {1: "copy", 2: "partial overwrite", 3: "rewrite", 4: "point mutation", 5: "copy of the partner"}
KIND = {"copy": "copy", "damage": "partial overwrite", "novel": "rewrite", "mut": "point mutation", "copyA": "copy of the partner"}
MADE = {"rewrite (neither copy nor ≥ 75% own)": "rewrite", "point mutation": "point mutation", "copy of the partner": "copy of the partner",
        "partial overwrite": "partial overwrite", "copy": "copy"}
NEITHER, CODE = "copies neither its tape nor the code it runs", "copies only the code it runs, confined"
MOTIF = re.compile(r"^(..) e3 21 e3 21 (e0|c0) \1 (e0|c0)")       # as lod_parts_first.py
W_MM, H_MM = 180.0, 146.0
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


def founder_marker(ax, x, y, entered, s=10):
    """One first founder: filled if the founding executor's pointer entered the partner in the founding encounter."""
    if entered:
        ax.scatter(x, y, s=s, marker="s", color=RED, lw=0, zorder=3)
    else:
        ax.scatter(x, y, s=s * 0.8, marker="s", facecolor="white", edgecolor=RED, lw=0.7, zorder=3)


def ring(ax, x, y):
    ax.scatter([x], [y], s=36, marker="o", facecolor="none", edgecolor=INK, lw=0.6, zorder=4)


# ------------------------------------------------------------------------------------------------------------- data
def load():
    F2 = pd.read_csv(os.path.join(LOD, "founders2.csv"))
    F1 = F2[F2["first"].astype(bool)]
    FD = pd.read_csv(os.path.join(LOD, "founding.csv"))
    M = F1.merge(FD, on="seed", suffixes=("", "_fd"))
    assert len(M) == len(F1) == len(FD) and (M.F_tape == M.F_tape_fd).all()
    M["code_copier"] = M.executor_exec_copier >= 0.5                           # FOUNDING.md: copier by executed bytes
    M["type"] = np.where(M.code_copier, CODE, NEITHER)
    M["entered"] = M.executor_entered_in_founding.astype(bool)
    M["producer"] = np.maximum(M.executor_writes_F, M.executor_self_F) >= 0.5
    assert (M.executor_class == "non-copier").all()
    assert (M[M.code_copier].executor_enters < 0.5).all() and M[M.code_copier].producer.all()   # the five are confined producers
    return M.reset_index(drop=True)


def choose_worlds(M):
    """Per executor type, the world whose chain is closest to the median first-founder chain (records and log steps)."""
    mr, ms = M.chain_records.median(), M.steps_elapsed.median()
    d = (M.chain_records - mr).abs() / mr + np.log(M.steps_elapsed / ms).abs()
    out = []
    for t in (NEITHER, CODE):
        g = M.assign(d=d)[M.type == t].sort_values("d")
        print(f"[ed_assembly] {t}: distance to the median chain ({mr:.0f} records, {ms:.0f} steps): " +
              ", ".join(f"{r.seed} {r.d:.2f}" for r in g.head(3).itertuples()))
        out.append(int(g.seed.iloc[0]))
    return out


def line_of(row):
    """Records of the first founder's chain: from the newest open copier older than F to F on the first line holding F
    (lod_founding.py's convention), and the partner of the founding encounter."""
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
    C = classify_tapes(sorted({tapes[i] for i in ids[fq:]}))
    lo = next(q for q in range(fq + 1, len(ids)) if C[tapes[ids[q]]]["class"] == "open copier")
    chain = [int(ids[q]) for q in range(lo, fq - 1, -1)]              # chain start ... F
    F = chain[-1]
    p1 = int(z["p1"][F])
    if int(z["kind"][F]) == 3:                                        # rewrite: own = F's cell before, oth = the other tape
        own, oth = (z["tape"][p1], z["p2_tape"][F]) if z["cell"][p1] == z["cell"][F] else (z["p2_tape"][F], z["tape"][p1])
    else:                                                             # copy of the partner: p1 = the partner
        own, oth = z["other"][F], z["tape"][p1]
    exe, par = (own, oth) if row.side == "executor's own half" else (oth, own)
    assert _hx(exe) == row.executor and _hx(exe) == tapes[chain[-2]], f"world {row.seed}: executor is not the record before F"
    ts = [tapes[i] for i in chain] + [_hx(par)]
    C = classify_tapes(sorted(set(ts)))
    R = roles(sorted(set(ts)))
    rec = pd.DataFrame({"step": [int(z["step"][i]) for i in chain], "tape": ts[:-1], "event": [EVENT[int(z["kind"][i])] for i in chain]})
    rec["cls"] = [C[t]["class"] for t in rec.tape]
    rec["start"] = [R[t]["start"] >= 0.5 for t in rec.tape]
    assert len(rec) - 1 == row.chain_records and rec.step.iloc[-1] - rec.step.iloc[0] == row.steps_elapsed
    assert rec.step.iloc[0] == row.last_open_step and rec.tape.iloc[0] == row.last_open_tape
    assert rec.cls.iloc[0] == "open copier" and rec.cls.iloc[-1] == "confined copier" and not (rec.cls.iloc[1:-1] == "open copier").any()
    assert ";".join(rec.event.iloc[1:-1]) == ";".join(KIND[k] for k in str(row.chain_events).split(";") if k != "nan")
    partner = {"tape": _hx(par), "cls": C[_hx(par)]["class"], "start": R[_hx(par)]["start"] >= 0.5}
    return rec, partner


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
    rec, partner = line_of(row)
    f_step = int(rec.step.iloc[-1])
    n = len(rec)
    y = 0
    for q, e in enumerate(rec.itertuples()):
        is_f = q == n - 1
        if is_f:                                                            # the partner of the founding encounter, off the line
            _row(ax, y, partner["tape"], partner["start"], partner["cls"], dashed=True)
            ax.text(-0.35, y + 0.5, "partner", fontsize=6, ha="right", va="center", color=INK)
            ax.text(20.1, y + 0.5, "off the line", fontsize=5.5, ha="left", va="center", color=GREY)
            y += 1
        _row(ax, y, e.tape, e.start, e.cls, box=is_f)
        d = e.step - f_step
        ax.text(19.55, y + 0.5, "0" if d == 0 else f"{MINUS}{-d:,}", fontsize=5.5, va="center", ha="right",
                color=INK if is_f else GREY, fontweight="bold" if is_f else "normal")
        if q > 0:
            same = rec.step.iloc[q - 1] == e.step
            ax.text(20.1, y + 0.5, e.event + (", same step" if same else ""), fontsize=5.5, va="center", ha="left",
                    color=INK if is_f else GREY, fontweight="bold" if is_f else "normal")
        lab = {0: "chain start", n - 2: "executor", n - 1: "founder"}.get(q)    # the founding executor is the record before F
        if lab:
            ax.text(-0.35, y + 0.5, lab, fontsize=6, ha="right", va="center", color=INK, fontweight="bold" if is_f else "normal")
        y += 1
    yf = y - 1
    # the return-closer motif in the founder (8 bytes, at its cyclic shift), labelled below its bracket
    ft = rec.tape.iloc[-1].split()
    for s in range(L):
        if MOTIF.search(" ".join(ft[s:] + ft[:s])):
            x0, x1, yb = s + 0.1, s + 7.9, yf + 1.3
            ax.plot([x0, x0, x1, x1], [yb - 0.2, yb, yb, yb - 0.2], color=RED, lw=0.8, clip_on=False, solid_capstyle="butt")
            ax.text((x0 + x1) / 2, yb + 0.25, "return-closer motif", fontsize=5.5, color=RED, ha="center", va="top")
            break
    # header
    ax.text(-6.0, -2.2, title, fontsize=6, color=INK, ha="left", va="center")
    ax.text(0.0, -0.75, f"byte position 0 … {L - 1} (hex)", fontsize=5.5, color=GREY, va="center")
    ax.text(19.55, -0.75, "step", fontsize=5.5, color=GREY, va="center", ha="right")
    ax.text(20.1, -0.75, "made by", fontsize=5.5, color=GREY, va="center", ha="left")
    if key:
        hy = -6.6
        pos = ((0.0, 0), (4.0, 0), (9.1, 0), (0.0, 1), (5.4, 1))
        for (lab, col), (x, k) in zip(BYTE_KEY, pos):
            yy = hy + k * 1.1
            ax.add_patch(plt.Rectangle((x + 0.06, yy + 0.08), 0.88, 0.84, facecolor=col, edgecolor="none", clip_on=False))
            ax.text(x + 1.25, yy + 0.5, lab, fontsize=5.5, va="center", color=INK)
        yy = hy + 1.1
        ax.add_patch(plt.Rectangle((9.1 + 0.06, yy + 0.08), 0.88, 0.84, facecolor="none", edgecolor=GREY, lw=0.6, ls=(0, (2, 1.5)), clip_on=False))
        ax.text(9.1 + 1.25, yy + 0.5, "off the line", fontsize=5.5, va="center", color=INK)
        yy = hy + 2.2
        ax.add_patch(plt.Rectangle((0.06, yy + 0.08), 0.88, 0.84, facecolor=SKY, edgecolor="none", clip_on=False))
        ax.add_patch(plt.Rectangle((1.1, yy + 0.12), 0.8, 0.76, facecolor="white", edgecolor=SKY, lw=0.7, clip_on=False))
        ax.text(2.25, yy + 0.5, "instruction start in ≥ / < half of 64 runs", fontsize=5.5, va="center", color=INK)
        for k, (lab, mk, fc, ec) in enumerate((("open copier", "o", SLATE, SLATE), ("non-copier", "o", "white", GREY), ("confined copier", "s", RED, RED))):
            yy = hy + k * 1.1 + 0.5
            ax.plot(16.75, yy, marker=mk, ms=3.0, mfc=fc, mec=ec, mew=0.6, ls="none", clip_on=False)
            ax.text(17.45, yy, lab, fontsize=5.5, va="center", color=INK)
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    ax.set_axis_off()
    print(f"[ed_assembly] line of world {row.seed} ({row.type}): {len(rec)} records, chain of {int(row.chain_records)} records over "
          f"{int(row.steps_elapsed)} steps ({int(rec.step.iloc[0]):,} to {f_step:,}); events {rec.event.iloc[1:].value_counts().to_dict()}; "
          f"between: {rec.cls.iloc[1:-1].value_counts().to_dict()}; executor at {int(rec.step.iloc[-2]) - f_step} ({rec.event.iloc[-2]}); "
          f"partner {partner['cls']}; founder on the {row.side}; executor makes F with {row.executor_soup_produces_F:.4f} of soup partners")
    return rec


def panel_soup(ax, M, shown):
    """Per first founder: the share of 512 soup partners with which the founding executor makes the founder (log axis, 0
    below a break), one row per kind of executor; filled if its pointer entered the partner in the founding encounter."""
    v = M.executor_soup_produces_F.values.astype(float)
    e0 = np.floor(np.log10(v[v > 0].min()))
    zero_x, lo_x = 10 ** (e0 - 0.6), 10 ** (e0 - 0.22)
    xv = np.where(v > 0, v, zero_x)
    base = {CODE: 0.0, NEITHER: 1.55}
    dy = 0.24
    seen, pos = {}, {}
    for i in sorted(range(len(M)), key=lambda i: (xv[i], M.seed.values[i])):
        r = base[M.type.values[i]]
        key = (r, round(np.log10(xv[i]) / 0.13))
        k = seen.get(key, 0)
        seen[key] = k + 1
        pos[i] = (xv[i], r + dy * k)
        founder_marker(ax, [xv[i]], [r + dy * k], M.entered.values[i])
    for sd in shown:
        ring(ax, *pos[int(np.where(M.seed.values == sd)[0][0])])
    for i in np.where(M.F_copy_of_open_copier.values)[0]:
        ax.annotate("copy of an\nopen copier", pos[i], xytext=(-4, -2), textcoords="offset points", fontsize=5.5, color=GREY, ha="right", va="top")
    n = M.type.value_counts()
    top = max(pos[i][1] for i in range(len(M)) if M.type.values[i] == NEITHER)
    ax.text(zero_x / 1.5, top + 0.42, f"{NEITHER} ({n[NEITHER]})", fontsize=5.5, color=INK, ha="left", va="center")
    ax.text(0.55, base[CODE] + 0.5, f"{CODE} ({n[CODE]})", fontsize=5.5, color=INK, ha="right", va="center")
    ax.set_xscale("log")
    ax.set_xlim(zero_x / 1.7, 1.7)
    ax.set_ylim(-0.4, top + 0.75)
    ticks = [t for t in (0.001, 0.01, 0.1, 1) if t > lo_x]
    ax.set_xticks([zero_x] + ticks, ["0"] + [f"{t:g}" for t in ticks])
    minor = [m * 10.0 ** e for e in range(-5, 1) for m in range(2, 10) if lo_x * 1.25 < m * 10.0 ** e < 1]
    ax.xaxis.set_minor_locator(plt.matplotlib.ticker.FixedLocator(minor))
    ax.xaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    brk = np.sqrt(zero_x * lo_x)                                      # axis break between 0 and the smallest share
    ax.spines["bottom"].set_bounds(brk * 1.2, 1.7)
    tr = plt.matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot([zero_x / 1.7, brk / 1.2], [0, 0], color=INK, lw=0.5, transform=tr, clip_on=False, solid_capstyle="butt")
    for xb in (brk / 1.2, brk * 1.2):
        ax.plot([xb / 1.07, xb * 1.07], [-0.04, 0.04], color=INK, lw=0.5, transform=tr, clip_on=False)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    fs.tidy(ax, "share of 512 soup partners with which\nthe founding executor makes the founder", None)
    h = [plt.Line2D([], [], marker="s", ls="none", ms=3.2, mfc=RED, mec=RED, mew=0, label="pointer entered the partner"),
         plt.Line2D([], [], marker="s", ls="none", ms=2.9, mfc="white", mec=RED, mew=0.7, label="did not"),
         plt.Line2D([], [], marker="o", ls="none", ms=5.0, mfc="none", mec=INK, mew=0.6, label="worlds in a, b")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=3, fontsize=5.5, frameon=False, handletextpad=0.2,
              columnspacing=0.8, borderaxespad=0.0)
    N = M[M.type == NEITHER]
    Np = N[~N.producer]
    print(f"[ed_assembly] panel c: {NEITHER} {len(N)}, {CODE} {int(M.code_copier.sum())}; soup share among the {len(N)}: median "
          f"{N.executor_soup_produces_F.median():.4f}, max {N.executor_soup_produces_F.max():.3f}; without the producer {N[N.producer].seed.tolist()}: "
          f"max {Np.executor_soup_produces_F.max():.3f}, zero in {int((Np.executor_soup_produces_F == 0).sum())}; code copiers "
          f"{M[M.code_copier].executor_soup_produces_F.min():.2f}-{M[M.code_copier].executor_soup_produces_F.max():.2f}, made by "
          f"{M[M.code_copier].producer_made_by.map(MADE).value_counts().to_dict()}, {M[M.code_copier].producer_steps_before_F.min():.0f}-"
          f"{M[M.code_copier].producer_steps_before_F.max():.0f} steps before; entered in the founding encounter {int(M.entered.sum())} of {len(M)} "
          f"(code copiers {int(M[M.code_copier].entered.sum())}); copy of an open copier {M[M.F_copy_of_open_copier.astype(bool)].seed.tolist()}")


def panel_events(ax, M):
    """Event kinds on the chains before the founder (records strictly between the chain start and F) against the founding events."""
    ev = pd.Series([e for x in M.chain_events.dropna() for e in x.split(";") if e]).map(KIND)
    fe = M.F_kind_fd.map(MADE)
    cats = ["rewrite", "partial overwrite", "point mutation", "copy", "copy of the partner"]
    assert ev.notna().all() and fe.notna().all() and set(ev) <= set(cats) and set(fe) <= set(cats)
    nc, nf = ev.value_counts(), fe.value_counts()
    pc, pf = [nc.get(c, 0) / len(ev) for c in cats], [nf.get(c, 0) / len(fe) for c in cats]
    y = np.arange(len(cats))
    for yy, a, b in zip(y, pc, pf):
        ax.plot([a, b], [yy, yy], color=RULE, lw=0.8, zorder=1)
    ax.scatter(pc, y, s=13, marker="D", color=CHAIN_C, lw=0, zorder=3, label=f"events on the chains before the founder ({len(ev)})")
    ax.scatter(pf, y, s=12, marker="s", color=RED, lw=0, zorder=3, label=f"founding events ({len(fe)})")
    ax.text(1.06, -0.85, "chain / founding", fontsize=5.5, color=GREY, ha="left", va="center", gid="allow-outside")
    for yy, c in zip(y, cats):
        ax.text(1.06, yy, f"{nc.get(c, 0)} / {nf.get(c, 0)}", fontsize=5.5, color=GREY, ha="left", va="center", gid="allow-outside")
    pm = nc.get("point mutation", 0) / len(ev)
    P = (1 - pm) ** len(fe)
    ax.annotate(f"0 of {len(fe)}: P = {P:.3f}", (pc[cats.index("point mutation")], cats.index("point mutation")), xytext=(6, 0),
                textcoords="offset points", fontsize=5.5, color=INK, ha="left", va="center")
    ax.set_yticks(y, cats)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(cats) - 0.4, -1.3)
    ax.set_xlim(-0.03, 1.0)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.spines["left"].set_visible(False)
    fs.tidy(ax, "share of events", None)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, fontsize=5.5, frameon=False, handletextpad=0.3, borderaxespad=0.0,
              labelspacing=0.25)
    print(f"[ed_assembly] panel d: chain events {len(ev)} {nc.to_dict()}; founding events {len(fe)} {nf.to_dict()}; point-mutation share "
          f"{pm:.4f}; P(0 of {len(fe)}) = (1 - {pm:.4f})^{len(fe)} = {P:.4f}")


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
    y = _swarm(np.log10(v), 0.075, 0.5)
    for i in range(len(v)):
        founder_marker(ax, [v[i]], [y[i]], M.entered.values[i])
    for sd in shown:
        i = int(np.where(M.seed.values == sd)[0][0])
        ring(ax, v[i], y[i])
    m = float(np.median(v))
    ax.axvline(m, color=GREY, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
    ax.text(m * 0.9, 1.45, f"median {m:.0f}", fontsize=5.5, color=GREY, ha="right", va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.7, 600)
    ax.set_xticks([1, 10, 100], ["1", "10", "100"])
    ax.set_ylim(-1.3, 1.8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    fs.tidy(ax, "steps from the newest open copier to the founder", None)
    print(f"[ed_assembly] panel e: steps median {m:.0f} (range {v.min():.0f}-{v.max():.0f}); records median {M.chain_records.median():.0f}; "
          f"only non-copiers between {int((M.all_between_noncopiers == True).sum())} of {len(M)}")  # noqa: E712


# ----------------------------------------------------------------------------------------------------------- figure
def _title(r):
    if r.type == NEITHER:
        return f"World {r.seed}: a program that copies neither its tape nor the code it runs writes the founder"
    return (f"World {r.seed}: a confined tape that copies only the code it runs arises by {MADE[r.producer_made_by]} "
            f"and writes the founder")


def ed_assembly(out):
    M = load()
    shown = choose_worlds(M)
    rows = [M[M.seed == s].iloc[0] for s in shown]
    fig = plt.figure(figsize=(W_MM * fs.MM, H_MM * fs.MM))
    xr = (-6.4, 27.3)
    x0, w = 1.0, (xr[1] - xr[0]) * CW
    # a, b: one line of descent per kind of executor (the key heads a)
    na, nb = int(rows[0].chain_records) + 2, int(rows[1].chain_records) + 2
    ya, yra = 4.0, (na + 2.1, -6.9)
    ha = (yra[0] - yra[1]) * RH
    yb, yrb = ya + ha + 3.0, (nb + 2.1, -2.9)
    hb = (yrb[0] - yrb[1]) * RH
    axa = axmm(fig, x0, ya, w, ha)
    axb = axmm(fig, x0, yb, w, hb)
    panel_line(axa, rows[0], _title(rows[0]), xr, yra, key=True)
    panel_line(axb, rows[1], _title(rows[1]), xr, yrb, key=False)
    # c-e: every first founder
    axc = axmm(fig, 118.0, 12.0, 60.0, 30.0)
    axd = axmm(fig, 131.0, 69.0, 27.0, 26.0)
    axe = axmm(fig, 118.0, 114.0, 58.0, 16.0)
    panel_soup(axc, M, shown)
    panel_events(axd, M)
    panel_steps(axe, M, shown)
    letter(fig, 0.5, 1.0, "a")
    letter(fig, 0.5, yb - 1.5, "b")
    letter(fig, 110.0, 1.0, "c")
    letter(fig, 110.0, 55.0, "d")
    letter(fig, 110.0, 106.0, "e")
    save(fig, os.path.join(out, "ed_assembly"))


if __name__ == "__main__":
    fs.setup()
    ed_assembly(os.path.join(HERE, "out"))
