"""The same seed under the four facing rules (scripts/free-look.mjs output).
  python docs/free-running/figures_facing.py <dir with run folders> <out dir>
Every square is one byte of the sheet; nothing is smoothed."""
import sys, json, colorsys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S, OUT = sys.argv[1], sys.argv[2]
W, CROP, T = 512, 128, 100000
lut = np.zeros((256, 3))
for v in range(1, 256):
    h = ((v * 2654435761) % 2**32) / 2**32
    lut[v] = colorsys.hsv_to_rgb(h, 0.62, 0.96)
lut[0] = (0.06, 0.06, 0.08)
INK, SOFT = "#1B2A33", "#6B7A83"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 9, "text.color": INK, "axes.edgecolor": SOFT})
panels = [
    ("f_same_s1", "All rightwards (the first version)"),
    ("f_rows_s1", "Along rows, either way round"),
    ("f_twoaxes_s1", "Rightwards or downwards"),
    ("a_seed1", "Any of eight ways (the rule now)"),
]
fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.25), dpi=200)
for ax, (run, title) in zip(axes, panels):
    a = np.fromfile(f"{S}/{run}/sheet_{T}.u8", dtype=np.uint8).reshape(W, W)[:CROP, :CROP]
    row = [r for r in json.load(open(f"{S}/{run}/look.json"))["rows"] if r["tick"] == T][0]
    ax.imshow(lut[a], interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10, loc="left", pad=4, fontweight="bold")
    ax.set_xlabel(f"same byte two places along the row: {row['sameRight'][1]*100:.0f}%\n"
                  f"same byte two places down the column: {row['sameDown'][1]*100:.0f}%",
                  fontsize=8.5, color=INK, labelpad=5, linespacing=1.35)
fig.suptitle(f"One starting sheet, four rules for which way a landing processor faces, after {T:,} ticks "
             f"(a {CROP} × {CROP}-byte corner of 512 × 512; dark = blank byte)",
             x=0.012, y=0.985, ha="left", fontsize=11, fontweight="bold")
fig.subplots_adjust(left=0.012, right=0.988, top=0.87, bottom=0.14, wspace=0.05)
fig.savefig(f"{OUT}/free_facing.png"); plt.close(fig)
