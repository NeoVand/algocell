"""The neighbourhood rule across the dissipation dial (scripts/around-look.mjs output).
  python docs/neighbourhood/figures_dial.py <dir with run folders r_g<gather>_d<dissipation>> <out png> <gather>
Each panel is the lattice as the app draws it (each cell's 16 bytes as a 4 x 4 block) at the last
saved update of one run. Nothing is smoothed."""
import sys, json, colorsys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S, out, gather = sys.argv[1], sys.argv[2], sys.argv[3]
dials = ["0", "0.1", "0.3", "0.7", "0.99", "1"]
lut = np.zeros((256, 3))
for v in range(1, 256):
    h = ((v * 2654435761) % 2**32) / 2**32
    lut[v] = colorsys.hsv_to_rgb(h, 0.62, 0.96)
lut[0] = (0.06, 0.06, 0.08)
INK, SOFT = "#1B2A33", "#6B7A83"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 9, "text.color": INK, "axes.edgecolor": SOFT})
fig, axes = plt.subplots(1, len(dials), figsize=(2.65 * len(dials), 3.75), dpi=200)
REPLICATOR = "e1e3" * 8
for ax, d in zip(axes, dials):
    run = f"{S}/r_g{gather}_d{d}"
    look = json.load(open(f"{run}/look.json"))
    q, row = look["q"], look["rows"][-1]
    W, H = q["width"], q["height"]
    a = np.fromfile(f"{run}/tapes_{row['update']}.u8", dtype=np.uint8).reshape(H, W, 16)
    img = a.reshape(H, W, 4, 4).transpose(0, 2, 1, 3).reshape(H * 4, W * 4)
    ax.imshow(lut[img], interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"dissipation {float(d) * 100:.0f}%", fontsize=10, loc="left", pad=4, fontweight="bold")
    share = dict((k, v) for k, v in row["topTapes"]).get(REPLICATOR, 0)
    ax.set_xlabel(f"replicator: {share * 100:.0f}% of cells\nbytes differing from memory: {row['heat'] * 100:.0f}%",
                  fontsize=8.5, labelpad=5, linespacing=1.35)
fig.suptitle(f"The neighbourhood rule with the two-layer law, {gather} neighbour{'s' if gather != '1' else ''} gathered, "
             f"after {row['update']:,} updates (one run per panel, {W} × {H} cells; replicator = the tape e1 e3 repeated)",
             x=0.012, y=0.985, ha="left", fontsize=10.5, fontweight="bold")
fig.subplots_adjust(left=0.012, right=0.988, top=0.86, bottom=0.15, wspace=0.05)
fig.savefig(out); plt.close(fig)
