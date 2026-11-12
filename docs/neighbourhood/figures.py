"""Growth of a replicator under the neighbourhood rule (scripts/around-look.mjs output).
  python docs/neighbourhood/figures.py <run folder> <out png> <updates,comma separated> <title>
Top row: the lattice as the app draws it, each cell's 16 bytes as a 4 x 4 block, one colour per byte
value. Bottom row: the cells that hold the tape which is commonest at the last time shown (either of
its two phases). Nothing is smoothed."""
import sys, json, colorsys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
run, out, ticks, title = sys.argv[1], sys.argv[2], [int(t) for t in sys.argv[3].split(",")], sys.argv[4]
q = json.load(open(f"{run}/look.json"))["q"]
W, H = q["width"], q["height"]
lut = np.zeros((256, 3))
for v in range(1, 256):
    h = ((v * 2654435761) % 2**32) / 2**32
    lut[v] = colorsys.hsv_to_rgb(h, 0.62, 0.96)
lut[0] = (0.06, 0.06, 0.08)
INK, SOFT, HOLD = "#1B2A33", "#6B7A83", "#2f6fd0"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 9, "text.color": INK, "axes.edgecolor": SOFT})
def tapes(t):
    return np.fromfile(f"{run}/tapes_{t}.u8", dtype=np.uint8).reshape(H, W, 16)
last = tapes(ticks[-1]).reshape(-1, 16)
vals, counts = np.unique(last, axis=0, return_counts=True)
top = vals[counts.argmax()]
phases = [top, np.roll(top, 1)]
fig, axes = plt.subplots(2, len(ticks), figsize=(3.3 * len(ticks), 5.9), dpi=200)
for c, t in enumerate(ticks):
    a = tapes(t)
    img = a.reshape(H, W, 4, 4).transpose(0, 2, 1, 3).reshape(H * 4, W * 4)
    ax = axes[0, c]
    ax.imshow(lut[img], interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{t:,} updates", fontsize=10, loc="left", pad=4, fontweight="bold")
    holds = np.zeros((H, W), bool)
    for p in phases:
        holds |= (a == p).all(axis=2)
    ax = axes[1, c]
    ax.imshow(np.where(holds[..., None], matplotlib.colors.to_rgb(HOLD), (0.93, 0.94, 0.95)), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"cells holding the replicator: {100 * holds.mean():.0f}%", fontsize=8.5, labelpad=5)
name = " ".join(f"{b:02x}" for b in top[:4]) + " …"
fig.suptitle(title, x=0.012, y=0.988, ha="left", fontsize=11, fontweight="bold")
fig.text(0.012, 0.928, f"Top: the lattice as the app draws it ({W} × {H} cells, 16 bytes each). Bottom, blue: cells whose tape is {name}",
         ha="left", fontsize=9, color=INK)
fig.subplots_adjust(left=0.012, right=0.988, top=0.875, bottom=0.05, wspace=0.05, hspace=0.1)
fig.savefig(out); plt.close(fig)
