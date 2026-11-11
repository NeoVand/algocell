"""The `written` facing rule: the sheet chooses an axis by itself (scripts/free-look.mjs output).
  python docs/free-running/figures_written.py <dir with run folders> <out dir>
Left four panels of each row: the whole sheet, each byte coloured by the axis along which it was last
written. Right panel: a corner of the sheet's bytes at the last time. Nothing is smoothed."""
import sys, colorsys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
S, OUT = sys.argv[1], sys.argv[2]
W, CROP = 512, 128
ROWS_C, COLS_C = "#2f6fd0", "#e07b2a"
axis_lut = np.array([matplotlib.colors.to_rgb(ROWS_C), matplotlib.colors.to_rgb(COLS_C)])
lut = np.zeros((256, 3))
for v in range(1, 256):
    h = ((v * 2654435761) % 2**32) / 2**32
    lut[v] = colorsys.hsv_to_rgb(h, 0.62, 0.96)
lut[0] = (0.06, 0.06, 0.08)
INK, SOFT = "#1B2A33", "#6B7A83"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 9, "text.color": INK, "axes.edgecolor": SOFT})
# (label, [(run folder, tick), ...]) -- one seed per row; the two folders of a row are one run.
rows = [
    ("Run A (seed 5)", [("w_s5", 5000), ("w_s5", 20000), ("w_s5", 50000), ("w_s5", 100000)]),
    ("Run B (seed 4)", [("w_s4", 20000), ("w_s4", 50000), ("w_s4", 100000), ("wl_s4", 200000)]),
]
fig, axes = plt.subplots(2, 5, figsize=(13.4, 6.05), dpi=200)
for r, (label, panels) in enumerate(rows):
    for c, (run, t) in enumerate(panels):
        ax = axes[r, c]
        g = np.fromfile(f"{S}/{run}/grain_{t}.u8", dtype=np.uint8).reshape(W, W)
        along_columns = g & 1  # facings 1, 3, 5, 7 read along a column
        ax.imshow(axis_lut[along_columns], interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{t:,} ticks", fontsize=10, loc="left", pad=4, fontweight="bold")
        ax.set_xlabel(f"written along rows: {100 * (1 - along_columns.mean()):.0f}% of bytes", fontsize=8.5, labelpad=5)
        if c == 0:
            ax.set_ylabel(label, fontsize=10, fontweight="bold", labelpad=6)
    run, t = panels[-1]
    ax = axes[r, 4]
    a = np.fromfile(f"{S}/{run}/sheet_{t}.u8", dtype=np.uint8).reshape(W, W)[:CROP, :CROP]
    ax.imshow(lut[a], interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"the bytes at {t:,} ticks", fontsize=10, loc="left", pad=4, fontweight="bold")
    ax.set_xlabel(f"a {CROP} × {CROP}-byte corner", fontsize=8.5, labelpad=5)
fig.suptitle("Two runs of one symmetric rule: each sheet settles on an axis by itself, then a replicator takes over along it "
             "(512 × 512 bytes, 4,096 processors)", x=0.012, y=0.985, ha="left", fontsize=11, fontweight="bold")
fig.legend(handles=[Patch(color=ROWS_C, label="byte last written by a processor reading along a row"),
                    Patch(color=COLS_C, label="… reading along a column")],
           loc="upper left", bbox_to_anchor=(0.008, 0.955), ncol=2, frameon=False, fontsize=9, handlelength=1.2, columnspacing=1.8)
fig.subplots_adjust(left=0.03, right=0.992, top=0.865, bottom=0.06, wspace=0.05, hspace=0.24)
fig.savefig(f"{OUT}/free_written.png"); plt.close(fig)
