"""Pictures of the free-running world from saved sheets (scripts/free-look.mjs output).
  python docs/free-running/figures.py <dir with run folders from scripts/free-look.mjs> <out dir>
Every pixel is one byte of the sheet; nothing is smoothed."""
import sys, colorsys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
S, OUT = sys.argv[1], sys.argv[2]
W = 512
lut = np.zeros((256, 3))
for v in range(1, 256):
    h = ((v * 2654435761) % 2**32) / 2**32
    lut[v] = colorsys.hsv_to_rgb(h, 0.62, 0.96)
lut[0] = (0.06, 0.06, 0.08)
INK, SOFT = "#1B2A33", "#6B7A83"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 9, "text.color": INK, "axes.edgecolor": SOFT})

def sheet(run, t):
    return np.fromfile(f"{S}/{run}/sheet_{t}.u8", dtype=np.uint8).reshape(W, W)
def heads(run, t):
    h = np.fromfile(f"{S}/{run}/heads_{t}.u32", dtype=np.uint32).reshape(-1, 40).astype(np.int64)
    s8 = lambda v: ((v + 128) % 256) - 128
    x = (h[:, 32] + s8(h[:, 19] & 0xff)) % W
    y = (h[:, 33] + s8(h[:, 19] >> 8)) % W
    return x, y

def strip(name, run, ticks, title, notes):
    fig, axes = plt.subplots(1, len(ticks), figsize=(3.3 * len(ticks), 3.95), dpi=200)
    for ax, t, note in zip(axes, ticks, notes):
        ax.imshow(lut[sheet(run, t)], interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{t:,} ticks", fontsize=10, loc="left", pad=4, fontweight="bold")
        ax.set_xlabel(note, fontsize=8.5, color=INK, labelpad=5)
    fig.suptitle(title, x=0.012, y=0.985, ha="left", fontsize=11.5, fontweight="bold")
    fig.subplots_adjust(left=0.012, right=0.988, top=0.86, bottom=0.1, wspace=0.05)
    fig.savefig(f"{OUT}/{name}.png"); plt.close(fig)

strip("free_takeover", "g_pic_s1", [2000, 4000, 4500, 5500],
      "A replicator appears and takes the sheet (512 × 512 bytes, 4,096 processors, one run)",
      ["waiting: 40% of bytes blank (dark)", "01 c5 has appeared: c5 is 6% of bytes", "c5 is 16% of bytes", "c5 is 49% of bytes"])
strip("free_second_takeover", "g_pic_s3", [60000, 65000, 80000, 100000],
      "Later, a variant that loops takes over from the first replicator (same sheet size, another run)",
      ["01 c5 everywhere", "18 f8 (a jump back) appears: 18 is 8% of bytes", "18 is 20% of bytes", "settled: the repeating unit is 6 bytes"])

# Close-ups with the bytes written out and the processors marked.
def closeup(ax, run, t, x0, y0, title, cols=32, rows=10):
    a = sheet(run, t)[y0:y0 + rows, x0:x0 + cols]
    ax.imshow(lut[a], interpolation="nearest", extent=(0, cols, rows, 0), aspect="equal")
    for y in range(rows):
        for x in range(cols):
            v = a[y, x]
            ax.text(x + 0.5, y + 0.53, f"{v:02x}", ha="center", va="center", fontsize=5.6, family="Menlo",
                    color="#E8ECEE" if v == 0 else "#10181D")
    hx, hy = heads(run, t)
    n = 0
    for x, y in zip(hx, hy):
        if x0 <= x < x0 + cols and y0 <= y < y0 + rows:
            ax.add_patch(Rectangle((x - x0 + 0.04, y - y0 + 0.04), 0.92, 0.92, fill=False, ec="white", lw=1.6))
            ax.add_patch(Rectangle((x - x0 + 0.04, y - y0 + 0.04), 0.92, 0.92, fill=False, ec="black", lw=0.6))
            n += 1
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title}   ({n} processors in view, outlined)", fontsize=9.5, loc="left", pad=4, fontweight="bold")
fig, axes = plt.subplots(3, 1, figsize=(9.6, 9.9), dpi=200)
closeup(axes[0], "g_pic_s1", 2000, 96, 120, "Waiting state, 2,000 ticks")
closeup(axes[1], "g_pic_s1", 10000, 96, 120, "First replicator, 10,000 ticks: 01 c5 repeated")
closeup(axes[2], "g_pic_s3", 100000, 96, 120, "Looping variant, 100,000 ticks: c5 01 c5 01 18 f8 repeated")
fig.suptitle("The same patch of sheet, byte by byte (32 × 10 bytes)", x=0.02, y=0.992, ha="left", fontsize=11.5, fontweight="bold")
fig.subplots_adjust(left=0.02, right=0.98, top=0.945, bottom=0.01, hspace=0.16)
fig.savefig(f"{OUT}/free_closeups.png"); plt.close(fig)
