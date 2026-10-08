"""Edited time-lapse of one soup world: variable speed, crossfades between stored snapshots, phase captions, a
synchronised chart and a step counter, rendered at 1920 × 1080 and encoded with ffmpeg.

    python manuscript/figures/soup_video.py --run runs/video/video_L16_st128_k4_s2001 [--fps 24] [--out manuscript/figures/out/stills]

Phases are detected from the run's own samples (zero-byte fraction, most common tape) and the timing of each phase is
set so that the eventful parts (the first replicator's spread, the closed takeover) play slowly and the long uneventful
reigns play fast. Snapshots are blended (crossfade) between stored steps, so the result is smooth even where snapshots
are sparse; where they are dense (every 5–25 steps) the blend is invisible.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import soup_stills as ss  # noqa: E402

W, H = 1920, 1080
INK = (28, 39, 51)
GREY = (107, 114, 128)
TEAL = (30, 138, 138)
RED = (232, 67, 31)
BG = (255, 255, 255)
FONT = "/System/Library/Fonts/Helvetica.ttc"


def font(size: int, bold: bool = False):
    try:
        return ImageFont.truetype(FONT, size, index=1 if bold else 0)
    except OSError:
        return ImageFont.load_default()


def load_samples(stem: str) -> list[dict]:
    rows = []
    for line in open(stem + ".jsonl"):
        if '"kind": "sample"' in line:
            r = json.loads(line)
            ex = r.get("exemplars") or [{}]
            rows.append({"step": r["step"], "zero": r.get("zero_frac", np.nan), "top": r.get("top_share", np.nan),
                         "unique": r.get("unique", np.nan), "tape": ex[0].get("tape", "")})
    rows.sort(key=lambda r: r["step"])
    return rows


def is_pusher(tape: str) -> bool:
    b = tape.split()
    return len(b) >= 4 and b[1].lower() in ("c5", "d5", "e5") and b[0].lower() in ("01", "11", "21") and b[2:4] == b[0:2]


def detect_phases(samples: list[dict], summary: dict, last_step: int) -> dict:
    steps = np.array([s["step"] for s in samples])
    zero = np.array([s["zero"] for s in samples], float)
    t_tar = int(steps[np.argmax(zero >= 0.25)]) if (zero >= 0.25).any() else 500
    t_first = summary.get("tq_10") or next((s["step"] for s in samples if is_pusher(s["tape"])), 1500)
    later = [(s["step"], s["zero"]) for s in samples if s["step"] > t_first + 2000]
    t_closed = next((st for st, z in later if z < 0.02), None)
    return {"t_tar": t_tar, "t_first": int(t_first), "t_closed": t_closed, "last": last_step}


def segments(ph: dict) -> list[dict]:
    """(step range, seconds, caption) in play order. Durations are editorial choices, stated here in one place."""
    tf, tc, last = ph["t_first"], ph["t_closed"], ph["last"]
    seg = [
        {"a": 0, "b": ph["t_tar"], "s": 5.0, "cap": "Twenty thousand random programs. Neighbours run each other.\nStack instructions push empty registers: zero bytes spread. Order, but nothing copies."},
        {"a": ph["t_tar"], "b": tf + 1500, "s": 14.0, "cap": "The first replicator: the two-byte word 01 c5, repeated. Executed, it writes itself\ninto its neighbour and runs on into the neighbour's code. It spreads as a wave."},
    ]
    if tc:
        seg += [
            {"a": tf + 1500, "b": max(tf + 1500, tc - 6000), "s": 9.0, "cap": "The open phase. Every copy depends on the neighbour it was made against:\na world of damaged copies, with pockets of zeros."},
            {"a": max(tf + 1500, tc - 6000), "b": min(last, tc + 6000), "s": 12.0, "cap": "A descendant with one extra instruction, a jump that returns execution\ninto its own code. It no longer reads its neighbour. Every copy is exact."},
            {"a": min(last, tc + 6000), "b": last, "s": 6.0, "cap": "Closed. The organism's future depends on itself alone;\nvariants of the same design compete, and the zeros are gone."},
        ]
    else:
        seg += [{"a": tf + 1500, "b": last, "s": 10.0, "cap": "The open phase: a world of damaged copies."}]
    return seg


class SnapCache:
    def __init__(self, snaps, L, cc):
        self.snaps, self.L, self.cc = snaps, L, cc
        self.steps = np.array([s for _, s, _ in snaps], float)
        self.cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def render(self, i: int):
        if i not in self.cache:
            soup = ss.load(self.snaps[i][2], self.L)
            b = np.asarray(ss.render_blocks(soup, scale=1))
            c = np.asarray(ss.render_classes(soup, self.cc, scale=1))
            self.cache[i] = (b, c)
            for k in [k for k in self.cache if k < i - 2]:
                del self.cache[k]
        return self.cache[i]

    def at(self, step: float):
        j = int(np.searchsorted(self.steps, step))
        if j <= 0:
            return self.render(0)
        if j >= len(self.steps):
            return self.render(len(self.steps) - 1)
        i0, i1 = j - 1, j
        t = (step - self.steps[i0]) / max(self.steps[i1] - self.steps[i0], 1e-9)
        (b0, c0), (b1, c1) = self.render(i0), self.render(i1)
        b = (b0.astype(np.float32) * (1 - t) + b1.astype(np.float32) * t).astype(np.uint8)
        c = (c0.astype(np.float32) * (1 - t) + c1.astype(np.float32) * t).astype(np.uint8)
        return b, c


def chart_image(samples: list[dict], last: int, size=(660, 300)):
    """The run's own statistics against step (log axis), rendered once; returns the image and a step → x mapper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dpi = 100
    fig, ax = plt.subplots(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    st = np.array([s["step"] for s in samples], float)
    st[st < 1] = 1
    ax.plot(st, [s["zero"] for s in samples], color="#9CA3AF", lw=1.6, label="zero bytes")
    ax.plot(st, [s["top"] for s in samples], color="#1E8A8A", lw=1.6, label="most common tape")
    ax.plot(st, [s["unique"] / 20000 for s in samples], color="#1C2733", lw=1.2, ls=":", label="distinct tapes / 20,000")
    ax.set_xscale("log")
    ax.set_xlim(10, last)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("step", fontsize=11)
    ax.tick_params(labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=11, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.2), ncol=3, columnspacing=1.2, handlelength=1.6)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.2)
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    tr = ax.transData
    x0 = tr.transform((10, 0))[0]
    x1 = tr.transform((last, 0))[0]
    y_top = size[1] - tr.transform((10, 1.0))[1]
    y_bot = size[1] - tr.transform((10, 0.0))[1]
    plt.close(fig)

    def step_to_x(step: float) -> float:
        step = max(step, 10)
        return x0 + (np.log10(step) - 1) / (np.log10(last) - 1) * (x1 - x0)

    return img, step_to_x, (y_top, y_bot)


def compose(frame_b: np.ndarray, frame_c: np.ndarray, step: float, caption: str, cap_alpha: float, chart, title: str, top_tape: str) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # title bar
    d.text((60, 36), title, fill=INK, font=font(34, bold=True))
    d.text((W - 60, 40), f"step {int(step):,}", fill=INK, font=font(32), anchor="ra")
    # main lattice (bytes view)
    main = Image.fromarray(frame_b).resize((1152, 900), Image.LANCZOS)
    img.paste(main, (60, 110))
    d.rectangle((60, 110, 60 + 1152, 110 + 900), outline=(180, 186, 193), width=1)
    d.text((60, 1020), "every tape a 4 × 4 block of its bytes at its lattice position · white = zero bytes", fill=GREY, font=font(20))
    # classes view
    cls = Image.fromarray(frame_c).resize((660, 516), Image.NEAREST)
    img.paste(cls, (1240, 110))
    d.rectangle((1240, 110, 1240 + 660, 110 + 516), outline=(180, 186, 193), width=1)
    d.text((1240, 636), "same soup, one pixel per tape, coloured by class · grey = unique random tape", fill=GREY, font=font(18))
    if top_tape:
        d.text((1240, 662), f"most common tape: {top_tape}", fill=INK, font=font(20))
    # chart with cursor
    cimg, step_to_x, (yt, yb) = chart
    cx, cy = 1240, 700
    img.paste(cimg, (cx, cy))
    xcur = cx + step_to_x(step)
    d.line((xcur, cy + yt, xcur, cy + yb), fill=RED, width=3)
    # caption
    if caption and cap_alpha > 0:
        import textwrap
        wrapped = textwrap.fill(" ".join(caption.split()), width=62)
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dl = ImageDraw.Draw(layer)
        a = int(255 * cap_alpha)
        dl.text((1240, 1000), wrapped, fill=INK + (a,), font=font(20), spacing=5)
        img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "stills"))
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--title", default="A soup of 20,000 random Z80 programs on a 160 × 125 lattice")
    a = ap.parse_args()
    stem = a.run if os.path.isabs(a.run) else os.path.join(EXP, a.run)
    L = a.L or int(__import__("re").search(r"_L(\d+)_", stem).group(1))
    summary = json.load(open(stem + ".summary.json"))
    samples = load_samples(stem)
    snaps = [(n, s, p) for n, s, p in ss.snapshot_files(stem) if s is not None]
    snaps = sorted({s: (n, s, p) for n, s, p in snaps}.values(), key=lambda x: x[1])   # one per step
    last = int(snaps[-1][1])
    ph = detect_phases(samples, summary, last)
    seg = segments(ph)
    print("phases", ph, "snapshots", len(snaps))
    cache = SnapCache(snaps, L, ss.ClassColours())
    chart = chart_image(samples, last)
    sample_steps = np.array([s["step"] for s in samples])

    def top_tape_at(step):
        k = int(np.clip(np.searchsorted(sample_steps, step) - 1, 0, len(samples) - 1))
        t = samples[k]["tape"]
        return (t[:23] + " …") if t else ""

    os.makedirs(a.out, exist_ok=True)
    base = os.path.basename(stem)
    mp4 = os.path.join(a.out, f"{base}.edited.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(a.crf), "-movflags", "+faststart", mp4]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n_frames = 0
    # title card
    card = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(card)
    d.text((W // 2, H // 2 - 60), a.title, fill=INK, font=font(44, bold=True), anchor="mm")
    d.text((W // 2, H // 2 + 20), "Each step, 8,192 random cells run their neighbours for 128 instructions. Nothing is selected.", fill=GREY, font=font(26), anchor="mm")
    d.text((W // 2, H // 2 + 70), "One world, 100,000 steps, played at variable speed.", fill=GREY, font=font(26), anchor="mm")
    for _ in range(int(3.0 * a.fps)):
        proc.stdin.write(card.tobytes()); n_frames += 1
    hold = 0.6   # seconds of hold at each phase boundary
    for sg in seg:
        nf = int(sg["s"] * a.fps)
        for k in range(nf):
            u = k / max(nf - 1, 1)
            step = sg["a"] + (sg["b"] - sg["a"]) * u
            fb, fc = cache.at(step)
            t_in, t_out = k / a.fps, (nf - k) / a.fps
            alpha = min(1.0, t_in / 0.6, t_out / 0.6)
            img = compose(fb, fc, step, sg["cap"], alpha, chart, a.title, top_tape_at(step))
            proc.stdin.write(img.tobytes()); n_frames += 1
        for _ in range(int(hold * a.fps)):
            proc.stdin.write(img.tobytes()); n_frames += 1
    end = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(end)
    d.text((W // 2, H // 2 - 30), "Heredity came first. Individuality was a return in control flow.", fill=INK, font=font(40, bold=True), anchor="mm")
    d.text((W // 2, H // 2 + 40), f"first replicator at step {ph['t_first']:,} · closed successor at step {ph['t_closed']:,}" if ph["t_closed"] else f"first replicator at step {ph['t_first']:,}", fill=GREY, font=font(26), anchor="mm")
    for _ in range(int(3.0 * a.fps)):
        proc.stdin.write(end.tobytes()); n_frames += 1
    proc.stdin.close(); proc.wait()
    print("wrote", mp4, n_frames, "frames", f"{n_frames / a.fps:.1f} s")


if __name__ == "__main__":
    main()
