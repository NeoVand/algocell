"""Stills and time-lapse frames of a soup from the stored snapshots (brotli-compressed uint8 arrays, shape (N, L)).

    python manuscript/figures/soup_stills.py --run runs/stageG/none@closure_L16_st128_k4_s2001 [--L 16] [--out manuscript/figures/out/stills]
    python manuscript/figures/soup_stills.py --run ... --video        # frames from every snapshot → mp4 via ffmpeg

Two renderings of the same soup, both with one cell per tape in the soup's own order (index i stays at the same place
over time, so takeovers appear as the picture filling in):
  bytes   every tape at its lattice position (160 × 125, the soup's real neighbourhood structure: pairs are lattice
          neighbours), drawn as a 4 × 4 block of its bytes coloured by value (zero, the tar, near-white; other values
          by a golden-angle hue permutation); the random soup is noise, the tar is empty, a replicator is a texture.
  classes one pixel per tape coloured by what it is: the k most abundant classes get fixed colours (ranked at each
          frame by abundance, colours assigned by first appearance so a class keeps its colour across frames),
          singletons are light grey, all-zero tapes are white.
"""

from __future__ import annotations

import argparse
import colorsys
import glob
import json
import os
import re
import subprocess

import brotli
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", ".."))

INK = (28, 39, 51)
CLASS_COLOURS = [(30, 138, 138), (232, 67, 31), (0, 114, 178), (230, 159, 0), (204, 121, 167), (0, 158, 115), (86, 180, 233), (213, 94, 0)]
SINGLETON = (226, 228, 231)
ZERO_TAPE = (255, 255, 255)


def byte_lut() -> np.ndarray:
    lut = np.zeros((256, 3), np.uint8)
    # golden-angle hue permutation: any two distinct byte values, however close, get well-separated hues
    for v in range(256):
        h = (v * 0.6180339887) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.70, 0.82 if v % 2 else 0.62)
        lut[v] = (int(r * 255), int(g * 255), int(b * 255))
    lut[0] = (250, 250, 250)
    return lut


LUT = byte_lut()


def load(path: str, L: int) -> np.ndarray:
    a = np.frombuffer(brotli.decompress(open(path, "rb").read()), np.uint8)
    return a.reshape(-1, L)


def snapshot_files(run_stem: str) -> list[tuple[str, int | None, str]]:
    """(name, step or None, path) sorted by step; 'emergence' placed by its step from summary.json, 'final' last."""
    files = glob.glob(run_stem + ".soup_*.u8.br")
    em_step = None
    sp = run_stem + ".summary.json"
    if os.path.exists(sp):
        s = json.load(open(sp))
        em_step = s.get("tq_10")
        if em_step is not None and em_step < 0:
            em_step = None
        final_step = s.get("steps_run")
    else:
        final_step = None
    out = []
    for f in files:
        name = re.search(r"\.soup_(.*)\.u8\.br$", f).group(1)
        if name.startswith("t"):
            out.append((name, int(name[1:]), f))
        elif name == "emergence":
            out.append((name, em_step, f))
        elif name == "final":
            out.append((name, final_step, f))
    out.sort(key=lambda x: (x[1] if x[1] is not None else 10**12, x[0] != "final"))
    return out


def render_bytes(soup: np.ndarray, cols: int = 50, scale: int = 1) -> Image.Image:
    N, L = soup.shape
    rows = -(-N // cols)
    pad = np.zeros((rows * cols, L), np.uint8)
    pad[:N] = soup
    img = LUT[pad.reshape(rows, cols * L)]
    im = Image.fromarray(img, "RGB")
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    return im


def render_blocks(soup: np.ndarray, width: int = 160, scale: int = 2) -> Image.Image:
    """Spatial byte view: tape i sits at lattice position (i % width, i // width), drawn as a square block of its
    bytes (side = ceil(sqrt(L)); unused block pixels white), so both the lattice and the byte texture are visible."""
    N, L = soup.shape
    side = int(np.ceil(np.sqrt(L)))
    rows = -(-N // width)
    blocks = np.full((rows * width, side * side), 0, np.uint8)
    blocks[:N, :L] = soup
    img = LUT[blocks]                                   # (cells, side*side, 3)
    img = img.reshape(rows, width, side, side, 3).transpose(0, 2, 1, 3, 4).reshape(rows * side, width * side, 3)
    if side * side > L:                                  # padding pixels white
        pad = np.zeros((rows * width, side * side), bool)
        pad[:, L:] = True
        pad = pad.reshape(rows, width, side, side).transpose(0, 2, 1, 3).reshape(rows * side, width * side)
        img[pad] = (255, 255, 255)
    im = Image.fromarray(np.ascontiguousarray(img), "RGB")
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST)


class ClassColours:
    """Stable colours for abundant classes across frames (assigned on first appearance in the top k)."""

    def __init__(self, k: int = 6):
        self.k = k
        self.assigned: dict[bytes, tuple[int, int, int]] = {}
        self.next = 0

    def colour(self, key: bytes) -> tuple[int, int, int]:
        if key not in self.assigned:
            self.assigned[key] = CLASS_COLOURS[self.next % len(CLASS_COLOURS)]
            self.next += 1
        return self.assigned[key]


def render_classes(soup: np.ndarray, cc: ClassColours, cols: int = 160, scale: int = 4, min_count: int = 2) -> Image.Image:
    N, L = soup.shape
    uniq, inv, counts = np.unique(soup, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    order = np.argsort(-counts, kind="stable")
    colour_of = np.tile(np.array(SINGLETON, np.uint8), (len(uniq), 1))
    zero = ~uniq.any(axis=1)
    colour_of[zero] = ZERO_TAPE
    top = [i for i in order[: cc.k] if counts[i] >= min_count and not zero[i]]
    for i in top:
        colour_of[i] = cc.colour(uniq[i].tobytes())
    rows = -(-N // cols)
    img = np.full((rows * cols, 3), 255, np.uint8)
    img[:N] = colour_of[inv]
    im = Image.fromarray(img.reshape(rows, cols, 3), "RGB")
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST)


def label(im: Image.Image, text: str, size: int = 28) -> Image.Image:
    """Add a white band with a step label under the image."""
    band = Image.new("RGB", (im.width, size + 16), (255, 255, 255))
    out = Image.new("RGB", (im.width, im.height + band.height), (255, 255, 255))
    out.paste(im, (0, 0))
    out.paste(band, (0, im.height))
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        font = ImageFont.load_default()
    d.text((8, im.height + 6), text, fill=INK, font=font)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run stem, e.g. runs/stageG/none@closure_L16_st128_k4_s2001")
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "stills"))
    ap.add_argument("--which", default="", help="comma-separated snapshot names to render (default: all)")
    ap.add_argument("--video", action="store_true", help="also assemble an mp4 from every snapshot (ffmpeg)")
    ap.add_argument("--fps", type=float, default=2.0)
    a = ap.parse_args()
    stem = a.run if os.path.isabs(a.run) else os.path.join(EXP, a.run)
    L = a.L or int(re.search(r"_L(\d+)_", stem).group(1))
    os.makedirs(a.out, exist_ok=True)
    base = os.path.basename(stem)
    snaps = snapshot_files(stem)
    want = set(a.which.split(",")) if a.which else None
    cc = ClassColours()
    frames_b, frames_c = [], []
    for name, step, path in snaps:
        soup = load(path, L)
        imb = render_blocks(soup)
        imc = render_classes(soup, cc)
        if want is None or name in want:
            imb.save(os.path.join(a.out, f"{base}.bytes.{name}.png"))
            imc.save(os.path.join(a.out, f"{base}.classes.{name}.png"))
        txt = f"step {step:,}" if step is not None else name
        if name == "emergence":
            txt += "  (first heritable replicator reaches 10%)"
        frames_b.append(label(imb.resize((imb.width, imb.height), Image.NEAREST), txt))
        frames_c.append(label(imc, txt))
        print(f"{name:10s} step={step} zero={np.mean(soup == 0):.3f}")
    if a.video and shutil_which("ffmpeg"):
        for tag, frames in (("bytes", frames_b), ("classes", frames_c)):
            fdir = os.path.join(a.out, f"frames_{tag}_{base}")
            os.makedirs(fdir, exist_ok=True)
            w, h = frames[0].size
            w, h = w - w % 2, h - h % 2
            for i, fr in enumerate(frames):
                fr.crop((0, 0, w, h)).save(os.path.join(fdir, f"f{i:05d}.png"))
            mp4 = os.path.join(a.out, f"{base}.{tag}.mp4")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(fdir, "f%05d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", mp4], check=True)
            print("wrote", mp4)


def shutil_which(x):
    from shutil import which
    return which(x)


if __name__ == "__main__":
    main()
