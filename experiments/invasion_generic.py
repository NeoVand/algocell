"""R4 C4b/C4c — invasions with self-initialising closers (REVISION_PREREG R4), one world per call.

cond = {L: 16, variant: None|'randreg', resident, invader ('pusher'|'SI13'|'R4'|'none'), seed, steps}. 'SI13' is the
self-initialising transmitter `21 00 00 11 10 00 01 10 00 ed b0 18 fe` + 3 random bytes per cell; 'R4' the regenerator
`44 5e ed b0`x4; 'pusher' `01 c5`x8. Every 250 steps: shares of cells carrying each design. Final soup saved.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SI_CORE = np.array([0x21, 0x00, 0x00, 0x11, 0x10, 0x00, 0x01, 0x10, 0x00, 0xED, 0xB0, 0x18, 0xFE], np.uint8)
R4 = np.array([0x44, 0x5E, 0xED, 0xB0] * 4, np.uint8)
PUSHER = np.array([0x01, 0xC5] * 8, np.uint8)
R_VALS = {0x04, 0x24, 0x44, 0x64, 0x84, 0xA4, 0x08, 0x48, 0x68, 0x88, 0xA8, 0xC8, 0xE8}


def conditions():
    out = []
    for s in range(1, 6):
        out += [{"L": 16, "variant": "randreg", "resident": "pusher", "invader": "SI13", "seed": s, "steps": 50000},
                {"L": 16, "variant": "randreg", "resident": "SI13", "invader": "pusher", "seed": s, "steps": 50000},
                {"L": 16, "variant": None, "resident": "R4", "invader": "SI13", "seed": s, "steps": 100000},
                {"L": 16, "variant": None, "resident": "SI13", "invader": "R4", "seed": s, "steps": 100000}]
    for s in range(1, 4):
        out += [{"L": 16, "variant": "randreg", "resident": "pusher", "invader": "none", "seed": s, "steps": 50000},
                {"L": 16, "variant": "randreg", "resident": "SI13", "invader": "none", "seed": s, "steps": 50000},
                {"L": 16, "variant": None, "resident": "R4", "invader": "none", "seed": s, "steps": 100000},
                {"L": 16, "variant": None, "resident": "SI13", "invader": "none", "seed": s, "steps": 100000}]
    return out


def stem(c):
    return f"{c['variant'] or 'std'}_{c['resident']}_{c['invader']}_s{c['seed']}"


def make(kind, n, rng):
    if kind == "SI13":
        return np.concatenate([np.tile(SI_CORE, (n, 1)), rng.integers(0, 256, size=(n, 3), dtype=np.uint8)], 1)
    return np.tile({"R4": R4, "pusher": PUSHER}[kind], (n, 1))


def shares(s):
    hp = np.full(len(s), 16)
    for sh in range(16):
        hp = np.minimum(hp, (s != np.roll(PUSHER, sh)).sum(1))
    si = (s[:, :11] != SI_CORE[:11]).sum(1) <= 1
    r4 = (s[:, 1] == 0x5E) & (s[:, 2] == 0xED) & (s[:, 3] == 0xB0) & np.isin(s[:, 0], list(R_VALS))
    return {"pusher": float((hp <= 4).mean()), "SI13": float(si.mean()), "R4": float(r4.mean()), "zero_frac": float((s == 0).mean())}


def run_world(c, outdir):
    from algocell_exp.soup import Soup
    os.makedirs(outdir, exist_ok=True)
    seed = 9100 + c["seed"] + 37 * ["pusher", "SI13", "R4"].index(c["resident"]) + 300 * (c["variant"] == "randreg")
    soup = Soup(160, 125, "square", 16, seed, 8192, 128, 4, [], shader_variant=c["variant"])
    rng = np.random.default_rng([seed, 5])
    N = soup.cell_count
    cells = make(c["resident"], N, rng)
    if c["invader"] != "none":
        idx = rng.choice(N, size=N // 100, replace=False)
        cells[idx] = make(c["invader"], len(idx), rng)
    soup.write_soup(cells.astype(np.uint8))
    step, t0 = 0, time.time()
    with open(os.path.join(outdir, stem(c) + ".jsonl"), "w") as f:
        f.write(json.dumps({"kind": "condition", **c, "soup_seed": seed}) + "\n")
        while step <= c["steps"]:
            f.write(json.dumps({"step": step, **shares(soup.read_soup())}) + "\n")
            dt = 50 if step < 2000 else 250
            soup.step(dt)
            step += dt
    np.save(os.path.join(outdir, stem(c) + "_final.npy"), soup.read_soup())
    return {**c, "wall_s": round(time.time() - t0, 1)}
