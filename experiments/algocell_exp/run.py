"""Run one condition headlessly and stream measurements as JSON lines.

    python -m algocell_exp.run --suppress "family:block-copy" --seed 6 --horizon 30000 --out run.jsonl

Every sample line has: step, species stats, byte entropy, high-order entropy,
top-3 exemplar tapes with their write mechanisms, elapsed seconds. The last
line (kind="summary") records emergence steps for several thresholds (-1 if
never reached within the horizon) so censoring is explicit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from .isa import count
from .metrics import exemplars, high_order_entropy, motif_share, quasispecies_share, species_stats
from .soup import Soup, adapter_summary

THRESHOLDS = (0.02, 0.10, 0.50)  # on exact-hash top species share
Q_THRESHOLDS = (0.10, 0.50)      # on q_share (quasispecies occupancy, Hamming radius L/4)


def run(
    *,
    grid: str = "square",
    tape: int | None = None,
    width: int = 160,
    height: int = 125,
    seed: int = 6,
    pairs: int = 8192,
    z80_steps: int = 128,
    noise_exp: int = 4,
    suppress: str | None = None,
    horizon: int = 30000,
    sample_every: int = 250,
    stop_share: float | None = 0.5,
    stop_after: int = 4,
    label: str = "",
    out=None,
    quiet: bool = False,
) -> dict:
    soup = Soup(width, height, grid, tape, seed, pairs, z80_steps, noise_exp, suppress)
    cond = {
        "label": label,
        "grid": grid,
        "tape_length": soup.tape_length,
        "width": width,
        "height": height,
        "cells": soup.cell_count,
        "seed": seed,
        "pairs": pairs,
        "z80_steps": z80_steps,
        "noise_exp": noise_exp,
        "mutation_rate": 1 / 2**noise_exp,
        "mutations_per_step": soup.mutation_count,
        "suppress": soup.patterns,
        "suppressed": count(soup.sets),
        "suppressed_by_page": {k: len(v) for k, v in soup.sets.items()},
        "horizon": horizon,
        "sample_every": sample_every,
        "adapter": adapter_summary(),
    }
    emit = (lambda d: (out.write(json.dumps(d) + "\n"), out.flush())) if out else (lambda d: None)
    emit({"kind": "condition", **cond})

    emergence = {f"t_{int(t*100):02d}": -1 for t in THRESHOLDS}
    emergence.update({f"tq_{int(t*100):02d}": -1 for t in Q_THRESHOLDS})
    first_mech = None
    t0 = time.perf_counter()
    step = 0
    stop_at = None
    samples = 0
    while step < horizon:
        n = min(sample_every, horizon - step)
        soup.step(n)
        step += n
        hashes = soup.read_hashes()
        sp = species_stats(hashes)
        soup_arr = soup.read_soup()
        hoe = high_order_entropy(soup_arr)
        ex = exemplars(soup_arr, hashes, sp["top3_hashes"])
        top_idx = int(np.argmax(hashes == np.uint32(sp["top_hash"])))
        qs = quasispecies_share(soup_arr, soup_arr[top_idx])
        ms = motif_share(soup_arr, soup_arr[top_idx])
        rec = {"kind": "sample", "step": step, **sp, **qs, **ms, **hoe, "exemplars": ex, "elapsed_s": round(time.perf_counter() - t0, 2)}
        emit(rec)
        samples += 1
        if not quiet:
            print(
                f"[{label or 'run'} s={seed}] step {step:>7} top {sp['top_share']:.4f} q {qs['q_share']:.3f} motif {ms['motif_share']:.3f} uniq {sp['unique']:>6} "
                f"H {sp['H_species']:.2f} hoe {hoe['hoe']:.3f} {ex[0]['tape'][:23]}.. {'+'.join(ex[0]['mechanisms']) or '-'}",
                file=sys.stderr,
            )
        for t in THRESHOLDS:
            key = f"t_{int(t*100):02d}"
            if emergence[key] < 0 and sp["top_share"] >= t:
                emergence[key] = step
                if first_mech is None:
                    first_mech = {"step": step, "tape": ex[0]["tape"], "mechanisms": ex[0]["mechanisms"], "share": sp["top_share"]}
        for t in Q_THRESHOLDS:
            key = f"tq_{int(t*100):02d}"
            if emergence[key] < 0 and qs["q_share"] >= t:
                emergence[key] = step
        if stop_share is not None and stop_at is None and qs["q_share"] >= stop_share:
            stop_at = step + stop_after * sample_every
        if stop_at is not None and step >= stop_at:
            break
    final_hashes = soup.read_hashes()
    sp = species_stats(final_hashes)
    soup_arr = soup.read_soup()
    top_idx = int(np.argmax(final_hashes == np.uint32(sp["top_hash"])))
    summary = {
        "kind": "summary",
        **cond,
        "steps_run": step,
        "censored": emergence["tq_10"] < 0,
        **emergence,
        "first_emergent": first_mech,
        "final": {
            **sp,
            **quasispecies_share(soup_arr, soup_arr[top_idx]),
            **motif_share(soup_arr, soup_arr[top_idx]),
            **high_order_entropy(soup_arr),
            "exemplars": exemplars(soup_arr, final_hashes, sp["top3_hashes"]),
        },
        "wall_s": round(time.perf_counter() - t0, 2),
        # Wall-clock throughput incl. sampling overhead (submit time alone would overstate it ~8x).
        "slots_per_s": round(step * pairs * z80_steps / max(time.perf_counter() - t0, 1e-9)),
    }
    emit(summary)
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="square", choices=["square", "hex"])
    ap.add_argument("--tape", type=int, default=None, help="bytes per cell (square grid): 4, 9, 16, 25, 36, 49, 64, 81, 100")
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=125)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--pairs", type=int, default=8192)
    ap.add_argument("--z80-steps", type=int, default=128)
    ap.add_argument("--noise-exp", type=int, default=4, help="mutation rate = 1/2^n")
    ap.add_argument("--suppress", default=None, help="';'-separated patterns (see z80-opcodes.ts)")
    ap.add_argument("--horizon", type=int, default=30000, help="simulation steps")
    ap.add_argument("--sample-every", type=int, default=250)
    ap.add_argument("--stop-share", type=float, default=0.5, help="stop a few samples after the quasispecies (q4) occupancy reaches this share (-1 = never)")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None, help="JSONL path (default: stdout)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    out = open(a.out, "a") if a.out else sys.stdout
    try:
        s = run(
            grid=a.grid, tape=a.tape, width=a.width, height=a.height, seed=a.seed, pairs=a.pairs, z80_steps=a.z80_steps,
            noise_exp=a.noise_exp, suppress=a.suppress, horizon=a.horizon, sample_every=a.sample_every,
            stop_share=None if a.stop_share < 0 else a.stop_share, label=a.label, out=out, quiet=a.quiet,
        )
    finally:
        if a.out:
            out.close()
    if a.out:
        print(json.dumps({k: s[k] for k in ("label", "seed", "steps_run", "t_02", "t_10", "tq_10", "tq_50", "wall_s", "slots_per_s")}), file=sys.stderr)


if __name__ == "__main__":
    main()
