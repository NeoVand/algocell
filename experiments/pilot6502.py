"""Local pilot of the 6502 soup (PREREG_6502_DRAFT.md, "Before registration" item 3): BUILT, NOT RUN.

    cd experiments
    python pilot6502.py --out results/pilot6502                         # the pilot: L = 16 and 32, seeds 1-4, 100,000 steps
    python pilot6502.py --smoke --out <scratch dir>                     # plumbing only: 200 steps, L = 16, one seed

The pilot is to be run only after the pre-registration's predictions are locked. Each world is Algocell's soup (ZSoup:
the original lattice, pairing, mutation and statistics kernels, 160 × 125, 8,192 draws per step, background mutation
k = 4) with the CPU replaced by the vendored 6502 machine `m6502_pair_maps_L<L>` (BRK faithful; `--brk halt` gives
the BRK-halts arm), 128 instructions per encounter. Per world it writes, under --out:

- `<world>.jsonl`: a condition line (machine and shader provenance), then every `--sample-every` steps (500) a sample:
  steps/s over the interval, the share of the most common exact tape (`top_share`, as run.py: hashes of all cells),
  the quasispecies share around it (`q_share`, Hamming radius L/4, and `q_half_share`, as run.py), the number of
  distinct tapes, the byte histogram of the soup (256 bins), the zero-byte fraction, and the active pairs of the
  last step; a summary line at the end;
- `<world>.t<step>.u8.br`: brotli-compressed soups (cells × L bytes) every `--snapshot-every` steps (25,000) and the
  final soup `<world>.final.u8.br`.

It records no disassembly, mechanism or replicator class: run.py's exemplars and mechanisms are Z80 code, and the
classification of 6502 forms belongs to the registered analysis, after the pilot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import brotli
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from algocell_exp.metrics import quasispecies_share, species_stats  # noqa: E402
from algocell_exp.soup import adapter_summary  # noqa: E402
from algocell_exp.zsoup import ZSoup  # noqa: E402


def machine_for(L: int, brk: str) -> str:
    return f"m6502_{'brkhalt' if brk == 'halt' else 'pair'}_maps_L{L}"


def run_world(L: int, seed: int, steps: int, out: Path, brk: str = "faithful", budget: int = 128, sample_every: int = 500,
              snapshot_every: int = 25000, submit: int = 50, label: str = "pilot6502") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    mname = machine_for(L, brk)
    world = f"{label}_{mname}_s{seed}_st{budget}"
    path = out / f"{world}.jsonl"
    if path.exists():
        raise FileExistsError(f"{path} exists; a world is never overwritten")
    soup = ZSoup(mname, seed=seed, z80_steps=budget)
    cond = {"kind": "condition", "world": world, "L": L, "seed": seed, "brk": brk, "budget": budget, "steps": steps,
            "width": soup.width, "height": soup.height, "pair_count": soup.pair_count, "noise_exp": soup.noise_exp,
            "sample_every": sample_every, "snapshot_every": snapshot_every, "adapter": adapter_summary(),
            "machine": soup.machine_provenance, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    f = open(path, "w")
    emit = lambda d: (f.write(json.dumps(d) + "\n"), f.flush())  # noqa: E731
    emit(cond)

    def snapshot(tag: str, cells: np.ndarray) -> None:
        (out / f"{world}.{tag}.u8.br").write_bytes(brotli.compress(np.ascontiguousarray(cells, np.uint8).tobytes(), quality=9))

    step = 0
    t_start = time.perf_counter()
    while step < steps:
        target = min(steps, (step // sample_every + 1) * sample_every)
        t0 = time.perf_counter()
        first = step
        while step < target:
            n = min(submit, target - step)
            soup.step(n)
            step += n
        soup.sync()
        dt = time.perf_counter() - t0
        hashes = soup.read_hashes()
        cells = soup.read_soup()
        sp = species_stats(hashes)
        top_idx = int(np.argmax(hashes == np.uint32(sp["top_hash"])))
        qs = quasispecies_share(cells, cells[top_idx])
        hist = soup.read_byte_counts()
        inter = soup.read_interactions()
        emit({"kind": "sample", "step": step, "steps_per_s": round((step - first) / dt, 2),
              "top_share": sp["top_share"], "top_hash": sp["top_hash"], "top_tape": cells[top_idx].tobytes().hex(), "unique": sp["unique"],
              "q_share": qs["q_share"], "q_half_share": qs["q_half_share"], "q_radius": qs["q_radius"],
              "byte_hist": hist.astype(int).tolist(), "zero_frac": float(hist[0] / max(int(hist.sum()), 1)),
              "active_pairs": int(inter["active"].sum()), "elapsed_s": round(time.perf_counter() - t_start, 2)})
        if snapshot_every and step % snapshot_every == 0 and step < steps:
            snapshot(f"t{step}", cells)
    cells = soup.read_soup()
    snapshot("final", cells)
    wall = time.perf_counter() - t_start
    summary = {"kind": "summary", "world": world, "steps_run": step, "wall_s": round(wall, 1), "steps_per_s": round(step / wall, 2),
               "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    emit(summary)
    f.close()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--L", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--brk", choices=["faithful", "halt"], default="faithful")
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--snapshot-every", type=int, default=25_000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true", help="plumbing only: 200 steps, L = 16, seed 1, samples every 100")
    ap.add_argument("--i-am-authorised", action="store_true", help="required for anything but --smoke (the pilot waits for the locked predictions)")
    a = ap.parse_args()
    if a.smoke:
        s = run_world(16, 1, 200, a.out, brk=a.brk, budget=a.budget, sample_every=100, snapshot_every=100, label="smoke6502")
        print(json.dumps(s))
        return
    if not a.i_am_authorised:
        sys.exit("refusing to run the pilot: it waits for the pre-registration's locked predictions (pass --i-am-authorised once they are)")
    for L in a.L:
        for seed in a.seeds:
            print(json.dumps(run_world(L, seed, a.steps, a.out, brk=a.brk, budget=a.budget, sample_every=a.sample_every,
                                       snapshot_every=a.snapshot_every)), flush=True)


if __name__ == "__main__":
    main()
