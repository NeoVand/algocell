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

import brotli
import numpy as np

from .isa import count
from .metrics import census, exemplars, high_order_entropy, motif_share, quasispecies_share, shift_occupancy, species_stats
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
    on_snapshot=None,
    random_tapes: int = 0,
    mutations_per_step: int | None = None,
    sample_every_early: int | None = None,
    early_until: int = 0,
    provenance: dict | None = None,
    sample_steps: list[int] | None = None,
    snapshot_steps: list[int] | None = None,
    census_pairs: int = 128,
    exemplar_count: int = 3,
    mem_length: int | None = None,
    zero_halts: bool = False,
) -> dict:
    """on_snapshot(name, soup_uint8_2d) is called with 'emergence' (first tq_10
    crossing) and 'final' so callers can persist soup snapshots.

    Sampling: every `sample_every_early` steps until `early_until`, then every
    `sample_every` (emergence often happens within the first few thousand steps,
    so a 500-step grid alone puts it at the resolution floor), plus every step listed
    in `sample_steps` (e.g. 1, 2, 3, 5, 8, 13, 21, 34: the zero flood forms within the
    first ~30 steps). At every step in `snapshot_steps` the full soup is passed to
    `on_snapshot(f"t{step}", soup)` and an interaction census is recorded: the before/after
    state of every pair that interacted in that step (aggregates + `census_pairs` raw pairs).
    Every sample also carries the shader's per-pair write counters summarised
    (`ix_*`). `mutations_per_step` overrides the default pair_count/2^noise_exp (control
    arms). `provenance` is stored verbatim in the condition record. `zero_halts` selects the lethal-tar rule (a zero
    byte fetched as an opcode halts the pair; Stage I control, square grid only) and is recorded in the condition record."""
    soup = Soup(width, height, grid, tape, seed, pairs, z80_steps, noise_exp, suppress, mutations_per_step=mutations_per_step, mem_length=mem_length, zero_halts=zero_halts)
    cond = {
        "label": label,
        "grid": grid,
        "zero_halts": soup.zero_halts,
        "tape_length": soup.tape_length,
        "mem_length": soup.mem_length,
        "width": width,
        "height": height,
        "cells": soup.cell_count,
        "seed": seed,
        "pairs": pairs,
        "z80_steps": z80_steps,
        "noise_exp": noise_exp,
        "mutation_rate": 1 / 2**noise_exp,  # legacy name: mutated bytes per pair slot per step
        "mutations_per_step": soup.mutation_count,
        "mutation_per_cell_per_step": soup.mutation_count / soup.cell_count,
        "mutation_per_byte_per_step": soup.mutation_count / (soup.cell_count * soup.tape_length),
        "suppress": soup.patterns,
        "suppressed": count(soup.sets),
        "suppressed_by_page": {k: len(v) for k, v in soup.sets.items()},
        "horizon": horizon,
        "sample_every": sample_every,
        "sample_every_early": sample_every_early,
        "early_until": early_until,
        "stop_share": stop_share,
        "stop_after": stop_after,
        "random_tapes": random_tapes,
        "sample_steps": sorted({int(x) for x in (sample_steps or []) if 0 < int(x) <= horizon}),
        "snapshot_steps": sorted({int(x) for x in (snapshot_steps or []) if 0 < int(x) <= horizon}),
        "census_pairs": census_pairs,
        "exemplar_count": exemplar_count,
        "adapter": adapter_summary(),
        **_provenance(soup, provenance),
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
    extra_steps = cond["sample_steps"]
    snaps = set(cond["snapshot_steps"])
    while step < horizon:
        target = next_sample_step(step, horizon, sample_every, sample_every_early, early_until, extra_steps)
        census_now = target in snaps and census_pairs > 0
        if census_now:
            # the census needs the soup BEFORE the last step of the interval and the pair memory after it
            if target - step > 1:
                soup.step(target - step - 1)
            before = soup.read_soup()
            soup.step(1)
        else:
            soup.step(target - step)
        step = target
        inter = soup.read_interactions()
        active_pairs = int(inter["active"].sum())
        byte_hist = soup.read_byte_counts()
        hashes = soup.read_hashes()
        sp = species_stats(hashes)
        soup_arr = soup.read_soup()
        hoe = high_order_entropy(soup_arr)
        ex = exemplars(soup_arr, hashes, sp["top10_hashes"][:exemplar_count], suppress=soup.sets)
        top_idx = int(np.argmax(hashes == np.uint32(sp["top_hash"])))
        qs = quasispecies_share(soup_arr, soup_arr[top_idx])
        qs.update(shift_occupancy(soup_arr, soup_arr[top_idx]))
        ms = motif_share(soup_arr, soup_arr[top_idx])
        cs = census(soup_arr)
        rec = {
            "kind": "sample", "step": step, **sp, **qs, **ms, **hoe, **cs, "exemplars": ex,
            "active_pairs": active_pairs,                      # pairs that interacted in the last step of this interval
            "zero_frac": float(byte_hist[0] / max(int(byte_hist.sum()), 1)),  # fraction of soup bytes equal to 0x00
            "byte_hist": byte_hist.astype(int).tolist(),       # 256-bin histogram of all soup bytes (padding excluded)
            **interaction_summary(inter, soup.tape_length),
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
        if random_tapes:
            # uniformly random cells, so diverse replicator clouds can be assayed post hoc
            ridx = np.random.default_rng([seed, step]).integers(0, soup_arr.shape[0], size=random_tapes)
            rec["random_tapes"] = [soup_arr[i].tobytes().hex(" ") for i in ridx]
        if census_now:
            rec["census"] = interaction_census(before, inter, soup.read_pair_memory(), soup.tape_length, census_pairs, seed, step)
            if on_snapshot:
                on_snapshot(f"t{step}", soup_arr)
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
                if key == "tq_10" and on_snapshot:
                    on_snapshot("emergence", soup_arr)
        if stop_share is not None and stop_share > 0 and stop_at is None and qs["q_share"] >= stop_share:
            stop_at = step + stop_after * (sample_every_early if (sample_every_early and step < early_until) else sample_every)
        if stop_at is not None and step >= stop_at:
            break
    final_hashes = soup.read_hashes()
    final_hist = soup.read_byte_counts()
    sp = species_stats(final_hashes)
    soup_arr = soup.read_soup()
    top_idx = int(np.argmax(final_hashes == np.uint32(sp["top_hash"])))
    if on_snapshot:
        on_snapshot("final", soup_arr)
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
            **shift_occupancy(soup_arr, soup_arr[top_idx], sample=soup_arr.shape[0]),
            **motif_share(soup_arr, soup_arr[top_idx]),
            **high_order_entropy(soup_arr),
            **census(soup_arr),
            "exemplars": exemplars(soup_arr, final_hashes, sp["top3_hashes"], suppress=soup.sets),
            "zero_frac": float(final_hist[0] / max(int(final_hist.sum()), 1)),
            "byte_hist": final_hist.astype(int).tolist(),
        },
        "samples": samples,
        "wall_s": round(time.perf_counter() - t0, 2),
        # Nominal pair slots per second incl. sampling overhead (≈57% of drawn pairs are active; see active_pairs).
        "slots_per_s": round(step * pairs * z80_steps / max(time.perf_counter() - t0, 1e-9)),
    }
    emit(summary)
    return summary


def next_sample_step(step: int, horizon: int, sample_every: int, sample_every_early: int | None, early_until: int, extra_steps) -> int:
    """The next step at which a sample is taken: the linear schedule (fine until early_until, then
    coarse), merged with the explicit `extra_steps`, never past the horizon."""
    # Regular samples sit on an aligned grid (multiples of the interval), so an explicit extra step
    # (e.g. 34) does not shift the grid; the early grid ends exactly at early_until.
    if sample_every_early and step < early_until:
        nxt = min((step // sample_every_early + 1) * sample_every_early, early_until)
    else:
        nxt = (step // sample_every + 1) * sample_every
    nxt = min(nxt, horizon)
    extra = next((x for x in extra_steps if x > step), None)
    return min(nxt, extra) if extra is not None else nxt


def sample_schedule(horizon: int, sample_every: int, sample_every_early: int | None = None, early_until: int = 0, extra_steps=()) -> list[int]:
    out, step = [], 0
    extra = sorted({int(x) for x in extra_steps if 0 < int(x) <= horizon})
    while step < horizon:
        step = next_sample_step(step, horizon, sample_every, sample_every_early, early_until, extra)
        out.append(step)
    return out


_WB_BINS = np.array([0, 1, 4, 16, 64, 256, 1 << 30])


def interaction_summary(inter: dict, L: int) -> dict:
    """Shader-counted byte writes per interaction of the last step (A = first program, B = second)."""
    a = inter["active"]
    wc = inter["write_counts"][a]
    if wc.shape[0] == 0:
        return {"ix_active": 0}
    wa, wb = wc[:, 0], wc[:, 1]
    return {
        "ix_active": int(a.sum()),
        "ix_silent_frac": float(((wa == 0) & (wb == 0)).mean()),       # interactions that wrote nothing anywhere
        "ix_wb_mean": float(wb.mean()), "ix_wa_mean": float(wa.mean()),  # writes landing in the partner / in oneself
        "ix_wb_ge_L_frac": float((wb >= L).mean()),                     # partner received at least a tape's worth of writes
        "ix_wb_hist": np.histogram(wb, bins=_WB_BINS)[0].astype(int).tolist(),  # bins: 0, 1-3, 4-15, 16-63, 64-255, 256+
    }


def interaction_census(before: np.ndarray, inter: dict, mem: np.ndarray, L: int, keep: int, seed: int, step: int) -> dict:
    """Before/after state of every pair that interacted in the last step: copy events (partner became
    >= 75% similar to the program under the best cyclic shift), bytes changed, zeros written, and a
    seeded subsample of `keep` raw pairs (hex) for post hoc work."""
    from .assay import _best_shift_rows

    a = inter["active"]
    i, j = inter["pairs"][a, 0], inter["pairs"][a, 1]
    A0, B0 = before[i], before[j]
    A1, B1 = mem[a, 0], mem[a, 1]
    simB0 = _best_shift_rows(B0, A0)[0]
    simB1, shB = _best_shift_rows(B1, A0)
    simA0 = _best_shift_rows(A0, B0)[0]
    simA1 = _best_shift_rows(A1, B0)[0]
    copyAB = (simB1 >= 0.75) & (simB0 < 0.75)
    copyBA = (simA1 >= 0.75) & (simA0 < 0.75)
    changedB = (B1 != B0)
    out = {
        "n": int(a.sum()),
        "copy_ab": int(copyAB.sum()), "copy_ba": int(copyBA.sum()),
        "partial_ab": int(((simB1 - simB0) >= 0.25).sum() - copyAB.sum()),   # gained >= 0.25 similarity without becoming a copy
        "bytes_changed_b_mean": float(changedB.sum(1).mean()), "bytes_changed_a_mean": float((A1 != A0).sum(1).mean()),
        "zero_writes_b": int((changedB & (B1 == 0)).sum()), "writes_b": int(changedB.sum()),
        "copy_offsets": {str(int(k)): int(v) for k, v in zip(*np.unique(shB[copyAB], return_counts=True))} if copyAB.any() else {},
        "destroyed_a": int(((A1 != A0).sum(1) >= L / 2).sum()),   # programs that lost at least half their bytes in the encounter
    }
    if keep > 0 and a.any():
        idx = np.random.default_rng([seed, step, 7]).choice(int(a.sum()), size=min(keep, int(a.sum())), replace=False)
        out["pairs"] = [[int(i[k]), int(j[k]), A0[k].tobytes().hex(), B0[k].tobytes().hex(), A1[k].tobytes().hex(), B1[k].tobytes().hex()] for k in idx]
    return out


def _provenance(soup, extra: dict | None) -> dict:
    """Everything needed to reproduce a run's software environment, stored in every condition record."""
    import hashlib
    import platform

    import brotli  # noqa: F401  (HOE depends on the brotli version)
    import wgpu

    from .soup import SHADER_DIR, _ADAPTER_INFO

    isa_sha = hashlib.sha256((SHADER_DIR / "isa.json").read_bytes()).hexdigest()[:16]
    try:
        from wgpu.backends.wgpu_native import lib_version_info

        wgpu_native_version = ".".join(map(str, lib_version_info))
    except Exception as e:  # noqa: BLE001
        print(f"warning: wgpu-native version unavailable: {e}", file=sys.stderr)
        wgpu_native_version = None
    prov = {
        "shader_file": soup.shader_file.name,
        "shader_sha256_16": soup.shader_sha256_16,
        "isa_sha256_16": isa_sha,
        "adapter_info": {k: str(v) for k, v in _ADAPTER_INFO.items()},
        "versions": {"wgpu": wgpu.__version__, "wgpu_native": wgpu_native_version, "numpy": np.__version__, "brotli": getattr(brotli, "__version__", None), "python": platform.python_version()},
        "mutations_per_step_override": soup.mutations_per_step,
    }
    if extra:
        prov.update(extra)
    return {"provenance": prov}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="square", choices=["square", "hex", "mixed"], help="mixed = square shader with the partner drawn uniformly from the whole soup (Stage H control)")
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
    ap.add_argument("--random-tapes", type=int, default=0, help="store this many random cell tapes per sample")
    ap.add_argument("--sample-every-early", type=int, default=None, help="finer sampling interval for the first --early-until steps")
    ap.add_argument("--early-until", type=int, default=0)
    ap.add_argument("--mutations-per-step", type=int, default=None, help="override pair_count/2^noise_exp (control arms)")
    ap.add_argument("--sample-steps", default="", help="comma-separated extra sample steps, e.g. 1,2,3,5,8,13,21,34")
    ap.add_argument("--snapshot-steps", default="", help="comma-separated steps at which the full soup is snapshotted and the interaction census taken")
    ap.add_argument("--zero-halts", action="store_true", help="lethal-tar rule: a zero byte fetched as an opcode halts the pair for the rest of the encounter (Stage I control; square grid only)")
    a = ap.parse_args(argv)
    out = open(a.out, "w") if a.out else sys.stdout
    try:
        s = run(
            grid=a.grid, tape=a.tape, width=a.width, height=a.height, seed=a.seed, pairs=a.pairs, z80_steps=a.z80_steps,
            noise_exp=a.noise_exp, suppress=a.suppress, horizon=a.horizon, sample_every=a.sample_every,
            stop_share=None if a.stop_share < 0 else a.stop_share, label=a.label, out=out, quiet=a.quiet, random_tapes=a.random_tapes,
            sample_every_early=a.sample_every_early, early_until=a.early_until, mutations_per_step=a.mutations_per_step,
            sample_steps=[int(x) for x in a.sample_steps.split(",") if x], snapshot_steps=[int(x) for x in a.snapshot_steps.split(",") if x],
            zero_halts=a.zero_halts,
        )
    finally:
        if a.out:
            out.close()
    if a.out:
        print(json.dumps({k: s[k] for k in ("label", "seed", "steps_run", "t_02", "t_10", "tq_10", "tq_50", "wall_s", "slots_per_s")}), file=sys.stderr)


if __name__ == "__main__":
    main()
