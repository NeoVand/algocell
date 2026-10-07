"""One condition → files on disk. Shared by the Modal worker and the local preflight so the
exact same code path (stale-file cleanup, snapshots, atomic summary) is exercised before a
sweep is paid for.

A condition is a dict of `algocell_exp.run.run()` keyword arguments plus `label` and, optionally,
`replicate` (for repeated runs of one seed) and `provenance` (injected by the launcher).
"""

from __future__ import annotations

import glob
import io
import json
import os

import brotli
import numpy as np

from .run import run


def run_stem(cond: dict) -> str:
    stem = f"{cond.get('label', 'run')}_L{cond.get('tape') or 16}_st{cond.get('z80_steps', 128)}_k{cond.get('noise_exp', 4)}_s{cond.get('seed', 0)}"
    if cond.get("replicate") is not None:
        stem += f"_r{cond['replicate']}"
    return stem


def run_to_dir(cond: dict, out_dir: str, provenance: dict | None = None) -> dict:
    """Run one condition, writing {stem}.jsonl, {stem}.soup_{emergence,final}.u8.br and, last and
    atomically, {stem}.summary.json into out_dir. Any file of this stem from an earlier attempt is
    removed first (runs are non-deterministic, so a stale emergence snapshot would belong to a
    different trajectory)."""
    os.makedirs(out_dir, exist_ok=True)
    stem = run_stem(cond)
    for stale in glob.glob(os.path.join(out_dir, f"{stem}.*")):
        os.remove(stale)
    kwargs = {k: v for k, v in cond.items() if k not in ("replicate",)}
    prov = dict(provenance or {})
    if cond.get("replicate") is not None:
        prov["replicate"] = cond["replicate"]
    if "provenance" in kwargs:
        prov.update(kwargs.pop("provenance") or {})
    kwargs["provenance"] = prov

    def snapshot(name: str, soup: np.ndarray) -> None:
        with open(os.path.join(out_dir, f"{stem}.soup_{name}.u8.br"), "wb") as f:
            f.write(brotli.compress(np.ascontiguousarray(soup).tobytes(), quality=9))

    buf = io.StringIO()
    summary = run(**kwargs, out=buf, quiet=True, on_snapshot=snapshot)
    with open(os.path.join(out_dir, f"{stem}.jsonl"), "w") as f:
        f.write(buf.getvalue())
    tmp = os.path.join(out_dir, f"{stem}.summary.json.tmp")
    with open(tmp, "w") as f:
        json.dump(summary, f)
    os.replace(tmp, os.path.join(out_dir, f"{stem}.summary.json"))
    return summary


def check_outputs(cond: dict, out_dir: str) -> list[str]:
    """Consistency checks on a finished condition's files; returns a list of problems (empty = OK)."""
    stem = run_stem(cond)
    problems = []
    sp = os.path.join(out_dir, f"{stem}.summary.json")
    if not os.path.exists(sp):
        return [f"{stem}: no summary"]
    s = json.load(open(sp))
    horizon = cond.get("horizon", s["horizon"])
    if s["steps_run"] != horizon and (cond.get("stop_share", 0.5) or 0) <= 0:
        problems.append(f"{stem}: steps_run {s['steps_run']} != horizon {horizon} although the early stop is disabled")
    recs = [json.loads(l) for l in open(os.path.join(out_dir, f"{stem}.jsonl"))]
    samples = [r for r in recs if r.get("kind") == "sample"]
    if len(samples) != s["samples"]:
        problems.append(f"{stem}: {len(samples)} sample lines vs summary.samples {s['samples']}")
    if cond.get("random_tapes") and any(len(r.get("random_tapes", [])) != cond["random_tapes"] for r in samples):
        problems.append(f"{stem}: random_tapes missing in some samples")
    early, until, every = cond.get("sample_every_early"), cond.get("early_until", 0), cond.get("sample_every", 250)
    expect = []
    t = 0
    while t < horizon:
        step = early if (early and t < until) else every
        step = min(step, horizon - t, (until - t) if (early and t < until) else step)
        t += step
        expect.append(t)
    got = [r["step"] for r in samples]
    if s["steps_run"] == horizon and got != expect:
        problems.append(f"{stem}: sample steps {got[:6]}… != expected {expect[:6]}…")
    if "byte_hist" not in samples[-1] or "active_pairs" not in samples[-1]:
        problems.append(f"{stem}: byte_hist/active_pairs missing")
    if not os.path.exists(os.path.join(out_dir, f"{stem}.soup_final.u8.br")):
        problems.append(f"{stem}: no final snapshot")
    if (s["tq_10"] >= 0) != os.path.exists(os.path.join(out_dir, f"{stem}.soup_emergence.u8.br")):
        problems.append(f"{stem}: emergence snapshot presence does not match tq_10")
    if s["tape_length"] != (cond.get("tape") or 16) or s["z80_steps"] != cond.get("z80_steps", 128) or s["noise_exp"] != cond.get("noise_exp", 4):
        problems.append(f"{stem}: summary parameters differ from the condition")
    if cond.get("mutations_per_step") is not None and s["mutations_per_step"] != cond["mutations_per_step"]:
        problems.append(f"{stem}: mutations_per_step override not applied")
    if cond.get("pairs") is not None and s["pairs"] != cond["pairs"]:
        problems.append(f"{stem}: pairs differ")
    return problems
