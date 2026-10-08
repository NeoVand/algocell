"""Stage H — run the well-mixed control conditions sequentially on the local GPU.

    .venv/bin/python stage_h_local.py                                   # all of conds/stageH.json -> runs/stageH (resumable)
    .venv/bin/python stage_h_local.py --only 1 --horizon 1000 --out runs/stageH_dry    # dry run (never into runs/stageH)

Every condition goes through algocell_exp.batch.run_to_dir, the file-writing path Stages A-G used on Modal,
with provenance {"stage": "H", "purpose": "well-mixed control"} plus the git commit, the condition file's hash,
the launch time and the adapter. A condition whose {stem}.summary.json already exists is skipped, so an
interrupted sweep resumes where it stopped (run_to_dir removes the partial files of a stem before re-running
it). Wall time and ms per step are printed per run; batch.check_outputs problems are printed and make the exit
status non-zero.

--horizon (and --only) exist for dry runs. A horizon override is refused for the default output directory so
that no truncated run can ever sit next to the real Stage H files; early_until, sample_every and the snapshot
steps are cut to the horizon exactly as preflight.py and the Modal smoke test do.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from algocell_exp.batch import check_outputs, run_stem, run_to_dir  # noqa: E402
from algocell_exp.soup import adapter_summary  # noqa: E402

DEFAULT_CONDS = os.path.join(ROOT, "conds", "stageH.json")
DEFAULT_OUT = os.path.join(ROOT, "runs", "stageH")
PROVENANCE = {"stage": "H", "purpose": "well-mixed control"}


def git_provenance() -> dict:
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "."], capture_output=True, text=True, cwd=ROOT).stdout.strip())
    except Exception:  # noqa: BLE001
        git, dirty = None, None
    return {"git_commit": git, "git_dirty_tracked": dirty}


def cut_to_horizon(cond: dict, horizon: int) -> dict:
    cc = dict(cond)
    cc["horizon"] = horizon
    if cc.get("early_until"):
        cc["early_until"] = min(cc["early_until"], horizon)
    cc["sample_every"] = min(cc.get("sample_every", 500), horizon)
    cc["snapshot_steps"] = [t for t in (cc.get("snapshot_steps") or []) if t <= horizon]
    return cc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conds", default=DEFAULT_CONDS)
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory (default runs/stageH)")
    ap.add_argument("--only", type=int, default=None, metavar="N", help="run only the first N conditions")
    ap.add_argument("--horizon", type=int, default=None, help="override the horizon (dry runs only; refused for the default --out)")
    a = ap.parse_args()
    if a.horizon is not None and os.path.abspath(a.out) == os.path.abspath(DEFAULT_OUT):
        print(f"refusing a horizon override into {DEFAULT_OUT}: dry runs go to another --out (e.g. runs/stageH_dry)", file=sys.stderr)
        return 2
    with open(a.conds) as f:
        conds = json.load(f)
    if a.only is not None:
        conds = conds[: a.only]
    prov = {
        **PROVENANCE,
        **git_provenance(),
        "conds_file": os.path.basename(a.conds),
        "conds_sha256_16": hashlib.sha256(open(a.conds, "rb").read()).hexdigest()[:16],
        "launched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": "local",
        "launcher": "stage_h_local.py",
    }
    if a.horizon is not None:
        prov["horizon_override"] = a.horizon
    os.makedirs(a.out, exist_ok=True)
    print(f"{len(conds)} condition(s) from {os.path.relpath(a.conds, ROOT)} -> {a.out} on {adapter_summary()}"
          f"{f' (horizon cut to {a.horizon})' if a.horizon is not None else ''}; commit {prov['git_commit']}{' DIRTY' if prov['git_dirty_tracked'] else ''}", flush=True)
    problems_total: list[str] = []
    done = skipped = 0
    t_all = time.perf_counter()
    for k, cond in enumerate(conds):
        stem = run_stem(cond)
        if os.path.exists(os.path.join(a.out, f"{stem}.summary.json")):
            print(f"[{k + 1}/{len(conds)}] {stem}: summary exists, skipped", flush=True)
            skipped += 1
            continue
        cc = cut_to_horizon(cond, a.horizon) if a.horizon is not None else cond
        t0 = time.perf_counter()
        s = run_to_dir(cc, a.out, prov)
        dt = time.perf_counter() - t0
        problems = check_outputs(cc, a.out)
        problems_total += problems
        done += 1
        ms = 1000 * dt / max(s["steps_run"], 1)
        print(f"[{k + 1}/{len(conds)}] {stem}: {s['steps_run']} steps in {dt:.1f} s wall ({ms:.3f} ms/step incl. sampling; "
              f"sim wall_s {s['wall_s']}), {s['samples']} samples, tq_10 {s['tq_10']}, final zero_frac {s['final']['zero_frac']:.3f}, "
              f"top share {s['final']['top_share']:.3f}; {'outputs OK' if not problems else 'PROBLEMS: ' + '; '.join(problems)}", flush=True)
        if a.horizon is not None:
            full = cond.get("horizon", 300_000)
            print(f"    projection at this rate: {full:,} steps ≈ {full * ms / 1000 / 60:.1f} min per run; "
                  f"{len(json.load(open(a.conds)))} runs ≈ {len(json.load(open(a.conds))) * full * ms / 1000 / 3600:.2f} h (upper bound: the dry run samples every 50 steps)", flush=True)
    print(f"done: {done} run, {skipped} skipped, {len(problems_total)} problem(s), {time.perf_counter() - t_all:.0f} s total")
    for pr in problems_total:
        print("  -", pr)
    return 1 if problems_total else 0


if __name__ == "__main__":
    sys.exit(main())
