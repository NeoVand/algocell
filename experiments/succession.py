"""Succession analysis from the per-sample census trajectories → analysis/succession.csv, succession_cells.csv.

Per run: when each replicator family first holds a 30% excess over the random-soup
baseline (headroom-normalised: (v − b)/(1 − b) ≥ 0.30, with b the per-L probability that
a random tape shows the pattern — the pre-registration's "30% excess over baseline"),
how long an LDIR invasion takes (first 10% → first 90% of cells carrying a block-copy
pair, raw fractions; censored at the last sample when 90% is never reached), which
family holds the soup at the LAST sample and at fixed steps (5k, 50k, 300k; NaN when the
run stopped earlier), the stack plateau before an invasion, and the zero-byte load.

Review 2026-10-07: `final_family` is evaluated at `last_step`, which is the early-stop
step (5k–20k) in ~20% of Stage A/B runs; use the fixed-step columns to compare soups
of equal age, and `stopped_early` to see which runs were cut.

    python succession.py runs/stageA
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

FAMILIES = {"push": "c_push2", "ex_sp": "c_ex_sp", "ldir": "c_blockcopy", "ld_hl": "c_ld_hl_w", "cb_hl": "c_cb_hl", "rst": "c_rst"}
TAKEOVER = 0.30       # headroom-normalised excess over the random-soup baseline
FIXED_STEPS = (5_000, 50_000, 300_000)

_BASE: dict[int, dict] = {}


def baseline(tape_length: int) -> dict:
    """Random-soup census baselines for this tape length (P(cell contains the pattern)
    grows with L: ≥ 2 PUSH bytes in 16 random bytes ≈ 2%, in 100 random bytes ≈ 46%).
    Estimated once per L from 20,000 random tapes (MC error ≈ 0.004)."""
    if tape_length not in _BASE:
        from algocell_exp.metrics import census

        rng = np.random.default_rng(12345)
        soup = rng.integers(0, 256, size=(20000, tape_length), dtype=np.uint8)
        c = census(soup)
        _BASE[tape_length] = {k: float(v) for k, v in c.items() if k.startswith("c_")}
    return _BASE[tape_length]


def excess(v: float, key: str, tape_length: int = 16) -> float:
    b = baseline(tape_length)[key]
    return max(0.0, (v - b) / max(1e-9, 1 - b))


def family_at(sample: dict, tape_length: int) -> tuple[str, float]:
    fam_vals = {fam: excess(sample.get(key, 0.0), key, tape_length) for fam, key in FAMILIES.items()}
    best = max(fam_vals, key=fam_vals.get)
    if fam_vals[best] >= TAKEOVER:
        return best, fam_vals[best]
    return ("flooded" if sample.get("c_zero8", 0) > 0.5 else "none"), fam_vals[best]


def analyze_run(jsonl_path: str, tape_length: int = 16, horizon: int | None = None) -> dict:
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("kind") == "sample":
                samples.append(d)
    if not samples:
        return {}
    out: dict = {}
    for fam, key in FAMILIES.items():
        out[f"t_{fam}_30"] = next((s["step"] for s in samples if excess(s.get(key, 0.0), key, tape_length) >= TAKEOVER), -1)
    t10 = next((s["step"] for s in samples if s.get("c_blockcopy", 0) >= 0.10), -1)
    t90 = next((s["step"] for s in samples if s.get("c_blockcopy", 0) >= 0.90), -1)
    out["ldir_10"], out["ldir_90"] = t10, t90
    out["ldir_invasion_steps"] = (t90 - t10) if (t10 > 0 and t90 > 0) else -1
    out["ldir_invasion_censored"] = bool(t10 > 0 and t90 < 0)
    out["ldir_invasion_censored_at"] = (samples[-1]["step"] - t10) if (t10 > 0 and t90 < 0) else -1
    t_stack = min([t for t in (out["t_push_30"], out["t_ex_sp_30"]) if t > 0], default=-1)
    end = t10 if t10 > 0 else samples[-1]["step"]
    win = [s for s in samples if t_stack > 0 and t_stack <= s["step"] <= end]
    if len(win) >= 3:
        vals = sorted(max(excess(s.get("c_push2", 0), "c_push2", tape_length), excess(s.get("c_ex_sp", 0), "c_ex_sp", tape_length)) for s in win)
        out["stack_plateau"] = vals[len(vals) // 2]          # headroom-normalised excess, comparable across L
        out["stack_plateau_steps"] = end - t_stack
    else:
        out["stack_plateau"] = float("nan")
        out["stack_plateau_steps"] = -1
    last = samples[-1]
    fam, exc = family_at(last, tape_length)
    out.update({"final_family": fam, "final_family_excess": round(exc, 3), "final_hoe": last.get("hoe"), "final_unique": last.get("unique"),
                "final_zero_frac": last.get("zero_frac", np.nan), "last_step": last["step"]})
    by_step = {s["step"]: s for s in samples}
    for t in FIXED_STEPS:
        s_t = by_step.get(t)
        out[f"family_{t//1000}k"] = family_at(s_t, tape_length)[0] if s_t else None
        out[f"ldir_{t//1000}k"] = s_t.get("c_blockcopy") if s_t else np.nan
        out[f"zero8_{t//1000}k"] = s_t.get("c_zero8") if s_t else np.nan
    if horizon is not None:
        out["stopped_early"] = last["step"] < horizon
    return out


def main(d: str) -> None:
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.summary.json"))):
        s = json.load(open(p))
        stem = p[: -len(".summary.json")]
        r = analyze_run(stem + ".jsonl", s.get("tape_length", 16), s.get("horizon")) if os.path.exists(stem + ".jsonl") else {}
        rows.append({"label": s["label"], "tape": s.get("tape_length", 16), "steps": s["z80_steps"], "k": s["noise_exp"], "seed": s["seed"], **r, "file": os.path.basename(p)})
    df = pd.DataFrame(rows)
    out = os.path.join(d, "analysis")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "succession.csv"), index=False)

    def agg(g: pd.DataFrame) -> pd.Series:
        fams = g["final_family"].value_counts().to_dict()
        inv = g[g["ldir_invasion_steps"] > 0]["ldir_invasion_steps"]
        ldir = g[g["t_ldir_30"] > 0]["t_ldir_30"]
        stack = g[(g["t_push_30"] > 0) | (g["t_ex_sp_30"] > 0)]
        t_stack = stack[["t_push_30", "t_ex_sp_30"]].replace(-1, float("nan")).min(axis=1)
        f300 = g["family_300k"].dropna()
        return pd.Series({
            "n": len(g),
            "stopped_early": int(g["stopped_early"].fillna(False).astype(bool).sum()) if "stopped_early" in g else -1,
            "last_step_min": int(g["last_step"].min()),
            "final_at_last": ", ".join(f"{k}:{v}" for k, v in sorted(fams.items(), key=lambda x: -x[1])),
            "family_300k": ", ".join(f"{k}:{v}" for k, v in sorted(f300.value_counts().to_dict().items(), key=lambda x: -x[1])) if len(f300) else "(no run reached 300k)",
            "stack_takeover_n": len(stack),
            "stack_takeover_med": float(t_stack.median()) if len(stack) else float("nan"),
            "ldir_takeover_n": len(ldir),
            "ldir_takeover_med": float(ldir.median()) if len(ldir) else float("nan"),
            "ldir_invasions_complete": len(inv),
            "ldir_invasions_censored": int(g["ldir_invasion_censored"].fillna(False).astype(bool).sum()),
            "ldir_invasion_med_steps": float(inv.median()) if len(inv) else float("nan"),
            "stack_plateau_med": float(g["stack_plateau"].median()),
        })

    table = df.groupby(["label", "tape", "steps", "k"]).apply(agg, include_groups=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    print(table.to_string(float_format=lambda x: f"{x:.0f}" if abs(x) >= 10 else f"{x:.2f}"))
    table.to_csv(os.path.join(out, "succession_cells.csv"))


if __name__ == "__main__":
    main(sys.argv[1])
