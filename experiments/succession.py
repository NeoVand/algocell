"""Succession analysis from the per-sample census trajectories.

Per run: when each replicator family first holds >= 30% of cells (by the
byte-pattern census), how long an LDIR invasion takes (10% -> 90% of cells
carrying a block-copy pair), which family holds the soup at the end, and the
occupancy plateau of the stack families before any invasion. Aggregated per
(label, tape, steps, k) cell.

    python succession.py runs/stageA
"""

from __future__ import annotations

import glob
import json
import os
import sys

import pandas as pd

FAMILIES = {"push": "c_push2", "ex_sp": "c_ex_sp", "ldir": "c_blockcopy", "ld_hl": "c_ld_hl_w", "cb_hl": "c_cb_hl", "rst": "c_rst"}
TAKEOVER = 0.30  # excess over the random-soup baseline, as a fraction of the headroom

_BASE: dict[int, dict] = {}


def baseline(tape_length: int) -> dict:
    """Random-soup census baselines for this tape length (P(cell contains the
    pattern) grows with L: e.g. >= 2 PUSH bytes in 16 random bytes ~ 2%, in 100
    random bytes ~ 46%). Estimated once per L from 20,000 random tapes."""
    if tape_length not in _BASE:
        import numpy as np

        from algocell_exp.metrics import census

        rng = np.random.default_rng(12345)
        soup = rng.integers(0, 256, size=(20000, tape_length), dtype=np.uint8)
        c = census(soup)
        _BASE[tape_length] = {k: float(v) for k, v in c.items() if k.startswith("c_")}
    return _BASE[tape_length]


def excess(v: float, key: str, tape_length: int = 16) -> float:
    b = baseline(tape_length)[key]
    return max(0.0, (v - b) / max(1e-9, 1 - b))


def analyze_run(jsonl_path: str, tape_length: int = 16) -> dict:
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("kind") == "sample":
                samples.append(d)
    if not samples:
        return {}
    out: dict = {}
    for fam, key in FAMILIES.items():
        t = next((s["step"] for s in samples if excess(s.get(key, 0.0), key, tape_length) >= TAKEOVER), -1)
        out[f"t_{fam}_30"] = t
    # LDIR invasion duration: first 10% -> first 90% (raw fraction of cells)
    t10 = next((s["step"] for s in samples if s.get("c_blockcopy", 0) >= 0.10), -1)
    t90 = next((s["step"] for s in samples if s.get("c_blockcopy", 0) >= 0.90), -1)
    out["ldir_10"] = t10
    out["ldir_90"] = t90
    out["ldir_invasion_steps"] = (t90 - t10) if (t10 > 0 and t90 > 0) else -1
    # Stack plateau: median push2/ex_sp over the window after stack takeover and before LDIR 10% (or end)
    t_stack = min([t for t in (out["t_push_30"], out["t_ex_sp_30"]) if t > 0], default=-1)
    end = t10 if t10 > 0 else samples[-1]["step"]
    win = [s for s in samples if t_stack > 0 and t_stack <= s["step"] <= end]
    if len(win) >= 3:
        vals = sorted(max(s.get("c_push2", 0), s.get("c_ex_sp", 0)) for s in win)
        out["stack_plateau"] = vals[len(vals) // 2]
        out["stack_plateau_steps"] = end - t_stack
    else:
        out["stack_plateau"] = float("nan")
        out["stack_plateau_steps"] = -1
    last = samples[-1]
    fam_vals = {fam: excess(last.get(key, 0.0), key, tape_length) for fam, key in FAMILIES.items()}
    best = max(fam_vals, key=fam_vals.get)
    out["final_family"] = best if fam_vals[best] >= TAKEOVER else ("flooded" if last.get("c_zero8", 0) > 0.5 else "none")
    out["final_family_excess"] = round(fam_vals[best], 3)
    out["final_hoe"] = last.get("hoe")
    out["final_unique"] = last.get("unique")
    out["last_step"] = last["step"]
    return out


def main(d: str) -> None:
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.summary.json"))):
        s = json.load(open(p))
        stem = p[: -len(".summary.json")]
        r = analyze_run(stem + ".jsonl", s.get("tape_length", 16)) if os.path.exists(stem + ".jsonl") else {}
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
        return pd.Series(
            {
                "n": len(g),
                "final": ", ".join(f"{k}:{v}" for k, v in sorted(fams.items(), key=lambda x: -x[1])),
                "stack_takeover_n": len(stack),
                "stack_takeover_med": float(t_stack.median()) if len(stack) else float("nan"),
                "ldir_takeover_n": len(ldir),
                "ldir_takeover_med": float(ldir.median()) if len(ldir) else float("nan"),
                "ldir_invasion_med_steps": float(inv.median()) if len(inv) else float("nan"),
                "stack_plateau_med": float(g["stack_plateau"].median()),
            }
        )

    table = df.groupby(["label", "tape", "steps", "k"]).apply(agg, include_groups=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 40)
    print(table.to_string(float_format=lambda x: f"{x:.0f}" if abs(x) >= 10 else f"{x:.2f}"))
    table.to_csv(os.path.join(out, "succession_cells.csv"))


if __name__ == "__main__":
    main(sys.argv[1])
