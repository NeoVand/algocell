"""Post hoc replication assay for a batch: for every run, assay the dominant tape
at the first tq_10 crossing (or the final dominant tape if censored) under that
run's own step budget and suppression set. Appends `assay` to a copy of the
summary and writes runs/<batch>/analysis/assays.csv.

    python assay_batch.py runs/stageA
"""

from __future__ import annotations

import glob
import json
import os
import sys

import brotli
import numpy as np
import pandas as pd

from algocell_exp.assay import assay
from algocell_exp.isa import parse_patterns
from algocell_exp.metrics import minimal_period


def load_snapshot(path: str, tape_len: int) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    raw = np.frombuffer(brotli.decompress(open(path, "rb").read()), dtype=np.uint8)
    return raw.reshape(-1, tape_len)


def exemplars_at(jsonl_path: str, step: int) -> list[str]:
    """Top-3 exemplar tapes (hex) at a given sample step."""
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("kind") == "sample" and d["step"] == step:
                return [e["tape"] for e in d["exemplars"]]
    return []


GEN2_MIN = 0.3
SHARE_MIN = 0.005


def t_rep(jsonl_path: str, z80_steps: int, sup: list[str], cache: dict) -> tuple[int, dict | None]:
    """First sample step at which a top-3 exemplar is a heritable replicator
    (gen2 >= GEN2_MIN) with exact share >= SHARE_MIN. Assays are cached by
    (tape, steps, suppression) so recurring floods/smears cost nothing."""
    key_sup = ";".join(sup)
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("kind") != "sample":
                continue
            for rank, (ex, share) in enumerate(zip(d["exemplars"], d["top3_shares"])):
                if share < SHARE_MIN:
                    continue
                key = (ex["tape"], z80_steps, key_sup)
                if key not in cache:
                    cache[key] = assay(bytes.fromhex(ex["tape"].replace(" ", "")), z80_steps=z80_steps, suppress=sup, n=64)
                r = cache[key]
                if r["gen2_score"] >= GEN2_MIN:
                    return d["step"], {"tape": ex["tape"], "rank": rank, "share": share, **{k: round(v, 3) for k, v in r.items()}}
    return -1, None


def main(d: str) -> None:
    rows = []
    cache: dict = {}
    for p in sorted(glob.glob(os.path.join(d, "*.summary.json"))):
        s = json.load(open(p))
        stem = p[: -len(".summary.json")]
        # The most common exact genotype can be the sterile offspring of a copier
        # (e.g. `21 e3 x8` written by `LD HL,$E321 ; PUSH HL`), so assay the top-3
        # exemplars and keep the best heritability score among them.
        where = "final"
        tapes: list[str] = []
        if s["tq_10"] > 0 and os.path.exists(stem + ".jsonl"):
            tapes = exemplars_at(stem + ".jsonl", s["tq_10"])
            where = "emergence"
        if not tapes:
            tapes = [e["tape"] for e in s["final"]["exemplars"]]
        sup = parse_patterns(";".join(s["suppress"])) if isinstance(s["suppress"], list) else parse_patterns(s["suppress"])
        best = None
        for rank, tape_hex in enumerate(tapes[:3]):
            r = assay(bytes.fromhex(tape_hex.replace(" ", "")), z80_steps=s["z80_steps"], suppress=sup)
            if best is None or r["gen2_score"] > best[1]["gen2_score"]:
                best = (rank, r, tape_hex)
        rank, r, tape_hex = best
        # In-situ assay of the final top-3 against the final soup snapshot (ecological truth).
        L = s.get("tape_length", 16)
        snap = load_snapshot(stem + ".soup_final.u8.br", L)
        insitu = None
        if snap is not None:
            for tape_hex2 in [e["tape"] for e in s["final"]["exemplars"]][:3]:
                r2 = assay(bytes.fromhex(tape_hex2.replace(" ", "")), z80_steps=s["z80_steps"], suppress=sup, neighbors=snap)
                if insitu is None or r2["gen2_score"] > insitu["gen2_score"]:
                    insitu = r2
        # Random-cell in-situ assay of the final soup: fraction of 16 random cells
        # that are heritable replicators against their own population. Catches
        # diverse clouds whose members never reach the top-3 exemplars.
        rep_frac = None
        if snap is not None:
            rng = np.random.default_rng(s["seed"])
            idx = rng.integers(0, snap.shape[0], size=16)
            hits = 0
            for i in idx:
                rr = assay(snap[i].tobytes(), z80_steps=s["z80_steps"], suppress=sup, n=32, neighbors=snap)
                hits += rr["gen2_score"] >= GEN2_MIN
            rep_frac = hits / 16
        trep, first = (t_rep(stem + ".jsonl", s["z80_steps"], sup, cache) if os.path.exists(stem + ".jsonl") else (-1, None))
        rows.append(
            {
                "label": s["label"], "tape_len": s.get("tape_length", 16), "steps": s["z80_steps"], "k": s["noise_exp"], "seed": s["seed"],
                "tq_10": s["tq_10"], "t_rep": trep, "t_rep_tape": first["tape"] if first else None, "t_rep_gen2": first["gen2_score"] if first else None,
                "where": where, "exemplar_rank": rank, "tape": tape_hex, "score": round(r["score"], 3), "gen2": round(r["gen2_score"], 3),
                "copies75": round(r["offspring_within_q"], 3), "self": round(r["self_preserved_as_A"], 3),
                "is_replicator": r["gen2_score"] >= GEN2_MIN,
                "final_gen2_insitu": round(insitu["gen2_score"], 3) if insitu else None,
                "final_score_insitu": round(insitu["score"], 3) if insitu else None,
                "final_replicator_insitu": (insitu["gen2_score"] >= GEN2_MIN) if insitu else None,
                "final_rep_fraction": rep_frac,
                "final_period": minimal_period(bytes.fromhex(s["final"]["exemplars"][0]["tape"].replace(" ", ""))),
                "t_rep_period": minimal_period(bytes.fromhex(first["tape"].replace(" ", ""))) if first else None,
                "final_c_blockcopy": s["final"].get("c_blockcopy"), "final_c_push2": s["final"].get("c_push2"), "final_hoe": s["final"]["hoe"],
                "steps_run": s["steps_run"], "horizon": s["horizon"], "file": os.path.basename(p),
            }
        )
        print(f"{s['label']:13s} L{s.get('tape_length',16):<3} st{s['z80_steps']:<3} k{s['noise_exp']} s{s['seed']:<2} tq10 {s['tq_10']:>7} t_rep {trep:>7} {(first['tape'][:23] if first else '-'):23s} | final gen2 rnd {r['gen2_score']:.2f} insitu {(insitu['gen2_score'] if insitu else float('nan')):.2f} repfrac {(rep_frac if rep_frac is not None else float('nan')):.2f} {tape_hex[:23]}", file=sys.stderr)
    df = pd.DataFrame(rows)
    out = os.path.join(d, "analysis")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "assays.csv"), index=False)
    print(df.groupby(["label", "tape_len", "steps", "k"]).agg(n=("seed", "size"), tq10_emerged=("tq_10", lambda x: int((x > 0).sum())), t_rep_emerged=("t_rep", lambda x: int((x > 0).sum())), t_rep_median=("t_rep", lambda x: float(x[x > 0].median()) if (x > 0).any() else float("nan")), final_rep_rnd=("is_replicator", "sum"), final_rep_insitu=("final_replicator_insitu", lambda x: int(x.fillna(False).astype(bool).sum())), final_rep_frac=("final_rep_fraction", "median"), final_blockcopy=("final_c_blockcopy", "median"), final_push2=("final_c_push2", "median")).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
