"""Post hoc replication assays for a batch of runs → analysis/assays.csv.

    python assay_batch.py runs/stageA            # needs the local GPU; ~1–3 h for 600 runs

Column blocks (review 2026-10-07: the old file mixed emergence-time and final-state
quantities under one name; every column now says which tape and which partners):

  identity      label, tape_len, steps, k, seed, file, horizon, steps_run, stopped_early, missing_jsonl
  pre-reg       tq_10, tq_50                      (quasispecies occupancy, from the run)
  t_rep block   t_rep + trep_*                    first sample at which a top-3 exact exemplar with share
                                                   >= 0.5% is HERITABLE (gen2 >= 0.3) against 64 random partners
  t_faith       t_faith + tfaith_tape             first sample at which such an exemplar is FAITHFUL
                                                   (gen2 >= 0.3 and >= 50% of partners became >= 75% copies)
  em block      em_*                              best-gen2 exemplar of the top-3 at the tq_10 sample (random partners)
  final block   final_*                           best-gen2 exemplar of the final top-3 (random partners), and the
                                                   SAME tape assayed in situ (partners drawn from the final soup)
  final_insitu_best_*                              best in-situ gen2 over the final top-3 (legacy comparability)
  func block    final_func_*                      32 uniformly random final cells assayed as A against random
                                                   partners (rnd) and against the soup (insitu); fractions heritable
  soup block    final_c_*, final_hoe, final_H0, final_zero_frac, final_q_share, final_q_shift_share

Periods are the tolerant period (>= 90% of positions match) of the tape in question, with the
match fraction next to it. Mechanisms are labelled under the run's own suppression set.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import brotli
import numpy as np
import pandas as pd

from algocell_exp.assay import GEN2_MIN, assay, assay_many
from algocell_exp.batch import select_summaries
from algocell_exp.isa import mechanisms, parse_patterns, resolve
from algocell_exp.metrics import tolerant_period

SHARE_MIN = 0.005     # pre-registered: an exemplar must hold >= 0.5% of cells to count (one random cell does not)
FUNC_CELLS = 32       # random cells per final soup for the functional fraction
FUNC_PARTNERS = 32


def read_jsonl(path: str) -> tuple[list[dict], bool]:
    """All parseable records; the flag says whether a line was unparseable (truncated write)."""
    recs, bad = [], False
    with open(path) as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                bad = True
    return recs, bad


def load_snapshot(path: str, L: int) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    raw = np.frombuffer(brotli.decompress(open(path, "rb").read()), dtype=np.uint8)
    return raw.reshape(-1, L)


def hexbytes(tape_hex: str) -> bytes:
    return bytes.fromhex(tape_hex.replace(" ", ""))


def period_fields(prefix: str, tape_hex: str | None) -> dict:
    if not tape_hex:
        return {f"{prefix}_period": np.nan, f"{prefix}_period_match": np.nan}
    p, m = tolerant_period(hexbytes(tape_hex))
    return {f"{prefix}_period": p, f"{prefix}_period_match": round(m, 3)}


def scan_emergence(samples: list[dict], z80_steps: int, patterns: list[str], sets: dict, cache: dict) -> dict:
    """First heritable and first faithful top-3 exemplar (share >= SHARE_MIN), scanning samples in time order."""
    out = {"t_rep": -1, "t_faith": -1}
    for d in samples:
        shares = d.get("top3_shares") or []
        for rank, ex in enumerate(d.get("exemplars") or []):
            if rank >= len(shares) or shares[rank] < SHARE_MIN:
                continue
            key = (ex["tape"], z80_steps, tuple(patterns))
            if key not in cache:
                cache[key] = assay(hexbytes(ex["tape"]), z80_steps=z80_steps, suppress=patterns, n=64)
            r = cache[key]
            if out["t_rep"] < 0 and r["is_replicator"]:
                out.update({
                    "t_rep": d["step"], "trep_tape": ex["tape"], "trep_rank": rank, "trep_share": round(shares[rank], 4),
                    "trep_score": round(r["score"], 3), "trep_gen2": round(r["gen2_score"], 3), "trep_gen2_cond": round(r["gen2_cond"], 3),
                    "trep_q75": round(r["offspring_within_q"], 3), "trep_faithful": r["faithful"], "trep_self_b": round(r["self_preserved_as_B"], 3),
                    "trep_offset": r.get("copy_offset"),
                    "trep_mechs": "+".join(mechanisms(hexbytes(ex["tape"]), sets)) or "-", **period_fields("trep", ex["tape"]),
                })
            if out["t_faith"] < 0 and r["faithful"]:
                out.update({"t_faith": d["step"], "tfaith_tape": ex["tape"]})
            if out["t_rep"] >= 0 and out["t_faith"] >= 0:
                return out
    return out


def best_of(tapes: list[str], z80_steps: int, patterns: list[str], neighbors=None, n: int = 64, seed: int = 0) -> tuple[int, dict, str]:
    best = None
    for rank, t in enumerate(tapes):
        r = assay(hexbytes(t), z80_steps=z80_steps, suppress=patterns, n=n, seed=seed, neighbors=neighbors)
        g = r["gen2_score"] if np.isfinite(r["gen2_score"]) else -np.inf
        if best is None or g > best[3]:
            best = (rank, r, t, g)
    rank, r, t, _ = best
    return rank, r, t


def main(d: str) -> None:
    rows: list[dict] = []
    cache: dict = {}
    files = select_summaries(d)
    for i, p in enumerate(files):
        s = json.load(open(p))
        stem = p[: -len(".summary.json")]
        L = s.get("tape_length", 16)
        patterns = s["suppress"] if isinstance(s["suppress"], list) else parse_patterns(s["suppress"])
        sets = resolve(patterns)
        prov = s.get("provenance") or {}
        row: dict = {
            "label": s["label"], "ablation": s["label"].split("@", 1)[0], "arm": s["label"].split("@", 1)[1] if "@" in s["label"] else "nominal",
            "tape_len": L, "steps": s["z80_steps"], "k": s["noise_exp"], "seed": s["seed"], "replicate": prov.get("replicate"), "file": os.path.basename(p),
            "horizon": s["horizon"], "steps_run": s["steps_run"], "stopped_early": s["steps_run"] < s["horizon"],
            "tq_10": s["tq_10"], "tq_50": s["tq_50"],
        }
        # ── emergence scan over the trajectory ──
        samples: list[dict] = []
        row["missing_jsonl"] = not os.path.exists(stem + ".jsonl")
        if not row["missing_jsonl"]:
            recs, bad = read_jsonl(stem + ".jsonl")
            samples = [r for r in recs if r.get("kind") == "sample"]
            row["truncated_jsonl"] = bad
        if samples:
            row.update(scan_emergence(samples, s["z80_steps"], patterns, sets, cache))
        else:
            row.update({"t_rep": np.nan, "t_faith": np.nan})
        # ── exemplar at the tq_10 sample (pre-registered emergence event) ──
        if s["tq_10"] > 0 and samples:
            at = next((r for r in samples if r["step"] == s["tq_10"]), None)
            if at is not None:
                rank, r, t = best_of([e["tape"] for e in at["exemplars"]][:3], s["z80_steps"], patterns)
                row.update({"em_tape": t, "em_rank": rank, "em_score": round(r["score"], 3), "em_gen2": round(r["gen2_score"], 3),
                            "em_q75": round(r["offspring_within_q"], 3), "em_faithful": r["faithful"]})
        # ── final state ──
        fin = s["final"]
        final_tapes = [e["tape"] for e in fin["exemplars"]][:3]
        rank, r, t = best_of(final_tapes, s["z80_steps"], patterns)
        row.update({
            "final_tape": t, "final_rank": rank, "final_share": round(fin["top3_shares"][rank], 4) if rank < len(fin.get("top3_shares", [])) else np.nan,
            "final_score": round(r["score"], 3), "final_gen2": round(r["gen2_score"], 3), "final_gen2_cond": round(r["gen2_cond"], 3),
            "final_q75": round(r["offspring_within_q"], 3), "final_faithful": r["faithful"], "final_replicator": r["is_replicator"], "final_offset": r.get("copy_offset"),
            "final_self_b": round(r["self_preserved_as_B"], 3), "final_mechs": "+".join(mechanisms(hexbytes(t), sets)) or "-",
            **period_fields("final", t),
        })
        snap = load_snapshot(stem + ".soup_final.u8.br", L)
        if snap is not None:
            ri = assay(hexbytes(t), z80_steps=s["z80_steps"], suppress=patterns, neighbors=snap, seed=s["seed"])
            row.update({"final_insitu_score": round(ri["score"], 3), "final_insitu_gen2": round(ri["gen2_score"], 3),
                        "final_insitu_n_inf": ri["n_informative"], "final_insitu_replicator": ri["is_replicator"]})
            rank_b, rb, tb = best_of(final_tapes, s["z80_steps"], patterns, neighbors=snap, seed=s["seed"])
            row.update({"final_insitu_best_gen2": round(rb["gen2_score"], 3), "final_insitu_best_tape": tb})
            # random-cell functional fraction (both partner types), vectorised
            rng = np.random.default_rng(s["seed"])
            cells = snap[rng.integers(0, snap.shape[0], size=FUNC_CELLS)]
            rnd = assay_many(cells, z80_steps=s["z80_steps"], suppress=patterns, n=FUNC_PARTNERS, seed=s["seed"])
            ins = assay_many(cells, z80_steps=s["z80_steps"], suppress=patterns, n=FUNC_PARTNERS, seed=s["seed"], neighbors=snap)
            informative = [x for x in ins if x["n_informative"] >= 8 and x.get("n_informative2", 8) >= 8]
            row.update({
                "final_func_n": FUNC_CELLS,
                "final_func_rnd": float(np.mean([x["is_replicator"] for x in rnd])),
                "final_func_rnd_faithful": float(np.mean([x["faithful"] for x in rnd])),
                "final_func_insitu": float(np.mean([x["is_replicator"] for x in informative])) if informative else np.nan,
                "final_func_insitu_n": len(informative),
            })
        row.update({
            "final_c_blockcopy": fin.get("c_blockcopy"), "final_c_push2": fin.get("c_push2"), "final_c_zero8": fin.get("c_zero8"),
            "final_hoe": fin.get("hoe"), "final_H0": fin.get("H0"), "final_zero_frac": fin.get("zero_frac", np.nan),
            "final_q_share": fin.get("q_share"), "final_q_shift_share": fin.get("q_shift_share", np.nan),
        })
        rows.append(row)
        print(f"[{i+1}/{len(files)}] {s['label']:14s} L{L:<3} st{s['z80_steps']:<3} k{s['noise_exp']} s{s['seed']:<4} tq10 {s['tq_10']:>7} t_rep {row.get('t_rep', -1)!s:>7} t_faith {row.get('t_faith', -1)!s:>7} | final gen2 {row['final_gen2']:.2f} faithful {row['final_faithful']!s:5} func_rnd {row.get('final_func_rnd', float('nan')):.2f} insitu {row.get('final_func_insitu', float('nan')):.2f}", file=sys.stderr)
    df = pd.DataFrame(rows)
    out = os.path.join(d, "analysis")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "assays.csv"), index=False)
    cell = df.groupby(["label", "tape_len", "steps", "k"]).agg(
        n=("seed", "size"),
        tq10=("tq_10", lambda x: int((x > 0).sum())),
        t_rep=("t_rep", lambda x: int((x > 0).sum())),
        t_faith=("t_faith", lambda x: int((x > 0).sum())),
        final_rep=("final_replicator", "sum"),
        final_faith=("final_faithful", "sum"),
        func_rnd=("final_func_rnd", "median"),
        func_insitu=("final_func_insitu", "median"),
        stopped=("stopped_early", "sum"),
    )
    pd.set_option("display.width", 200)
    print(cell.to_string())


if __name__ == "__main__":
    main(sys.argv[1])
