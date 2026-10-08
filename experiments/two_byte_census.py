"""Census of every two-byte word as a tiled organism (THEORY.md Lemma 1, finite search).

    python two_byte_census.py [--L 16] [--partners 32] [--out runs/census2]

All 65,536 words (a, b) tiled to L bytes are run through the culture test (assay_many: n random partners, 128 Z80 steps,
gen2 from the produced copies); every heritable word (gen2 ≥ 0.3) then gets the partner-independence test (256 random
partners: fraction copied ≥ 75%, fraction of encounters with ≥ 25% self-damage) and a disassembly. Writes census2.csv,
heritable.csv and NUMBERS_CENSUS2.md.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from algocell_exp.assay import assay_many
from algocell_exp.isa import disassemble
from closure import control_flow, partner_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--partners", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--out", default="runs/census2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    words = np.array([(x, y) for x in range(256) for y in range(256)], dtype=np.uint8)
    tapes = np.tile(words, (1, a.L // 2))
    rows = []
    for i in range(0, len(tapes), a.chunk):
        res = assay_many(tapes[i:i + a.chunk], z80_steps=128, suppress=(), n=a.partners, seed=0)
        for j, r in enumerate(res):
            rows.append({"a": int(words[i + j, 0]), "b": int(words[i + j, 1]), "word": f"{words[i + j, 0]:02x} {words[i + j, 1]:02x}",
                         "score": r["score"], "gen2": r["gen2_score"], "heritable": bool(r["is_replicator"]), "faithful": bool(r["faithful"])})
        print(f"{i + len(res)}/{len(tapes)}", flush=True)
    C = pd.DataFrame(rows)
    C.to_csv(os.path.join(a.out, "census2.csv"), index=False)
    H = C[C["heritable"]].copy()
    rng = np.random.default_rng(0)
    cf, bl, cop, dmg, sim, mn = [], [], [], [], [], []
    for _, r in H.iterrows():
        hx = " ".join(f"{v:02x}" for v in tapes[r["a"] * 256 + r["b"]])
        c, d, s = partner_test(tapes[r["a"] * 256 + r["b"]], rng)
        cf.append("+".join(control_flow(hx)) or "-")
        ins = disassemble(tapes[r["a"] * 256 + r["b"]])
        mn.append(" ; ".join(dict.fromkeys(x["mnemonic"] for x in ins[:4])))
        cop.append(c); dmg.append(d); sim.append(s)
    H["control_flow"], H["mnemonics"], H["copied"], H["damaged"], H["mean_similarity"] = cf, mn, cop, dmg, sim
    H = H.sort_values("gen2", ascending=False)
    H.to_csv(os.path.join(a.out, "heritable.csv"), index=False)
    md = [f"# Two-byte word census at L = {a.L} (generated)\n",
          f"- {len(C):,} words tiled to {a.L} bytes, culture test with {a.partners} random partners, 128 steps: heritable (gen2 ≥ 0.3) {len(H)}, faithful {int(C['faithful'].sum())}, "
          f"score ≥ 0.5 {int((C['score'] >= 0.5).sum())}",
          f"- heritable words with a control-flow instruction: {int((H['control_flow'] != '-').sum())}/{len(H)}; partner test (256 partners): copied median {H['copied'].median():.2f} "
          f"(max {H['copied'].max():.2f}), self-damage median {H['damaged'].median():.2f} (min {H['damaged'].min():.2f}); words copying ≥ 0.95 of partners: {int((H['copied'] >= 0.95).sum())}\n" if len(H) else "- no heritable word",
          H[["word", "mnemonics", "control_flow", "score", "gen2", "faithful", "copied", "damaged"]].head(40).to_markdown(index=False, floatfmt=".2f") if len(H) else ""]
    with open(os.path.join(a.out, "NUMBERS_CENSUS2.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
