"""Supplementary table for X1 (branching transfer) from results/branching/branching_summary.csv and branching_mutants.csv.gz.

    .venv/bin/python branching_table.py      # results/branching/SUPP_TABLE.md
"""
from __future__ import annotations

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "results", "branching")
NAMES = {"pusher16": "pusher, L = 16", "pusher64": "pusher, L = 64", "ret16": "return closer, L = 16",
         "ldir32": "block-copy tiling, L = 32", "genome20_s6003": "transmitter, L = 20"}


def main():
    S = pd.read_csv(os.path.join(D, "branching_summary.csv"))
    M = pd.read_csv(os.path.join(D, "branching_mutants.csv.gz"))
    rows = ["| genotype | partners | unmutated alive, g = 1 / 4 / 8 | sites, g = 1 / 2 / 4 / 8 | mutants carrying, g = 8 |",
            "|---|---|---|---|---|"]
    for (key, arm), d in S.groupby(["key", "arm"], sort=False):
        d = d.set_index("gen")
        m8 = M[(M.key == key) & (M.arm == arm) & (M.gen == 8) & M.identifiable]
        carry = int((m8.retained > 0).sum())
        rows.append(f"| {NAMES.get(key, key)} | {'uniform random' if arm == 'U' else 'open-phase soup'} | "
                    f"{d.ctrl_alive[1]:.2f} / {d.ctrl_alive[4]:.2f} / {d.ctrl_alive[8]:.2f} | "
                    f"{int(d.sites[1])} / {int(d.sites[2])} / {int(d.sites[4])} / {int(d.sites[8])} | {carry} of {len(m8)} |")
    open(os.path.join(D, "SUPP_TABLE.md"), "w").write("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
