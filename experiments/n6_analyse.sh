#!/bin/zsh
# N6: frozen analysis pipeline on the replication worlds (paths only differ from N3). Run from anywhere.
set -e
E=/Users/neo/repos/algocell/experiments
export LOD_IND=$E/runs/lod_n6_modal/lod_n6
export LOD_OUT=$E/results/lod_n6
mkdir -p $LOD_OUT
cd $E
.venv/bin/python lod_chain.py --indir $LOD_IND --out $LOD_OUT --tag _v6b
.venv/bin/python lod_founders2.py
.venv/bin/python lod_parts.py --indir $LOD_IND --chain $LOD_OUT/chain_v6b.csv --out $LOD_OUT --tag _v6
.venv/bin/python lod_parts_first.py
.venv/bin/python lod_founding.py
.venv/bin/python lod_founding_summary.py
.venv/bin/python lod_reach.py
.venv/bin/python n6_score.py
