#!/bin/sh
# Fetch the BFF soups from the Modal volume into runs/bff_modal/bff/, analyse, publish to results/bff/.
# usage: micro/analyze_bff.sh [--no-fetch]
set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python
if [ "${1:-}" != "--no-fetch" ]; then
  mkdir -p runs/bff_modal
  $PY -m modal volume get algocell-atlas-runs bff/ runs/bff_modal/ --force
fi
echo "$(ls runs/bff_modal/bff/*/summary.json 2>/dev/null | wc -l | tr -d ' ') finished soups in runs/bff_modal/bff"
$PY -m micro.bff_analysis --dir runs/bff_modal/bff --out runs/bff_modal/analysis > runs/bff_modal_analysis.log 2>&1
mkdir -p results/bff
cp runs/bff_modal/analysis/NUMBERS_BFF.md runs/bff_modal/analysis/runs.csv results/bff/
cp runs/bff_modal/analysis/*.png runs/bff_modal/analysis/*.pdf runs/bff_modal/analysis/*.svg results/bff/ 2>/dev/null || true
# cross-machine reproducibility: same seed locally (Apple GPU) and on Modal (L40S) must give identical epoch statistics
$PY - <<'PYEOF'
import os, pandas as pd
rows = []
for d in sorted(os.listdir("runs/bff")):
    a, b = f"runs/bff/{d}/epochs.csv", f"runs/bff_modal/bff/{d}/epochs.csv"
    if os.path.isdir(f"runs/bff/{d}") and os.path.exists(a) and os.path.exists(b):
        try:
            A, B = pd.read_csv(a), pd.read_csv(b)
        except pd.errors.EmptyDataError:
            continue
        n = min(len(A), len(B))
        cols = ["executed_mean", "frac_entered", "writesB_mean", "copy_frac", "zero_frac"]
        same = all((A[c].values[:n] == B[c].values[:n]).all() for c in cols)
        rows.append({"run": d, "epochs_compared": n, "identical": same})
R = pd.DataFrame(rows)
print(R.to_string(index=False) if len(R) else "no overlapping runs yet")
if len(R):
    R.to_csv("results/bff/reproducibility_local_vs_modal.csv", index=False)
PYEOF
echo "analysis complete: results/bff/NUMBERS_BFF.md"
