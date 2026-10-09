#!/bin/sh
# Stage K post-processing (runs from experiments/): assays → c4 census → per-world closure table → zoo → findings, then the
# pre-registered K1–K3 scoring. Logs under runs/stageK_*.log. Run files live in runs/stageK/stageK (as fetched from Modal).
set -eu
cd "$(dirname "$0")"
PY=.venv/bin/python
D=runs/stageK/stageK
$PY assay_batch.py $D > runs/stageK_assay.log 2>&1
$PY c4_functional.py $D > runs/stageK_c4.log 2>&1
$PY stage_g.py --dir $D --out $D/analysis/stageK --no-verdicts > runs/stageK_stage_g.log 2>&1
$PY zoo.py $D > runs/stageK_zoo.log 2>&1 || echo "zoo failed (non-fatal)"
$PY findings.py $D --out $D/analysis/NUMBERS.md > runs/stageK_findings.log 2>&1 || echo "findings failed (non-fatal)"
mkdir -p results/stageK
cp -R $D/analysis/. results/stageK/ 2>/dev/null || true
cp $D/../stageK_summaries.json results/stageK/ 2>/dev/null || cp runs/stageK/stageK_summaries.json results/stageK/ 2>/dev/null || true
$PY - <<'PYEOF'
import pandas as pd, numpy as np, os
p = "runs/stageK/stageK/analysis/stageK/stage_g_runs.csv"
g = pd.read_csv(p)
def is_push(t):
    b = str(t).split(); return len(b) >= 4 and b[0] in ("01", "11", "21") and b[1] in ("c5", "d5", "e5") and b[2:4] == b[0:2]
k1 = int(g.first_tape.map(is_push).sum())
closed = (g.final_has_cf | g.final_has_block) & (g.final_copied >= 0.95) & (g.final_damaged <= 0.0)
k2 = int(closed.sum())
med = float(np.nanmedian(g.t_rep[g.t_rep > 0]))
lines = ["# Stage K (L = 32, aligned ring, 20 worlds, 1,000,000 steps): pre-registered scoring", "",
         f"- K1 first replicator a load–push word: {k1}/20 (needs ≥ 18) → {'MET' if k1 >= 18 else 'NOT MET'}",
         f"- K2 closed successor (loop-bearing, copies ≥ 0.95, damage 0.00) by 1M steps: {k2}/20 (needs ≥ 15; kill < 10) → {'MET' if k2 >= 15 else ('KILLED' if k2 < 10 else 'NOT MET')}",
         f"- K3 median first-replicator step {med:,.0f} (predicted 300–1,000) → {'MET' if 300 <= med <= 1000 else 'NOT MET'}",
         "", "first replicators:", g.first_tape.str[:14].value_counts().to_string(), "", "final dominants (loop / block / copied / damaged):",
         g[["seed", "final_cf", "final_block", "final_copied", "final_damaged", "final_share"]].to_string(index=False)]
out = "\n".join(lines); print(out)
os.makedirs("results/stageK", exist_ok=True); open("results/stageK/SCORING.md", "w").write(out + "\n")
PYEOF
echo "STAGE K PIPELINE DONE"
