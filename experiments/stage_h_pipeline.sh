#!/bin/sh
# Stage H post-processing in one go (runs from experiments/): assays → c4 census → per-world closure table → standard
# chain → publish → pre-registered scoring. Logs under runs/stageH_*.log. See results/stageH/COMMANDS.md.
set -eu
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY assay_batch.py runs/stageH > runs/stageH_assay.log 2>&1
$PY c4_functional.py runs/stageH > runs/stageH_c4.log 2>&1
$PY stage_g.py --dir runs/stageH --out runs/stageH/analysis/stageH --no-verdicts > runs/stageH_stage_g.log 2>&1
$PY succession.py runs/stageH > runs/stageH_succession.log 2>&1 || echo "succession failed (non-fatal)"
$PY zoo.py runs/stageH > runs/stageH_zoo.log 2>&1 || echo "zoo failed (non-fatal)"
$PY analyze.py runs/stageH > runs/stageH_analyze.log 2>&1 || echo "analyze failed (non-fatal)"
$PY findings.py runs/stageH --out runs/stageH/analysis/NUMBERS.md > runs/stageH_findings.log 2>&1 || echo "findings failed (non-fatal)"
$PY report.py runs/stageH --out runs/REPORT_stageH.md > runs/stageH_report.log 2>&1 || echo "report failed (non-fatal)"
./publish_results.sh stageH
$PY stage_h_analysis.py --h-runs results/stageH/stageH/stage_g_runs.csv
echo "STAGE H PIPELINE DONE"
