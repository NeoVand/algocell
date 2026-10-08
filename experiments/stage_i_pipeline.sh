#!/bin/sh
# Stage I post-processing (runs from experiments/): assays → c4 census → per-world table → standard chain → publish → scoring.
set -eu
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY assay_batch.py runs/stageI > runs/stageI_assay.log 2>&1
$PY c4_functional.py runs/stageI > runs/stageI_c4.log 2>&1
$PY stage_g.py --dir runs/stageI --out runs/stageI/analysis/stageI --no-verdicts > runs/stageI_stage_g.log 2>&1
$PY succession.py runs/stageI > runs/stageI_succession.log 2>&1 || echo "succession failed (non-fatal)"
$PY zoo.py runs/stageI > runs/stageI_zoo.log 2>&1 || echo "zoo failed (non-fatal)"
$PY analyze.py runs/stageI > runs/stageI_analyze.log 2>&1 || echo "analyze failed (non-fatal)"
$PY findings.py runs/stageI --out runs/stageI/analysis/NUMBERS.md > runs/stageI_findings.log 2>&1 || echo "findings failed (non-fatal)"
$PY report.py runs/stageI --out runs/REPORT_stageI.md > runs/stageI_report.log 2>&1 || echo "report failed (non-fatal)"
./publish_results.sh stageI
$PY stage_i_analysis.py
echo "STAGE I PIPELINE DONE"
