#!/bin/sh
# Generic post-processing for a fetched Modal stage: assays → c4 census → per-world closure table → zoo → findings, copied to
# results/<stage>/. usage: ./stage_pipeline.sh stageM runs/stageM/stageM
set -eu
cd "$(dirname "$0")"
S="$1"; D="$2"; PY=.venv/bin/python
$PY assay_batch.py $D > runs/${S}_assay.log 2>&1
$PY c4_functional.py $D > runs/${S}_c4.log 2>&1
$PY stage_g.py --dir $D --out $D/analysis/$S --no-verdicts > runs/${S}_stage_g.log 2>&1
$PY zoo.py $D > runs/${S}_zoo.log 2>&1 || echo "zoo failed (non-fatal)"
$PY findings.py $D --out $D/analysis/NUMBERS.md > runs/${S}_findings.log 2>&1 || echo "findings failed (non-fatal)"
mkdir -p results/$S && cp -R $D/analysis/. results/$S/ 2>/dev/null || true
echo "PIPELINE $S DONE"
