#!/bin/sh
# Fetch a finished stage from the Modal volume and run the whole post hoc chain.
# usage: ./analyze_stage.sh stageD [--no-fetch]
set -eu
B="$1"; shift || true
cd "$(dirname "$0")"
if [ "${1:-}" != "--no-fetch" ]; then ./fetch.sh "$B"; fi
PY=.venv/bin/python
$PY assay_batch.py "runs/$B" > "runs/${B}_assay.log" 2>&1
$PY succession.py "runs/$B" > "runs/${B}_succession.log" 2>&1
$PY zoo.py "runs/$B" > "runs/${B}_zoo.log" 2>&1
$PY analyze.py "runs/$B" > "runs/${B}_analyze.log" 2>&1
$PY findings.py "runs/$B" --out "runs/$B/analysis/NUMBERS.md" > "runs/${B}_findings.log" 2>&1
$PY report.py "runs/$B" --out "runs/REPORT_$B.md" > "runs/${B}_report.log" 2>&1
./publish_results.sh "$B"
echo "analysis of $B complete: runs/REPORT_$B.md, results/$B/"
