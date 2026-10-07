#!/bin/sh
# Download a batch's results from the Modal volume into runs/<batch>/ and summarize.
# usage: ./fetch.sh stageA
set -e
B="$1"; [ -n "$B" ] || { echo "usage: $0 <batch>"; exit 1; }
cd "$(dirname "$0")"
mkdir -p "runs/$B"
.venv/bin/modal volume get algocell-atlas-runs "$B/" runs/ --force >/dev/null 2>&1 || true
echo "$(ls runs/$B/*.summary.json 2>/dev/null | wc -l | tr -d ' ') summaries in runs/$B"
