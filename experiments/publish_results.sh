#!/bin/sh
# Copy a batch's analysis outputs (small tables, figures, report) from the
# gitignored runs/ tree into the tracked results/ tree.
# usage: ./publish_results.sh stageA
set -e
B="$1"; [ -n "$B" ] || { echo "usage: $0 <batch>"; exit 1; }
cd "$(dirname "$0")"
mkdir -p "results/$B"
cp runs/$B/analysis/*.csv runs/$B/analysis/*.png runs/$B/analysis/ZOO.md "results/$B/" 2>/dev/null || true
[ -f "runs/REPORT_$B.md" ] && sed "s#$B/analysis/#$B/#g" "runs/REPORT_$B.md" > "results/REPORT_$B.md"
du -sh "results/$B" | cut -f1 | xargs -I{} echo "results/$B: {}"
