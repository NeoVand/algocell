#!/bin/sh
# Copy a batch's analysis outputs (tables, figures, zoo, report) from the gitignored runs/
# tree into the tracked results/ tree. Fails loudly on any missing input.
# usage: ./publish_results.sh stageA
set -eu
B="$1"
cd "$(dirname "$0")"
SRC="runs/$B/analysis"
[ -d "$SRC" ] || { echo "no $SRC"; exit 1; }
rm -rf "results/$B"
mkdir -p "results/$B"
cp "$SRC"/*.csv "$SRC"/*.png "$SRC"/*.pdf "results/$B/" 2>/dev/null || true
[ -f "$SRC/ZOO.md" ] && cp "$SRC/ZOO.md" "results/$B/"
for sub in "$SRC"/*/; do
  [ -d "$sub" ] || continue
  name=$(basename "$sub")
  mkdir -p "results/$B/$name"
  cp "$sub"/* "results/$B/$name/"
done
if [ -f "runs/REPORT_$B.md" ]; then
  sed "s#$B/analysis/#$B/#g" "runs/REPORT_$B.md" > "results/REPORT_$B.md"
fi
[ -f "runs/$B/analysis/NUMBERS.md" ] && cp "runs/$B/analysis/NUMBERS.md" "results/$B/NUMBERS.md"
du -sh "results/$B" | cut -f1 | xargs -I{} echo "results/$B: {}"
