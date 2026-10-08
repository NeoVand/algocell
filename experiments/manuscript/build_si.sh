#!/bin/sh
# Assemble the Supplementary Information from the tracked documents (no figures: Nature puts figures in Extended Data).
set -eu
cd "$(dirname "$0")/.."
OUT=manuscript/SI.md
{
  echo "# Supplementary Information"
  echo
  echo "Open replicators evolve closure in a digital primordial soup — supplementary text, assembled $(date +%Y-%m-%d) from the tracked documents of the experiments repository."
  echo
  echo "## S1. Theorems, propositions and the model (THEOREMS.md)"; echo; sed 's/^# /### /' THEOREMS.md
  echo; echo "## S2. Formal definitions, predictions and their pre-registered outcomes (THEORY.md)"; echo; sed 's/^# /### /' THEORY.md
  echo; echo "## S3. Pre-registration and change log (PLAN.md)"; echo; sed 's/^# /### /' PLAN.md
  echo; echo "## S4. Generated number tables"; echo
  for f in results/stageG/stageG/NUMBERS_G.md results/stageG/NUMBERS.md results/stageC/stage_c/NUMBERS_C.md results/stageD/stage_d/NUMBERS_D.md results/stageE/stage_e/NUMBERS_E.md results/stageF/stage_f/NUMBERS_F.md results/closure/NUMBERS_CLOSURE.md results/census2/NUMBERS_CENSUS2.md results/bff/NUMBERS_BFF.md results/bff_search/NUMBERS_SEARCH.md results/bff_closed_search/NUMBERS_CLOSED_SEARCH_p10.md results/stageG/c4/NUMBERS_C4.md; do
    [ -f "$f" ] && { echo "### $f"; echo; sed 's/^# /#### /' "$f"; echo; }
  done
  echo; echo "## S5. Literature notes (LITERATURE.md)"; echo; sed 's/^# /### /' LITERATURE.md
} > "$OUT"
echo "wrote $OUT ($(wc -w < "$OUT" | tr -d ' ') words)"
