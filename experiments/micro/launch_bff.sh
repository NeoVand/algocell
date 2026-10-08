#!/bin/sh
# BFF soups for THEORY.md P1: 12 seeds × {standard, wrap}, 2^17 programs, 16,384 epochs, in 6 concurrent streams.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p runs/bff
stream() {
  for spec in "$@"; do
    variant=${spec%% *}; seed=${spec##* }; flag=""
    [ "$variant" = wrap ] && flag="--ip-wrap"
    $PY -m micro.bff_soup --out "runs/bff/${variant}_s${seed}" --seed "$seed" $flag > "runs/bff/${variant}_s${seed}.log" 2>&1
  done
}
stream "std 1" "wrap 1" "std 7" "wrap 7" &
stream "std 2" "wrap 2" "std 8" "wrap 8" &
stream "std 3" "wrap 3" "std 9" "wrap 9" &
stream "std 4" "wrap 4" "std 10" "wrap 10" &
stream "std 5" "wrap 5" "std 11" "wrap 11" &
stream "std 6" "wrap 6" "std 12" "wrap 12" &
wait
echo "all BFF streams finished"
