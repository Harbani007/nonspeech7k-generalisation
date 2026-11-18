#!/usr/bin/env bash
# Controlled 2x2 AST experiment on the cached frozen tokens (ast_finetune.py cache must have finished). Resumable.
#   sampler {standard, source}  x  loss {cross-entropy, cross-entropy + cross-source SupCon}
# Every condition at a given seed shares the head/projection initialisation, the RNG stream, the step budget
# (EPOCHS x SPE samples, batch BATCH) and the optimiser. lambda and tau are fixed a priori (no tuning on test).
# Plus one frozen-AST (linear head only) reference per seed.
# usage: bash run_ast_suite.sh <python> <data dir> <cache dir> <runs dir> [seeds...]
PY=$1; DATA=$2; CACHE=$3; RUNS=$4; shift 4; SEEDS=${@:-0}
EPOCHS=4; SPE=768; BATCH=32; LAMBDA=0.5; TAU=0.1
for seed in $SEEDS; do
  for cond in "finetune standard 0 ast_ft" "finetune source 0 ast_ft" "finetune standard $LAMBDA ast_ftsc" \
              "finetune source $LAMBDA ast_ftsc" "frozen standard 0 ast_frozen"; do
    set -- $cond
    out="$RUNS/${4}_source_disjoint_${2}_s${seed}"
    [ -f "$out/test_metrics.json" ] && { echo "skip $out"; continue; }
    mkdir -p "$out"
    "$PY" ast_finetune.py train --data "$DATA" --cache "$CACHE" --mode "$1" --sampler "$2" --supcon "$3" --tau $TAU \
      --seed "$seed" --epochs $EPOCHS --samples-per-epoch $SPE --batch $BATCH --out "$out" > "$out/train.log" 2>&1
    echo "$out: $(tail -1 "$out/train.log")"
  done
done
echo SUITE_DONE
