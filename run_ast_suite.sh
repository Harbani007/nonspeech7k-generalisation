#!/usr/bin/env bash
# AST experiments on the cached frozen tokens (ast_finetune.py cache must have finished). Resumable.
# usage: bash run_ast_suite.sh <python> <data dir> <cache dir> <runs dir> [seeds...]
PY=$1; DATA=$2; CACHE=$3; RUNS=$4; shift 4; SEEDS=${@:-0}
for seed in $SEEDS; do
  for ms in "frozen standard" "finetune standard" "finetune source"; do
    set -- $ms
    name=$([ "$1" = frozen ] && echo ast_frozen || echo ast_ft)
    out="$RUNS/${name}_source_disjoint_${2}_s${seed}"
    [ -f "$out/test_metrics.json" ] && { echo "skip $out"; continue; }
    mkdir -p "$out"
    "$PY" ast_finetune.py train --data "$DATA" --cache "$CACHE" --mode "$1" --sampler "$2" --seed "$seed" --out "$out" > "$out/train.log" 2>&1
    tail -1 "$out/train.log"
  done
done
echo SUITE_DONE
