#!/usr/bin/env bash
# Runs the CNN experiments sequentially; skips runs whose test_metrics.json exists (resumable).
# usage: bash run_cnn_suite.sh <python> <data dir> <runs dir>
PY=$1; DATA=$2; RUNS=$3
for seed in 0 1 2; do
  for proto_sampler in "source_disjoint standard" "source_disjoint source" "official standard"; do
    set -- $proto_sampler
    out="$RUNS/cnn_${1}_${2}_s${seed}"
    [ -f "$out/test_metrics.json" ] && { echo "skip $out"; continue; }
    mkdir -p "$out"
    "$PY" train_cnn.py --data "$DATA" --protocol "$1" --sampler "$2" --seed "$seed" --epochs 25 \
      --samples-per-epoch 1024 --out "$out" > "$out/train.log" 2>&1
    tail -1 "$out/train.log"
  done
done
echo SUITE_DONE
