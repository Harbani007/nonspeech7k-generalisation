#!/usr/bin/env bash
# Clean rerun of the log-mel CNN (from scratch) on the source-disjoint split, GPU, 3 seeds x 2 samplers.
# Same settings as the original suite (25 epochs x 1024 samples, batch 32, AdamW 2e-3, OneCycle, no early stopping).
# Writes to a FRESH runs folder so nothing mixes with the old (possibly overlapped) CPU runs. Finished runs are skipped.
# usage: bash run_gpu_cnn.sh <python> <dl data dir with splits.csv + logmel.npy> <runs dir> [seeds...]
PY=$1; DATA=$2; RUNS=$3; shift 3; SEEDS=${@:-0 1 2}
mkdir -p "$RUNS"
for seed in $SEEDS; do
  for sampler in standard source; do
    out="$RUNS/cnn_source_disjoint_${sampler}_s${seed}"
    [ -f "$out/test_metrics.json" ] && { echo "skip $out (finished)"; continue; }
    rm -rf "$out"; mkdir -p "$out"   # the CNN trainer has no resume: an interrupted run restarts from scratch
    "$PY" train_cnn.py --data "$DATA" --protocol source_disjoint --sampler "$sampler" --seed "$seed" --epochs 25 \
      --samples-per-epoch 1024 --batch 32 --device cuda --out "$out" > "$out/train.log" 2>&1
    echo "$out: $(tail -1 "$out/train.log")"
  done
done
echo CNN_DONE
