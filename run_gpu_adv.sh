#!/usr/bin/env bash
# EXPLORATORY follow-up (not pre-registered): ordinary AST fine-tuning + recording-identity adversary (gradient reversal).
# Writes into the SAME runs dir as run_gpu_suite.sh, so it is paired seed-by-seed with the finished "ordinary" runs.
# usage: bash run_gpu_adv.sh <python> <data dir> <fbank dir> <runs dir> <K> [seeds...]   env: ADV (default 0.3)
PY=$1; DATA=$2; FB=$3; RUNS=$4; K=$5; shift 5; SEEDS=${@:-0 1 2}
EPOCHS=${EPOCHS:-4}; SPE=${SPE:-768}; BATCH=${BATCH:-32}; LR=${LR:-3e-5}; ADV=${ADV:-0.3}
for seed in $SEEDS; do
  out="$RUNS/ast_ftadv_source_disjoint_standard_s${seed}"
  [ -f "$out/test_metrics.json" ] && { echo "skip $out (finished)"; continue; }
  mkdir -p "$out"
  "$PY" ast_finetune.py train --data "$DATA" --cache "$FB" --inputs fbank --amp --freeze-k "$K" --mode finetune \
    --sampler standard --adv $ADV --lr $LR --seed "$seed" --epochs $EPOCHS --samples-per-epoch $SPE --batch $BATCH \
    --out "$out" >> "$out/train.log" 2>&1
  echo "$out: $(tail -1 "$out/train.log")"
done
echo ADV_DONE
