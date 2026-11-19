#!/usr/bin/env bash
# Background-swap experiment (GPU). Pre-specified in docs/PREREGISTRATION_BGSWAP.md.
#   control : ordinary fine-tuning on the SAME waveform inputs (bgswap 0)
#   bgswap  : each training clip's background replaced, with probability BGSWAP, by the background of a clip from a
#             different recording (vocalisation and label kept)
# Same seeds, init, budget and optimiser as run_gpu_suite.sh. Resumable (finished runs skipped, last.pt resumed).
# usage: bash run_gpu_bgswap.sh <python> <data dir> <waves dir> <runs dir> <K> [seeds...]
PY=$1; DATA=$2; WV=$3; RUNS=$4; K=$5; shift 5; SEEDS=${@:-0 1 2}
EPOCHS=${EPOCHS:-4}; SPE=${SPE:-768}; BATCH=${BATCH:-32}; LR=${LR:-3e-5}; BGSWAP=${BGSWAP:-0.5}
mkdir -p "$RUNS"
for seed in $SEEDS; do
  for cond in "0 ast_wv" "$BGSWAP ast_bgs"; do
    set -- $cond
    out="$RUNS/${2}_source_disjoint_standard_s${seed}"
    [ -f "$out/test_metrics.json" ] && { echo "skip $out (finished)"; continue; }
    mkdir -p "$out"
    "$PY" ast_finetune.py train --data "$DATA" --cache "$WV" --inputs wave --bgswap "$1" --amp --freeze-k "$K" \
      --mode finetune --sampler standard --lr $LR --seed "$seed" --epochs $EPOCHS --samples-per-epoch $SPE \
      --batch $BATCH --out "$out" >> "$out/train.log" 2>&1
    echo "$out: $(tail -1 "$out/train.log")"
  done
done
echo BGSWAP_DONE
