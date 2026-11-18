#!/usr/bin/env bash
# GPU version of the controlled 2x2 AST experiment (same design as run_ast_suite.sh), for Colab or any CUDA machine.
#   sampler {standard, source}  x  loss {cross-entropy, cross-entropy + cross-source SupCon}  + frozen-AST reference
# Inputs are AST filterbanks (ast_finetune.py fbank); transformer layers K..11 are fine-tuned live, K=0 = all 12 layers.
# Resumable: a finished run (test_metrics.json) is skipped, an interrupted run resumes from its last epoch (last.pt).
# Keep RUNS on persistent storage (Google Drive) so a Colab disconnect loses at most one epoch.
# usage: bash run_gpu_suite.sh <python> <data dir> <fbank dir> <runs dir> <K> [seeds...]
#   env overrides: EPOCHS SPE BATCH LAMBDA TAU LR
PY=$1; DATA=$2; FB=$3; RUNS=$4; K=$5; shift 5; SEEDS=${@:-0 1 2}
EPOCHS=${EPOCHS:-4}; SPE=${SPE:-768}; BATCH=${BATCH:-32}; LAMBDA=${LAMBDA:-0.5}; TAU=${TAU:-0.1}; LR=${LR:-3e-5}
mkdir -p "$RUNS"
for seed in $SEEDS; do
  for cond in "finetune standard 0 ast_ft" "finetune source 0 ast_ft" "finetune standard $LAMBDA ast_ftsc" \
              "finetune source $LAMBDA ast_ftsc" "frozen standard 0 ast_frozen"; do
    set -- $cond
    out="$RUNS/${4}_source_disjoint_${2}_s${seed}"
    [ -f "$out/test_metrics.json" ] && { echo "skip $out (finished)"; continue; }
    mkdir -p "$out"
    "$PY" ast_finetune.py train --data "$DATA" --cache "$FB" --inputs fbank --amp --freeze-k "$K" --mode "$1" \
      --sampler "$2" --supcon "$3" --tau $TAU --lr $LR --seed "$seed" --epochs $EPOCHS --samples-per-epoch $SPE \
      --batch $BATCH --out "$out" >> "$out/train.log" 2>&1
    echo "$out: $(tail -1 "$out/train.log")"
  done
done
echo SUITE_DONE
