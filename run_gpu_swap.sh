#!/usr/bin/env bash
# Test-time background swap on existing AST checkpoints (no training). Pre-specified in docs/PREREGISTRATION_SWAP_PROBE.md.
# Step 1 loads every checkpoint and checks that 'orig' reproduces its stored test predictions; step 2 is the full run.
# usage: bash run_gpu_swap.sh <python> <data dir with splits.csv> <clips.csv> <audio root> <runs dir> <K> <out dir>
set -e
PY=$1; DATA=$2; CLIPS=$3; AUDIO=$4; RUNS=$5; K=$6; OUT=$7
"$PY" swap_probe.py --data "$DATA" --clips "$CLIPS" --audio-root "$AUDIO" --runs "$RUNS" --expect-freeze-k "$K" \
  --out "$OUT/check" --check-only --check-n 256 --device cuda
"$PY" swap_probe.py --data "$DATA" --clips "$CLIPS" --audio-root "$AUDIO" --runs "$RUNS" --expect-freeze-k "$K" \
  --out "$OUT" --device cuda --batch 64
echo SWAP_DONE
