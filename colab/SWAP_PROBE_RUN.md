# Colab run: test-time background swap (no training; T4)

Pre-registered in `docs/PREREGISTRATION_SWAP_PROBE.md` (committed before any result). Primary model: the 3 GPU
`ast_ft_source_disjoint_standard_s*` checkpoints (full fine-tuning, FREEZE_K = 0). The other 9 GPU checkpoints are
secondary. Nothing is trained; the test partition is only scored.

## Setup
Run the usual setup cell first (Drive mounted; `WORK`, `FB`, `RUNS`, `FREEZE_K = 0` defined; audio unzipped to
/content/data/audio; /content/splits/splits.csv present). Then pull the newest code:
```python
%cd /content
!rm -rf /content/ns7k-src && git clone -q -b analysis/source-identity https://github.com/Harbani007/nonspeech7k-generalisation.git /content/ns7k-src
%cd /content/ns7k-src
!git log -1 --format="%h %s"
!ls "{RUNS}" | grep -E "^ast_ft(sc)?_source_disjoint_(standard|source)_s[012]$"   # expect 12 lines
```

## Run (check step a few minutes; full run estimated 20-60 min on a T4, not yet measured)
```python
SWAP = f"{WORK}/swap_probe_k{FREEZE_K}"
!bash run_gpu_swap.sh python /content/splits "{FB}/clips.csv" /content/data/audio "{RUNS}" {FREEZE_K} "{SWAP}" 2>&1 \
   | grep -vE "Warning|UNEXPECTED|^Key|^-+\+|^Notes|can be ignored|^$"
```
What to check in the output, in order:
1. The reproduction table: 12 rows, `freeze_k` 0, `agreement` 1.0 (or > 0.99; fp16 can flip a near-tie). If a run is
   lower than 0.99, stop and send me the table.
2. `donor_balance.csv` lines (|smd| ideally < 0.25) and the donor pool per class.
3. The `condition_scores` and `swap_effects` tables, the residue audit, and the `DECISION` line at the end.

If Colab disconnects, rerun the cell: the check step is repeated and the full run starts again (it takes no
training state; it refuses to overwrite only a run that finished, i.e. one with `decision.json`).

## Send me
The printed output of the cell, and from `{SWAP}`: `decision.json`, `swap_effects.csv`, `condition_scores.csv`,
`donor_balance.csv`, `residue_audit.csv`, `attraction_by_class.csv`, `reproduction_check.csv`, `G11_swap_probe.png`,
and the `examples/` folder (zip it: `!cd "{SWAP}" && zip -qr examples.zip examples`).

## Listening task for you (the audio-quality audit)
Open `examples/*.png` and listen to the matching `.wav` files:
- `*__orig.wav` vs `*__fg.wav`: is the whole vocalisation kept in `fg`? (If parts are cut, the mask is too tight.)
- `*__same_m20.wav` / `*__diff_m20.wav`: does the vocalisation sound identical in both, with only the background changed?
- `suspicious_donor_bg__*__bg_m20.wav` / `__bg_m30.wav`: can you still hear the donor's vocalisation? These are the
  donor backgrounds the model finds most class-like, so they are the most likely to contain residue.
Write down what you hear per file (yes/no + a word). That goes into the report as the manual audit.
