# Background probe results (Colab T4, 19 Nov). Transcribed from `colab_background_probe_run.ipynb` (cell 3 output)

Models: the 12 full-fine-tuning checkpoints (FREEZE_K=0, 3 seeds x 4 conditions) + frozen-AST logistic-regression probe.
Test: source-disjoint test clips whose foreground covers 5-80% of the clip. Macro-F1, 95% CI = bootstrap over test
source groups (seeds averaged inside each resample). Chance = 95th percentile of 200 prediction permutations.

## Primary: foreground = within 20 dB of the loudest frame (645 test clips)
| system | full | background | foreground | random control |
|---|---|---|---|---|
| ordinary | 0.788 [0.720, 0.837] | 0.304 [0.257, 0.341] | 0.732 [0.665, 0.783] | 0.671 [0.598, 0.719] |
| source-balanced | 0.797 [0.731, 0.844] | 0.265 [0.224, 0.301] | 0.741 [0.673, 0.794] | 0.662 [0.587, 0.718] |
| contrastive | 0.785 [0.719, 0.831] | 0.304 [0.260, 0.340] | 0.729 [0.666, 0.776] | 0.657 [0.585, 0.705] |
| combined | 0.784 [0.715, 0.834] | 0.277 [0.231, 0.319] | 0.744 [0.675, 0.800] | 0.655 [0.584, 0.713] |
| frozen probe, trained = tested variant | 0.761 [0.695, 0.817] | 0.387 [0.327, 0.437] | 0.750 [0.676, 0.807] | 0.654 [0.569, 0.713] |
| frozen probe trained on full | 0.761 | 0.284 [0.233, 0.329] | 0.738 [0.672, 0.795] | - |
Chance (95th pct) 0.125-0.170 in every row; every background CI lies above it.

Paired differences (b - a), fine-tuned models: full -> background -0.48 to -0.53; background -> random +0.35 to +0.40;
full -> foreground -0.040 to -0.056 (all CIs exclude 0).

## Sensitivity: within 10 dB (1,037 clips, less strict, more vocalisation residue)
background 0.440-0.489; foreground 0.660-0.675; full 0.791-0.820; frozen probe trained on background 0.575 [0.512, 0.620].
