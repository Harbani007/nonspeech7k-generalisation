# Phase 0: test-time background swap (college A100, 28 Nov 2026)

Pre-registered in `docs/PREREGISTRATION_SWAP_PROBE.md` (+ amendment 1: edge-held masks). Run: `run_gpu_swap.sh` on the
12 fully fine-tuned AST checkpoints retrained on the A100 (`/workspace/work/runs_k0_pc200`, FREEZE_K = 0), code commit
d96267c. No training; the test partition is only scored. Files in this folder are copied from the server's
`/workspace/work/swap_probe_k0` (full zip incl. listening examples: `Downloads\swap_probe_results_A100.zip`).

**Reproduction check:** all 12 checkpoints reproduce their stored test predictions (agreement 1.000 or 0.999; fp16).
**Design:** each of 643 test vocalisations is kept unchanged and its background is replaced by the background of a
**same-class** donor (control) or **different-class** donors (level-matched, 20 dB and 30 dB masks; also raw).
Primary metric: delta = macro-recall(same-class donor) - macro-recall(different-class donor); 95% CI from a
recording-level bootstrap.

## Decision (pre-registered rule): DETECTED BUT SMALL
GO required delta >= 0.05 with CI > 0 in BOTH the 20 dB and 30 dB arms; it holds at 20 dB only.

| model (3 seeds) | delta 20 dB | delta 30 dB | delta raw (no level match) | splice damage (orig - same, 20 dB) |
|---|---|---|---|---|
| **AST ordinary (primary)** | **+0.051 [0.030, 0.077]** | **+0.031 [0.011, 0.052]** | +0.072 [0.048, 0.098] | +0.019 [-0.010, 0.043] |
| AST source-balanced | +0.063 [0.041, 0.088] | +0.031 [0.013, 0.052] | +0.092 [0.069, 0.118] | +0.021 [-0.007, 0.046] |
| AST contrastive | +0.063 [0.039, 0.093] | +0.033 [0.015, 0.051] | +0.086 [0.063, 0.113] | +0.023 [-0.010, 0.052] |
| AST combined | +0.046 [0.026, 0.070] | +0.022 [0.001, 0.044] | +0.082 [0.060, 0.104] | +0.026 [0.001, 0.050] |

Primary model, 20 dB arm: 13.9% of predictions change when the background comes from another class
[11.7, 16.6]%; 5.5% switch to the background's class, against 3.9% expected without any pull (attraction excess
+0.017 [0.012, 0.023]). At 30 dB: 10.2% change, attraction excess +0.008 [0.005, 0.012].

Residue audit (primary model on donor backgrounds alone): macro-F1 for the donor's class 0.323 (20 dB mask) and 0.208
(30 dB mask); the loudest remaining frame is always at least 20 / 30 dB below the clip's peak (by construction).

## What it means
1. Background content pulls the predictions, but only a little: replacing the background with one from another
   class lowers recall by about 3-6 points; splicing itself does not measurably hurt (same-class control).
2. None of the training variants removes this pull: source-balanced sampling and the contrastive loss leave the swap
   effect unchanged (all CIs overlap).
3. Consistent with the background probe (vocalisation removed: about 2x chance at 20 dB, 1.5x at 30 dB).
4. Consequence for the plan: background-swap *training* (Phase 2) is not justified by the pre-registered rule and is
   skipped; the speech-model comparison (Phase 3) goes ahead mainly as a model comparison, with this swap test as a
   secondary measure of background reliance.
