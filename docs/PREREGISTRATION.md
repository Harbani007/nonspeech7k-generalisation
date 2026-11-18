# Pre-specified analysis: source-aware AST fine-tuning (written 18 Nov 2026, before any run of this experiment)

Committed before the runs start, so hypotheses, hyperparameters and the primary metric cannot be adjusted after
seeing results.

## Motivation (verified before this experiment)

Corrected leave-one-website-out (RESULTS.md section 8): with training size, class mix and number of distinct sources
matched, holding a website out costs about 0.06 macro-F1 for frozen-AST features on both testable websites
(Aigei 0.057 [0.043, 0.072], Freesound 0.056 [0.035, 0.075]). Source/website shift is real, so a source-invariance
objective is worth testing. It is also modest, so a null result is plausible.

## Design: 2 x 2 factorial, AST partial fine-tuning (`ast_finetune.py`, `run_ast_suite.sh`)

| | cross-entropy only | cross-entropy + 0.5 x cross-source SupCon |
|---|---|---|
| class-balanced, clip-uniform sampling | **ordinary** | **contrastive** |
| class-balanced, source-uniform sampling | **source-balanced** | **combined** |

- Model: MIT/ast-finetuned-audioset-10-10-0.4593, 4 s input; layers 0-10 frozen (cached), pretrained layer 11 +
  final LayerNorm + new 7-class head trained (7.09 M trainable parameters). CPU only, so full fine-tuning is not
  possible.
- Data: source-disjoint protocol (`dl_data.py`); training pool <= 200 clips per class (1,355 clips), all 727
  validation clips, all 1,160 test clips (300 test source groups). Duplicates are merged into source groups.
- Budget, identical in every condition: 4 epochs x 768 samples, batch 32, AdamW (encoder lr 3e-5, head and projection
  lr 1e-3, wd 0.01), OneCycle. The same initialisation and RNG stream are used at a given seed; the projection head is
  always built.
- Contrastive term fixed a priori, not tuned: lambda = 0.5, tau = 0.1, projection 768-768-128. Positives are same
  class and different source group; same-class same-source pairs are excluded from the denominator.
- Checkpoint = the epoch with the best validation macro-F1. The test set is scored once per run.
- Seeds 0, 1, 2. A frozen-AST linear-head reference is run at each seed.

## Hypotheses and decision rule

Primary metric: macro-F1 on the source-disjoint test set. Uncertainty: paired bootstrap over test source groups
(2,000 resamples) with seeds averaged inside each resample (`dl_report.py`, `ast_2x2_effects.csv`).

- **H1 (contrastive):** the contrastive main effect is > 0. Supported only if its 95% CI excludes 0.
- **H2 (sampling):** the source-balanced main effect is > 0. Same rule.
- **H3 (complementarity):** interaction > 0. Exploratory: the experiment is underpowered for interactions.
- **Mechanism check (secondary):** if cross-source contrast works as intended, the within-class nearest neighbour
  of a test embedding should come from the same source less often (`G9`). A drop without an F1 gain means the
  representation changed without helping classification.
- Secondary, descriptive only: per-website test macro-F1 and per-class recall.

If no CI excludes 0, the result is reported as **no detectable effect at this budget**, not as evidence that the
method fails in general. No hyperparameter is changed after test results are seen; any later variant is labelled
exploratory.
