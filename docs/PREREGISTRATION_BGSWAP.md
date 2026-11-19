# Pre-specified analysis: background-swap training (written 19 Nov 2026, before any run)

## Motivation (verified)
Background probe (20 dB cut, 645 unseen-recording test clips, full fine-tuning, 3 seeds): the fine-tuned AST scores
0.27-0.30 macro-F1 with the vocalisation removed (chance <= 0.16), against 0.73-0.74 with only the vocalisation and
0.78-0.80 on the full clip. A 10 dB cut gives 0.44-0.49, so part of the "background" signal may be faint vocalisation.

## Gate (run first)
Background probe at a 30 dB cut. If background-only macro-F1 falls to the chance band (95% CI overlaps the
permutation 95th percentile), the background shortcut is mostly vocalisation residue: the swap experiment is then
reported as exploratory only.

## Design
- Control: ordinary fine-tuning on waveform inputs (`--inputs wave --bgswap 0`), so both arms share exactly the same
  input pipeline. Treatment: `--bgswap 0.5`. With probability 0.5 a training clip becomes x*g + donor*(1 - g_donor):
  vocalisation and label kept, background taken from a random training clip of a different recording group.
- FREEZE_K = 0 (all 12 layers), 200 training clips per class, 4 epochs x 768 samples, batch 32, lr 3e-5, seeds 0-2.
  The same init and sample stream per seed; swap draws use their own RNG. p = 0.5 and the 20 dB / 50 ms mask are fixed a priori.

## Hypotheses (decision rule: 95% CI from the recording-level paired bootstrap excludes 0)
- **B1 (shortcut removed):** background-only macro-F1 is lower for bgswap than for the control.
- **B2 (no cost on unseen recordings):** source-disjoint test macro-F1 for bgswap minus control is >= -0.01 (point estimate),
  and B2+ (improvement) if its CI excludes 0 in the positive direction.
- **B3 (exploratory, only if compute allows):** a smaller website gap in leave-one-website-out fine-tuning.

## Honest prior (written before results)
B1 is likely: the augmentation directly makes background uninformative. B2+ is uncertain: vocalisation-only performance
(0.73) is below full-clip performance (0.79), so some of the information the swap removes is useful and may also hold on
unseen recordings. Many clips are 95-100% vocalisation, where the swap can only add a donor background underneath.
Possible outcomes, each reported as is: (a) shortcut removed and accuracy kept or improved; (b) shortcut removed at a small
accuracy cost (a robustness/accuracy trade-off); (c) no change.
