# Pre-specified analysis: test-time background swap (written 19 Nov 2026, before any swap result exists)

Committed before the GPU run and before any full run of `swap_probe.py`, so the hypotheses, the primary model, the
metrics and the decision rule cannot be adjusted after seeing results. A CPU smoke test on a few clips (to check that
the code runs) is the only thing run before this commit.

## Question
The background probe (`docs/results/background_probe/SUMMARY.md`) shows that background audio alone carries class
information (fine-tuned AST 0.27-0.30 macro-F1 with the vocalisation removed, chance <= 0.16). That is a statement
about the DATA. It does not show that the trained model USES the background when the vocalisation is present.

**Does a fine-tuned AST change its decision when only the background behind an unchanged vocalisation is changed?**

No training. Existing checkpoints only. Test partition only (recordings never seen in training).

## Manipulation
For a target test clip x (16 kHz, native length up to 4 s) with vocalisation mask g (20 dB rule of
`background_probe.masks_for`: frames within 20 dB of the loudest frame, dilated 50 ms, 10 ms raised-cosine edges):

    swapped = x * g  +  s * d * (1 - g_d)

- `x * g` is the target's vocalisation, identical in every condition (checked numerically, see Checks).
- `d * (1 - g_d)` is the background of a donor test clip d, with the donor's own vocalisation removed by the donor
  mask g_d at **20 dB** or at the stricter **30 dB** rule (two arms). If the donor is shorter than the target, the
  missing part is silence; donors at least as long as the target are preferred (see pairing).
- `s` = level matching: RMS of the target's own background (samples with 1 - g > 0.5) divided by the RMS of the
  donor background (samples with 1 - g_d > 0.5), clipped to [0.01, 100]. So the vocalisation-to-background level
  ratio of the target is kept. Sensitivity arm: s = 1 (donor background at its natural level), 20 dB donor mask only.

Conditions per target: `orig` (x), `fg` (x * g, no background), and for each arm {20 dB matched (PRIMARY),
30 dB matched, 20 dB unmatched}: one **same-class** donor and six **different-class** donors (one per other class).

## Eligibility and paired donor selection (fixed, seed 0)
- Targets: test clips whose 20 dB foreground covers 5-80% of the clip (as in the background probe; about 645 clips).
- Donor pool: test clips whose foreground covers 5-80% of the clip under BOTH the 20 and the 30 dB rule (so the
  same donors are used in both arms and the only difference between arms is the mask), whose 30 dB background RMS
  is above -70 dBFS (no digital-silence backgrounds, which level matching would amplify).
- Source independence: a donor never comes from the target's recording group (`sources.py` groups, duplicates merged).
- Pairing: the same-class donor is drawn at random (seed 0) from same-class donors of other recording groups that are
  at least as long as the target (if none, the longest). Each different-class donor (one per other class) is the
  nearest neighbour of THAT same-class donor in standardised donor-background features: background fraction
  (30 dB rule), log background RMS, background spectral centroid, duration; same group and length restrictions.
  Same-class and different-class donors are therefore matched on duration, energy and acoustic character.
- Balance check reported: standardised mean differences (SMD) of the four features between same-class and
  different-class donors. |SMD| > 0.25 on any feature is reported as a limitation (not a reason to re-draw).

## Models
- **PRIMARY: AST-FT ordinary, full fine-tuning on GPU (FREEZE_K = 0), 3 seeds** (`ast_ft_source_disjoint_standard_s{0,1,2}`).
- Secondary (descriptive, same analysis): the other three GPU conditions (source-balanced, contrastive, combined;
  3 seeds each). Exploratory: the CPU layer-11 checkpoints (same names, FREEZE_K = 11), which may be run on the laptop
  before the GPU run; they do not enter the decision.

## Metrics
Per run, per condition: prediction for every target. Seeds are averaged inside every bootstrap resample.
1. **Macro-recall of the target's own class** (balanced accuracy; for the six different-class donors the per-target
   correctness is the mean over the six).
2. **Swap effect (PRIMARY): delta = macro-recall(same-class donor) - macro-recall(different-class donors)**, 20 dB
   matched arm. Positive = the model is pulled away from the vocalisation's class when the background belongs to
   another class. Same-class donors are the control: both arms are spliced the same way, so splicing artefacts cancel.
3. Manipulation check: macro-recall(orig) - macro-recall(same-class donor), and macro-recall(fg) for reference.
4. **Donor attraction (secondary)**: for target i and wrong class c,
   A_ic = 1[pred = c | donor of class c] - mean over the other wrong classes c' of 1[pred = c | donor of class c'].
   The null expectation is the rate at which the SAME target is predicted as c when the donor is of some other
   class, so it controls the model's class biases and the target's own prediction tendencies. Reported as the mean
   over targets of mean_c A_ic (0 under no attraction), plus per donor class.
5. Flip rates: share of different-class swaps whose prediction differs from the same-class swap, and share that
   flips to the donor's class.
6. Residue audit (descriptive): (a) primary model on donor backgrounds alone (`d * (1 - g_d)`), macro-F1 for the
   donor's class at 20 and 30 dB: above chance means the donor background still carries class information, which
   can be context OR residual vocalisation and is not separable by this number; (b) loudest remaining frame of each
   donor background relative to the donor's peak (checks the mask implementation); (c) listening/spectrogram set.

Uncertainty: 95% CI from 2,000 bootstrap resamples of target recording groups (the unit of independence).

## Decision rule (primary model, matched arms)
- **GO (substantial background reliance):** delta >= 0.05 AND its 95% CI excludes 0, in BOTH the 20 dB and the
  30 dB arm. Then the speech-model comparison and swap-based training are justified.
- **NEGLIGIBLE:** the upper end of the 95% CI of delta is below 0.02 in the 20 dB arm (an effect of 0.02 or more is
  ruled out at this precision). Then background reliance is not a substantial failure mode of this model when the
  vocalisation is present.
- **DETECTED BUT SMALL:** CI excludes 0 but the point estimate is below 0.05 (or GO holds at 20 dB only).
- **INCONCLUSIVE:** anything else (CI includes 0 and its upper end is >= 0.02): not evidence of absence; the
  precision is insufficient.
The 0.05 and 0.02 thresholds are practical-effect choices made for this study, not established standards.

## Honest prior and known limits (written before results)
- The manipulation is imperfect: the target's own background stays under the vocalisation (soft-mask edges and
  overlapping frames), the donor's vocalisation leaves a silent hole, and residual donor vocalisation below the
  mask threshold cannot be ruled out by the 30 dB arm alone. Same-class donors control the first two but not the
  third. The listening set and the residue audit are there to make this visible, not to prove it absent.
- Any effect is the model's response to this specific manipulation on 4 s clips of Nonspeech7k test recordings.
- A GO does not show that background reliance hurts accuracy on natural recordings; it shows sensitivity.

## Amendment 1 (19 Nov 2026, after the CPU smoke test on 14 targets, before any full or GPU run)
The smoke test's mask check found leftover vocalisation in donor backgrounds, all of it at the clip edges:
`masks_for` smooths the mask with a convolution that zero-pads the clip ends (and leaves up to 159 samples after the
last full frame unmarked), so a vocalisation touching the clip start/end kept about half its amplitude for ~10 ms.
Before the fix, 6.5% (20 dB) and 33% (30 dB) of donor backgrounds had a frame above the cut threshold; after it, 0% and 0%.
Change: `swap_probe.py` uses `masks_for(..., edge_hold=True)` (the edge value is held instead of zero-padded). The old
behaviour stays the default, so the published background-probe numbers and the cached training masks are unchanged
(they carry this small edge leak; noted as a limitation of the earlier probe).
Consequences, fixed now: 643 eligible targets (was about 645); donor pool 312 clips (breath 7 clips from 6
recordings, screaming 14 from 8: same-class donors for these classes are reused often; reported, not changed).
Nothing else changes: hypotheses, models, metrics, thresholds and the decision rule are as above.
