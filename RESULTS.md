# Results: first full run on Nonspeech7k

All 7,014 clips (6,289 train, 725 test). Class-balanced logistic regression on frozen features, 5 seeds, training
budgets of 300 / 500 / 700 clips. Numbers are macro-F1, mean over seeds; seed-to-seed standard deviation is about
0.01-0.02 on the official test and up to 0.05 on the recording-disjoint (group) splits, so differences below about
0.03 should be read as noise. Raw tables: `docs/summary.txt`, `docs/probe.txt`, `docs/audit.txt`,
`docs/burst_tuning.txt`.

## 1. The official split is almost clean

Only 6 of the 407 test recordings also appear in train; 4.4% of test clips come from them. The "leaky benchmark"
story we started with is mostly **not** true for the official split.

Random-clip vs recording-disjoint splits differ by only 0.00-0.05 macro-F1 for every feature set
(`docs/figures/2_protocol_gap.png`). Splitting by recording costs little here.

## 2. Website is tangled up with class

The official test set is 100% Freesound. The train set is 47% Freesound, 28% Aigei, 26% YouTube, spread very
unevenly: 79% of train coughs are Aigei; 63% of train breaths come from YouTube, and all 1,615 YouTube clips come
from just 15 recordings.

Evidence that a classifier uses this as a shortcut:

| Breath F1, recording-disjoint split, 700 clips | random sampling | max 3 clips per recording |
|---|---|---|
| spectral (MFCC etc.) | 0.96 | 0.36 |
| structure | 0.74 | 0.90 |
| AST | 0.98 | 0.99 |

YouTube recordings make up 46% of the clips in the recording-disjoint test folds. Random sampling puts about 11%
YouTube clips in training, while "max 3 clips per recording" puts about 2%. The spectral features' breath score collapses
when YouTube clips are rare in training, which fits the reading that they recognise YouTube audio rather than
breathing. With only 15 YouTube recordings, this is a strong hint, not proof.

A within-class probe on recordings never seen in training predicts the website above chance with every feature set
(balanced-accuracy lift over chance 0.5: spectral 0.16, structure 0.17, AST 0.22).

Training only on Freesound clips (same budget) does **not** improve the all-Freesound official test; for AST it
is slightly worse (-0.02 to -0.04). Mixed-website data helps more than it hurts.

## 3. Structure features: complementary to spectral, nothing added to AST

| Official test, 700 clips | macro-F1 |
|---|---|
| structure | 0.46 |
| spectral | 0.47 |
| **spectral + structure** | **0.61** |
| AST | 0.82 |
| AST + structure | 0.81 |
| AST + spectral + structure | 0.81 |

Structure and spectral features are good at different classes (breath: spectral 0.73 vs structure 0.41; laugh:
structure 0.71 vs spectral 0.36), so together they gain 0.14. On top of AST they add nothing. The hypothesis
that structure features generalise better to unseen recordings is **not supported**: their random-vs-group gap
(about 0.05) is no smaller than the spectral one (about 0.01).

## 4. Recording identity

Within a class, a linear probe identifies the source recording of a clip with accuracy 0.97 (AST), 0.93
(spectral), 0.72 (structure); chance is 0.13. Structure features carry the least recording identity, yet that
does not translate into better generalisation. AST carries the most and still generalises best. Encoding the
recording is not the same as relying on it.

## 5. Burst thresholds

Chosen on the train set only (grouped CV, burst features alone): 3 dB prominence, 0.2 s minimum gap, replacing
the initial guess of 6 dB / 0.08 s (0.371 vs 0.334, SE 0.031). The test set was never used.

## What this means for the project

- The method claim (structure features generalise better) is not supported. Say so.
- The defensible findings are analytical: the official split is clean; website is confounded with class; a
  simple feature set can score breath almost perfectly by recognising the website; the recording-disjoint
  folds are dominated by 15 YouTube recordings, which inflates breath scores under that protocol.
- Next experiments that would firm this up: leave-one-website-out evaluation; balancing the website mix within
  each class; a cross-corpus test on VocalSound.

## Caveats

- AST was pretrained on AudioSet, which is drawn from YouTube; it may have heard some of the same YouTube
  recordings. Its breath scores should be read with that in mind.
- Logistic regression on frozen features only; no fine-tuning.
- 5 seeds, one fold per seed.

## 6. Leave-one-website-out (added after the first run)

> **Superseded (18 Nov, see section 7).** The grouping used here still merged unrelated recordings: Aigei File_IDs
> are counters that restart for every filename family, so '79-cough-aigei', '79-sneeze-aigei' and Freesound sound 79
> were one group (97 groups mixed classes). The numbers below are kept as a record and must not be cited.

`website_holdout.py`. For each website W, W's sources are split into 5 source-disjoint folds. Each fold is
classified by a model trained without W ("holdout") and by a control trained with W's *other* sources
("control"). Neither ever sees the test sources. Classes with fewer than 5 sources in W are not scored.
5 seeds; 95% intervals from one paired bootstrap over test sources, all seeds averaged inside each resample.

Two matching modes:
- **clips:** same clips per class in both conditions.
- **recordings (strict):** same number of distinct sources per class, one clip each, so training size, class
  mix and source diversity are identical and only the website mix differs.

### Grouping had to be fixed first (`find_duplicates.py`, `common.load_meta`)

- **File_ID is not a recording ID for Aigei.** Filenames such as `142=KKKId0LDwVc-192_2_0.wav` name the video
  or sound each clip was cut from. 958 Aigei crying "recordings" are 70 sources; 376 clips come from one YouTube
  video. `load_meta` now groups by that source (2,304 -> 1,398 groups; `groups="file_id"` restores the old
  behaviour).
- **Re-uploaded audio.** Nearest neighbours in AST space from a different source (cosine > 0.98), confirmed by
  waveform cross-correlation > 0.9: 101 same-audio pairs, 50 across websites, **22 across the official
  train/test split** (leakage the File_ID audit cannot see), 2 with different labels. No bit-identical copies.
  These pairs are merged into one group (1,398 -> 1,335). Uploader identity is not in the metadata, so
  uploader-level leakage cannot be checked.

### Results with corrected groups, strict (recordings-matched)

| Held-out website (scored classes, test sources) | Features | control | holdout | gap [95% CI] |
|---|---|---|---|---|
| Aigei (5 classes, 261) | AST | 0.84 | 0.77 | 0.067 [0.039, 0.089] |
| | spectral | 0.46 | 0.45 | 0.008 [-0.041, 0.079] |
| | structure | 0.56 | 0.51 | 0.044 [0.009, 0.068] |
| Freesound (6 classes, 904) | AST | 0.77 | 0.69 | 0.075 [0.046, 0.106] |
| | spectral | 0.36 | 0.32 | 0.044 [0.020, 0.065] |
| | structure | 0.44 | 0.35 | 0.088 [0.067, 0.106] |
| YouTube (laugh only, 5 sources; underpowered) | AST | 0.92 | 0.92 | 0.000 [-0.021, 0.010] |

Clip-matched gaps with corrected groups are larger (AST: Aigei 0.128, Freesound 0.097), so part of the
clip-matched gap is source diversity, but not all of it.

- **A website gap survives every control**: 0.04-0.09 macro-F1 for AST on both testable websites, with source
  diversity, class mix and training size matched and duplicates merged.
- **AST is not more robust; on Aigei it loses more than the handcrafted features** (structure minus AST gap
  -0.023 [-0.045, -0.005]). Pretraining does not protect against website shift here.
- The loss is class-specific: crying when Aigei is held out (AST recall -0.25, 65 sources); laugh, screaming
  and yawn when Freesound is held out (AST -0.11 to -0.15).
- With File_ID grouping the Freesound strict gap looked near zero (AST 0.028); correct grouping changes the
  conclusion, which is itself a reason to report grouping explicitly.
- Sections 1-5 used File_ID groups. Their recording-disjoint numbers should be re-run with corrected groups.
- Exploratory note: earlier checks (section 3 and a residual-correction test) looked at the official test set;
  later method choices should use source-disjoint validation only.

## 7. Corrected source identity and duplicate audit (v3, 18 Nov)

Code: `sources.py` (tested in `tests/test_sources.py`), `find_duplicates.py`, `source_audit.py`. Outputs:
`results/v3/` (`source_manifest.csv`, `duplicates.csv`, `groupings_and_leakage.csv`, `source_concentration_*.csv`,
`figures/A1-A4`).

**Source keys.** Freesound File_ID = Freesound sound ID; YouTube File_ID = video ID (recovered in full from the
filename when truncated); Aigei '<n>=<key>-<k>' names give the source (11-character YouTube IDs share the `yt:`
namespace with the YouTube website; numeric keys that may be Freesound IDs are kept separate unless audio confirms
it); other Aigei File_IDs are namespaced by filename family. Every clip carries the rule that produced its key and a
known/inferred flag.

| Grouping | groups | groups that mix classes | official test clips sharing a group with train |
|---|---|---|---|
| File_ID as published | 2,304 | 97 | 32 (4.4%) |
| File_ID, collision-free | 2,475 | 4 | 29 (4.0%) |
| reconstructed source | 1,586 | 4 | 29 (4.0%) |
| source + confirmed duplicates | 1,503 | 5 | **52 (7.2%)** |

- Earlier counts of "2,304 -> 1,398 groups" and "6 shared test recordings" were both distorted by ID collisions.
  The historical Aigei crying figures reproduce: 958 clips, 70 sources, 376 clips from `yt:KKKId0LDwVc`.
- One YouTube video (`5StJa6Ia6bI`) is both a YouTube-website source and the source of 46 Aigei clips: a
  cross-website source link visible only through filename IDs.
- Concentration: breath has 1,850 clips but an effective number of sources (inverse Simpson) of 4.6; 827 breath
  clips come from one video. Crying: 1,055 published File_IDs, 166 sources, effective number 20.

**Duplicates.** Top-5 AST neighbours from other sources with cosine >= 0.95, confirmed by peak normalised waveform
cross-correlation over all lags (all clips are 32 kHz mono, checked) >= 0.9:

- 277 candidates -> **137 confirmed**, 31 uncertain (0.6 <= xcorr < 0.9, kept apart, never merged), 109 rejected.
- Confirmed: 53 cross-website, **23 cross the official train/test split**, 2 with conflicting labels
  (`228-cough-aigei` matches two Aigei sneeze clips), 0 bit-identical. 22 are time-shifted by > 100 ms and 28
  differ in length, so exact-hash or zero-lag checks would miss them.
- Control: 500 random same-class different-source pairs never exceed xcorr 0.41 (99th percentile 0.22), so the
  0.9 threshold produced no false positives in this control.
- Sensitivity (`duplicates_sensitivity.csv`): confirmed pairs range from 79 (cosine 0.99) to 160 (cosine 0.95,
  xcorr 0.7); cross-split pairs stay between 21 and 24 under every setting. The old top-1 / 0.98 rule finds 102.
- Limitation: recall is bounded by AST screening; duplicates whose embeddings differ more than cosine 0.95 (heavy
  re-encoding, partial overlaps) are not searched.

## 8. Leave-one-website-out with corrected sources (v3, replaces section 6)

Same design as section 6 (`website_holdout.py`, 5 source-disjoint folds per website, 5 seeds, paired bootstrap
over test sources with seeds averaged inside each resample), now grouped by `sources.py` keys with confirmed
duplicates merged (1,503 groups). Outputs: `results/v3/website_holdout_{recordings,clips}*.csv/.txt`.

Strict (recordings-matched: same training size, class mix and number of distinct sources per class; only the
website mix differs). Macro-F1 over the classes the held-out website can score:

| Held-out website (scored classes, test sources) | Features | control | holdout | gap [95% CI] |
|---|---|---|---|---|
| Aigei (5 classes, 427) | AST | 0.849 | 0.792 | **0.057 [0.043, 0.072]** |
| | structure | 0.546 | 0.504 | 0.042 [0.012, 0.064] |
| | spectral | 0.473 | 0.446 | 0.026 [-0.010, 0.080] |
| Freesound (6 classes, 905) | AST | 0.764 | 0.708 | **0.056 [0.035, 0.075]** |
| | structure | 0.420 | 0.349 | 0.071 [0.053, 0.087] |
| | spectral | 0.358 | 0.310 | 0.048 [0.018, 0.074] |
| YouTube (laugh only, 5 sources; underpowered) | AST | 0.900 | 0.909 | -0.008 [-0.018, 0.000] |

- **The website gap survives correct grouping:** about 0.06 macro-F1 for frozen-AST features on both testable
  websites, with training size, class mix and source diversity matched and duplicates merged.
- **Retracted from section 6:** "AST loses more than the handcrafted features on Aigei". With corrected groups the
  gap differences between feature sets all include 0 (Aigei structure - AST -0.015 [-0.048, 0.009]; Freesound
  structure - AST +0.016 [-0.007, 0.040]). Pretrained features are neither more nor less robust to website shift.
- Clip-matched gaps are larger (AST: Aigei 0.112, Freesound 0.098), so part of the clip-matched gap is lost source
  diversity rather than website.
- The loss is class-specific. Aigei held out: crying recall drops 0.18 (AST), other classes <= 0.03. Freesound held
  out: yawn 0.11, laugh 0.09, crying 0.08, screaming 0.08.
- Breath can never be scored (one website), and YouTube only for laugh, so this test covers 5-6 of 7 classes.

## 9. Robustness check of the duplicate scores (19 Nov)

`find_duplicates.peak_xcorr` divides the peak cross-correlation of whole-clip-standardised signals by the shorter
clip's length, which is not bounded by 1 when clips differ in length or loudness profile (33 of 137 confirmed pairs
scored > 1). `recheck_duplicates.py` recomputes every candidate with a bounded score: cosine similarity of the
overlapping segments at each lag (overlap >= half the shorter clip and >= 10% of each clip's energy).

- 133 of the 137 confirmed pairs also score >= 0.9 (136 >= 0.8; minimum 0.77). Random-pair control: max 0.31.
- Applying the bounded score at 0.9 to all 277 candidates would confirm 151 pairs (54 cross-website, 24 across the
  official split, 1 label conflict); agreement with the original status: 255/277.
- The splits and all experiments keep the original merge (137 pairs). The difference (about 4 pairs fewer, 18 more)
  is negligible relative to 1,503 groups. Outputs: `docs/results/audit/duplicates_recheck*.csv`.
