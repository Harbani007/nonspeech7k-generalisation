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
