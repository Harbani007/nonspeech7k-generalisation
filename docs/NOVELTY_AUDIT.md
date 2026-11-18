# Novelty audit (18 Nov 2026)

Checked before any claim is made. Searches were quick web searches; full papers were not read, so overlaps may be
larger than listed here.

## The proposed objective is NOT new as a loss

"Supervised contrastive learning whose positives are same-class examples from a *different* domain" already exists:

| Work | Domain definition | Backbone / data |
|---|---|---|
| Kim et al., *Stethoscope-guided Supervised Contrastive Learning for Cross-domain Adaptation on Respiratory Sound Classification*, ICASSP 2024 ([arXiv 2312.09603](https://arxiv.org/abs/2312.09603)) | stethoscope (device) | AST, ICBHI |
| Moummad et al., [arXiv 2409.08589](https://arxiv.org/abs/2409.08589) (domain-invariant bird-sound representations; exact title not checked) | focal vs soundscape recordings | SupCon / ProtoCLR, bioacoustics |
| [arXiv 2505.23132](https://arxiv.org/abs/2505.23132) (not read in full) | patient / domain | AST, respiratory sounds |
| SelfReg (Kim et al., ICCV 2021) and other contrastive domain-generalisation methods | image domains | vision |

So `cross_source_supcon` (in `ast_finetune.py`) must be described as an **application of known cross-domain
supervised contrastive learning**, not a new loss.

## What may still be a contribution (to be checked further before any write-up)

1. **"Domain" = reconstructed recording source, not metadata.** Nonspeech7k has no device, uploader or recording
   labels. Sources are reconstructed from filename IDs (`sources.py`) and same-audio duplicates found by
   waveform cross-correlation are merged, so a re-upload of the same audio never counts as a cross-source positive.
2. **Same-class, same-source pairs are excluded from the denominator** (neither positive nor negative). Small
   design detail; the cited works may already do something equivalent.
3. **Evaluation**: the effect is measured on a source-disjoint test set with source-grouped paired bootstrap CIs.
   The question asked is falsifiable: does cross-source contrast improve generalisation to unseen sources beyond
   source-balanced sampling at the same budget?

## Dataset-integrity findings: what is and is not new

- The original Nonspeech7k paper (Rashid, Li & Du, IET Signal Processing 2023,
  [DOAJ record](https://doaj.org/article/906206f71b99408cb214534ca09b6bd8); data
  [Zenodo 6967442](https://zenodo.org/records/6967442)) describes chunking longer recordings into clips. **Source
  reuse is therefore documented and must not be presented as a discovery.**
- What this project adds is quantification: how concentrated each class is (e.g. breath: 827 of 1,850 clips from
  one video, effective number of sources 4.6), cross-website re-uploads (137 confirmed same-audio pairs, 53 across
  websites), how much of the official test set shares a source with training (52/725 = 7.2%), and whether that
  contamination changes reported scores (for the baselines tested so far: no detectable change).
- Not yet checked: whether later papers using Nonspeech7k already report source-disjoint splits.
- **Source-aware splitting is a known general principle in audio**: UrbanSound8K ships predefined folds for this
  reason; recent work benchmarks leakage and generalisation in audio classification
  ([Western Sydney record](https://researchers.westernsydney.edu.au/en/publications/benchmarking-data-leakage-and-generalization-in-audio-classificat/))
  and the problems of deriving benchmarks from long-form recordings ([arXiv 2607.03201](https://arxiv.org/abs/2607.03201)).
  A quick search found no Nonspeech7k-specific source-disjoint audit; voc2vec ([arXiv 2502.16298](https://arxiv.org/abs/2502.16298))
  uses Nonspeech7k as a benchmark, but how it was split was not checked. Absence in a quick search is not proof of absence.
