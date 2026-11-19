# Project state (handoff) — 19 Nov 2026

Read this first in a new session, then **docs/ai/NEXT_PLAN.md** (what to do next, in order). Then `professor_demo/RESEARCH_STUDY_GUIDE.md` (START HERE, Ch. 13, professor section),
`RESULTS.md` sections 7-8, `docs/PREREGISTRATION.md`, `docs/NOVELTY_AUDIT.md`.

## Research question
Does source-aware training (source-balanced sampling, cross-source supervised contrastive loss) improve a fine-tuned AST
on Nonspeech7k recordings never seen in training? Motivation: website shift exists after corrected source grouping.

## Verified results (evidence in `docs/results/`)
| Finding | Evidence |
|---|---|
| Published File_IDs collide (Aigei counters per filename family); 1,586 reconstructed sources, 1,503 after merging 137 confirmed duplicate pairs | `docs/results/audit/` |
| 52/725 official test clips (7.2%) share a source/duplicate with train; removing them changes baselines by < 0.01 (CIs include 0) | `RESULTS.md` s7 |
| Leave-one-website-out (frozen AST + LR, recordings-matched): gap Aigei 0.057 [0.043, 0.072], Freesound 0.056 [0.035, 0.075] | `docs/results/website_holdout/` |
| **GPU full fine-tuning 2x2 (Colab, FREEZE_K=0, 3 seeds): null.** contrastive -0.006 [-0.013, 0.002]; balanced +0.019 [-0.002, 0.038] | `docs/results/colab_k0_pc200/` |
| Test embeddings cluster by recording: same-recording nearest neighbour 70-72% vs 7.6% chance, for all models, including contrastive | `docs/results/colab_k0_pc200/G9_*.csv` |
| **CPU 2x2 (layer 11 only, 3 seeds): contrastive null -0.002 [-0.005, 0.002]; source-balanced +0.035 [0.012, 0.055] (significant)**; GPU shows the same direction for balanced (+0.019) but its CI touches 0 | `docs/results/cpu_k11_final/ast_2x2_effects.csv` |
| Test errors cluster by recording: error ICC 0.27-0.36 (permutation p < 0.002), design effect about 2, effective test size about half, worst 10% of recordings hold about 80% of errors, clip-bootstrap CIs about 2x too narrow | `docs/results/error_clustering/` |
| Background probe (20 dB cut, 645 clips): fine-tuned AST full 0.79, vocalisation-only 0.73, background-only 0.27-0.30 (chance <= 0.16), random-cut control 0.66; probe trained on background 0.39 | `docs/results/background_probe/SUMMARY.md` |
| Contrastive loss verified active (pairs, losses, gradients logged; real-data smoke test) | `docs/results/audit/smoke_contrastive_real.txt` |

## In progress / ready to run
- Phase 1: clean CNN rerun on Colab (`run_gpu_cnn.sh`; commands in NEXT_PLAN.md).
- Phase 2: background-swap experiment, 30 dB gate first (`colab/BGSWAP_RUN.md`, `docs/PREREGISTRATION_BGSWAP.md`).
- Phase 3: advanced models (WavLM, Whisper encoder, BEATs) plus the cross-model background probe; not coded yet.
- Lab report and slides: `report/make_lab_report.py`, `report/make_presentation.py` (regenerate after new results).
- Background probe at 30 dB (does background-only performance plateau above chance, i.e. context, or fall to chance, i.e. vocalisation residue?): `background_probe.py --fg-db 30`.
- EXPLORATORY: recording-adversarial AST (`--adv`, `run_gpu_adv.sh`), also in `colab/MORNING_RUN.md`.

## How to run on a new machine
```bash
# data: Zenodo record 6967442 -> train.zip, test.zip; unzip both into ../data/audio (any folder layout)
pip install -r requirements.txt torch transformers nbformat nbconvert jupyter
python -m unittest discover tests          # 23 tests
mkdir -p ../results/v3/dl && cp docs/results/splits/splits.csv ../results/v3/dl/
# GPU: see colab/README.md (fbank -> run_gpu_suite.sh -> dl_report.py)
```
Demo notebooks (`professor_demo/LIVE_*.ipynb`) need trained checkpoints under `../results/v3/runs/` (not in git;
checkpoints are too large). The `*_executed.ipynb` copies show all outputs without rerunning.

## Rules (from the user)
- Commits under the user's identity, no co-author or Claude lines. Never push; the user pushes. Branch + PR, never commit to main directly.
- Never tune on the test set; lambda=0.5, tau=0.1 fixed a priori. Report null results. Never claim the loss is novel.
- Source reuse is documented in the original paper (chunking); claim quantification, not discovery.

## Sensible next steps
1. Finish/report the CPU replication (if the old laptop is available).
2. Rerun `website_holdout.py ... --save-preds` to store per-clip predictions behind the 0.057/0.056 gaps.
3. Exploratory (label as such): larger lambda chosen on validation; all training clips (PER_CLASS_TRAIN=100000);
   fine-tuned-AST website hold-out; cross-corpus test (VocalSound).
