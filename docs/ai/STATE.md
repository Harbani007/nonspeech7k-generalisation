# Project state (handoff) — 19 Nov 2026

Read this first in a new session. Then `professor_demo/RESEARCH_STUDY_GUIDE.md` (START HERE, Ch. 13, professor section),
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
| Contrastive loss verified active (pairs, losses, gradients logged; real-data smoke test) | `docs/results/audit/smoke_contrastive_real.txt` |

## In progress when this was written
- CPU 2x2 (layer 11 only, cached tokens) on the old laptop: 9 of 15 runs done (seeds 0-1; seed 2 running).
  Interim per-run lines: `docs/results/cpu_k11_interim/ast_suite.log`. This run needs the token cache
  (`results/v3/ast_cache`, 7 GB, not in git) and can only be finished on the old laptop, or redone with
  `ast_finetune.py cache` (about 1-2 h CPU) and `run_ast_suite.sh`. It is a replication; the GPU run already answers the question.

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
