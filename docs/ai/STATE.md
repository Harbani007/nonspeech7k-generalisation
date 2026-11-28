# Project state (handoff) — 19 Nov 2026

Read this first in a new session, then **docs/ai/NEXT_PLAN.md** (what to do next, in order). Then `professor_demo/RESEARCH_STUDY_GUIDE.md` (START HERE, Ch. 13, professor section),
`RESULTS.md` sections 7-8, `docs/PREREGISTRATION.md`, `docs/NOVELTY_AUDIT.md`.

## Research question (changed 19 Nov 2026)
**Current:** what do the trained models rely on: the vocalisation, or the recording background around it?
Next experiment (highest priority, approved): test-time background swap on the existing AST checkpoints,
`swap_probe.py`, pre-registered in `docs/PREREGISTRATION_SWAP_PROBE.md` (commit f154c4f, before any result),
Colab steps in `colab/SWAP_PROBE_RUN.md`. Go/no-go decides whether Whisper/WavLM and swap-based training are worth running.
Do NOT start Whisper/WavLM or swap training before the primary swap result is reviewed with the user.

**Earlier (answered, mostly null):** does source-aware training (source-balanced sampling, cross-source supervised contrastive loss) improve a fine-tuned AST
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
| **Clean CNN rerun (GPU, 3 seeds): standard 0.762 [0.704, 0.812], source-balanced 0.792 [0.741, 0.834]; balanced - standard +0.030 [+0.005, +0.057] (significant)**. Source-balanced sampling now helps in two architectures (CNN +0.030, AST laptop +0.035; AST full FT +0.019 n.s.) | `docs/results/cnn_gpu/SUMMARY.md` |
| Contrastive loss verified active (pairs, losses, gradients logged; real-data smoke test) | `docs/results/audit/smoke_contrastive_real.txt` |

## In progress / ready to run
- **Phase 0 (now): background-swap probe on Colab** (`run_gpu_swap.sh`, `colab/SWAP_PROBE_RUN.md`). Built and
  smoke-tested on the laptop CPU checkpoints (all 12 load; 'orig' reproduces stored test predictions 1.0); GPU run
  not done; needs a push of this branch (the user pushes). Pre-registration amendment 1 (commit cdee4ce): edge-held
  masks. The smoke check found that `masks_for` leaked ~10 ms of vocalisation at clip edges at about half amplitude
  (zero-padded ramp + unmarked tail samples). The fix is opt-in (`edge_hold=True`), so the PUBLISHED background-probe
  numbers and the cached training masks still carry this small edge leak: a limitation to state when citing them.
  A full CPU (layer-11) swap run would take about 5 h on this laptop (7 targets x 3 models = 174 s); not run.
- **Edge-leak fix now switchable everywhere (28 Nov):** `background_probe.py --edge-hold` and `ast_finetune.py waves --edge-hold`
  (default off reproduces earlier results). Rebuilding the masks with the fix changed 2,508 of 3,242 clips (914 of 1,160
  test clips; 2,305 at the start, 1,147 at the end), only at clip edges. Corrected probe rerun: `colab/EDGE_FIX_RUN.md`
  (20 and 30 dB); until it runs, treat the published 0.27-0.30 background-only score as carrying this leak.
- Whisper note: transformers' WhisperEncoder refuses inputs other than 3000 mel frames (modeling_whisper.py),
  so 4 s inputs need either 30 s padding (faithful) or a validated positional-embedding cut. Not decided.
- Phase 1 DONE (clean CNN rerun, see results table).
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
