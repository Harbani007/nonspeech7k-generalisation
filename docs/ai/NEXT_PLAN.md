# Forward plan (written 19 Nov 2026). A new session starts here, after docs/ai/STATE.md

Goal for the course: a complete, fair **deep-learning comparison** on leakage-safe evaluation, with one strong finding
about what the models rely on. All models: same source-disjoint split (`docs/results/splits/splits.csv`), same test
set (1,160 clips / 300 recordings), 3 seeds, recording-level bootstrap CIs, model selection on validation only.

## Where we are (verified, see STATE.md)
| Model | Macro-F1 (unseen recordings) | Status |
|---|---|---|
| Log-mel CNN from scratch | 0.767 (1 run, CPU, possibly overlapped) | **rerun: Phase 1** |
| AST frozen + linear head | 0.777 (3 seeds) | done |
| AST fine-tuned (ordinary / balanced / contrastive / combined), laptop | 0.788 / 0.821 / 0.784 / 0.822 | done |
| AST fully fine-tuned (GPU), same four | 0.816 / 0.837 / 0.812 / 0.830 | done |
| Background probe (vocalisation removed) | fine-tuned AST still 0.27-0.30 (chance <= 0.16) | done; 30 dB gate pending |

## Phase 1: clean CNN rerun (Colab GPU, about 30-45 min)
```python
# after the usual setup cell (Drive mounted, WORK defined, repo cloned to /content/ns7k-src, audio unzipped)
%cd /content/ns7k-src
!python dl_data.py --audio-root /content/data/audio --splits docs/results/splits/splits.csv --out "{WORK}/dl"   # log-mels, ~5 min
!bash run_gpu_cnn.sh python "{WORK}/dl" "{WORK}/runs_cnn" 0 1 2
!python dl_report.py --data "{WORK}/dl" --runs "{WORK}/runs_cnn" --audit /content/splits --out "{WORK}/report_cnn" 2>&1 | grep -A12 "system"
```
Done when: 3 seeds x {standard, source-balanced} finished; report the table with CIs and the source-balanced effect for the CNN.
(Verified on the laptop: `dl_data.py --splits` reproduces the original `logmel.npy` exactly; `train_cnn.py --device cuda` added.)

## Phase 2: background-swap experiment (Colab, about 2 h): `colab/BGSWAP_RUN.md`
Gate first (30 dB probe). Pre-registered in `docs/PREREGISTRATION_BGSWAP.md`.

## Phase 3: advanced models to "one-up" AST (new code, about 3-4 h coding + 2-3 h Colab)
Research question, and the reason this is more than a bigger model:
**Do models pretrained on human SPEECH understand human NON-speech vocalisations better than a general-audio model,
and do they rely less on the recording background?**

| Model | Pretraining | Why | Availability |
|---|---|---|---|
| WavLM Base+ (`microsoft/wavlm-base-plus`) | 94k h of speech, self-supervised; strong on paralinguistics | speech prior: vocal tract, breath, laughter | Hugging Face transformers |
| Whisper-small encoder (`openai/whisper-small`) | 680k h of weakly labelled web audio, multilingual | huge, diverse, noisy web audio (similar to Nonspeech7k's sources) | transformers (pads input to 30 s) |
| BEATs (Microsoft, 2022) | AudioSet, iterative audio tokenizer; state of the art on AudioSet/ESC-50 | strongest general-audio SSL model | checkpoint from the microsoft/unilm GitHub (more setup) |
| Not used: LAION-CLAP | trained on many web audio sets | a Hugging Face dataset "CLAPv2/Nonspeech7k" exists, so test leakage is likely | - |

Implementation (one new script, `ssl_finetune.py`), reusing what exists:
1. Inputs: the 16 kHz waveform cache already built for Phase 2 (`ast_finetune.py waves`, `waves.f16`), same clip selection.
2. Model: pretrained encoder + mean-pooled hidden states + new 7-class head; fine-tune the top N layers
   (WavLM 12 layers, Whisper-small 12 layers); log trainable parameters, gradients and weight drift as for AST.
3. The same budget, optimiser, seeds, checkpointing/resume/locks and outputs (`test_predictions.csv`, `history.csv`,
   `test_embeddings.npy`, `best.pt`), so `dl_report.py`, `error_clustering.py` and the source-clustering diagnostic work unchanged.
4. Extend `background_probe.py` with a model-agnostic loader so every model gets the same "vocalisation removed /
   only / random" test. **The cross-architecture comparison of background reliance is the main new result.**
5. Smoke test on the laptop (tiny subset, CPU), then Colab: 3 seeds per model.

## Phase 4: write-up (about 2 h)
Update the report (`report/make_lab_report.py`) and slides (`report/make_presentation.py`): final model table
(CNN, AST variants, WavLM, Whisper/BEATs), the background-probe comparison across models, conclusions.

## Stop rule
Stop when: every model has 3 seeds on the same split with CIs; the background-probe result is settled (30 dB gate +
cross-model comparison); report and slides are regenerated. Do not start new methods after that.

## Honest expectations
- The CNN rerun will most likely land near 0.77 (the earlier single run).
- Whether a speech model beats AST is genuinely open: speech SSL models are strong on laughter/cough-type sounds, but AST
  was pretrained on AudioSet, which contains these classes. Either answer is reportable.
- The most likely "headline": models differ in how much they rely on the background, measured with the same probe.
