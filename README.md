# nonspeech7k-generalisation

Does accuracy on [Nonspeech7k](https://zenodo.org/records/6967442) (7 classes of human non-speech sound:
breath, cough, cry, laugh, scream, sneeze, yawn) survive when the test clips come from **recordings the model has
never seen**? And do features describing a sound's *temporal structure* (bursts, rhythm, modulation) hold up on
unseen recordings better than spectral features or pretrained embeddings?

Nonspeech7k was built by cutting longer recordings into 0.5-4 s clips, and every clip carries a `File_ID` naming its
source recording. Clips from one recording share a microphone, room and noise, so a random split can let a model
win by recognising the recording instead of the sound.

**Status.** The pipeline is complete and runs end to end on synthetic audio. It has **not** yet been run on the real
dataset, and the AST/CLAP embedding extraction has not been run at all. Any result you see from `--synthetic` is a
smoke test, and its figures are watermarked as such. No claim about Nonspeech7k is made here yet.

## What the code does

1. `audit.py` - do train and test share source recordings? How concentrated is each class on a few recordings?
2. `extract.py` - per-clip features, cached to `feats/`:
   - `struct` (38): burst count/rate, gaps between bursts, attack/decay, rhythm strength, modulation spectrum, envelope shape
   - `spec` (35): control set, MFCC mean/std and spectral statistics
   - `ast`, `clap`: frozen embeddings (optional, `--embeddings`)
3. `run_experiments.py` - class-balanced logistic regression under three protocols (`official`, `random` clip split,
   `group` split by `File_ID`), at training budgets of 300/500/700 clips, sampled randomly or with at most k clips
   per recording. Reports macro-F1, balanced accuracy and per-class F1 over several seeds.
4. `probe_source.py` - how well can each feature set identify *which recording* a clip came from, within a class?
5. `plots.py`, `class_profile.py` - figures. `run_all.py` runs everything.

The key number is the **protocol gap**: macro-F1 under the random split minus macro-F1 under the group split. A small
gap means the features depend less on recording identity.

## Quick demo (no dataset needed)

```bash
pip install -r requirements.txt
python run_all.py --synthetic --workdir out_demo
```

This generates 1,344 synthetic clips, runs the pipeline and writes `out_demo/figures/*.png` stamped
"SYNTHETIC DATA".

## Real run

Download `train.zip`, `test.zip` and the two metadata CSVs from [Zenodo](https://zenodo.org/records/6967442)
(about 2.5 GB, non-commercial academic use) and unzip the audio into one folder.

```bash
python run_all.py --meta-train "metadata of train set .csv" --meta-test "metadata of test set.csv" --audio-root <audio folder> --workdir out
```

Run `audit.py` first on its own if you only want the split check:

```bash
python audit.py "metadata of train set .csv" "metadata of test set.csv"
```

Add `--embeddings` to `run_all.py` for AST/CLAP (needs `torch` and `transformers`; weights download on first use).

## Outputs (in `--workdir`)

| File | Contents |
|---|---|
| `audit.txt` | split overlap and per-class concentration |
| `summary.txt`, `results.csv` | macro-F1 tables and every individual run |
| `probe.txt`, `probe.csv` | recording-identity probe |
| `figures/1_protocols.png` | accuracy by protocol and feature set |
| `figures/2_protocol_gap.png` | the protocol gap |
| `figures/3_budget.png` | training budget, random vs recording-balanced sampling |
| `figures/4_per_class.png` | per-class F1 on unseen recordings |
| `figures/5_class_profile.png` | what the structure features see in each class |

## Limitations

- Nothing has been run on the real data, and the embedding code is untested.
- Burst-detection thresholds in `features.py` (6 dB prominence, 80 ms spacing) are untuned guesses. Fix them before
  looking at group-split results.
- The hypothesis that structure features generalise better may be false. Related work is mixed and none tests it on
  this dataset.
- Published descriptions of the official split disagree, and none says whether clips from one recording stay on the
  same side. `audit.py` answers that.
- Do not use voc2vec for main claims: it was pretrained on Nonspeech7k.
- Never commit the audio. `.gitignore` already excludes `*.wav`, `*.zip` and `feats/`.

## Layout

`features.py` feature definitions, `common.py` metadata and cache helpers, `make_synthetic.py` smoke-test data,
`make_brief.py` rebuilds the PDF project brief (a copy is in `docs/project_brief.pdf`).
