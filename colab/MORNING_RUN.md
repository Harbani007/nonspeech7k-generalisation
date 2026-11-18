# Morning Colab run: background probe ("what is the model listening to?") + adversarial follow-up

Prerequisite: the Colab notebook session from yesterday (cells 1-3 run: settings with FREEZE_K = 0, GPU check,
Drive mounted). `RUNS`, `FB` and `REPORT` then point at the finished full-fine-tuning runs on Drive.

## 1. Get the newest code (after `git push` from the laptop)
```python
!rm -rf /content/ns7k-src && git clone -b analysis/source-identity https://github.com/Harbani007/nonspeech7k-generalisation.git /content/ns7k-src
%cd /content/ns7k-src
!pip -q install "transformers>=4.40" soundfile
```

## 2. Background probe (main new experiment, about 15-30 min on a T4)
No training: it re-scores your trained checkpoints on modified test audio.
```python
!python background_probe.py --data /content/splits --clips "{FB}/clips.csv" --audio-root /content/data/audio \
    --runs "{RUNS}" --out "{WORK}/background_probe_db20" --fg-db 20 --device cuda
```
Sensitivity check, a less strict removal (optional, same time again):
```python
!python background_probe.py --data /content/splits --clips "{FB}/clips.csv" --audio-root /content/data/audio \
    --runs "{RUNS}" --out "{WORK}/background_probe_db10" --fg-db 10 --device cuda
```
Show the figure and listen to examples:
```python
from IPython.display import Image, Audio, display
import pandas as pd, glob
display(Image(f"{WORK}/background_probe_db20/G10_background_probe.png"))
display(pd.read_csv(f"{WORK}/background_probe_db20/background_probe_paired_differences.csv").round(3))
for f in sorted(glob.glob(f"{WORK}/background_probe_db20/examples/*.wav"))[:6]:
    print(f.split("/")[-1]); display(Audio(f))
```

## 3. Adversarial follow-up (optional, about 30-45 min)
```python
!ADV=0.3 EPOCHS={EPOCHS} SPE={SPE} BATCH={BATCH} LR={LR} bash run_gpu_adv.sh python /content/splits "{FB}" "{RUNS}" {FREEZE_K} 0 1 2
!python dl_report.py --data /content/splits --runs "{RUNS}" --audit /content/splits --out "{REPORT}_adv" 2>&1 | grep -A12 "adversarial\|source clustering"
```
If you run this BEFORE step 2, the background probe also scores the adversarial models.

## Send back
- the printed table from step 2 (results and paired differences);
- `background_probe_db20/background_probe_results.csv`, `background_probe_paired_differences.csv`, `mask_summary.csv`;
- (if run) `{REPORT}_adv/paired_differences.csv` and `G9_source_clustering_of_embeddings.csv`.

## How to read step 2
- `full` vs `background`: how much performance is lost when the vocalisation is cut out.
- `background` vs `random` (same amount of audio removed elsewhere): proves the loss comes from removing the
  vocalisation specifically.
- `background` above `chance_95` (column `above_chance`): the model still recognises the class from what is
  left. If true on these unseen recordings, it's the striking result.
- `frozen-AST probe (train background)` tested on `background`: whether class information exists in the background
  at all, independent of the fine-tuned models.
