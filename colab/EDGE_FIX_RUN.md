# Corrected background probe (edge-leak fix): Colab, about 25 min on a T4

Why: the original masks left ~10 ms of vocalisation at about half volume at the clip start and up to ~90% of the last
5 ms at the clip end (zero-padded edge smoothing + 80 unframed samples). Rebuilding the masks with the fix changed
2,508 of 3,242 clips (914 of 1,160 test clips), only at the clip edges. This rerun gives corrected numbers for the
"vocalisation removed / only / random cut" probe on the same 12 fully fine-tuned AST checkpoints (FREEZE_K = 0).

## Cell 1: setup (works after a runtime restart)
```python
DRIVE_DIR = "/content/drive/MyDrive/ns7k"
WORK = f"{DRIVE_DIR}/work"
FB = f"{WORK}/fbank_pc200"
RUNS = f"{WORK}/runs_k0_pc200"
from google.colab import drive
drive.mount("/content/drive")
import os, shutil, zipfile, glob
%cd /content
shutil.rmtree("/content/ns7k-src", ignore_errors=True)
!git clone -q -b analysis/source-identity https://github.com/Harbani007/nonspeech7k-generalisation.git /content/ns7k-src
%cd /content/ns7k-src
!pip -q install "transformers>=4.40" soundfile
os.makedirs("/content/splits", exist_ok=True)
shutil.copy("docs/results/splits/splits.csv", "/content/splits/splits.csv")
if not os.path.exists("/content/data/audio/.unpacked"):
    for z in ["train.zip", "test.zip"]:
        zipfile.ZipFile(f"{DRIVE_DIR}/{z}").extractall("/content/data/audio")
    open("/content/data/audio/.unpacked", "w").write("ok")
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE: switch runtime to GPU")
print("edge-hold switch present:", "--edge-hold" in open("background_probe.py").read())
print("finished AST runs:", len(glob.glob(f"{RUNS}/ast_*/test_metrics.json")), "(expect 15)")
print("clips.csv:", os.path.exists(f"{FB}/clips.csv"))
```
All checks must pass: a GPU name, `True`, 15, `True`.

## Cell 2: corrected probe at 20 dB (the published setting), about 12 min
```python
!python background_probe.py --data /content/splits --clips "{FB}/clips.csv" --audio-root /content/data/audio \
    --runs "{RUNS}" --out "{WORK}/background_probe_db20_edgefix" --fg-db 20 --edge-hold --device cuda 2>&1 | grep -A60 "system    variant"
```

## Cell 3: corrected probe at 30 dB (stricter cut), about 12 min
```python
!python background_probe.py --data /content/splits --clips "{FB}/clips.csv" --audio-root /content/data/audio \
    --runs "{RUNS}" --out "{WORK}/background_probe_db30_edgefix" --fg-db 30 --edge-hold --device cuda 2>&1 | grep -A60 "system    variant"
```

## Cell 4: old vs corrected, side by side
```python
import pandas as pd
old = pd.read_csv(f"{WORK}/background_probe_db20/background_probe_results.csv")
new = pd.read_csv(f"{WORK}/background_probe_db20_edgefix/background_probe_results.csv")
n30 = pd.read_csv(f"{WORK}/background_probe_db30_edgefix/background_probe_results.csv")
k = ["system", "variant"]
t = old[k + ["macro_f1"]].merge(new[k + ["macro_f1", "ci_lo", "ci_hi", "chance_95"]], on=k, suffixes=("_old20", "_fixed20"))
t = t.merge(n30[k + ["macro_f1", "ci_lo", "chance_95"]].rename(columns={"macro_f1": "macro_f1_fixed30", "ci_lo": "ci_lo_fixed30",
            "chance_95": "chance_95_fixed30"}), on=k)
display(t[t.variant == "background"].round(3))
display(t.round(3))
```

## What to send back
The printed tables of cells 2-4, plus these files from Drive:
`background_probe_db20_edgefix/background_probe_results.csv`, `background_probe_db20_edgefix/background_probe_paired_differences.csv`,
`background_probe_db30_edgefix/background_probe_results.csv`, and both folders' `mask_summary.csv`.

## How to read it
Look at the `background` rows (vocalisation removed):
- **Corrected 20 dB stays near the old 0.27-0.30 and above chance:** the leak did not drive the finding; say so.
- **Corrected value drops toward `chance_95`:** part of the "twice chance" result was the leaked vocalisation; the
  slides and report must be corrected.
- **30 dB:** if it stays above chance as well, the leftover audio really carries class information (background
  context); if it falls to chance, the earlier signal was faint vocalisation, not context.

## Also affected (only if you run Phase 2, background-swap training)
The waveform cache used for training masks (`ast_finetune.py waves`) has the same leak. Rebuild it with the fix before
training: `!python ast_finetune.py waves --data /content/splits --audio-root /content/data/audio --edge-hold --out "{WORK}/waves_pc200_edgefix"`
and pass that folder instead of the old one.
