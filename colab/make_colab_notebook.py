"""Generate colab/NS7K_AST_GPU_COLAB.ipynb:  python colab/make_colab_notebook.py"""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb = nbf.v4.new_notebook()
nb.cells = [
    md("""# Nonspeech7k: source-aware AST fine-tuning on a GPU (Google Colab)

Runs the controlled 2 x 2 experiment of this project on a GPU:

| | cross-entropy | cross-entropy + cross-source SupCon |
|---|---|---|
| clip-uniform sampling | ordinary | contrastive |
| source-uniform sampling | source-balanced | combined |

plus a frozen-AST reference, 3 seeds, on the **source-disjoint** split (`splits.csv`, built by `dl_data.py`).

**Before you start, put these 3 files in Google Drive, folder `MyDrive/ns7k/`:**
1. `ns7k_colab_bundle.zip`: the code plus `splits.csv` (made on the laptop, see `colab/README.md`)
2. `train.zip` and 3. `test.zip`: the Nonspeech7k audio (the Zenodo files, record 6967442)

Then: *Runtime -> Change runtime type -> GPU* and run the cells top to bottom.
Everything that matters (filterbanks, checkpoints, logs, report) is written to `MyDrive/ns7k/work`, so after a
disconnect you just rerun all cells: finished runs are skipped and an interrupted run resumes from its last epoch."""),
    md("## 1. Settings (edit here)"),
    code(r'''DRIVE_DIR = "/content/drive/MyDrive/ns7k"   # where the 3 uploaded files are
FREEZE_K = 0        # 0 = fine-tune all 12 transformer layers (full fine-tuning); 11 = same as the CPU experiment
SEEDS = "0 1 2"
PER_CLASS_TRAIN = 200   # 200 = same training pool as the CPU experiment; 100000 = every training clip
EPOCHS, SPE, BATCH = 4, 768, 32   # same budget as the CPU experiment; lower BATCH to 16 if the GPU runs out of memory
LR = "3e-5"            # encoder learning rate (head 1e-3); for full fine-tuning 1e-5 is a common safer choice
import os
WORK = f"{DRIVE_DIR}/work"
FB = f"{WORK}/fbank_pc{PER_CLASS_TRAIN}"
RUNS = f"{WORK}/runs_k{FREEZE_K}_pc{PER_CLASS_TRAIN}"
REPORT = f"{WORK}/report_k{FREEZE_K}_pc{PER_CLASS_TRAIN}"
print(RUNS)'''),
    md("## 2. GPU check"),
    code(r'''!nvidia-smi
import torch
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU: Runtime -> Change runtime type -> GPU")'''),
    md("## 3. Mount Drive, unpack code and audio"),
    code(r'''from google.colab import drive
drive.mount("/content/drive")
import os, zipfile
os.makedirs(WORK, exist_ok=True)
for f in ["ns7k_colab_bundle.zip", "train.zip", "test.zip"]:
    assert os.path.exists(f"{DRIVE_DIR}/{f}"), f"missing {DRIVE_DIR}/{f}"
zipfile.ZipFile(f"{DRIVE_DIR}/ns7k_colab_bundle.zip").extractall("/content")
if not os.path.exists("/content/data/audio/.unpacked"):
    for z in ["train.zip", "test.zip"]:
        zipfile.ZipFile(f"{DRIVE_DIR}/{z}").extractall("/content/data/audio")
    open("/content/data/audio/.unpacked", "w").write("ok")
n = sum(f.endswith(".wav") for _, _, fs in os.walk("/content/data/audio") for f in fs)
print(n, "wav files (expected 7014)")'''),
    code(r'''%cd /content/ns7k-src
!pip -q install "transformers>=4.40" soundfile nbformat
!python -m unittest discover tests 2>&1 | tail -3'''),
    md("## 4. AST input filterbanks (once; saved to Drive, resumable)"),
    code(r'''!python ast_finetune.py fbank --data /content/splits --audio-root /content/data/audio --per-class-train {PER_CLASS_TRAIN} --out "{FB}" 2>&1 | tail -2'''),
    md("""## 5. Train the 2 x 2 experiment

Rerun this cell after any disconnect. Progress per run: `RUNS/<run>/train.log`. Each step logs cross-entropy,
contrastive loss, number of cross-source positive pairs, share of anchors with a positive and gradient norms
(`step_log.csv`); gradient norms per pretrained layer are in `grad_norms.csv`, weight change in `weight_drift.csv`."""),
    code(r'''!EPOCHS={EPOCHS} SPE={SPE} BATCH={BATCH} LR={LR} bash run_gpu_suite.sh python /content/splits "{FB}" "{RUNS}" {FREEZE_K} {SEEDS}'''),
    code(r'''# quick look at one run while others train
import pandas as pd, glob
for d in sorted(glob.glob(f"{RUNS}/*"))[:20]:
    h = os.path.join(d, "history.csv")
    print(os.path.basename(d), "finished" if os.path.exists(os.path.join(d, "test_metrics.json")) else "not finished",
          pd.read_csv(h)[["epoch", "train_ce", "train_supcon", "cross_source_pairs_per_batch", "val_macro_f1"]].round(3).to_dict("records")[-1] if os.path.exists(h) else "")'''),
    md("## 6. Report: 2 x 2 effects with source-grouped bootstrap CIs, curves, confusions, per-class, per-website, source clustering"),
    code(r'''!python dl_report.py --data /content/splits --runs "{RUNS}" --audit /content/splits --out "{REPORT}" 2>&1 | tail -40'''),
    code(r'''from IPython.display import Image, display
import pandas as pd
display(pd.read_csv(f"{REPORT}/ast_2x2_effects.csv").round(4))
for f in ["G6_ast_2x2_source_aware_training", "G1_G2_learning_curves_ast_2x2", "G3_confusion_matrices",
          "G4_per_class_recall", "G9_source_clustering_of_embeddings"]:
    p = f"{REPORT}/{f}.png"
    if os.path.exists(p):
        print(f); display(Image(p))'''),
    md("## 7. Live inference with a trained checkpoint on a real test recording (raw softmax, not calibrated)"),
    code(r'''import sys, numpy as np
sys.path.insert(0, "/content/ns7k-src")
from inference import ASTClassifier
from common import index_audio
from features import load_mono
from dl_data import CLASSES
from IPython.display import Audio
sp = pd.read_csv("/content/splits/splits.csv"); audio = index_audio("/content/data/audio")
clf = ASTClassifier(f"{RUNS}/ast_ftsc_source_disjoint_source_s0/best.pt")
row = sp[sp.source_disjoint == "test"].sample(1, random_state=1).iloc[0]
w, sr = load_mono(audio[row.filename]); display(Audio(w, rate=sr))
s, t = clf.predict(w)
print(row.filename, "| true:", row.cls, "| predicted:", CLASSES[s.argmax()], f"| {1000*t:.0f} ms")
pd.Series(s, index=CLASSES).round(3)'''),
    md("""## Notes
* `FREEZE_K=0` fine-tunes all 12 transformer layers; patch and position embeddings stay pretrained in every setting.
* Results with a different `FREEZE_K`, `PER_CLASS_TRAIN` or budget are a **different experiment** from the CPU run
  (layer 11 only, 200 clips per class): report them separately, never mixed in one table.
* Hyperparameters (lambda 0.5, tau 0.1) are fixed in advance (`docs/PREREGISTRATION.md`); do not tune them on the test set.
  If you want to tune, select on validation macro-F1 only and label the result exploratory."""),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["accelerator"] = "GPU"
out = os.path.join(HERE, "NS7K_AST_GPU_COLAB.ipynb")
nbf.write(nb, out)
print("wrote", out)
