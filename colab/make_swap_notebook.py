"""Generate colab/SWAP_PROBE_COLAB.ipynb:  python colab/make_swap_notebook.py"""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb = nbf.v4.new_notebook()
nb.cells = [
    md("""# Nonspeech7k: test-time background swap (no training)

Pre-registered in `docs/PREREGISTRATION_SWAP_PROBE.md`. Scores the 12 existing GPU AST checkpoints
(`runs_k0_pc200`, full fine-tuning) on test clips whose vocalisation is kept and whose background is replaced by the
background of a same-class (control) or different-class donor from another recording.
Primary model: `ast_ft_source_disjoint_standard` (3 seeds).

Needs in `MyDrive/ns7k/`: `ns7k_colab_bundle.zip`, `train.zip`, `test.zip` (as before) and the finished runs in
`work/runs_k0_pc200`. *Runtime -> Change runtime type -> T4 GPU*, then run all cells. Results go to
`MyDrive/ns7k/work/swap_probe_k0` and survive disconnects."""),
    md("## 1. Settings"),
    code(r'''DRIVE_DIR = "/content/drive/MyDrive/ns7k"
FREEZE_K = 0
PER_CLASS_TRAIN = 200
WORK = f"{DRIVE_DIR}/work"
FB = f"{WORK}/fbank_pc{PER_CLASS_TRAIN}"
WV = f"{WORK}/waves_pc{PER_CLASS_TRAIN}"
RUNS = f"{WORK}/runs_k{FREEZE_K}_pc{PER_CLASS_TRAIN}"
SWAP = f"{WORK}/swap_probe_k{FREEZE_K}"
print(RUNS, "->", SWAP)'''),
    md("## 2. GPU check"),
    code(r'''!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch
print("CUDA", torch.cuda.is_available())'''),
    md("## 3. Drive, audio, newest code"),
    code(r'''from google.colab import drive
drive.mount("/content/drive")
import os, zipfile
for f in ["ns7k_colab_bundle.zip", "train.zip", "test.zip"]:
    assert os.path.exists(f"{DRIVE_DIR}/{f}"), f"missing {DRIVE_DIR}/{f}"
zipfile.ZipFile(f"{DRIVE_DIR}/ns7k_colab_bundle.zip").extractall("/content")   # provides /content/splits/splits.csv
if not os.path.exists("/content/data/audio/.unpacked"):
    for z in ["train.zip", "test.zip"]:
        zipfile.ZipFile(f"{DRIVE_DIR}/{z}").extractall("/content/data/audio")
    open("/content/data/audio/.unpacked", "w").write("ok")
print(sum(f.endswith(".wav") for _, _, fs in os.walk("/content/data/audio") for f in fs), "wav files (expected 7014)")
assert os.path.exists("/content/splits/splits.csv")'''),
    code(r'''%cd /content
!rm -rf /content/ns7k-src && git clone -q -b analysis/source-identity https://github.com/Harbani007/nonspeech7k-generalisation.git /content/ns7k-src
%cd /content/ns7k-src
!git log -1 --format="code at %h: %s"
!pip -q install "transformers>=4.40" soundfile
!python -m unittest discover tests 2>&1 | tail -3'''),
    md("## 4. Inputs present? (12 finished checkpoints and the clip list)"),
    code(r'''import glob
runs = sorted(d for d in glob.glob(f"{RUNS}/ast_ft*_source_disjoint_*_s[012]") if os.path.exists(f"{d}/test_metrics.json"))
print(len(runs), "finished checkpoints (expected 12):", [os.path.basename(d) for d in runs])
CLIPS = f"{FB}/clips.csv" if os.path.exists(f"{FB}/clips.csv") else f"{WV}/clips.csv"
if not os.path.exists(CLIPS):   # same clip selection, rebuilt in about 3 min
    !python ast_finetune.py waves --data /content/splits --audio-root /content/data/audio --out "{WV}" 2>&1 | tail -1
    CLIPS = f"{WV}/clips.csv"
print("clip list:", CLIPS)'''),
    md("""## 5. Run (check step first, then the full run)

Check in the output: the reproduction table (12 rows, `freeze_k` 0, `agreement` >= 0.99), the donor balance,
then the result tables and the `DECISION` line. If Colab disconnects, rerun this cell."""),
    code(r'''!bash run_gpu_swap.sh python /content/splits "{CLIPS}" /content/data/audio "{RUNS}" {FREEZE_K} "{SWAP}" 2>&1 \
   | grep -vE "Warning|UNEXPECTED|^Key|^-+\+|^Notes|can be ignored|^\s*$|log10"'''),
    md("## 6. Results"),
    code(r'''import pandas as pd, json
from IPython.display import Image, display
pd.set_option("display.width", 250)
print(json.dumps(json.load(open(f"{SWAP}/decision.json")), indent=1))
E = pd.read_csv(f"{SWAP}/swap_effects.csv")
display(E[E.system == "ast_ft_source_disjoint_standard"].round(4))
display(pd.read_csv(f"{SWAP}/condition_scores.csv").round(3))
display(pd.read_csv(f"{SWAP}/donor_balance.csv").round(3))
display(pd.read_csv(f"{SWAP}/residue_audit.csv").round(3).T)
display(pd.read_csv(f"{SWAP}/reproduction_check.csv"))
display(Image(f"{SWAP}/G11_swap_probe.png"))'''),
    md("## 7. Pack the results to send back (one zip on Drive)"),
    code(r'''!cd "{WORK}" && zip -qr swap_probe_k{FREEZE_K}_results.zip "swap_probe_k{FREEZE_K}" -x "*/runs/*" && ls -la swap_probe_k{FREEZE_K}_results.zip
# per-checkpoint predictions (runs/*/predictions.csv.gz) stay on Drive; zip them too if asked'''),
]
nbf.write(nb, os.path.join(HERE, "SWAP_PROBE_COLAB.ipynb"))
print("wrote", os.path.join(HERE, "SWAP_PROBE_COLAB.ipynb"))
