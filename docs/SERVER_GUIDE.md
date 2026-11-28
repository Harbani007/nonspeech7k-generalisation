# How to run the project on the college GPU server (JupyterLab) yourself

Written for Harbani and Ina: everything Claude did on the A100 on 28 Nov 2026, step by step, so you can do it alone.

## 0. What the server is
- A JupyterLab page at `http://172.16.224.131:1003` (college network). Behind it: a Linux container with an
  **NVIDIA A100 (80 GB)**, Python 3.12 and an NVIDIA build of PyTorch. You are `root` inside the container.
- Your files live in `/workspace` (the JupyterLab file panel shows this folder as `/`).
- **The container can be wiped when your session ends.** Always download results before you log off (section 6).
- It is shared: run one big job at a time.

## 1. The three ways to run things (and when to use which)
| Way | How | Use it for |
|---|---|---|
| Notebook cell | File -> New -> Notebook, paste code, `Shift + Enter` | short steps (seconds to a few minutes) |
| Background job from a notebook | `subprocess.Popen(..., start_new_session=True)` (section 3) | long training/evaluation (minutes to hours) |
| Terminal | Launcher (`+` tab) -> Other -> **Terminal** | shell commands; `nohup ... &` works here |

Rule of thumb: anything longer than ~10 minutes goes in the background and writes a **log file** you can read later.

## 2. One-time setup in a new container
Notebook cell 1 (packages + code):
```python
import sys, torch
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
!{sys.executable} -m pip install -q "transformers>=4.40" soundfile scikit-learn openpyxl
%cd /workspace
!rm -rf /workspace/ns7k-src && git clone -q -b analysis/source-identity https://github.com/Harbani007/nonspeech7k-generalisation.git /workspace/ns7k-src
%cd /workspace/ns7k-src
```
- Check the torch version printed is still `2.14.0a0+...nv26.09` afterwards: never reinstall torch here.
- **Do not** use `importlib.reload(torch)`: the NVIDIA build crashes ("Only a single TORCH_LIBRARY...").
  If you need a clean start: menu **Kernel -> Restart Kernel**.

Notebook cell 2 (dataset from Zenodo, ~5-10 min; see section 2 of colab/README for the cell used on 28 Nov):
download the files listed by `https://zenodo.org/api/records/6967442`, unzip into `/workspace/data/audio`,
check `wav files: 7014`, copy `docs/results/splits/splits.csv` to `/workspace/splits/`.

Notebook cell 3 (AST input features, ~5 min):
```python
W = "/workspace/work"
!{sys.executable} ast_finetune.py fbank --data /workspace/splits --audio-root /workspace/data/audio --out {W}/fbank_pc200 2>&1 | tail -2
```

## 3. Starting a long job in the background (the important trick)
`!command &` does **not** work in this Jupyter (error: `OSError ... background processes not supported`). Use:
```python
import subprocess, sys, os
W = "/workspace/work"; os.makedirs(W, exist_ok=True)
log = open(f"{W}/ast_suite.log", "w")
cmd = ["bash", "run_gpu_suite.sh", sys.executable, "/workspace/splits", f"{W}/fbank_pc200", f"{W}/runs_k0_pc200", "0", "0", "1", "2"]
p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd="/workspace/ns7k-src")
print("started, pid", p.pid)
```
`start_new_session=True` detaches the job: it keeps running if you close the tab or restart the notebook.
Alternative: open a **Terminal** and type `cd /workspace/ns7k-src && nohup bash run_gpu_suite.sh python3 ... > /workspace/work/ast_suite.log 2>&1 &`

## 4. Checking progress
```python
import glob, os
W = "/workspace/work"
print("finished runs:", len(glob.glob(f"{W}/runs_k0_pc200/*/test_metrics.json")), "of 15")
print(open(f"{W}/ast_suite.log").read()[-1500:])
logs = sorted(glob.glob(f"{W}/runs_k0_pc200/*/train.log"), key=os.path.getmtime)
print("CURRENT RUN:", logs[-1] if logs else None); print(open(logs[-1]).read()[-1200:] if logs else "")
```
- The suite log only gets a line when a run **finishes**; the running run writes to its own `train.log`.
- In a terminal: `tail -f /workspace/work/ast_suite.log` (Ctrl+C stops watching, not the job).
- Is it still running? Terminal: `ps aux | grep -E "ast_finetune|swap_probe|background_probe" | grep -v grep`.
- GPU busy? `nvidia-smi` (on this server it may say "Insufficient Permissions" for details; that is normal).

## 5. Chaining steps so they run unattended
Write a small shell script that waits for the training to finish and then runs the next steps, and start it once.
Example used on 28 Nov (`/workspace/work/post.sh`, simplified):
```bash
#!/bin/bash
cd /workspace/ns7k-src; W=/workspace/work
until [ "$(ls $W/runs_k0_pc200/*/test_metrics.json 2>/dev/null | wc -l)" -ge 15 ]; do sleep 30; done
python3 dl_report.py --data /workspace/splits --runs $W/runs_k0_pc200 --audit /workspace/splits --out $W/report_k0 > $W/post_report.log 2>&1
python3 background_probe.py --data /workspace/splits --clips $W/fbank_pc200/clips.csv --audio-root /workspace/data/audio \
    --runs $W/runs_k0_pc200 --out $W/probe_db20_edgefix --fg-db 20 --edge-hold --device cuda > $W/post_probe.log 2>&1
echo ALL_DONE > $W/post_status.txt
```
Create it with JupyterLab: File -> New -> Text File, paste, save as `/workspace/work/post.sh`; then in a terminal:
`nohup bash /workspace/work/post.sh > /workspace/work/post_main.log 2>&1 &`

## 6. Getting results onto your laptop
1. Zip the small files (never the `.pt` model weights):
   ```python
   import zipfile, glob, os
   W = "/workspace/work"
   with zipfile.ZipFile("/workspace/results_for_laptop.zip", "w", zipfile.ZIP_DEFLATED) as z:
       for pat in ["report_k0/*", "probe_*/*.csv", "probe_*/*.png", "runs_k0_pc200/*/test_metrics.json",
                   "runs_k0_pc200/*/test_predictions.csv", "*.log"]:
           for f in glob.glob(os.path.join(W, pat)):
               z.write(f, os.path.relpath(f, W))
   ```
2. In the JupyterLab file panel: right-click `results_for_laptop.zip` -> **Download**.
   Opening `http://172.16.224.131:1003/files/...` directly in a new tab gives **403 Forbidden** (the server only
   trusts requests coming from the JupyterLab page itself), so always download from the file panel.

## 7. Problems we hit, and the fix
| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: Only a single TORCH_LIBRARY ...` | `importlib.reload(torch)` | Kernel -> Restart Kernel; never reload torch |
| `OSError` on a `!... &` cell | Jupyter refuses background `!` commands | `subprocess.Popen(..., start_new_session=True)` or a terminal |
| Cell stuck at `[*]` while the kernel says Idle | the same notebook open in two tabs/windows | close the other tab, or Kernel -> Interrupt and run again |
| `NameError: W is not defined` | new notebook or restarted kernel (variables are per notebook) | rerun the setup lines (`import sys`, `W = "/workspace/work"`, `%cd /workspace/ns7k-src`) |
| `FileNotFoundError` for `dl_data.py` etc. | notebook not in the code folder | `%cd /workspace/ns7k-src` |
| `KeyError: '<file>.wav'` | audio not unzipped in this container | rerun the Zenodo/unzip cell |
| 403 Forbidden on a file link | direct link outside JupyterLab | download from the file panel |

## 8. What ran on 28 Nov (for reference)
- 15 AST runs (all 12 layers fine-tuned, 3 seeds x 5 conditions): about 1 minute each on the A100.
- Report + three background probes (old masks 20 dB, fixed 20 dB, fixed 30 dB): about 25 minutes per probe.
- Phase 0 test-time background swap (`run_gpu_swap.sh`): started 10:39 UTC.
Results: `docs/results/a100_rerun/`.
