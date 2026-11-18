# Running the AST experiments on a GPU (Google Colab)

## One-time preparation on the laptop

```powershell
cd C:\Users\Administrator\Downloads\nonspeech7k-generalisation\ns7k-src
C:\Users\ADMINI~1\AppData\Local\Temp\ns7kvenv\Scripts\python.exe colab\make_bundle.py --splits ..\results\v3\dl\splits.csv --out ..\colab_upload\ns7k_colab_bundle.zip
```

(Any Python with no extra packages works for `make_bundle.py`.)

Upload to Google Drive, folder `MyDrive/ns7k/`:

| file | from |
|---|---|
| `ns7k_colab_bundle.zip` | `nonspeech7k-generalisation\colab_upload\` (made above, < 1 MB) |
| `train.zip` (2.3 GB) | `nonspeech7k-generalisation\data\train.zip` (or Zenodo record 6967442) |
| `test.zip` (0.2 GB) | `nonspeech7k-generalisation\data\test.zip` |

## In Colab

1. Upload `colab/NS7K_AST_GPU_COLAB.ipynb` (File -> Upload notebook), or open it from Drive.
2. Runtime -> Change runtime type -> **T4 GPU** (or better).
3. Set `FREEZE_K` in cell 1: `0` = full fine-tuning of all 12 layers, `11` = same as the CPU experiment.
4. Run all cells. Outputs go to `MyDrive/ns7k/work/` and survive disconnects: rerun all cells to resume.

## Same thing on any Linux GPU machine (no notebook)

```bash
pip install torch transformers soundfile scipy pandas scikit-learn matplotlib
unzip ns7k_colab_bundle.zip -d ~/ns7k && mkdir -p ~/ns7k/data/audio
unzip train.zip -d ~/ns7k/data/audio && unzip test.zip -d ~/ns7k/data/audio
cd ~/ns7k/ns7k-src
python ast_finetune.py fbank --data ../splits --audio-root ../data/audio --out ../work/fbank_pc200
bash run_gpu_suite.sh python ../splits ../work/fbank_pc200 ../work/runs_k0_pc200 0 0 1 2
python dl_report.py --data ../splits --runs ../work/runs_k0_pc200 --audit ../splits --out ../work/report_k0_pc200
```

## What is identical to the CPU experiment, and what is not

- Identical: the split (`splits.csv`), clip selection (seed 0, up to 200 training clips per class unless changed),
  the loss (`cross_source_supcon`), samplers, budget, initialisation per seed, logging, checkpointing, report.
- Different by design: the CPU experiment trains layer 11 on cached tokens; on GPU the filterbanks are fed through
  the network live, so any number of layers can be trained (`FREEZE_K`). The GPU run uses fp16 mixed precision.
  Treat each (FREEZE_K, training pool) setting as its own experiment.
