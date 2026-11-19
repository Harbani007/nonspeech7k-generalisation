# Colab run: background-swap experiment (about 1.5-2 h on a T4)

Run the setup cell from yesterday first (Drive mounted; `WORK`, `FB`, `RUNS`, `FREEZE_K` defined; code cloned to
/content/ns7k-src; audio unzipped; /content/splits/splits.csv present). Pull the newest code:
```python
%cd /content
!rm -rf /content/ns7k-src && git clone -q -b analysis/source-identity https://github.com/Harbani007/nonspeech7k-generalisation.git /content/ns7k-src
%cd /content/ns7k-src
WV = f"{WORK}/waves_pc200"
```

## Step 1: gate, background probe at 30 dB (about 10 min)
```python
!python background_probe.py --data /content/splits --clips "{FB}/clips.csv" --audio-root /content/data/audio \
    --runs "{RUNS}" --out "{WORK}/background_probe_db30" --fg-db 30 --device cuda 2>&1 | grep -A40 "system    variant"
```
Look at the `background` rows: is `ci_lo` above `chance_95`? Send me the table.

## Step 2: waveform cache (about 3 min, once)
```python
!python ast_finetune.py waves --data /content/splits --audio-root /content/data/audio --out "{WV}"
```

## Step 3: train control + background swap, 3 seeds each (about 1-1.5 h; rerun after a disconnect, it resumes)
```python
!EPOCHS=4 SPE=768 BATCH=32 LR=3e-5 BGSWAP=0.5 bash run_gpu_bgswap.sh python /content/splits "{WV}" "{RUNS}" {FREEZE_K} 0 1 2
```

## Step 4: evaluate (about 15 min)
```python
!python dl_report.py --data /content/splits --runs "{RUNS}" --audit /content/splits --out "{WORK}/report_bgswap" 2>&1 | grep -B2 -A3 "background-swap"
!python background_probe.py --data /content/splits --clips "{FB}/clips.csv" --audio-root /content/data/audio \
    --runs "{RUNS}" --out "{WORK}/background_probe_bgswap" --fg-db 20 --device cuda \
    --run-globs "ast_wv_source_disjoint_standard_s*" "ast_bgs_source_disjoint_standard_s*"     --compare ast_wv_source_disjoint_standard ast_bgs_source_disjoint_standard 2>&1 | grep -A60 "system    variant"
```
Send me both printed tables, plus `report_bgswap/paired_differences.csv` and
`background_probe_bgswap/background_probe_results.csv`.
