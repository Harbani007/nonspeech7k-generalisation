# Clean CNN rerun (Colab T4, 19 Nov). Transcribed from `colab_cnn_run.ipynb` (cell 4 output)

Log-mel CNN from scratch (`train_cnn.py`, 293,175 weights), source-disjoint split (the committed
`docs/results/splits/splits.csv`; inputs rebuilt with `dl_data.py --splits`, verified identical to the laptop cache),
25 epochs x 1,024 samples, batch 32, AdamW 2e-3, OneCycle, no early stopping, seeds 0-2, `run_gpu_cnn.sh`.

| run | best epoch | accuracy | macro-F1 | balanced acc |
|---|---|---|---|---|
| standard s0 | 25 | 0.754 | 0.759 | 0.804 |
| standard s1 | 23 | 0.750 | 0.751 | 0.808 |
| standard s2 | 21 | 0.787 | 0.776 | 0.816 |
| source-balanced s0 | 21 | 0.791 | 0.787 | 0.809 |
| source-balanced s1 | 23 | 0.813 | 0.811 | 0.831 |
| source-balanced s2 | 23 | 0.784 | 0.776 | 0.815 |

dl_report.py (1,160 test clips, 300 recordings, recording-level bootstrap, seeds averaged per resample):
| system | macro-F1 [95% CI] | seed SD | accuracy | balanced acc |
|---|---|---|---|---|
| CNN standard | 0.762 [0.704, 0.812] | 0.010 | 0.764 | 0.809 |
| CNN source-balanced | 0.792 [0.741, 0.834] | 0.015 | 0.796 | 0.818 |

**Source-balanced minus standard: +0.030 [+0.005, +0.057] (CI excludes 0).**
The earlier single CPU run (0.767) is consistent with the clean standard result (0.762).
