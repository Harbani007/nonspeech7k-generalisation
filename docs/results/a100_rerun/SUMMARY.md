# College A100 rerun (28 Nov 2026): AST full fine-tuning + background probe, old vs edge-fixed masks

Server: NVIDIA A100-SXM4-80GB, PyTorch 2.14 (NVIDIA build), transformers 5.24.1. Code: branch analysis/source-identity
(commit d96267c). Data downloaded from Zenodo 6967442 (7,014 wavs); split = docs/results/splits/splits.csv.
Values transcribed from the server files /workspace/work/report_k0/ast_2x2_effects.csv and
/workspace/work/probe_{db20_old,db20_edgefix,db30_edgefix}/background_probe_results.csv (read via the Jupyter API).
Full result files: /workspace/results_for_laptop.zip on the server (to be copied here).

## 1. AST, all 12 layers fine-tuned, 3 seeds (reproduces the Colab T4 run)
| condition | A100 macro-F1 [95% CI] | Colab T4 macro-F1 |
|---|---|---|
| ordinary | 0.816 [0.765, 0.855] | 0.816 |
| source-balanced | 0.837 [0.789, 0.870] | 0.837 |
| contrastive | 0.812 [0.764, 0.849] | 0.812 |
| combined | 0.835 [0.787, 0.870] | 0.830 |
| balanced main effect | +0.022 [+0.002, +0.041] | +0.019 [-0.002, +0.038] |
| contrastive main effect | -0.003 [-0.011, +0.005] | -0.006 [-0.013, +0.002] |
| interaction | +0.002 [-0.013, +0.019] | -0.003 [-0.020, +0.013] |

## 2. Background probe: vocalisation removed ("background"), macro-F1 [CI low], chance = 95th pct of permutations
| model | old masks 20 dB | edge-fixed 20 dB | edge-fixed 30 dB |
|---|---|---|---|
| AST ordinary | 0.304 [0.258], ch 0.139 | 0.305 [0.259], ch 0.138 | 0.197 [0.154], ch 0.116 |
| AST source-balanced | 0.265 [0.224], ch 0.126 | 0.263 [0.223], ch 0.127 | 0.162 [0.117], ch 0.104 |
| AST contrastive | 0.303 [0.260], ch 0.140 | 0.303 [0.260], ch 0.138 | 0.202 [0.160], ch 0.122 |
| AST combined | 0.273 [0.228], ch 0.125 | 0.275 [0.230], ch 0.125 | 0.174 [0.125], ch 0.102 |
| linear probe trained on background only | 0.387 [0.327, 0.437], ch 0.157 | 0.389 [0.330, 0.438], ch 0.156 | 0.271 [0.213, 0.319], ch 0.158 |

Vocalisation-only ("foreground"): 0.727-0.744 (20 dB), 0.716-0.758 (30 dB). Test clips eligible: 645 clips / 186
recordings at 20 dB (643 with edge fix); **348 clips / 123 recordings at 30 dB** (breath 9, cough 62, crying 110,
laugh 108, screaming 14, sneeze 28, yawn 17): the 30 dB subset is smaller and differently composed.

## Conclusions
1. The edge leak did not drive the finding: fixing it changes the 20 dB background-only score by at most 0.002.
2. With the stricter 30 dB cut, background-only performance falls from about 2x chance to about 1.5x chance
   (0.16-0.20; CI lower bounds 0.117-0.160 vs chance 0.10-0.12, two of the four only just above). So much of the 20 dB
   "background" signal was quiet vocalisation 20-30 dB below the peak.
3. A classifier trained only on audio more than 30 dB below the vocalisation peak still reaches 0.271 [0.213, 0.319]
   against chance 0.158: some class information remains in the recording context.
4. Caveat: the 30 dB result is on a smaller, differently composed subset (few breath and screaming clips).
