"""Workflow block diagram for the lab report: python report/flowchart.py report/build/flowchart.png"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

steps = [("Nonspeech7k audio\n7,014 clips, 7 classes, 32 kHz", "#E8EEF7"),
         ("Source reconstruction\nfilename IDs -> recordings (sources.py)", "#E8EEF7"),
         ("Duplicate detection\nAST screening + waveform cross-correlation", "#E8EEF7"),
         ("Source-disjoint split\ntrain / validation / test, no shared recording", "#FFF3D6"),
         ("Features\nlog-mel (CNN) | AST filterbank 128 x 400", "#FFF3D6"),
         ("Models\nLR baselines | CNN | AST frozen | AST fine-tuned 2x2", "#E3F2E1"),
         ("Selection on validation macro-F1\n(test set used once)", "#E3F2E1"),
         ("Evaluation\nAcc, P, R, F1, confusion, recording-level bootstrap CIs", "#F7E3E3"),
         ("Diagnostics\nwebsite hold-out | error clustering | background probe", "#F7E3E3")]
fig, ax = plt.subplots(figsize=(6.2, 9.2)); ax.axis("off")
n = len(steps); h = 1.0 / n
for i, (t, c) in enumerate(steps):
    y = 1 - (i + 1) * h + 0.25 * h
    ax.add_patch(FancyBboxPatch((0.08, y), 0.84, 0.6 * h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=c, ec="#333", lw=1))
    ax.text(0.5, y + 0.3 * h, t, ha="center", va="center", fontsize=9.5, family="serif")
    if i < n - 1:
        ax.annotate("", xy=(0.5, y - 0.4 * h + 0.012), xytext=(0.5, y - 0.012), arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.savefig(sys.argv[1], dpi=200, bbox_inches="tight")
