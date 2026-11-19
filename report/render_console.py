"""Render saved notebook text output as a console-style image (used as execution screenshots in the report)."""
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def render(lines, title, out, width_chars=110):
    lines = [ANSI.sub("", l)[:width_chars] for l in lines]
    h = 0.17 * (len(lines) + 2) + 0.3
    fig = plt.figure(figsize=(0.083 * width_chars, h))
    fig.patch.set_facecolor("#1E1E1E")
    fig.text(0.01, 1 - 0.12 / h, title, color="#9CDCFE", family="monospace", fontsize=8.5, va="top", weight="bold")
    fig.text(0.01, 1 - 0.36 / h, "\n".join(lines), color="#D4D4D4", family="monospace", fontsize=8, va="top", linespacing=1.32)
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def pick(path, start, stop=None, skip=("Loading", "UNEXPECTED", "classifier.", "LOAD REPORT", "Key   ", "-----", "Notes", "warn", "can be ignored")):
    txt = open(path, encoding="utf-8").read().splitlines()
    out, on = [], False
    for l in txt:
        if start in l:
            on = True
        if on and not any(s in l for s in skip):
            out.append(l)
        if on and stop and stop in l:
            break
    return out


if __name__ == "__main__":
    S = "report/screens"
    render(pick(f"{S}/colab_probe_cell3_0.txt", "system    variant", "done in")[:24],
           "Google Colab (T4 GPU)  |  background_probe.py --fg-db 20  |  printed results", f"{S}/console_colab_probe.png")
    demo = pick(f"{S}/demo_cell1_0.txt", "torch") + pick(f"{S}/demo_cell3_0.txt", "file") + [""] + \
        [l for l in open(f"{S}/demo_cell9_1.txt", encoding="utf-8").read().splitlines() if l.startswith(("trainable", "CNN prediction"))] + \
        [l for l in open(f"{S}/demo_cell11_1.txt", encoding="utf-8").read().splitlines() if l.startswith(("AST:", "combined model", "AST ordinary"))] + [""] + \
        pick(f"{S}/demo_cell13_2.txt", "cnn:")
    render(demo, "LIVE_DL_DEMO.ipynb  |  live inference on real test recordings (laptop CPU)", f"{S}/console_demo.png")
    tr = pick(f"{S}/train_cell6_0.txt", "logits") + [""] + pick(f"{S}/train_cell8_0.txt", "epoch 1")
    render(tr, "LIVE_TRAINING_DEMO.ipynb  |  forward, loss, backward, optimiser step, training loop", f"{S}/console_training.png")
    print("ok")
