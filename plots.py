"""Figures from results.csv written by run_experiments.py.

  python plots.py results.csv --out figures [--budget 500] [--synthetic]

--synthetic stamps every figure so smoke-test output cannot be mistaken for Nonspeech7k results.
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASS_NAMES = ["breath", "cough", "cry", "laugh", "scream", "sneeze", "yawn"]
PROT_COLORS = {"official": "#999999", "random": "#E69F00", "group": "#0072B2"}
SET_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titleweight": "bold", "axes.titlesize": 11})


def stamp(fig, synthetic):
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC DATA\nNOT NONSPEECH7K RESULTS", ha="center", va="center", fontsize=26,
                 color="red", alpha=0.18, rotation=25, weight="bold")


def save(fig, path, synthetic):
    stamp(fig, synthetic)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("wrote", path)


def fig_protocols(df, budget, out, synthetic):
    d = df[(df["mode"] == "random") & (df["budget"] == budget)]
    sets = list(dict.fromkeys(d["featureset"]))
    g = d.groupby(["featureset", "protocol"]).macro_f1.agg(["mean", "std"]).fillna(0)
    prots = [p for p in ["official", "random", "group"] if p in d["protocol"].unique()]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    w = 0.8 / len(prots)
    for i, p in enumerate(prots):
        m = [g.loc[(s, p), "mean"] for s in sets]
        e = [g.loc[(s, p), "std"] for s in sets]
        ax.bar(np.arange(len(sets)) + (i - (len(prots) - 1) / 2) * w, m, w, yerr=e, capsize=2,
               color=PROT_COLORS[p], label={"group": "group (unseen recordings)", "random": "random clips",
                                            "official": "official split"}[p])
    ax.set_xticks(range(len(sets)), sets)
    ax.set_ylabel("macro-F1 (mean +/- std over seeds)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Accuracy by evaluation protocol ({budget} training clips)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    save(fig, os.path.join(out, "1_protocols.png"), synthetic)


def fig_gap(df, out, synthetic):
    d = df[(df["mode"] == "random") & df["protocol"].isin(["random", "group"])]
    t = d.pivot_table(index=["featureset", "budget", "seed"], columns="protocol", values="macro_f1")
    if not {"random", "group"} <= set(t.columns):
        return
    gap = (t["random"] - t["group"]).groupby(["featureset", "budget"]).agg(["mean", "std"]).fillna(0)
    sets = list(dict.fromkeys(d["featureset"]))
    budgets = sorted(d["budget"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 4))
    w = 0.8 / len(budgets)
    for i, b in enumerate(budgets):
        ax.bar(np.arange(len(sets)) + (i - (len(budgets) - 1) / 2) * w, [gap.loc[(s, b), "mean"] for s in sets], w,
               yerr=[gap.loc[(s, b), "std"] for s in sets], capsize=2, color=SET_COLORS[i % 6], label=f"{b} clips")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(sets)), sets)
    ax.set_ylabel("protocol gap = F1(random) - F1(group)")
    ax.set_title("How much accuracy depends on seeing the recording\n(smaller = more robust to unseen recordings)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, os.path.join(out, "2_protocol_gap.png"), synthetic)


def fig_budget(df, out, synthetic):
    d = df[df["protocol"] == "group"]
    if d.empty:
        return
    sets = list(dict.fromkeys(d["featureset"]))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, s in enumerate(sets):
        for mode, ls in [("random", "-"), ("balanced", "--")]:
            g = d[(d["featureset"] == s) & (d["mode"] == mode)].groupby("budget").macro_f1.agg(["mean", "std"]).fillna(0)
            if len(g):
                ax.errorbar(g.index, g["mean"], g["std"], color=SET_COLORS[i % 6], ls=ls, marker="o", ms=4, capsize=2,
                            label=f"{s} ({mode})")
    ax.set_xlabel("training clips")
    ax.set_ylabel("macro-F1 on unseen recordings")
    ax.set_title("Training budget, random clips vs at most k clips per recording")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save(fig, os.path.join(out, "3_budget.png"), synthetic)


def fig_per_class(df, budget, out, synthetic):
    d = df[(df["protocol"] == "group") & (df["mode"] == "random") & (df["budget"] == budget)]
    cols = sorted(c for c in d.columns if c.startswith("f1_c"))
    if d.empty or not cols:
        return
    m = d.groupby("featureset")[cols].mean()
    fig, ax = plt.subplots(figsize=(7.5, 0.7 * len(m) + 1.6))
    im = ax.imshow(m.values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(cols)), [CLASS_NAMES[int(c[4:])] for c in cols])
    ax.set_yticks(range(len(m)), m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if m.values[i, j] > 0.6 else "black", fontsize=8)
    ax.spines[:].set_visible(False)
    ax.set_title(f"Per-class F1 on unseen recordings ({budget} training clips)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    save(fig, os.path.join(out, "4_per_class.png"), synthetic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    df = pd.read_csv(a.results)
    os.makedirs(a.out, exist_ok=True)
    budget = a.budget if a.budget in df["budget"].unique() else int(sorted(df["budget"].unique())[-1])
    fig_protocols(df, budget, a.out, a.synthetic)
    fig_gap(df, a.out, a.synthetic)
    fig_budget(df, a.out, a.synthetic)
    fig_per_class(df, budget, a.out, a.synthetic)


if __name__ == "__main__":
    main()
