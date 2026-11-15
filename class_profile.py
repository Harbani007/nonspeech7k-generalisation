"""What the structure features 'see' in each sound class: per-class feature means as a heatmap.

  python class_profile.py --meta-train ... --meta-test ... --feats feats --out figures [--synthetic]
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import load_feats, load_meta  # noqa: E402
from plots import CLASS_NAMES, stamp  # noqa: E402

KEY = ["burst_per_s", "burst_n", "burst_ibi_cv", "burst_attack_s", "burst_decay_s", "acf_peak",
       "dur_active_s", "dur_active_frac", "env_flatness", "env_tcentroid", "mod_slow", "mod_mid", "mod_fast"]
LABELS = {"burst_per_s": "bursts / s", "burst_n": "burst count", "burst_ibi_cv": "gap irregularity",
          "burst_attack_s": "attack time", "burst_decay_s": "decay time", "acf_peak": "rhythm strength",
          "dur_active_s": "active duration", "dur_active_frac": "active fraction", "env_flatness": "envelope flatness",
          "env_tcentroid": "energy centroid (time)", "mod_slow": "modulation 0.5-2 Hz", "mod_mid": "modulation 2-4 Hz",
          "mod_fast": "modulation 4-16 Hz"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-train", required=True)
    ap.add_argument("--meta-test", required=True)
    ap.add_argument("--feats", default="feats")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    meta = load_meta(a.meta_train, a.meta_test)
    X, cols = load_feats(a.feats, meta, "struct")
    df = pd.DataFrame(X, columns=cols)

    def band_sum(*ranges):
        use = [c for c in cols if c.startswith("mod_b") and any(c.endswith(f"_{r}hz") for r in ranges)]
        return df[use].mean(axis=1)

    df["mod_slow"] = band_sum("0.5-1", "1-2")
    df["mod_mid"] = band_sum("2-4")
    df["mod_fast"] = band_sum("4-8", "8-16")
    df["y"] = meta["y"].values
    means = df.groupby("y")[KEY].mean()
    os.makedirs(a.out, exist_ok=True)
    means.index = [CLASS_NAMES[i] for i in means.index]
    means.to_csv(os.path.join(a.out, "class_profile.csv"))
    z = (means - means.mean()) / (means.std() + 1e-12)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(KEY)), [LABELS[k] for k in KEY], rotation=40, ha="right")
    ax.set_yticks(range(len(z)), z.index)
    ax.spines[:].set_visible(False)
    ax.set_title("What the structure features see in each class\n(class mean, standardised across classes)")
    fig.colorbar(im, ax=ax, fraction=0.03, label="std. devs from average class")
    stamp(fig, a.synthetic)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "5_class_profile.png"), dpi=160)
    print("wrote", os.path.join(a.out, "5_class_profile.png"))


if __name__ == "__main__":
    main()
