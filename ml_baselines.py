"""Traditional-ML baselines on exactly the deep-learning splits (dl_data.py), written in the same run format.

Features (cached by extract.py): spec (35 handcrafted spectral), struct-nf (structure, noise-floor columns
dropped), ast (frozen AST, 768-d pooled embedding of the 10.24 s padded input). Classifier: class-balanced
logistic regression, C chosen by grouped 3-fold CV on the training partition (run_experiments.fit_predict).
Standardisation is fitted on the training partition only. The val partition is not used (no epochs to select).

  python ml_baselines.py --data results/v3/dl --feats results/feats --runs results/v3/runs
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from dl_data import CLASSES
from dl_metrics import metrics
from run_experiments import fit_predict, get_blocks, prep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--sets", nargs="+", default=["spec", "struct-nf", "ast"])
    ap.add_argument("--protocols", nargs="+", default=["source_disjoint", "official"])
    a = ap.parse_args()
    sp = pd.read_csv(os.path.join(a.data, "splits.csv"))
    y, g = sp.y.values, sp.group.values
    for proto in a.protocols:
        tr, te = np.where(sp[proto] == "train")[0], np.where(sp[proto] == "test")[0]
        for s in a.sets:
            out = os.path.join(a.runs, f"ml_{s}_{proto}")
            os.makedirs(out, exist_ok=True)
            blocks = get_blocks(s, sp, a.feats)
            Xtr, Xte = prep([b[tr] for b in blocks], [b[te] for b in blocks])
            pred = fit_predict(Xtr, y[tr], g[tr], Xte, True)
            res = metrics(y[te], pred, len(CLASSES))
            res.update(dict(featureset=s, protocol=proto, n_train=len(tr), n_test=len(te)))
            json.dump(res, open(os.path.join(out, "test_metrics.json"), "w"), indent=1)
            p = sp.iloc[te][["filename", "y", "cls", "group", "source"]].reset_index(drop=True)
            p["pred"] = pred
            p.to_csv(os.path.join(out, "test_predictions.csv"), index=False)
            print(f"{proto:16s} {s:10s} acc {res['accuracy']:.3f} macroF1 {res['macro_f1']:.3f} "
                  f"bal_acc {res['balanced_accuracy']:.3f}", flush=True)


if __name__ == "__main__":
    main()
