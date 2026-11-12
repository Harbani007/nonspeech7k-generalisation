"""Recording-identity probe: how well can each feature set tell *which recording* a clip came from,
among recordings of the same class?  Higher accuracy above chance = the features carry more recording
identity, i.e. more room for a model to cheat on a random clip split.

  python probe_source.py --meta-train ... --meta-test ... --feats feats --sets spec struct [ast] --out probe.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import load_feats, load_meta

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-train", required=True)
    ap.add_argument("--meta-test", required=True)
    ap.add_argument("--feats", default="feats")
    ap.add_argument("--sets", nargs="+", default=["spec", "struct"])
    ap.add_argument("--min-clips", type=int, default=20, help="only recordings with at least this many clips")
    ap.add_argument("--max-recs", type=int, default=15, help="at most this many recordings per class (largest first)")
    ap.add_argument("--out", default="probe.csv")
    a = ap.parse_args()

    meta = load_meta(a.meta_train, a.meta_test)
    rows = []
    for s in a.sets:
        X, _ = load_feats(a.feats, meta, s)
        for c, sub in meta.groupby("y"):
            counts = sub["group"].value_counts()
            keep = counts[counts >= a.min_clips].index[:a.max_recs]
            if len(keep) < 3:
                continue
            idx = sub.index[sub["group"].isin(keep)].values
            g = meta.loc[idx, "group"].values
            Xi = X[idx]
            accs = []
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xi, g):
                sc = StandardScaler().fit(Xi[tr])
                m = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(Xi[tr]), g[tr])
                accs.append((m.predict(sc.transform(Xi[te])) == g[te]).mean())
            rows.append(dict(featureset=s, y=int(c), n_recordings=len(keep), n_clips=len(idx),
                             acc=float(np.mean(accs)), chance=1.0 / len(keep)))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    if df.empty:
        print("no class has >= 3 recordings with enough clips; lower --min-clips")
        return
    df["lift"] = df["acc"] - df["chance"]
    print("Recording-identity probe (within class). lift = accuracy - chance; higher = more recording identity in the features")
    print(df.groupby("featureset")[["acc", "chance", "lift"]].mean().round(3))


if __name__ == "__main__":
    main()
