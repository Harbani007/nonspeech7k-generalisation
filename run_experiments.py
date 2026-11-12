"""Evaluate feature sets under three protocols and several training budgets.

Protocols
  official  : train on the official train clips, test on the official test clips
  random    : stratified random clip split of the pooled data (recordings leak across the split)
  group     : stratified split disjoint by File_ID (test recordings never seen in training)

For each (protocol, seed) one split is drawn; training clips are then subsampled to `budget`
clips ('random' = random clips per class, 'balanced' = at most --kmax clips per recording).

Feature-set spec: names joined by '+', e.g. struct, struct-nf (drops nf_* columns), spec, ast, ast+struct.
Blocks are standardised on the training subset and scaled by 1/sqrt(dim) so a 25-d block is not
drowned by a 768-d block, then fed to a class-balanced logistic regression.
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

from common import load_feats, load_meta

warnings.filterwarnings("ignore", category=ConvergenceWarning)
CS = [0.3, 3.0, 30.0]


def get_blocks(spec, meta, feat_dir):
    blocks = []
    for tok in spec.split("+"):
        drop_nf = tok.endswith("-nf")
        X, cols = load_feats(feat_dir, meta, tok[:-3] if drop_nf else tok)
        if drop_nf:
            keep = [i for i, c in enumerate(cols) if not c.startswith("nf_")]
            X = X[:, keep]
        blocks.append(X)
    return blocks


def make_split(protocol, meta, seed):
    y, g = meta["y"].values, meta["group"].values
    if protocol == "official":
        return np.where(meta["split"] == "train")[0], np.where(meta["split"] == "test")[0]
    if protocol == "random":
        cv = StratifiedKFold(5, shuffle=True, random_state=seed).split(y, y)
    else:
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(y, y, g)
    return next(iter(cv))


def sample_budget(meta, idx, n, mode, rng, kmax):
    y, g = meta["y"].values, meta["group"].values
    classes = np.unique(y[idx])
    quota = n // len(classes)
    chosen = []
    for c in classes:
        ic = idx[y[idx] == c]
        if mode == "random":
            chosen.extend(rng.choice(ic, min(quota, len(ic)), replace=False))
            continue
        groups = rng.permutation(np.unique(g[ic]))
        pick = []
        for gr in groups:
            members = ic[g[ic] == gr]
            pick.extend(rng.choice(members, min(kmax, len(members)), replace=False))
            if len(pick) >= quota:
                break
        pick = pick[:quota]
        if len(pick) < quota:  # not enough recordings: top up with unused clips
            rest = np.setdiff1d(ic, pick)
            pick.extend(rng.choice(rest, min(quota - len(pick), len(rest)), replace=False))
        chosen.extend(pick)
    return np.array(chosen)


def prep(blocks_tr, blocks_te):
    Atr, Ate = [], []
    for A, B in zip(blocks_tr, blocks_te):
        mu, sd = A.mean(0), A.std(0) + 1e-6
        s = np.sqrt(A.shape[1])
        Atr.append((A - mu) / sd / s)
        Ate.append((B - mu) / sd / s)
    return np.hstack(Atr), np.hstack(Ate)


def fit_predict(Xtr, ytr, gtr, Xte, inner_cv):
    best_c = 3.0
    if inner_cv and len(np.unique(gtr)) >= 3:
        folds = list(GroupKFold(3).split(Xtr, ytr, gtr))
        scores = {}
        for c in CS:
            s = []
            for a, b in folds:
                m = LogisticRegression(C=c, max_iter=1000, class_weight="balanced").fit(Xtr[a], ytr[a])
                s.append(f1_score(ytr[b], m.predict(Xtr[b]), average="macro"))
            scores[c] = np.mean(s)
        best_c = max(scores, key=scores.get)
    m = LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    return m.predict(Xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-train", required=True)
    ap.add_argument("--meta-test", required=True)
    ap.add_argument("--feats", default="feats")
    ap.add_argument("--sets", nargs="+", default=["spec", "struct", "struct-nf"])
    ap.add_argument("--protocols", nargs="+", default=["official", "random", "group"])
    ap.add_argument("--budgets", nargs="+", type=int, default=[300, 500, 700])
    ap.add_argument("--modes", nargs="+", default=["random", "balanced"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--kmax", type=int, default=3)
    ap.add_argument("--no-inner-cv", action="store_true")
    ap.add_argument("--out", default="results.csv")
    a = ap.parse_args()

    meta = load_meta(a.meta_train, a.meta_test)
    blocks = {s: get_blocks(s, meta, a.feats) for s in a.sets}
    y, g = meta["y"].values, meta["group"].values
    classes = np.unique(y)
    rows = []
    for protocol in a.protocols:
        for seed in range(a.seeds):
            tr_all, te = make_split(protocol, meta, seed)
            for budget in a.budgets:
                for mode in a.modes:
                    rng = np.random.default_rng(seed * 100003 + budget)
                    tr = sample_budget(meta, tr_all, budget, mode, rng, a.kmax)
                    for s in a.sets:
                        Xtr, Xte = prep([b[tr] for b in blocks[s]], [b[te] for b in blocks[s]])
                        pred = fit_predict(Xtr, y[tr], g[tr], Xte, not a.no_inner_cv)
                        per_class = f1_score(y[te], pred, average=None, labels=classes)
                        rows.append(dict(
                            protocol=protocol, seed=seed, budget=budget, mode=mode, featureset=s,
                            **{f"f1_c{c}": v for c, v in zip(classes, per_class)},
                            macro_f1=f1_score(y[te], pred, average="macro"),
                            bal_acc=balanced_accuracy_score(y[te], pred),
                            n_train=len(tr), n_train_recordings=len(np.unique(g[tr])),
                            n_test=len(te), test_recordings_seen=int(np.isin(g[te], g[tr]).mean() * 100)))
            print(f"done {protocol} seed {seed}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    pd.set_option("display.width", 200)
    r = df[df["mode"] == "random"]
    print("\nmacro-F1 (mean over seeds), mode=random")
    print(r.pivot_table(index=["featureset", "budget"], columns="protocol", values="macro_f1").round(3))
    if {"random", "group"} <= set(a.protocols):
        t = r.pivot_table(index=["featureset", "budget"], columns="protocol", values="macro_f1")
        print("\nprotocol gap (random - group), smaller = more robust to unseen recordings")
        print((t["random"] - t["group"]).round(3).unstack("budget"))
    print("\nmacro-F1 under group protocol: random vs recording-balanced sampling")
    print(df[df["protocol"] == "group"].pivot_table(index=["featureset", "budget"], columns="mode",
                                                    values="macro_f1").round(3))
    print("\nmean distinct training recordings per budget/mode (group protocol)")
    print(df[df["protocol"] == "group"].pivot_table(index="budget", columns="mode",
                                                    values="n_train_recordings").round(1))


if __name__ == "__main__":
    main()
