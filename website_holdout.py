"""Leave-one-website-out test: does accuracy drop when a website is absent from training?

For each held-out website W, its recordings are split into 5 folds grouped by File_ID. Each test fold is
classified twice, by models trained on the same number of clips per class:
  holdout : training clips from the OTHER websites only
  control : training clips from the other websites AND the remaining recordings of W (source-matched)
Both see only recordings disjoint from the test fold, so the gap (control - holdout) measures what is lost by
never having heard website W, not by never having heard the test recordings.

Only classes with at least --min-recs recordings in W (and in the other websites) are scored; the rest are
listed as untestable. 95% intervals come from a bootstrap over test recordings, paired between conditions.

  python website_holdout.py --meta-train ... --meta-test ... --feats feats --sets spec struct-nf ast
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from common import load_meta
from run_experiments import fit_predict, get_blocks, prep

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def sample(idx, y, cap, rng, quota=None):
    out, quota = [], quota or {}
    for c in np.unique(y[idx]):
        ic = idx[y[idx] == c]
        out.extend(rng.choice(ic, min(quota.get(c, cap), len(ic)), replace=False))
    return np.array(out)


def boot_gap(yt, pc, ph, rec, labels, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    recs = np.unique(rec)
    pos = {r: np.where(rec == r)[0] for r in recs}
    gaps = []
    for _ in range(n):
        i = np.concatenate([pos[r] for r in rng.choice(recs, len(recs))])
        gaps.append(f1_score(yt[i], pc[i], labels=labels, average="macro")
                    - f1_score(yt[i], ph[i], labels=labels, average="macro"))
    return np.percentile(gaps, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-train", required=True)
    ap.add_argument("--meta-test", required=True)
    ap.add_argument("--feats", default="feats")
    ap.add_argument("--sets", nargs="+", default=["spec", "struct-nf"])
    ap.add_argument("--cap", type=int, default=100, help="max training clips per class")
    ap.add_argument("--min-recs", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="website_holdout.csv")
    a = ap.parse_args()

    meta = load_meta(a.meta_train, a.meta_test)  # pooled: the official split plays no role here
    y, g, src = meta["y"].values, meta["group"].values, meta["source"].values
    names = meta.groupby("y")["cls"].first()
    recs = meta.groupby(["source", "y"])["group"].nunique()
    blocks = {s: get_blocks(s, meta, a.feats) for s in a.sets}
    rows, per_class = [], []
    for w in sorted(set(src)):
        other = np.where(src != w)[0]
        ok = [c for c in names.index if recs.get((w, c), 0) >= a.min_recs
              and meta.iloc[other].query("y == @c")["group"].nunique() >= a.min_recs]
        skipped = [names[c] for c in names.index if c not in ok]
        print(f"\nheld-out website {w}: scored classes {[names[c] for c in ok]}; untestable {skipped}")
        if not ok:
            continue
        test_pool = np.where((src == w) & np.isin(y, ok))[0]
        n_folds = min(5, a.min_recs)
        for seed in range(a.seeds):
            rng = np.random.default_rng(seed)
            pred = {(s, k): np.empty(len(test_pool), int) for s in a.sets for k in ("control", "holdout")}
            perm = rng.permutation(np.unique(g[test_pool]))
            fold_of = {r: i % n_folds for i, r in enumerate(perm)}
            folds = np.array([fold_of[r] for r in g[test_pool]])
            for f in range(n_folds):
                te = test_pool[folds == f]
                keep = ~np.isin(g, g[te])  # never train on a test recording, even across websites
                hold_pool = np.where(keep & (src != w))[0]
                ctrl_pool = np.where(keep)[0]  # other websites + W's recordings outside this test fold
                tr_h = sample(hold_pool, y, a.cap, rng)
                quota = {c: int((y[tr_h] == c).sum()) for c in np.unique(y[tr_h])}
                tr_c = sample(ctrl_pool[np.isin(y[ctrl_pool], list(quota))], y, a.cap, rng, quota)
                for s in a.sets:
                    for k, tr in (("holdout", tr_h), ("control", tr_c)):
                        Xtr, Xte = prep([b[tr] for b in blocks[s]], [b[te] for b in blocks[s]])
                        pred[(s, k)][folds == f] = fit_predict(Xtr, y[tr], g[tr], Xte, True)
            yt, rt = y[test_pool], g[test_pool]
            for s in a.sets:
                pc, ph = pred[(s, "control")], pred[(s, "holdout")]
                lo, hi = boot_gap(yt, pc, ph, rt, ok, seed=seed)
                rows.append(dict(heldout=w, featureset=s, seed=seed, n_test=len(test_pool),
                                 n_test_recordings=len(np.unique(rt)), classes=len(ok),
                                 control=f1_score(yt, pc, labels=ok, average="macro"),
                                 holdout=f1_score(yt, ph, labels=ok, average="macro"), ci_lo=lo, ci_hi=hi))
                for c in ok:
                    m = yt == c
                    per_class.append(dict(heldout=w, featureset=s, seed=seed, cls=names[c],
                                          control_recall=(pc[m] == c).mean(), holdout_recall=(ph[m] == c).mean()))
            print(f"  seed {seed} done", flush=True)

    df = pd.DataFrame(rows)
    df["gap"] = df["control"] - df["holdout"]
    df.to_csv(a.out, index=False)
    pc_df = pd.DataFrame(per_class)
    pc_df.to_csv(a.out.replace(".csv", "_per_class.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\nmacro-F1 over scored classes, mean over seeds. gap = control - holdout; CI = mean of per-seed "
          "recording-bootstrap 95% intervals")
    print(df.groupby(["heldout", "featureset"])[["control", "holdout", "gap", "ci_lo", "ci_hi"]].mean().round(3))
    print("\nper-class recall drop (control - holdout), mean over seeds")
    pc_df["drop"] = pc_df["control_recall"] - pc_df["holdout_recall"]
    print(pc_df.pivot_table(index=["heldout", "cls"], columns="featureset", values="drop").round(2))


if __name__ == "__main__":
    main()
